#!/usr/bin/env python
"""
detect_and_cache.py

Phase 1 of the temporal actor track pipeline.
Runs YOLOv8s person detection on required frames per video and saves
one detection cache .pkl per video.

Supports:
  - Intelligent frame selection (detect only required or all, based on coverage)
  - Resume (skip videos with existing cache)
  - --force_redetect to override resume
  - Atomic writes to prevent corruption on crash

Cache format per video:
  dict[int, list[tuple[float, float, float, float, float]]]
  Keys   = 1-based frame indices
  Values = list of (x1, y1, x2, y2, conf) tuples, normalized coords [0,1]
"""

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 1: Per-video person detection and caching."
    )
    p.add_argument("--split", required=True, choices=["train", "val"],
                   help="Dataset split to process.")
    p.add_argument("--actor_csv", required=True, type=str,
                   help="Path to processed actor CSV.")
    p.add_argument("--frame_root", required=True, type=str,
                   help="Root directory containing per-video frame folders.")
    p.add_argument("--cache_dir", required=True, type=str,
                   help="Output directory for per-video detection cache files.")
    p.add_argument("--clip_len", type=int, default=8,
                   help="Temporal clip length T (default: 8).")
    p.add_argument("--clip_stride", type=int, default=1,
                   help="Temporal stride within clip (default: 1).")
    p.add_argument("--det_model", type=str, default="yolov8s.pt",
                   help="YOLO model name or path (default: yolov8s.pt).")
    p.add_argument("--det_imgsz", type=int, default=256,
                   help="Detection input resolution (default: 256).")
    p.add_argument("--det_conf_thresh", type=float, default=0.3,
                   help="Minimum detection confidence (default: 0.3).")
    p.add_argument("--det_batch_size", type=int, default=64,
                   help="Frames per model.predict() call (default: 64).")
    p.add_argument("--coverage_thresh", type=float, default=0.95,
                   help="If required-frame coverage >= this, detect ALL "
                        "frames in that video for simplicity (default: 0.95).")
    p.add_argument("--force_detect_all", action="store_true",
                   help="Always detect all frames, ignoring coverage logic.")
    p.add_argument("--force_redetect", action="store_true",
                   help="Re-run detection even if cache .pkl already exists.")
    p.add_argument("--log_file", type=str, default=None,
                   help="Optional path to write log output to a file.")
    return p.parse_args()


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging to console and optionally to a file."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ──────────────────────────────────────────────
# Actor CSV loading
# ──────────────────────────────────────────────

def load_actor_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the processed actor CSV.

    Handles both with-header and headerless formats by checking
    whether expected column names appear in the first row.

    Returns:
        DataFrame with at least 'video_id' and 'center_frame' columns.
    """
    col_names = (
        ["sample_id", "video_id", "timestamp", "person_id",
         "x1", "y1", "x2", "y2", "center_frame"]
        + [f"label_{i}" for i in range(1, 81)]
    )
    # Peek at first row to detect header
    df_peek = pd.read_csv(csv_path, nrows=1)
    if "video_id" in df_peek.columns and "center_frame" in df_peek.columns:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=col_names)

    logger.info(f"Loaded actor CSV: {len(df)} actor instances from {csv_path}")
    return df


# ──────────────────────────────────────────────
# Required-frame computation
# ──────────────────────────────────────────────

def compute_required_frames_per_video(
    df: pd.DataFrame,
    clip_len: int,
    clip_stride: int,
) -> Dict[str, Set[int]]:
    """
    For each video, compute the union of all frame indices needed
    across every actor instance's temporal clip window.

    Frame indices are NOT clamped here because max_frame is unknown
    at this stage.  Clamping is applied per-video in main().

    Returns:
        dict mapping video_id -> set of (unclamped) frame indices.
    """
    half = clip_len // 2
    required: Dict[str, Set[int]] = {}

    for video_id, group in df.groupby("video_id"):
        frames: Set[int] = set()
        for cf in group["center_frame"].unique():
            cf_int = int(cf)
            for i in range(clip_len):
                fi = cf_int + (i - half) * clip_stride
                frames.add(fi)
        required[str(video_id)] = frames

    return required


# ──────────────────────────────────────────────
# Frame counting
# ──────────────────────────────────────────────

def get_max_frame_count(video_dir: Path) -> int:
    """
    Count .jpg files in a video directory.

    Returns the count, which equals the max 1-based frame index
    (filenames are 000001.jpg … NNNNNN.jpg with no gaps).
    """
    return sum(1 for f in os.listdir(video_dir) if f.endswith(".jpg"))


# ──────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────

def detect_video_frames(
    model,
    frame_paths: List[str],
    frame_indices: List[int],
    det_imgsz: int,
    det_conf_thresh: float,
    det_batch_size: int,
) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    """
    Run YOLOv8 person detection on a list of frames.

    Args:
        model:           Loaded YOLO model instance.
        frame_paths:     Ordered list of frame file paths (strings).
        frame_indices:   Corresponding 1-based frame indices (same length).
        det_imgsz:       Input resolution for YOLO.
        det_conf_thresh: Confidence threshold (already set in predict, but
                         passed here for documentation clarity).
        det_batch_size:  Number of paths per model.predict() call.

    Returns:
        dict mapping frame_index -> list of (x1, y1, x2, y2, conf).
        Coordinates are normalized to [0, 1].
        Frames with no person detections map to an empty list.
    """
    cache: Dict[int, List[Tuple[float, float, float, float, float]]] = {}

    for batch_start in range(0, len(frame_paths), det_batch_size):
        batch_end = min(batch_start + det_batch_size, len(frame_paths))
        batch_paths = frame_paths[batch_start:batch_end]
        batch_indices = frame_indices[batch_start:batch_end]

        results = model.predict(
            source=batch_paths,
            imgsz=det_imgsz,
            conf=det_conf_thresh,
            classes=[0],       # COCO person class
            verbose=False,
            device=0,
        )

        for fi, result in zip(batch_indices, results):
            boxes_norm = result.boxes.xyxyn.cpu().tolist()   # [[x1,y1,x2,y2],…]
            confs = result.boxes.conf.cpu().tolist()          # [conf, …]
            detections = [
                (b[0], b[1], b[2], b[3], c)
                for b, c in zip(boxes_norm, confs)
            ]
            cache[fi] = detections

    return cache


# ──────────────────────────────────────────────
# Atomic save
# ──────────────────────────────────────────────

def save_cache_atomic(cache: dict, cache_path: Path) -> None:
    """
    Pickle the detection cache with an atomic write.

    Writes to a .tmp.pkl first, then renames.  If the process crashes
    mid-write, only the .tmp.pkl is left (ignored by resume logic).
    """
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(tmp_path), str(cache_path))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    """Run Phase 1: detect persons in video frames, cache results to disk."""
    args = parse_args()
    setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info(f"Phase 1: Detection caching — split={args.split}")
    logger.info(f"  Actor CSV       : {args.actor_csv}")
    logger.info(f"  Frame root      : {args.frame_root}")
    logger.info(f"  Cache dir       : {args.cache_dir}")
    logger.info(f"  Model           : {args.det_model}")
    logger.info(f"  imgsz           : {args.det_imgsz}")
    logger.info(f"  conf_thresh     : {args.det_conf_thresh}")
    logger.info(f"  batch_size      : {args.det_batch_size}")
    logger.info(f"  clip_len        : {args.clip_len}")
    logger.info(f"  clip_stride     : {args.clip_stride}")
    logger.info(f"  coverage_thresh : {args.coverage_thresh}")
    logger.info(f"  force_detect_all: {args.force_detect_all}")
    logger.info(f"  force_redetect  : {args.force_redetect}")
    logger.info("=" * 60)

    # ── Setup paths ──
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_root = Path(args.frame_root)

    # ── Load actor CSV ──
    df = load_actor_csv(args.actor_csv)

    # ── Compute required frames per video ──
    required_frames = compute_required_frames_per_video(
        df, args.clip_len, args.clip_stride
    )

    # ── Enumerate videos ──
    video_ids = sorted(df["video_id"].unique().astype(str))
    total_videos = len(video_ids)
    logger.info(f"Total videos in split: {total_videos}")

    # ── Load YOLO model (downloads weights on first run) ──
    logger.info(f"Loading YOLO model: {args.det_model} ...")
    from ultralytics import YOLO
    model = YOLO(args.det_model)
    logger.info("YOLO model loaded.")

    # ── Counters ──
    videos_processed = 0
    videos_skipped = 0
    videos_failed = 0
    total_frames_detected = 0
    total_detection_time = 0.0

    # ── Per-video loop ──
    for idx, video_id in enumerate(video_ids, 1):
        cache_path = cache_dir / f"{video_id}.pkl"

        # Resume: skip if cache already exists
        if cache_path.exists() and not args.force_redetect:
            logger.info(
                f"[{idx}/{total_videos}] Skipping {video_id}: cache exists"
            )
            videos_skipped += 1
            continue

        # Locate video frame directory
        video_dir = frame_root / video_id
        if not video_dir.is_dir():
            logger.warning(
                f"[{idx}/{total_videos}] Directory not found: {video_dir}"
            )
            videos_failed += 1
            continue

        # Count frames in this video
        max_frame = get_max_frame_count(video_dir)
        if max_frame == 0:
            logger.warning(
                f"[{idx}/{total_videos}] No .jpg frames in {video_dir}"
            )
            videos_failed += 1
            continue

        # Determine which frames to detect
        raw_required = required_frames.get(video_id, set())
        clamped_required = {max(1, min(f, max_frame)) for f in raw_required}
        coverage = len(clamped_required) / max_frame

        if args.force_detect_all or coverage >= args.coverage_thresh:
            frames_to_detect = list(range(1, max_frame + 1))
            detect_mode = "all"
        else:
            frames_to_detect = sorted(clamped_required)
            detect_mode = "required_only"

        n_detect = len(frames_to_detect)
        logger.info(
            f"[{idx}/{total_videos}] {video_id}: "
            f"{n_detect} frames ({detect_mode}), "
            f"required={len(clamped_required)}, "
            f"coverage={coverage:.3f}, "
            f"max_frame={max_frame}"
        )

        # Build ordered frame paths
        frame_paths = [
            str(video_dir / f"{fi:06d}.jpg") for fi in frames_to_detect
        ]

        # Run detection
        try:
            t_start = time.time()
            cache = detect_video_frames(
                model=model,
                frame_paths=frame_paths,
                frame_indices=frames_to_detect,
                det_imgsz=args.det_imgsz,
                det_conf_thresh=args.det_conf_thresh,
                det_batch_size=args.det_batch_size,
            )
            t_elapsed = time.time() - t_start
        except Exception as e:
            logger.error(
                f"[{idx}/{total_videos}] FAILED {video_id}: {e}",
                exc_info=True,
            )
            videos_failed += 1
            continue

        # Save atomically
        save_cache_atomic(cache, cache_path)

        # Per-video statistics
        total_dets = sum(len(dets) for dets in cache.values())
        avg_dets = total_dets / max(n_detect, 1)
        fps = n_detect / max(t_elapsed, 1e-9)

        logger.info(
            f"  -> Done: {t_elapsed:.1f}s, {fps:.0f} FPS, "
            f"{total_dets} total dets, avg {avg_dets:.1f} dets/frame"
        )

        videos_processed += 1
        total_frames_detected += n_detect
        total_detection_time += t_elapsed

    # ── Final summary ──
    logger.info("=" * 60)
    logger.info("Phase 1 COMPLETE")
    logger.info(f"  Split              : {args.split}")
    logger.info(f"  Videos processed   : {videos_processed}")
    logger.info(f"  Videos skipped     : {videos_skipped} (cache existed)")
    logger.info(f"  Videos failed      : {videos_failed}")
    logger.info(f"  Total frames       : {total_frames_detected}")
    total_min = total_detection_time / 60.0
    logger.info(f"  Total time         : {total_detection_time:.1f}s "
                f"({total_min:.1f} min)")
    if total_detection_time > 0:
        avg_fps = total_frames_detected / total_detection_time
        logger.info(f"  Average throughput : {avg_fps:.0f} FPS")
    logger.info(f"  Cache directory    : {cache_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()