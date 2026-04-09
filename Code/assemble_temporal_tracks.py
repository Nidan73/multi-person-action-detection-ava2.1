#!/usr/bin/env python
"""
assemble_temporal_tracks.py

Phase 2 of the temporal actor track pipeline.
Reads actor CSV and per-video detection caches from detect_and_cache.py,
builds T=8 temporal tracks for each actor instance, and produces:
  - Per-video sidecar .pkl (list of track records)
  - Merged split-level sidecar .pkl (dict keyed by sample_id)
  - Summary .json with mandatory statistics

Track building per (video_id, timestamp) group:
  1. GT box at center frame (always)
  2. Outward propagation with Hungarian matching
  3. Gap filling: interpolation (both sides) or repeat (one side)
"""

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 2: Temporal track assembly from detection caches."
    )
    p.add_argument("--split", required=True, choices=["train", "val"],
                   help="Dataset split.")
    p.add_argument("--actor_csv", required=True, type=str,
                   help="Path to processed actor CSV.")
    p.add_argument("--frame_root", required=True, type=str,
                   help="Root directory of per-video frame folders.")
    p.add_argument("--cache_dir", required=True, type=str,
                   help="Detection cache directory from detect_and_cache.py.")
    p.add_argument("--sidecar_dir", required=True, type=str,
                   help="Output directory for per-video sidecar .pkl files.")
    p.add_argument("--output_pkl", required=True, type=str,
                   help="Path for merged split-level sidecar .pkl.")
    p.add_argument("--output_summary", required=True, type=str,
                   help="Path for summary .json.")
    p.add_argument("--clip_len", type=int, default=8,
                   help="Temporal clip length T (default: 8).")
    p.add_argument("--clip_stride", type=int, default=1,
                   help="Temporal stride within clip (default: 1).")
    p.add_argument("--assoc_iou_thresh", type=float, default=0.3,
                   help="Min IoU for cross-frame association (default: 0.3).")
    p.add_argument("--center_iou_thresh", type=float, default=0.5,
                   help="IoU threshold for center-frame diagnostic (default: 0.5).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-video sidecars.")
    p.add_argument("--log_file", type=str, default=None,
                   help="Optional log file path.")
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
# Data loading
# ──────────────────────────────────────────────

def load_actor_csv(csv_path: str) -> pd.DataFrame:
    """Load processed actor CSV with header auto-detection."""
    col_names = (
        ["sample_id", "video_id", "timestamp", "person_id",
         "x1", "y1", "x2", "y2", "center_frame"]
        + [f"label_{i}" for i in range(1, 81)]
    )
    df_peek = pd.read_csv(csv_path, nrows=1)
    if "video_id" in df_peek.columns and "center_frame" in df_peek.columns:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=col_names)
    logger.info(f"Loaded actor CSV: {len(df)} instances from {csv_path}")
    return df


def load_detection_cache(cache_path: Path) -> Dict[int, List[Tuple]]:
    """Load a per-video detection cache .pkl."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_max_frame_count(video_dir: Path) -> int:
    """Count .jpg files in a video directory."""
    return sum(1 for f in os.listdir(video_dir) if f.endswith(".jpg"))


# ──────────────────────────────────────────────
# IoU
# ──────────────────────────────────────────────

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes (normalized coords)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / union


# ──────────────────────────────────────────────
# Hungarian matching
# ──────────────────────────────────────────────

def hungarian_match(
    ref_boxes: List[List[float]],
    detections: List[Tuple[float, float, float, float, float]],
    iou_thresh: float,
) -> Dict[int, Tuple[List[float], float, float]]:
    """
    Match actor reference boxes to detections via Hungarian algorithm.

    Args:
        ref_boxes:  [n_actors] list of [x1,y1,x2,y2] reference boxes.
        detections: [n_dets] list of (x1,y1,x2,y2,conf) tuples.
        iou_thresh: Minimum IoU to accept a match.

    Returns:
        Dict mapping actor_index -> (matched_box, confidence, iou).
        Unmatched actors are absent.
    """
    n_actors = len(ref_boxes)
    n_dets = len(detections)

    if n_actors == 0 or n_dets == 0:
        return {}

    # Build IoU matrix
    iou_matrix = np.zeros((n_actors, n_dets), dtype=np.float64)
    for i, ref in enumerate(ref_boxes):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = compute_iou(ref, list(det[:4]))

    # Cost = 1 - IoU  (minimize cost = maximize IoU)
    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments: Dict[int, Tuple[List[float], float, float]] = {}
    for r, c in zip(row_ind, col_ind):
        iou_val = float(iou_matrix[r, c])
        if iou_val >= iou_thresh:
            det = detections[c]
            assignments[r] = (list(det[:4]), float(det[4]), iou_val)

    return assignments


# ──────────────────────────────────────────────
# Gap filling
# ──────────────────────────────────────────────

def fill_gaps(
    boxes: List[Optional[List[float]]],
    sources: List[Optional[str]],
    valid_mask: List[int],
    det_scores: List[float],
    assoc_ious: List[float],
) -> None:
    """
    Fill None positions with interpolation or nearest-repeat.

    Uses only originally-observed positions (gt / det) as anchors.
    Modifies all lists in place.
    """
    T = len(boxes)

    # Indices that have real observations (not None)
    anchors = [i for i in range(T) if boxes[i] is not None]

    for pos in range(T):
        if boxes[pos] is not None:
            continue

        # Nearest anchor on the left
        left: Optional[int] = None
        for a in reversed(anchors):
            if a < pos:
                left = a
                break

        # Nearest anchor on the right
        right: Optional[int] = None
        for a in anchors:
            if a > pos:
                right = a
                break

        if left is not None and right is not None:
            # Linear interpolation per coordinate
            alpha = (pos - left) / (right - left)
            box = [
                boxes[left][c] + (boxes[right][c] - boxes[left][c]) * alpha
                for c in range(4)
            ]
            boxes[pos] = box
            sources[pos] = "interp"
        elif left is not None:
            boxes[pos] = list(boxes[left])
            sources[pos] = "repeat"
        elif right is not None:
            boxes[pos] = list(boxes[right])
            sources[pos] = "repeat"
        else:
            # Should never happen — center is always an anchor
            raise RuntimeError(f"No anchor box found for position {pos}")

        valid_mask[pos] = 0
        det_scores[pos] = 0.0
        assoc_ious[pos] = 0.0


# ──────────────────────────────────────────────
# Track building — per timestamp group
# ──────────────────────────────────────────────

def build_tracks_for_timestamp_group(
    group_df: pd.DataFrame,
    det_cache: Dict[int, List[Tuple]],
    clip_len: int,
    clip_stride: int,
    max_frame: int,
    assoc_iou_thresh: float,
) -> List[Dict[str, Any]]:
    """
    Build temporal tracks for all actors sharing (video_id, timestamp).

    All actors share the same center_frame and frame window.
    Hungarian matching ensures one-to-one detection assignment.
    """
    center_idx = clip_len // 2
    center_frame = int(group_df.iloc[0]["center_frame"])

    # ── Frame indices with clamping ──
    frame_indices: List[int] = []
    clamped: List[bool] = []
    for i in range(clip_len):
        fi = center_frame + (i - center_idx) * clip_stride
        is_clamped = (fi < 1 or fi > max_frame)
        fi = max(1, min(fi, max_frame))
        frame_indices.append(fi)
        clamped.append(is_clamped)

    # ── Actor data ──
    actor_rows = group_df.to_dict("records")
    n_actors = len(actor_rows)

    # Per-actor arrays (initialised to None / zero)
    all_boxes: List[List[Optional[List[float]]]] = [
        [None] * clip_len for _ in range(n_actors)
    ]
    all_sources: List[List[Optional[str]]] = [
        [None] * clip_len for _ in range(n_actors)
    ]
    all_valid: List[List[int]] = [
        [0] * clip_len for _ in range(n_actors)
    ]
    all_det_scores: List[List[float]] = [
        [0.0] * clip_len for _ in range(n_actors)
    ]
    all_assoc_ious: List[List[float]] = [
        [0.0] * clip_len for _ in range(n_actors)
    ]

    # ── Centre frame: always GT ──
    gt_boxes: List[List[float]] = []
    for a_idx, row in enumerate(actor_rows):
        gt_box = [
            float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"]),
        ]
        gt_boxes.append(gt_box)
        all_boxes[a_idx][center_idx] = gt_box
        all_sources[a_idx][center_idx] = "gt"
        all_valid[a_idx][center_idx] = 1
        all_det_scores[a_idx][center_idx] = 1.0
        all_assoc_ious[a_idx][center_idx] = 0.0  # not a det match

    # ── Centre detection IoU (diagnostic) ──
    center_dets = det_cache.get(frame_indices[center_idx], [])
    center_det_ious: List[float] = []
    for gt_box in gt_boxes:
        if center_dets:
            best = max(compute_iou(gt_box, list(d[:4])) for d in center_dets)
        else:
            best = 0.0
        center_det_ious.append(best)

    # ── Propagate BACKWARD (center-1 → 0) ──
    ref_bwd = [list(gt) for gt in gt_boxes]          # start from GT
    for pos in range(center_idx - 1, -1, -1):
        if clamped[pos]:
            continue
        dets = det_cache.get(frame_indices[pos], [])
        assignments = hungarian_match(ref_bwd, dets, assoc_iou_thresh) if dets else {}
        for a_idx in range(n_actors):
            if a_idx in assignments:
                box, conf, iou = assignments[a_idx]
                all_boxes[a_idx][pos] = box
                all_sources[a_idx][pos] = "det"
                all_valid[a_idx][pos] = 1
                all_det_scores[a_idx][pos] = conf
                all_assoc_ious[a_idx][pos] = iou
                ref_bwd[a_idx] = box                  # update reference

    # ── Propagate FORWARD (center+1 → T-1) ──
    ref_fwd = [list(gt) for gt in gt_boxes]           # reset to GT
    for pos in range(center_idx + 1, clip_len):
        if clamped[pos]:
            continue
        dets = det_cache.get(frame_indices[pos], [])
        assignments = hungarian_match(ref_fwd, dets, assoc_iou_thresh) if dets else {}
        for a_idx in range(n_actors):
            if a_idx in assignments:
                box, conf, iou = assignments[a_idx]
                all_boxes[a_idx][pos] = box
                all_sources[a_idx][pos] = "det"
                all_valid[a_idx][pos] = 1
                all_det_scores[a_idx][pos] = conf
                all_assoc_ious[a_idx][pos] = iou
                ref_fwd[a_idx] = box

    # ── Fill gaps ──
    for a_idx in range(n_actors):
        fill_gaps(
            all_boxes[a_idx], all_sources[a_idx], all_valid[a_idx],
            all_det_scores[a_idx], all_assoc_ious[a_idx],
        )

    # ── Assemble records ──
    records: List[Dict[str, Any]] = []
    for a_idx, row in enumerate(actor_rows):
        track_quality = sum(all_valid[a_idx]) / clip_len
        records.append({
            "sample_id":      str(row["sample_id"]),
            "video_id":       str(row["video_id"]),
            "timestamp":      int(row["timestamp"]),
            "person_id":      int(row["person_id"]),
            "center_frame":   int(center_frame),
            "center_idx":     int(center_idx),
            "frame_indices":  list(frame_indices),
            "boxes":          all_boxes[a_idx],
            "valid_mask":     all_valid[a_idx],
            "sources":        all_sources[a_idx],
            "det_scores":     all_det_scores[a_idx],
            "assoc_ious":     all_assoc_ious[a_idx],
            "center_det_iou": center_det_ious[a_idx],
            "track_quality":  track_quality,
        })

    return records


# ──────────────────────────────────────────────
# Track building — per video
# ──────────────────────────────────────────────

def build_tracks_for_video(
    video_df: pd.DataFrame,
    det_cache: Dict[int, List[Tuple]],
    clip_len: int,
    clip_stride: int,
    max_frame: int,
    assoc_iou_thresh: float,
) -> List[Dict[str, Any]]:
    """Build tracks for every actor instance in one video."""
    records: List[Dict[str, Any]] = []
    for _ts, group_df in video_df.groupby("timestamp"):
        group_records = build_tracks_for_timestamp_group(
            group_df, det_cache, clip_len, clip_stride,
            max_frame, assoc_iou_thresh,
        )
        records.extend(group_records)
    return records


# ──────────────────────────────────────────────
# Atomic save
# ──────────────────────────────────────────────

def save_atomic(data: Any, path: Path) -> None:
    """Atomic pickle write: write .tmp.pkl then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(tmp_path), str(path))


# ──────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────

def percentile_stats(values: List[float]) -> Dict[str, float]:
    """Standard percentile breakdown for a list of floats."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0,
                "p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean":   round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "min":    round(float(np.min(arr)), 6),
        "p5":     round(float(np.percentile(arr, 5)), 6),
        "p25":    round(float(np.percentile(arr, 25)), 6),
        "p75":    round(float(np.percentile(arr, 75)), 6),
        "p95":    round(float(np.percentile(arr, 95)), 6),
    }


def histogram_bins(
    values: List[float], n_bins: int = 10,
    low: float = 0.0, high: float = 1.0,
) -> List[Dict[str, Any]]:
    """Simple histogram with fixed range."""
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=n_bins, range=(low, high))
    return [
        {"bin_low": round(float(edges[i]), 2),
         "bin_high": round(float(edges[i + 1]), 2),
         "count": int(counts[i])}
        for i in range(n_bins)
    ]


def compute_summary(
    records: List[Dict[str, Any]],
    clip_len: int,
    center_iou_thresh: float,
) -> Dict[str, Any]:
    """Compute mandatory summary statistics across all track records."""
    center_idx = clip_len // 2
    n_samples = len(records)

    if n_samples == 0:
        return {"total_samples": 0, "error": "no records"}

    # ── Per-sample scalars ──
    track_qualities = [r["track_quality"] for r in records]
    center_det_ious = [r["center_det_iou"] for r in records]

    # ── Per-position aggregation ──
    source_counts: Dict[str, int] = {"gt": 0, "det": 0, "interp": 0, "repeat": 0}
    total_positions = n_samples * clip_len

    # Offsets: -(clip_len//2) … +(clip_len//2 - 1)  e.g. -4…+3 for T=8
    per_offset: Dict[int, Dict[str, int]] = {}
    for off in range(-(clip_len // 2), clip_len - clip_len // 2):
        per_offset[off] = {"gt": 0, "det": 0, "interp": 0, "repeat": 0, "total": 0}

    all_assoc_ious_det: List[float] = []

    # Per-video accumulator
    video_tqs: Dict[str, List[float]] = {}

    for r in records:
        vid = r["video_id"]
        video_tqs.setdefault(vid, []).append(r["track_quality"])

        for pos in range(clip_len):
            src = r["sources"][pos]
            source_counts[src] += 1

            offset = pos - center_idx
            per_offset[offset][src] += 1
            per_offset[offset]["total"] += 1

            if src == "det":
                all_assoc_ious_det.append(r["assoc_ious"][pos])

    # ── Per-offset summary ──
    per_offset_summary: Dict[str, Dict[str, float]] = {}
    for off in sorted(per_offset.keys()):
        d = per_offset[off]
        t = d["total"]
        key = f"+{off}" if off > 0 else str(off)
        if t > 0:
            per_offset_summary[key] = {
                "gt_frac":         round(d["gt"] / t, 4),
                "det_frac":        round(d["det"] / t, 4),
                "interp_frac":     round(d["interp"] / t, 4),
                "repeat_frac":     round(d["repeat"] / t, 4),
                "unmatched_rate":  round((d["interp"] + d["repeat"]) / t, 4),
            }
        else:
            per_offset_summary[key] = {}

    # ── Per-video summary ──
    per_video_summary: Dict[str, Dict[str, Any]] = {}
    low_quality_videos: List[str] = []
    for vid in sorted(video_tqs.keys()):
        tqs = video_tqs[vid]
        mean_tq = float(np.mean(tqs))
        n_low = sum(1 for tq in tqs if tq < 0.5)
        per_video_summary[vid] = {
            "mean_track_quality": round(mean_tq, 4),
            "n_samples": len(tqs),
            "n_low_quality": n_low,
        }
        if mean_tq < 0.5:
            low_quality_videos.append(vid)

    # ── Center IoU below threshold ──
    n_below = sum(1 for v in center_det_ious if v < center_iou_thresh)

    # ── Assemble ──
    return {
        "total_samples": n_samples,
        "clip_len": clip_len,
        "track_quality": {
            **percentile_stats(track_qualities),
            "histogram": histogram_bins(track_qualities),
        },
        "center_det_iou": {
            **percentile_stats(center_det_ious),
            "threshold": center_iou_thresh,
            "n_below_threshold": n_below,
            "below_threshold_frac": round(n_below / n_samples, 4),
            "histogram": histogram_bins(center_det_ious),
        },
        "assoc_iou_det_only": {
            **percentile_stats(all_assoc_ious_det),
            "count": len(all_assoc_ious_det),
            "histogram": histogram_bins(all_assoc_ious_det),
        },
        "source_distribution": {
            src: {"count": cnt, "fraction": round(cnt / total_positions, 4)}
            for src, cnt in source_counts.items()
        },
        "per_offset_from_center": per_offset_summary,
        "per_video": per_video_summary,
        "low_quality_videos": low_quality_videos,
        "n_low_quality_videos": len(low_quality_videos),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    """Run Phase 2: assemble temporal tracks from detection caches."""
    args = parse_args()
    setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info(f"Phase 2: Track assembly — split={args.split}")
    logger.info(f"  Actor CSV        : {args.actor_csv}")
    logger.info(f"  Frame root       : {args.frame_root}")
    logger.info(f"  Cache dir        : {args.cache_dir}")
    logger.info(f"  Sidecar dir      : {args.sidecar_dir}")
    logger.info(f"  Output pkl       : {args.output_pkl}")
    logger.info(f"  Output summary   : {args.output_summary}")
    logger.info(f"  clip_len         : {args.clip_len}")
    logger.info(f"  clip_stride      : {args.clip_stride}")
    logger.info(f"  assoc_iou_thresh : {args.assoc_iou_thresh}")
    logger.info(f"  center_iou_thresh: {args.center_iou_thresh}")
    logger.info(f"  overwrite        : {args.overwrite}")
    logger.info("=" * 60)

    cache_dir = Path(args.cache_dir)
    sidecar_dir = Path(args.sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    frame_root = Path(args.frame_root)

    # ── Load actor CSV ──
    df = load_actor_csv(args.actor_csv)
    df["video_id"] = df["video_id"].astype(str)

    video_ids = sorted(df["video_id"].unique())
    total_videos = len(video_ids)
    logger.info(f"Total videos: {total_videos}")

    # ────────────────────────────
    # STAGE 1: per-video sidecars
    # ────────────────────────────
    videos_processed = 0
    videos_skipped = 0
    videos_failed = 0
    tracks_built = 0
    stage1_time = 0.0

    for idx, video_id in enumerate(video_ids, 1):
        sidecar_path = sidecar_dir / f"{video_id}.pkl"

        # Resume
        if sidecar_path.exists() and not args.overwrite:
            logger.info(
                f"[{idx}/{total_videos}] Skipping {video_id}: sidecar exists"
            )
            videos_skipped += 1
            continue

        # Detection cache
        cache_path = cache_dir / f"{video_id}.pkl"
        if not cache_path.exists():
            logger.error(
                f"[{idx}/{total_videos}] Missing detection cache: {cache_path}"
            )
            videos_failed += 1
            continue

        det_cache = load_detection_cache(cache_path)

        # Frame count
        video_dir = frame_root / video_id
        if not video_dir.is_dir():
            logger.error(
                f"[{idx}/{total_videos}] Missing video dir: {video_dir}"
            )
            videos_failed += 1
            continue
        max_frame = get_max_frame_count(video_dir)

        # Actors for this video
        video_df = df[df["video_id"] == video_id]
        if len(video_df) == 0:
            logger.warning(f"[{idx}/{total_videos}] No actors for {video_id}")
            videos_failed += 1
            continue

        # Build tracks
        t0 = time.time()
        records = build_tracks_for_video(
            video_df, det_cache, args.clip_len, args.clip_stride,
            max_frame, args.assoc_iou_thresh,
        )
        elapsed = time.time() - t0

        # Save
        save_atomic(records, sidecar_path)

        n = len(records)
        mean_tq = float(np.mean([r["track_quality"] for r in records])) if n else 0.0
        mean_cdi = float(np.mean([r["center_det_iou"] for r in records])) if n else 0.0

        logger.info(
            f"[{idx}/{total_videos}] {video_id}: "
            f"{n} tracks, quality={mean_tq:.3f}, "
            f"center_iou={mean_cdi:.3f}, {elapsed:.1f}s"
        )

        videos_processed += 1
        tracks_built += n
        stage1_time += elapsed

    logger.info("-" * 60)
    logger.info(
        f"Stage 1 done: processed={videos_processed}, "
        f"skipped={videos_skipped}, failed={videos_failed}, "
        f"tracks={tracks_built}, time={stage1_time:.1f}s"
    )

    # ─────────────────────────────────
    # STAGE 2: merge per-video sidecars
    # ─────────────────────────────────
    logger.info("Merging per-video sidecars ...")
    all_records: List[Dict[str, Any]] = []
    merge_missing = 0

    for video_id in video_ids:
        sidecar_path = sidecar_dir / f"{video_id}.pkl"
        if not sidecar_path.exists():
            logger.warning(f"Missing sidecar for merge: {video_id}")
            merge_missing += 1
            continue
        with open(sidecar_path, "rb") as f:
            records = pickle.load(f)
        all_records.extend(records)

    logger.info(
        f"Merged: {len(all_records)} records "
        f"({merge_missing} videos missing)"
    )

    # Sanity check
    expected = len(df)
    if len(all_records) != expected:
        logger.warning(
            f"Count mismatch: merged={len(all_records)}, "
            f"actor_csv={expected}"
        )

    merged = {r["sample_id"]: r for r in all_records}
    output_pkl = Path(args.output_pkl)
    save_atomic(merged, output_pkl)
    logger.info(f"Saved merged sidecar: {output_pkl} ({len(merged)} samples)")

    # ──────────────────
    # STAGE 3: summary
    # ──────────────────
    logger.info("Computing summary statistics ...")
    summary = compute_summary(all_records, args.clip_len, args.center_iou_thresh)

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {output_summary}")

    # ── Print key numbers ──
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Total samples     : {summary['total_samples']}")

    tq = summary["track_quality"]
    logger.info(
        f"  Track quality     : mean={tq['mean']:.4f}  "
        f"median={tq['median']:.4f}  min={tq['min']:.4f}  "
        f"p5={tq['p5']:.4f}"
    )

    cdi = summary["center_det_iou"]
    logger.info(
        f"  Center det IoU    : mean={cdi['mean']:.4f}  "
        f"median={cdi['median']:.4f}  p5={cdi['p5']:.4f}  "
        f"below_{args.center_iou_thresh}={cdi['n_below_threshold']}"
    )

    ai = summary["assoc_iou_det_only"]
    logger.info(
        f"  Assoc IoU (det)   : mean={ai['mean']:.4f}  "
        f"median={ai['median']:.4f}  p5={ai['p5']:.4f}  "
        f"n={ai['count']}"
    )

    sd = summary["source_distribution"]
    logger.info(
        f"  Sources           : "
        f"gt={sd['gt']['fraction']:.4f}  "
        f"det={sd['det']['fraction']:.4f}  "
        f"interp={sd['interp']['fraction']:.4f}  "
        f"repeat={sd['repeat']['fraction']:.4f}"
    )

    logger.info(
        f"  Low-quality videos: {summary['n_low_quality_videos']}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()