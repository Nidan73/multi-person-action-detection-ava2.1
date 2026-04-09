#!/usr/bin/env python3
"""
preprocess_ava_actor_csv.py

Reads a raw AVA-style headerless CSV (8 columns) and produces a processed
actor-level CSV: one row per (video_id, timestamp, person_id) with 80-dim
multi-hot labels and a computed center_frame index.

Run once per split (train / val).
"""

import argparse
import csv
import hashlib
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ──────────────────────────── helpers ────────────────────────────

def compute_center_frame(
    timestamp: int, ava_start_sec: int, fps: int, frame_offset: int
) -> int:
    """Map AVA timestamp (seconds) to 1-based frame index."""
    return (timestamp - ava_start_sec) * fps + frame_offset


def compute_sample_id(
    video_id: str, timestamp: int, person_id: int,
    x1: float, y1: float, x2: float, y2: float,
) -> str:
    """Deterministic 16-char hex hash for a unique actor instance."""
    raw = f"{video_id}|{timestamp}|{person_id}|{x1:.6f}|{y1:.6f}|{x2:.6f}|{y2:.6f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def frame_path(frames_root: Path, video_id: str, frame_idx: int) -> Path:
    return frames_root / video_id / f"{frame_idx:06d}.jpg"


# ──────────────────────────── CLI ────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw AVA CSV → actor-level processed CSV"
    )
    p.add_argument("--raw_csv", type=str, required=True,
                   help="Path to raw AVA-style CSV (headerless, 8 columns)")
    p.add_argument("--output_csv", type=str, required=True,
                   help="Output path for processed actor-level CSV")
    p.add_argument("--frames_root", type=str, required=True,
                   help="Root dir of extracted frames (contains <video_id>/ subfolders)")
    p.add_argument("--fps", type=int, default=8,
                   help="Extraction frame rate (default: 8)")
    p.add_argument("--ava_start_sec", type=int, default=900,
                   help="First annotated second in the source video (default: 900)")
    p.add_argument("--frame_offset", type=int, default=1,
                   help="Frame numbering start (1 for 1-based, default: 1)")
    p.add_argument("--verify_n", type=int, default=500,
                   help="Random samples checked for frame existence (default: 500)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for verification sampling (default: 42)")
    return p.parse_args()


# ──────────────────────────── main ───────────────────────────────

def main() -> None:
    args = parse_args()

    raw_csv_path   = Path(args.raw_csv)
    output_csv_path = Path(args.output_csv)
    frames_root     = Path(args.frames_root)

    # ── print mapping formula ──
    print("=" * 60)
    print("[INFO] Frame-index formula:")
    print(f"       center_frame = (timestamp - {args.ava_start_sec}) "
          f"* {args.fps} + {args.frame_offset}")
    for ts_ex in [900, 902, 1000, 1798]:
        cf = compute_center_frame(ts_ex, args.ava_start_sec, args.fps, args.frame_offset)
        print(f"       timestamp {ts_ex} -> frame {cf:06d}.jpg")
    print("=" * 60)

    # ── 1. read raw CSV ──
    print(f"\n[READ] Raw CSV: {raw_csv_path}")
    raw_rows: List[Tuple[str, int, float, float, float, float, int, int]] = []
    with open(raw_csv_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, cols in enumerate(reader, start=1):
            if len(cols) != 8:
                print(f"[WARN] line {line_no}: expected 8 columns, got {len(cols)} -> skip")
                continue
            try:
                video_id  = cols[0]
                timestamp = int(cols[1])
                x1, y1    = float(cols[2]), float(cols[3])
                x2, y2    = float(cols[4]), float(cols[5])
                action_id = int(cols[6])
                person_id = int(cols[7])
            except ValueError as exc:
                print(f"[WARN] line {line_no}: parse error ({exc}) -> skip")
                continue
            raw_rows.append((video_id, timestamp, x1, y1, x2, y2, action_id, person_id))

    raw_count = len(raw_rows)
    print(f"[READ] {raw_count:,} raw rows parsed")

    # ── 2. validate action-ID range ──
    all_action_ids = {r[6] for r in raw_rows}
    bad_ids = sorted(a for a in all_action_ids if a < 1 or a > 80)
    if bad_ids:
        print(f"[ERROR] action_ids outside [1,80]: {bad_ids}  — aborting")
        return
    print(f"[CHECK] action_id range: {min(all_action_ids)}–{max(all_action_ids)}, "
          f"{len(all_action_ids)} unique values")

    # ── 3. group by (video_id, timestamp, person_id) ──
    groups: Dict[Tuple[str, int, int], List] = defaultdict(list)
    for row in raw_rows:
        key = (row[0], row[1], row[7])          # video_id, timestamp, person_id
        groups[key].append(row)
    print(f"[GROUP] {len(groups):,} unique (video_id, timestamp, person_id) groups")

    # ── 4. process each group ──
    processed: List[Dict[str, Any]] = []
    n_box_mismatch = 0
    n_degenerate   = 0
    n_oob          = 0

    for key in sorted(groups.keys()):
        video_id, timestamp, person_id = key
        rows_in_group = groups[key]

        # --- box consistency check ---
        unique_boxes = {(r[2], r[3], r[4], r[5]) for r in rows_in_group}
        if len(unique_boxes) > 1:
            print(f"[WARN] box mismatch: ({video_id}, t={timestamp}, pid={person_id}) "
                  f"has {len(unique_boxes)} distinct boxes -> skip group")
            n_box_mismatch += 1
            continue

        x1, y1, x2, y2 = unique_boxes.pop()

        # --- degenerate box ---
        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] degenerate box ({x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}) "
                  f"at ({video_id}, t={timestamp}, pid={person_id}) -> drop")
            n_degenerate += 1
            continue

        # --- out-of-bounds box ---
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and
                0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            print(f"[WARN] OOB box ({x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}) "
                  f"at ({video_id}, t={timestamp}, pid={person_id}) -> drop")
            n_oob += 1
            continue

        # --- multi-hot label vector ---
        labels = [0] * 80
        for r in rows_in_group:
            labels[r[6] - 1] = 1                # action_id 1 -> index 0

        if sum(labels) == 0:                     # should never happen after checks
            print(f"[WARN] zero labels at ({video_id}, t={timestamp}, pid={person_id}) -> skip")
            continue

        # --- center frame ---
        cf = compute_center_frame(timestamp, args.ava_start_sec, args.fps, args.frame_offset)

        # --- sample id ---
        sid = compute_sample_id(video_id, timestamp, person_id, x1, y1, x2, y2)

        processed.append({
            "sample_id": sid,
            "video_id": video_id,
            "timestamp": timestamp,
            "person_id": person_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_frame": cf,
            "labels": labels,
        })

    # ── 5. sanity: processed < raw ──
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] Raw rows:             {raw_count:>10,}")
    print(f"[SUMMARY] Actor instances:       {len(processed):>10,}")
    print(f"[SUMMARY] Box-mismatch skipped:  {n_box_mismatch:>10,}")
    print(f"[SUMMARY] Degenerate dropped:    {n_degenerate:>10,}")
    print(f"[SUMMARY] Out-of-bounds dropped: {n_oob:>10,}")
    if len(processed) >= raw_count:
        print("[WARN] processed >= raw — multi-label grouping may have failed!")

    # ── 6. write processed CSV ──
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = (["sample_id", "video_id", "timestamp", "person_id",
               "x1", "y1", "x2", "y2", "center_frame"]
              + [f"label_{i}" for i in range(1, 81)])

    with open(output_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for entry in processed:
            row = [
                entry["sample_id"],
                entry["video_id"],
                entry["timestamp"],
                entry["person_id"],
                f"{entry['x1']:.6f}",
                f"{entry['y1']:.6f}",
                f"{entry['x2']:.6f}",
                f"{entry['y2']:.6f}",
                entry["center_frame"],
            ] + entry["labels"]
            writer.writerow(row)

    print(f"\n[WRITE] {len(processed):,} rows -> {output_csv_path}")

    # ── 7. summary stats ──
    unique_videos     = sorted({e["video_id"] for e in processed})
    labels_per_inst   = [sum(e["labels"]) for e in processed]
    all_timestamps    = [e["timestamp"] for e in processed]
    all_center_frames = [e["center_frame"] for e in processed]

    print(f"\n[STATS] Unique videos:        {len(unique_videos)}")
    print(f"[STATS] Labels per instance:  "
          f"min={min(labels_per_inst)}  max={max(labels_per_inst)}  "
          f"mean={statistics.mean(labels_per_inst):.2f}  "
          f"median={statistics.median(labels_per_inst):.1f}")
    print(f"[STATS] Timestamp range:      {min(all_timestamps)} – {max(all_timestamps)}")
    print(f"[STATS] Center-frame range:   {min(all_center_frames)} – {max(all_center_frames)}")

    # label frequency
    label_freq = [0] * 80
    for e in processed:
        for i, v in enumerate(e["labels"]):
            label_freq[i] += v

    print(f"\n[STATS] Label frequency (action_id: count):")
    for i in range(80):
        bar = "#" * min(int(label_freq[i] / max(max(label_freq), 1) * 40), 40)
        print(f"  action_{i+1:2d}: {label_freq[i]:>7,}  {bar}")

    # ── 8. frame existence verification ──
    n_verify = min(args.verify_n, len(processed))
    print(f"\n[VERIFY] Checking {n_verify} random samples for frame existence …")
    random.seed(args.seed)
    sample_indices = random.sample(range(len(processed)), n_verify)

    n_missing = 0
    for idx in sample_indices:
        e = processed[idx]
        fp = frame_path(frames_root, e["video_id"], e["center_frame"])
        if not fp.exists():
            n_missing += 1
            if n_missing <= 10:
                print(f"  [MISS] {fp}")

    miss_pct = n_missing / n_verify * 100 if n_verify else 0
    print(f"[VERIFY] Checked {n_verify}, missing {n_missing} ({miss_pct:.1f}%)")

    if miss_pct > 5.0:
        print(f"[ERROR] >5 % missing!  Your mapping constants are likely wrong.")
        print(f"        ava_start_sec={args.ava_start_sec}  fps={args.fps}  "
              f"frame_offset={args.frame_offset}")
    elif n_missing > 0:
        print(f"[WARN] Some frames missing (<5 %).  Investigate specific cases above.")
    else:
        print(f"[OK] All {n_verify} sampled frames exist on disk.")

    print(f"\n{'=' * 60}")
    print("[DONE]")


if __name__ == "__main__":
    main()