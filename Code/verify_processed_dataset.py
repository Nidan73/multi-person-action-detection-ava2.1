#!/usr/bin/env python3
"""
verify_processed_dataset.py

Runs all smoke-test checks on a processed actor-level CSV:

1. Every center_frame maps to an existing .jpg
2. Every sample has ≥1 positive label
3. Processed row count < raw row count
4. All action labels in {0, 1}  (IDs implicitly in [1, 80] by column design)
5. No degenerate / OOB boxes
7. No NaN or empty values

Also prints full summary statistics.
"""

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify processed actor-level AVA CSV")
    p.add_argument("--processed_csv", type=str, required=True,
                   help="Processed actor-level CSV (with header)")
    p.add_argument("--raw_csv", type=str, required=True,
                   help="Original raw AVA CSV (for row-count comparison)")
    p.add_argument("--frames_root", type=str, required=True,
                   help="Frame root directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processed_path = Path(args.processed_csv)
    raw_path       = Path(args.raw_csv)
    frames_root    = Path(args.frames_root)

    # ── count raw rows ──
    with open(raw_path, "r") as fh:
        raw_count = sum(1 for _ in fh)
    print(f"[INFO] Raw CSV rows: {raw_count:,}")

    # ── load processed CSV ──
    rows: List[dict] = []
    with open(processed_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    proc_count = len(rows)
    print(f"[INFO] Processed CSV rows: {proc_count:,}")

    # ── accumulators ──
    n_fail_frame      = 0
    n_fail_zero_label = 0
    n_fail_degenerate = 0
    n_fail_oob        = 0
    n_fail_nan        = 0
    n_fail_label_val  = 0

    label_freq        = Counter()
    labels_per_inst: List[int]  = []
    timestamps:      List[int]  = []
    video_ids        = set()

    # deduplicate frame checks for speed
    checked_frames: set = set()
    missing_frames: set = set()

    for row_i, row in enumerate(rows):
        # ── check 7: NaN / empty ──
        for col, val in row.items():
            if val is None or val.strip() == "" or val.strip().lower() == "nan":
                if n_fail_nan < 5:
                    print(f"  [NaN] row {row_i}, col='{col}', val='{val}'")
                n_fail_nan += 1

        video_id     = row["video_id"]
        timestamp    = int(row["timestamp"])
        center_frame = int(row["center_frame"])
        x1 = float(row["x1"]); y1 = float(row["y1"])
        x2 = float(row["x2"]); y2 = float(row["y2"])

        video_ids.add(video_id)
        timestamps.append(timestamp)

        # ── check 5a: degenerate box ──
        if x2 <= x1 or y2 <= y1:
            if n_fail_degenerate < 5:
                print(f"  [DEGEN] row {row_i}: ({x1},{y1},{x2},{y2})")
            n_fail_degenerate += 1

        # ── check 5b: out-of-bounds ──
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and
                0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            if n_fail_oob < 5:
                print(f"  [OOB] row {row_i}: ({x1},{y1},{x2},{y2})")
            n_fail_oob += 1

        # ── check 2 + 4: labels ──
        label_sum = 0
        for j in range(1, 81):
            val = int(row[f"label_{j}"])
            if val not in (0, 1):
                n_fail_label_val += 1
            label_sum += val
            if val == 1:
                label_freq[j] += 1

        labels_per_inst.append(label_sum)
        if label_sum == 0:
            if n_fail_zero_label < 5:
                print(f"  [ZERO-LABEL] row {row_i}")
            n_fail_zero_label += 1

        # ── check 1: frame exists (deduplicated) ──
        fkey = (video_id, center_frame)
        if fkey not in checked_frames:
            checked_frames.add(fkey)
            fp = frames_root / video_id / f"{center_frame:06d}.jpg"
            if not fp.exists():
                missing_frames.add(fkey)
                if len(missing_frames) <= 10:
                    print(f"  [MISS] {fp}")

    # count samples whose frame is missing
    n_fail_frame = sum(
        1 for row in rows
        if (row["video_id"], int(row["center_frame"])) in missing_frames
    )

    # ── results ──
    print(f"\n{'=' * 60}")
    print(f"  CHECK 1  frames exist on disk:    "
          f"{'OK' if n_fail_frame == 0 else f'FAIL ({n_fail_frame} samples, {len(missing_frames)} unique frames missing)'}")
    print(f"  CHECK 2  all have ≥1 label:       "
          f"{'OK' if n_fail_zero_label == 0 else f'FAIL ({n_fail_zero_label})'}")
    print(f"  CHECK 3  processed < raw:         "
          f"{'OK' if proc_count < raw_count else 'FAIL'}"
          f"  ({proc_count:,} vs {raw_count:,})")
    print(f"  CHECK 4  label values in {{0,1}}:   "
          f"{'OK' if n_fail_label_val == 0 else f'FAIL ({n_fail_label_val})'}")
    print(f"  CHECK 5  no degenerate/OOB boxes: "
          f"{'OK' if (n_fail_degenerate + n_fail_oob) == 0 else f'FAIL (degen={n_fail_degenerate}, oob={n_fail_oob})'}")
    print(f"  CHECK 7  no NaN / empty:          "
          f"{'OK' if n_fail_nan == 0 else f'FAIL ({n_fail_nan})'}")
    print(f"{'=' * 60}")

    all_pass = (n_fail_frame == 0 and n_fail_zero_label == 0 and
                proc_count < raw_count and n_fail_label_val == 0 and
                n_fail_degenerate == 0 and n_fail_oob == 0 and
                n_fail_nan == 0)
    print(f"\n  OVERALL: {'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")

    # ── stats ──
    print(f"\n{'=' * 60}")
    print(f"[STATS] Total samples:         {proc_count:,}")
    print(f"[STATS] Unique videos:         {len(video_ids)}")
    print(f"[STATS] Unique (vid, frame):   {len(checked_frames):,}")
    print(f"[STATS] Labels per instance:")
    print(f"         min    = {min(labels_per_inst)}")
    print(f"         max    = {max(labels_per_inst)}")
    print(f"         mean   = {statistics.mean(labels_per_inst):.2f}")
    print(f"         median = {statistics.median(labels_per_inst):.1f}")
    print(f"[STATS] Timestamp range:       {min(timestamps)} – {max(timestamps)}")

    print(f"\n[STATS] Label frequency:")
    for aid in range(1, 81):
        cnt = label_freq.get(aid, 0)
        bar = "#" * min(int(cnt / max(max(label_freq.values()), 1) * 40), 40)
        print(f"  action_{aid:2d}: {cnt:>7,}  {bar}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()