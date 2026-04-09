#!/usr/bin/env python
"""
verify_temporal_tracks.py

Verification script for temporal actor track sidecar files.
Cross-references the merged sidecar .pkl against the actor CSV,
runs 17 mandatory structural and statistical checks, and prints
a detailed summary with pass/fail verdicts.
"""

import argparse
import logging
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Verify temporal track sidecar against actor CSV."
    )
    p.add_argument("--sidecar_pkl", required=True, type=str,
                   help="Path to merged sidecar .pkl.")
    p.add_argument("--actor_csv", required=True, type=str,
                   help="Path to processed actor CSV.")
    p.add_argument("--frame_root", required=True, type=str,
                   help="Root directory of per-video frame folders.")
    p.add_argument("--clip_len", type=int, default=8,
                   help="Expected clip length T (default: 8).")
    p.add_argument("--frame_spot_check", type=int, default=500,
                   help="Number of random frame paths to verify (default: 500).")
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

def load_sidecar(pkl_path: str) -> Dict[str, Dict[str, Any]]:
    """Load merged sidecar pickle (dict keyed by sample_id)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded sidecar: {len(data)} samples from {pkl_path}")
    return data


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
    # Ensure string types for join keys
    df["sample_id"] = df["sample_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    logger.info(f"Loaded actor CSV: {len(df)} instances from {csv_path}")
    return df


def get_video_frame_counts(
    frame_root: Path, video_ids: Set[str],
) -> Dict[str, int]:
    """Count .jpg files per video directory. Cache results."""
    counts: Dict[str, int] = {}
    for vid in video_ids:
        vid_dir = frame_root / vid
        if vid_dir.is_dir():
            counts[vid] = sum(1 for f in os.listdir(vid_dir) if f.endswith(".jpg"))
        else:
            counts[vid] = 0
    return counts


# ──────────────────────────────────────────────
# Check functions
# ──────────────────────────────────────────────

def check_coverage(
    sidecar: Dict[str, Any], csv_ids: Set[str],
) -> Tuple[bool, str]:
    """Check 1: every actor CSV sample_id exists in sidecar."""
    sidecar_ids = set(sidecar.keys())
    missing = csv_ids - sidecar_ids
    n_covered = len(csv_ids) - len(missing)
    total = len(csv_ids)
    passed = len(missing) == 0
    msg = f"{n_covered} / {total} samples covered"
    if not passed:
        msg += f" — {len(missing)} MISSING"
        examples = sorted(missing)[:5]
        msg += f" (first 5: {examples})"
    return passed, msg


def check_no_extras(
    sidecar: Dict[str, Any], csv_ids: Set[str],
) -> Tuple[bool, str]:
    """Check 2: no sample_id in sidecar that is not in actor CSV."""
    sidecar_ids = set(sidecar.keys())
    extras = sidecar_ids - csv_ids
    passed = len(extras) == 0
    msg = f"{len(extras)} extra samples in sidecar"
    if not passed:
        examples = sorted(extras)[:5]
        msg += f" (first 5: {examples})"
    return passed, msg


def check_metadata_match(
    sidecar: Dict[str, Any], csv_lookup: Dict[str, Dict],
) -> Tuple[bool, str]:
    """Check 3: video_id, timestamp, person_id, center_frame match."""
    fields = ["video_id", "timestamp", "person_id", "center_frame"]
    mismatches: List[str] = []
    for sid, rec in sidecar.items():
        if sid not in csv_lookup:
            continue
        csv_row = csv_lookup[sid]
        for f in fields:
            sv = rec[f]
            cv = csv_row[f]
            # Normalize types for comparison
            if isinstance(sv, (int, float)) and isinstance(cv, (int, float)):
                if int(sv) != int(cv):
                    mismatches.append(f"{sid}: {f} sidecar={sv} csv={cv}")
            else:
                if str(sv) != str(cv):
                    mismatches.append(f"{sid}: {f} sidecar={sv} csv={cv}")
        if len(mismatches) >= 5:
            break
    passed = len(mismatches) == 0
    msg = f"{len(mismatches)} metadata mismatches"
    if not passed:
        for m in mismatches[:5]:
            msg += f"\n    {m}"
    return passed, msg


def check_shape_consistency(
    sidecar: Dict[str, Any], clip_len: int,
) -> Tuple[bool, str]:
    """Check 4: all list fields have length == clip_len."""
    list_fields = ["frame_indices", "boxes", "valid_mask", "sources", "det_scores"]
    bad: List[str] = []
    for sid, rec in sidecar.items():
        for f in list_fields:
            if len(rec[f]) != clip_len:
                bad.append(f"{sid}: {f} len={len(rec[f])} expected={clip_len}")
        if len(bad) >= 5:
            break
    passed = len(bad) == 0
    msg = f"{len(bad)} shape inconsistencies"
    if not passed:
        for b in bad[:5]:
            msg += f"\n    {b}"
    return passed, msg


def check_center_frame_correctness(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 5: frame_indices[center_idx] == center_frame."""
    bad_count = 0
    for rec in sidecar.values():
        ci = rec["center_idx"]
        if rec["frame_indices"][ci] != rec["center_frame"]:
            bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} samples with wrong center frame position"
    return passed, msg


def check_center_source(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 6: sources[center_idx] == 'gt'."""
    bad_count = 0
    for rec in sidecar.values():
        ci = rec["center_idx"]
        if rec["sources"][ci] != "gt":
            bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} samples with center source != 'gt'"
    return passed, msg


def check_center_valid_mask(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 7: valid_mask[center_idx] == 1."""
    bad_count = 0
    for rec in sidecar.values():
        ci = rec["center_idx"]
        if rec["valid_mask"][ci] != 1:
            bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} samples with center valid_mask != 1"
    return passed, msg


def check_center_box_match(
    sidecar: Dict[str, Any], csv_lookup: Dict[str, Dict],
) -> Tuple[bool, str]:
    """Check 8: boxes[center_idx] matches GT box from actor CSV."""
    tol = 1e-6
    bad: List[str] = []
    for sid, rec in sidecar.items():
        if sid not in csv_lookup:
            continue
        ci = rec["center_idx"]
        sbox = rec["boxes"][ci]
        crow = csv_lookup[sid]
        cbox = [float(crow["x1"]), float(crow["y1"]),
                float(crow["x2"]), float(crow["y2"])]
        for k in range(4):
            if abs(sbox[k] - cbox[k]) > tol:
                bad.append(
                    f"{sid}: sidecar={sbox} csv={cbox}"
                )
                break
        if len(bad) >= 5:
            break
    passed = len(bad) == 0
    msg = f"{len(bad)} center box mismatches"
    if not passed:
        for b in bad[:5]:
            msg += f"\n    {b}"
    return passed, msg


def check_box_range(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 9: all box coordinates in [0.0, 1.0]."""
    out_count = 0
    for rec in sidecar.values():
        for box in rec["boxes"]:
            for coord in box:
                if coord < 0.0 or coord > 1.0:
                    out_count += 1
    passed = out_count == 0
    msg = f"{out_count} out-of-range coordinates"
    return passed, msg


def check_box_validity(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 10: x2 > x1 and y2 > y1 for all boxes."""
    bad_count = 0
    for rec in sidecar.values():
        for box in rec["boxes"]:
            if box[2] <= box[0] or box[3] <= box[1]:
                bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} degenerate boxes"
    return passed, msg


def check_frame_indices_range(
    sidecar: Dict[str, Any],
    frame_counts: Dict[str, int],
) -> Tuple[bool, str]:
    """Check 11: all frame indices in [1, max_frame]."""
    bad_count = 0
    for rec in sidecar.values():
        vid = rec["video_id"]
        max_f = frame_counts.get(vid, 0)
        for fi in rec["frame_indices"]:
            if fi < 1 or fi > max_f:
                bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} frame indices out of range"
    return passed, msg


def check_frame_existence_spot(
    sidecar: Dict[str, Any],
    frame_root: Path,
    n_checks: int,
) -> Tuple[bool, str]:
    """Check 12: random sample of frame paths exist on disk."""
    # Collect all (video_id, frame_idx) pairs
    all_pairs: List[Tuple[str, int]] = []
    for rec in sidecar.values():
        vid = rec["video_id"]
        for fi in rec["frame_indices"]:
            all_pairs.append((vid, fi))

    if len(all_pairs) == 0:
        return False, "No frame pairs to check"

    n_checks = min(n_checks, len(all_pairs))
    sample = random.sample(all_pairs, n_checks)

    missing = 0
    for vid, fi in sample:
        path = frame_root / vid / f"{fi:06d}.jpg"
        if not path.exists():
            missing += 1

    passed = missing == 0
    msg = f"{n_checks - missing} / {n_checks} spot-checked frames exist"
    if not passed:
        msg += f" — {missing} MISSING"
    return passed, msg


def check_valid_mask_values(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 13: all valid_mask values are 0 or 1."""
    bad_count = 0
    for rec in sidecar.values():
        for v in rec["valid_mask"]:
            if v not in (0, 1):
                bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} invalid valid_mask values"
    return passed, msg


def check_sources_values(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 14: all source values in allowed set."""
    allowed = {"gt", "det", "interp", "repeat"}
    bad_count = 0
    bad_vals: Set[str] = set()
    for rec in sidecar.values():
        for s in rec["sources"]:
            if s not in allowed:
                bad_count += 1
                bad_vals.add(str(s))
    passed = bad_count == 0
    msg = f"{bad_count} invalid source values"
    if not passed:
        msg += f" (found: {bad_vals})"
    return passed, msg


def check_track_quality_consistency(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 15: track_quality == mean(valid_mask)."""
    tol = 1e-6
    bad_count = 0
    for rec in sidecar.values():
        expected = sum(rec["valid_mask"]) / len(rec["valid_mask"])
        if abs(rec["track_quality"] - expected) > tol:
            bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} track_quality mismatches"
    return passed, msg


def check_at_least_one_label(
    sidecar: Dict[str, Any], csv_lookup: Dict[str, Dict],
) -> Tuple[bool, str]:
    """Check 16: every sample has at least one positive label in actor CSV."""
    label_cols = [f"label_{i}" for i in range(1, 81)]
    bad_count = 0
    for sid in sidecar.keys():
        if sid not in csv_lookup:
            continue
        row = csv_lookup[sid]
        total = sum(int(row.get(lc, 0)) for lc in label_cols)
        if total == 0:
            bad_count += 1
    passed = bad_count == 0
    msg = f"{bad_count} samples with zero labels"
    return passed, msg


def check_no_nan(
    sidecar: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check 17: no NaN in numeric fields."""
    nan_count = 0
    for rec in sidecar.values():
        # boxes
        for box in rec["boxes"]:
            for v in box:
                if math.isnan(v):
                    nan_count += 1
        # det_scores
        for v in rec["det_scores"]:
            if math.isnan(v):
                nan_count += 1
        # track_quality
        if math.isnan(rec["track_quality"]):
            nan_count += 1
        # center_det_iou
        if math.isnan(rec["center_det_iou"]):
            nan_count += 1
        # assoc_ious
        if "assoc_ious" in rec:
            for v in rec["assoc_ious"]:
                if math.isnan(v):
                    nan_count += 1
    passed = nan_count == 0
    msg = f"{nan_count} NaN values found"
    return passed, msg


# ──────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────

def print_summary(
    sidecar: Dict[str, Any], clip_len: int,
) -> None:
    """Print detailed summary statistics."""
    center_idx = clip_len // 2
    n = len(sidecar)

    # ── Basic counts ──
    video_ids = set(rec["video_id"] for rec in sidecar.values())
    logger.info(f"  Total samples  : {n}")
    logger.info(f"  Unique videos  : {len(video_ids)}")

    # ── Track quality ──
    tqs = np.array([r["track_quality"] for r in sidecar.values()])
    logger.info("")
    logger.info("  Track quality:")
    logger.info(f"    mean   = {np.mean(tqs):.4f}")
    logger.info(f"    median = {np.median(tqs):.4f}")
    logger.info(f"    min    = {np.min(tqs):.4f}")
    logger.info(f"    max    = {np.max(tqs):.4f}")
    logger.info(f"    p5     = {np.percentile(tqs, 5):.4f}")
    logger.info(f"    p25    = {np.percentile(tqs, 25):.4f}")
    logger.info(f"    p75    = {np.percentile(tqs, 75):.4f}")
    logger.info(f"    p95    = {np.percentile(tqs, 95):.4f}")

    # ── Center det IoU ──
    cdis = np.array([r["center_det_iou"] for r in sidecar.values()])
    frac_ge_05 = float(np.mean(cdis >= 0.5))
    frac_lt_03 = float(np.mean(cdis < 0.3))
    logger.info("")
    logger.info("  Center detection IoU:")
    logger.info(f"    mean   = {np.mean(cdis):.4f}")
    logger.info(f"    median = {np.median(cdis):.4f}")
    logger.info(f"    min    = {np.min(cdis):.4f}")
    logger.info(f"    max    = {np.max(cdis):.4f}")
    logger.info(f"    >= 0.5 : {frac_ge_05:.4f} ({int(frac_ge_05 * n)} samples)")
    logger.info(f"    < 0.3  : {frac_lt_03:.4f} ({int(frac_lt_03 * n)} samples)")

    # ── Source distribution ──
    source_counts: Dict[str, int] = {"gt": 0, "det": 0, "interp": 0, "repeat": 0}
    total_positions = n * clip_len

    per_offset: Dict[int, Dict[str, int]] = {}
    for off in range(-(clip_len // 2), clip_len - clip_len // 2):
        per_offset[off] = {"gt": 0, "det": 0, "interp": 0, "repeat": 0, "total": 0}

    for rec in sidecar.values():
        for pos in range(clip_len):
            src = rec["sources"][pos]
            source_counts[src] += 1
            offset = pos - center_idx
            per_offset[offset][src] += 1
            per_offset[offset]["total"] += 1

    logger.info("")
    logger.info("  Source distribution (all positions):")
    for src in ["gt", "det", "interp", "repeat"]:
        cnt = source_counts[src]
        frac = cnt / total_positions if total_positions > 0 else 0
        logger.info(f"    {src:8s}: {cnt:>10d} ({frac:.4f})")

    # ── Per-offset ──
    logger.info("")
    logger.info("  Per-offset-from-center source fractions:")
    logger.info(f"    {'offset':>6s}  {'det%':>7s}  {'interp%':>7s}  "
                f"{'repeat%':>7s}  {'unmatched%':>10s}")
    for off in sorted(per_offset.keys()):
        if off == 0:
            continue
        d = per_offset[off]
        t = d["total"]
        if t == 0:
            continue
        det_f = d["det"] / t
        int_f = d["interp"] / t
        rep_f = d["repeat"] / t
        unm_f = (d["interp"] + d["repeat"]) / t
        label = f"+{off}" if off > 0 else str(off)
        logger.info(f"    {label:>6s}  {det_f:>7.4f}  {int_f:>7.4f}  "
                    f"{rep_f:>7.4f}  {unm_f:>10.4f}")

    # ── Per-video quality ──
    video_tqs: Dict[str, List[float]] = {}
    for rec in sidecar.values():
        vid = rec["video_id"]
        video_tqs.setdefault(vid, []).append(rec["track_quality"])

    low_quality_videos = []
    for vid in sorted(video_tqs.keys()):
        mean_tq = float(np.mean(video_tqs[vid]))
        if mean_tq < 0.5:
            low_quality_videos.append((vid, mean_tq, len(video_tqs[vid])))

    logger.info("")
    logger.info(f"  Videos with mean track_quality < 0.5: "
                f"{len(low_quality_videos)}")
    for vid, mtq, cnt in low_quality_videos:
        logger.info(f"    {vid}: mean_tq={mtq:.3f}, n_samples={cnt}")

    # ── Sample quality buckets ──
    n_below_05 = int(np.sum(tqs < 0.5))
    n_perfect = int(np.sum(tqs == 1.0))
    logger.info("")
    logger.info(f"  Samples with track_quality < 0.5 : "
                f"{n_below_05} ({n_below_05 / n:.4f})")
    logger.info(f"  Samples with track_quality == 1.0: "
                f"{n_perfect} ({n_perfect / n:.4f})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    """Run all verification checks on the temporal track sidecar."""
    args = parse_args()
    setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Temporal Track Sidecar Verification")
    logger.info(f"  Sidecar pkl     : {args.sidecar_pkl}")
    logger.info(f"  Actor CSV       : {args.actor_csv}")
    logger.info(f"  Frame root      : {args.frame_root}")
    logger.info(f"  Clip len        : {args.clip_len}")
    logger.info(f"  Spot check      : {args.frame_spot_check}")
    logger.info("=" * 60)

    # ── Load data ──
    sidecar = load_sidecar(args.sidecar_pkl)
    df = load_actor_csv(args.actor_csv)
    frame_root = Path(args.frame_root)

    csv_ids = set(df["sample_id"].astype(str))
    csv_lookup: Dict[str, Dict] = {
        str(row["sample_id"]): row
        for _, row in df.iterrows()
    }

    # ── Frame counts (cached) ──
    all_video_ids = set(rec["video_id"] for rec in sidecar.values())
    all_video_ids |= set(df["video_id"].astype(str))
    logger.info("Counting frames per video ...")
    frame_counts = get_video_frame_counts(frame_root, all_video_ids)
    logger.info(f"Frame counts for {len(frame_counts)} videos cached.")

    # ── Run checks ──
    checks: List[Tuple[str, bool, str]] = []

    def run_check(name: str, passed: bool, msg: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append((name, passed, msg))
        logger.info(f"  [{status}] {name}: {msg}")

    logger.info("")
    logger.info("-" * 60)
    logger.info("RUNNING 17 CHECKS")
    logger.info("-" * 60)

    # 1
    p, m = check_coverage(sidecar, csv_ids)
    run_check("1. Coverage", p, m)

    # 2
    p, m = check_no_extras(sidecar, csv_ids)
    run_check("2. No extras", p, m)

    # 3
    p, m = check_metadata_match(sidecar, csv_lookup)
    run_check("3. Metadata match", p, m)

    # 4
    p, m = check_shape_consistency(sidecar, args.clip_len)
    run_check("4. Shape consistency", p, m)

    # 5
    p, m = check_center_frame_correctness(sidecar)
    run_check("5. Center frame correctness", p, m)

    # 6
    p, m = check_center_source(sidecar)
    run_check("6. Center source", p, m)

    # 7
    p, m = check_center_valid_mask(sidecar)
    run_check("7. Center valid_mask", p, m)

    # 8
    p, m = check_center_box_match(sidecar, csv_lookup)
    run_check("8. Center box match", p, m)

    # 9
    p, m = check_box_range(sidecar)
    run_check("9. Box range", p, m)

    # 10
    p, m = check_box_validity(sidecar)
    run_check("10. Box validity", p, m)

    # 11
    p, m = check_frame_indices_range(sidecar, frame_counts)
    run_check("11. Frame indices range", p, m)

    # 12
    random.seed(42)
    p, m = check_frame_existence_spot(sidecar, frame_root, args.frame_spot_check)
    run_check("12. Frame existence spot check", p, m)

    # 13
    p, m = check_valid_mask_values(sidecar)
    run_check("13. Valid_mask values", p, m)

    # 14
    p, m = check_sources_values(sidecar)
    run_check("14. Sources values", p, m)

    # 15
    p, m = check_track_quality_consistency(sidecar)
    run_check("15. Track quality consistency", p, m)

    # 16
    p, m = check_at_least_one_label(sidecar, csv_lookup)
    run_check("16. At least one label", p, m)

    # 17
    p, m = check_no_nan(sidecar)
    run_check("17. No NaN", p, m)

    # ── Summary statistics ──
    logger.info("")
    logger.info("-" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("-" * 60)
    print_summary(sidecar, args.clip_len)

    # ── Final verdict ──
    n_passed = sum(1 for _, p, _ in checks if p)
    n_failed = len(checks) - n_passed
    failed_names = [name for name, p, _ in checks if not p]

    logger.info("")
    logger.info("=" * 60)
    if n_failed == 0:
        logger.info(f"ALL {len(checks)} CHECKS PASSED — sidecar is verified")
    else:
        logger.info(f"{n_failed} / {len(checks)} CHECKS FAILED — sidecar has issues")
        for fn in failed_names:
            logger.info(f"  FAILED: {fn}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()