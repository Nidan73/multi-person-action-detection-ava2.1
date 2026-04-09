#!/usr/bin/env python
"""
debug_visualize_temporal_track.py

Visual debugging of temporal actor tracks.
Draws color-coded bounding boxes on clip frames and saves
horizontal strip images for inspection.

Colors:
    green  = "gt"     (ground truth at center frame)
    blue   = "det"    (detection-matched)
    orange = "interp" (interpolated)
    red    = "repeat" (nearest-repeat fallback)
"""

import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Source → color mapping (RGB)
SOURCE_COLORS = {
    "gt":     (0, 200, 0),
    "det":    (50, 120, 255),
    "interp": (255, 165, 0),
    "repeat": (220, 30, 30),
}

# Source → label for display
SOURCE_LABELS = {
    "gt":     "GT",
    "det":    "DET",
    "interp": "INTERP",
    "repeat": "REPEAT",
}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Visualize temporal actor tracks from sidecar."
    )
    p.add_argument("--sidecar_pkl", required=True, type=str,
                   help="Path to merged sidecar .pkl.")
    p.add_argument("--actor_csv", required=True, type=str,
                   help="Path to processed actor CSV.")
    p.add_argument("--frame_root", required=True, type=str,
                   help="Root directory of per-video frame folders.")
    p.add_argument("--output_dir", required=True, type=str,
                   help="Directory to save visualization strips.")
    p.add_argument("--mode", choices=["random", "worst", "best", "sample_id"],
                   default="random",
                   help="Sample selection mode (default: random).")
    p.add_argument("--n", type=int, default=10,
                   help="Number of samples for random/worst/best (default: 10).")
    p.add_argument("--sample_ids", nargs="+", default=None,
                   help="Explicit sample_id(s) when mode=sample_id.")
    p.add_argument("--log_file", type=str, default=None,
                   help="Optional log file path.")
    return p.parse_args()


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging."""
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
    """Load merged sidecar pickle."""
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
    df["sample_id"] = df["sample_id"].astype(str)
    logger.info(f"Loaded actor CSV: {len(df)} instances from {csv_path}")
    return df


# ──────────────────────────────────────────────
# Sample selection
# ──────────────────────────────────────────────

def select_samples(
    sidecar: Dict[str, Any],
    mode: str,
    n: int,
    sample_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Select sample_ids based on mode.

    Modes:
        random:    n random samples
        worst:     n lowest track_quality
        best:      n highest track_quality
        sample_id: explicit list
    """
    if mode == "sample_id":
        if not sample_ids:
            raise ValueError("--sample_ids required when mode=sample_id")
        missing = [s for s in sample_ids if s not in sidecar]
        if missing:
            logger.warning(f"{len(missing)} sample_ids not found in sidecar")
        return [s for s in sample_ids if s in sidecar]

    all_ids = list(sidecar.keys())

    if mode == "random":
        random.seed(42)
        return random.sample(all_ids, min(n, len(all_ids)))

    if mode == "worst":
        sorted_ids = sorted(all_ids, key=lambda s: sidecar[s]["track_quality"])
        return sorted_ids[:n]

    if mode == "best":
        sorted_ids = sorted(
            all_ids, key=lambda s: sidecar[s]["track_quality"], reverse=True,
        )
        return sorted_ids[:n]

    raise ValueError(f"Unknown mode: {mode}")


# ──────────────────────────────────────────────
# Frame loading
# ──────────────────────────────────────────────

def load_frame(frame_root: Path, video_id: str, frame_idx: int) -> Image.Image:
    """Load a single frame as a PIL Image. Returns grey image if missing."""
    path = frame_root / video_id / f"{frame_idx:06d}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    else:
        logger.warning(f"Missing frame: {path}")
        return Image.new("RGB", (256, 256), (128, 128, 128))


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────

def get_font(size: int = 12) -> ImageFont.FreeTypeFont:
    """Try to load a monospace font; fall back to default."""
    font_names = [
        "consola.ttf",    # Windows Consolas
        "cour.ttf",       # Windows Courier
        "arial.ttf",      # Windows Arial
        "DejaVuSansMono.ttf",  # Linux
    ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_annotated_frame(
    img: Image.Image,
    box: List[float],
    source: str,
    det_score: float,
    frame_idx: int,
    pos_in_clip: int,
    center_idx: int,
) -> Image.Image:
    """
    Draw a color-coded bounding box and annotation text on a frame.

    Args:
        img:         PIL Image (will be copied).
        box:         [x1, y1, x2, y2] normalized.
        source:      "gt", "det", "interp", or "repeat".
        det_score:   Detection confidence (0.0 if not "det").
        frame_idx:   1-based frame number.
        pos_in_clip: Position index in clip (0..T-1).
        center_idx:  Center position index.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = get_font(11)
    font_small = get_font(10)

    color = SOURCE_COLORS.get(source, (255, 255, 255))
    label = SOURCE_LABELS.get(source, source.upper())

    # Pixel coordinates
    px1 = int(box[0] * w)
    py1 = int(box[1] * h)
    px2 = int(box[2] * w)
    py2 = int(box[3] * h)

    # Draw box (2px width)
    for offset in range(2):
        draw.rectangle(
            [px1 - offset, py1 - offset, px2 + offset, py2 + offset],
            outline=color,
        )

    # Top-left label background
    text_line1 = f"f={frame_idx}"
    text_line2 = label
    if source == "det":
        text_line2 += f" {det_score:.2f}"

    # Background rectangle for text
    text_y = max(py1 - 24, 0)
    draw.rectangle([px1, text_y, px1 + 80, text_y + 22], fill=(0, 0, 0, 180))
    draw.text((px1 + 2, text_y + 1), text_line1, fill=(255, 255, 255), font=font_small)
    draw.text((px1 + 2, text_y + 11), text_line2, fill=color, font=font_small)

    # Mark center frame with a small indicator
    if pos_in_clip == center_idx:
        draw.rectangle([0, 0, 3, h - 1], fill=(0, 255, 0))
        draw.rectangle([w - 4, 0, w - 1, h - 1], fill=(0, 255, 0))

    return img


# ──────────────────────────────────────────────
# Strip building
# ──────────────────────────────────────────────

def build_strip(
    record: Dict[str, Any],
    frame_root: Path,
    label_str: str = "",
) -> Image.Image:
    """
    Build a horizontal strip of T annotated frames with a title bar.

    Returns a single PIL Image.
    """
    video_id = record["video_id"]
    frame_indices = record["frame_indices"]
    boxes = record["boxes"]
    sources = record["sources"]
    det_scores = record["det_scores"]
    center_idx = record["center_idx"]
    T = len(frame_indices)

    # Load and annotate each frame
    annotated: List[Image.Image] = []
    for pos in range(T):
        img = load_frame(frame_root, video_id, frame_indices[pos])
        img = img.resize((256, 256), Image.BILINEAR)
        img = draw_annotated_frame(
            img, boxes[pos], sources[pos], det_scores[pos],
            frame_indices[pos], pos, center_idx,
        )
        annotated.append(img)

    # Concatenate horizontally
    strip_w = 256 * T
    strip_h = 256
    strip = Image.new("RGB", (strip_w, strip_h))
    for i, img in enumerate(annotated):
        strip.paste(img, (i * 256, 0))

    # Title bar
    title_h = 36
    title_bar = Image.new("RGB", (strip_w, title_h), (30, 30, 30))
    draw = ImageDraw.Draw(title_bar)
    font = get_font(13)

    title_text = (
        f"sample={record['sample_id']}  "
        f"video={video_id}  "
        f"ts={record['timestamp']}  "
        f"person={record['person_id']}  "
        f"quality={record['track_quality']:.3f}  "
        f"center_iou={record['center_det_iou']:.3f}"
    )
    draw.text((6, 2), title_text, fill=(255, 255, 255), font=font)

    if label_str:
        draw.text((6, 18), f"labels: {label_str}", fill=(200, 200, 200), font=font)

    # Legend
    legend_x = strip_w - 340
    legend_items = [("GT", SOURCE_COLORS["gt"]),
                    ("DET", SOURCE_COLORS["det"]),
                    ("INTERP", SOURCE_COLORS["interp"]),
                    ("REPEAT", SOURCE_COLORS["repeat"])]
    for i, (lbl, clr) in enumerate(legend_items):
        x = legend_x + i * 80
        draw.rectangle([x, 4, x + 10, 14], fill=clr)
        draw.text((x + 14, 2), lbl, fill=clr, font=font)

    # Combine title + strip
    full = Image.new("RGB", (strip_w, title_h + strip_h))
    full.paste(title_bar, (0, 0))
    full.paste(strip, (0, title_h))

    return full


# ──────────────────────────────────────────────
# Label formatting
# ──────────────────────────────────────────────

def get_label_string(csv_row: Dict, max_labels: int = 10) -> str:
    """Format active label IDs as a compact string."""
    label_cols = [f"label_{i}" for i in range(1, 81)]
    active = []
    for lc in label_cols:
        if lc in csv_row and int(csv_row[lc]) == 1:
            active.append(lc.replace("label_", ""))
    if len(active) > max_labels:
        return ", ".join(active[:max_labels]) + f" (+{len(active) - max_labels})"
    return ", ".join(active) if active else "none"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    """Generate visualization strips for selected temporal tracks."""
    args = parse_args()
    setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Temporal Track Visualization")
    logger.info(f"  Sidecar     : {args.sidecar_pkl}")
    logger.info(f"  Actor CSV   : {args.actor_csv}")
    logger.info(f"  Frame root  : {args.frame_root}")
    logger.info(f"  Output dir  : {args.output_dir}")
    logger.info(f"  Mode        : {args.mode}")
    logger.info(f"  N           : {args.n}")
    logger.info("=" * 60)

    sidecar = load_sidecar(args.sidecar_pkl)
    df = load_actor_csv(args.actor_csv)
    csv_lookup = {str(row["sample_id"]): row for _, row in df.iterrows()}

    frame_root = Path(args.frame_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select samples
    selected = select_samples(sidecar, args.mode, args.n, args.sample_ids)
    logger.info(f"Selected {len(selected)} samples ({args.mode})")

    # Generate strips
    for i, sid in enumerate(selected, 1):
        rec = sidecar[sid]
        csv_row = csv_lookup.get(sid, None)
        label_str = get_label_string(csv_row) if csv_row is not None else ""

        strip = build_strip(rec, frame_root, label_str)

        # Sanitize filename (sample_id may contain special chars)
        safe_name = sid.replace("/", "_").replace("\\", "_")
        out_path = output_dir / f"{safe_name}.jpg"
        strip.save(str(out_path), quality=90)

        logger.info(
            f"  [{i}/{len(selected)}] {sid} "
            f"quality={rec['track_quality']:.3f} -> {out_path.name}"
        )

    logger.info(f"Done. {len(selected)} strips saved to {output_dir}")


if __name__ == "__main__":
    main()