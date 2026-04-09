#!/usr/bin/env python3
"""
debug_visualize_sample.py

Loads one sample from AVAActorDataset, draws the actor box on the center
frame, prints metadata and labels, and saves the result to disk.
No GUI / display required.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise one AVA actor-dataset sample")
    p.add_argument("--processed_csv", type=str, required=True)
    p.add_argument("--frames_root", type=str, required=True)
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index of the sample to visualise (default: 0)")
    p.add_argument("--output", type=str, default="debug_sample.jpg",
                   help="Output image path (default: debug_sample.jpg)")
    p.add_argument("--clip_length", type=int, default=8)
    p.add_argument("--temporal_stride", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── import dataset (same directory or PYTHONPATH) ──
    from ava_actor_dataset import AVAActorDataset

    print(f"[INFO] Loading dataset from {args.processed_csv}")
    ds = AVAActorDataset(
        processed_csv=args.processed_csv,
        frames_root=args.frames_root,
        clip_length=args.clip_length,
        temporal_stride=args.temporal_stride,
        transform=None,
    )

    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        print(f"[ERROR] sample_idx={args.sample_idx} out of range [0, {len(ds) - 1}]")
        sys.exit(1)

    clip, box, labels, meta = ds[args.sample_idx]

    # ── print info ──
    print(f"\n{'=' * 50}")
    print(f"  sample_idx:    {args.sample_idx}")
    print(f"  sample_id:     {meta['sample_id']}")
    print(f"  video_id:      {meta['video_id']}")
    print(f"  timestamp:     {meta['timestamp']}s")
    print(f"  person_id:     {meta['person_id']}")
    print(f"  center_frame:  {meta['center_frame']}")
    print(f"  frame_indices: {meta['frame_indices']}")
    print(f"  clip shape:    {tuple(clip.shape)}  (C, T, H, W)")
    print(f"  box (norm):    [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")

    active_ids: List[int] = [i + 1 for i in range(80) if labels[i] > 0.5]
    print(f"  active labels: {active_ids}  ({len(active_ids)} actions)")
    print(f"{'=' * 50}\n")

    # ── find center frame in clip ──
    # center_frame should match one of frame_indices; fall back to middle
    fi = meta["frame_indices"]
    cf = meta["center_frame"]
    try:
        clip_pos = fi.index(cf)
    except ValueError:
        clip_pos = len(fi) // 2

    # ── extract image ──
    center_tensor = clip[:, clip_pos, :, :]                        # (C, H, W)
    img_np = (center_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np, mode="RGB")

    # ── draw box ──
    W, H = img.size                                                 # PIL: (W, H)
    x1_px = int(box[0].item() * W)
    y1_px = int(box[1].item() * H)
    x2_px = int(box[2].item() * W)
    y2_px = int(box[3].item() * H)

    draw = ImageDraw.Draw(img)
    for offset in range(3):                                         # thick outline
        draw.rectangle(
            [x1_px - offset, y1_px - offset, x2_px + offset, y2_px + offset],
            outline="red",
        )

    # ── label text ──
    label_str = ", ".join(str(a) for a in active_ids)
    text = f"Actions: [{label_str}]"
    text_y = max(0, y1_px - 16)
    draw.text((x1_px, text_y), text, fill="yellow")

    # ── frame-index annotation (top-left) ──
    fi_str = f"frame {cf:06d}  clip: {fi}"
    draw.text((4, 4), fi_str, fill="cyan")

    # ── save ──
    out_path = Path(args.output)
    img.save(out_path, quality=95)
    print(f"[SAVED] {out_path.resolve()}")


if __name__ == "__main__":
    main()