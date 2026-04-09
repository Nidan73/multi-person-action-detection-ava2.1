#!/usr/bin/env python3
"""
ava_actor_dataset.py

PyTorch Dataset that reads a processed actor-level CSV (produced by
preprocess_ava_actor_csv.py) and loads clips of frames from on-disk
frame folders.

Returns per sample:
    clip   : (C, T, H, W) float32 tensor in [0, 1]
    box    : (4,) float32 tensor  [x1, y1, x2, y2] normalised
    labels : (80,) float32 multi-hot tensor
    meta   : dict with video_id, timestamp, person_id, center_frame,
             sample_id, frame_indices
"""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AVAActorDataset(Dataset):
    """Actor-centric clip dataset for AVA-style action classification."""

    def __init__(
        self,
        processed_csv: str,
        frames_root: str,
        clip_length: int = 8,
        temporal_stride: int = 1,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            processed_csv:   Path to the processed actor-level CSV (with header).
            frames_root:     Root dir that contains <video_id>/<frame>.jpg folders.
            clip_length:     Number of frames per clip (T).
            temporal_stride: Temporal stride between sampled frames.
            transform:       Optional callable applied to the clip tensor (C,T,H,W)
                             *after* [0,1] scaling.  Use this for normalisation later.
        """
        self.frames_root     = Path(frames_root)
        self.clip_length     = clip_length
        self.temporal_stride = temporal_stride
        self.transform       = transform

        self.samples: List[Dict[str, Any]] = self._load_csv(processed_csv)
        self.frame_counts: Dict[str, int]  = self._scan_frame_counts()

    # ------------------------------------------------------------------ io

    def _load_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Parse processed CSV into a list of sample dicts."""
        samples: List[Dict[str, Any]] = []
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {path}")

        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                labels = torch.zeros(80, dtype=torch.float32)
                for i in range(1, 81):
                    labels[i - 1] = float(row[f"label_{i}"])

                samples.append({
                    "sample_id":    row["sample_id"],
                    "video_id":     row["video_id"],
                    "timestamp":    int(row["timestamp"]),
                    "person_id":    int(row["person_id"]),
                    "x1":           float(row["x1"]),
                    "y1":           float(row["y1"]),
                    "x2":           float(row["x2"]),
                    "y2":           float(row["y2"]),
                    "center_frame": int(row["center_frame"]),
                    "labels":       labels,
                })

        print(f"[AVAActorDataset] Loaded {len(samples):,} samples from {path.name}")
        return samples

    def _scan_frame_counts(self) -> Dict[str, int]:
        """Count .jpg files in each video folder (runs once at init)."""
        video_ids = sorted({s["video_id"] for s in self.samples})
        counts: Dict[str, int] = {}
        missing_dirs = 0
        for vid in video_ids:
            vid_dir = self.frames_root / vid
            if vid_dir.is_dir():
                counts[vid] = sum(1 for _ in vid_dir.glob("*.jpg"))
            else:
                print(f"[WARN] Video directory missing: {vid_dir}")
                counts[vid] = 0
                missing_dirs += 1

        print(f"[AVAActorDataset] Scanned {len(counts)} video dirs "
              f"({missing_dirs} missing)")
        return counts

    # -------------------------------------------------------- frame logic

    def _get_clip_indices(self, center: int, max_frame: int) -> List[int]:
        """
        Compute T frame indices centred on *center*, clamped to [1, max_frame].

        For T=8, stride=1:  indices = [center-3 .. center+4]   (8 frames)
        """
        total_span = (self.clip_length - 1) * self.temporal_stride
        start = center - total_span // 2
        indices = [start + i * self.temporal_stride for i in range(self.clip_length)]
        return [max(1, min(idx, max_frame)) for idx in indices]

    def _load_frame(self, video_id: str, frame_idx: int) -> torch.Tensor:
        """Load one frame -> float32 tensor (C, H, W) in [0, 1]."""
        fp = self.frames_root / video_id / f"{frame_idx:06d}.jpg"
        if not fp.exists():
            raise FileNotFoundError(
                f"Frame not found: {fp}  "
                f"(video={video_id}, idx={frame_idx})"
            )
        img = Image.open(fp).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)

    # -------------------------------------------------------- Dataset API

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        s = self.samples[idx]
        video_id     = s["video_id"]
        center_frame = s["center_frame"]
        max_frame    = self.frame_counts.get(video_id, 7200)

        # --- clip ---
        frame_indices = self._get_clip_indices(center_frame, max_frame)
        frames = [self._load_frame(video_id, fi) for fi in frame_indices]
        clip = torch.stack(frames, dim=1)          # (C, T, H, W)

        if self.transform is not None:
            clip = self.transform(clip)

        # --- box ---
        box = torch.tensor(
            [s["x1"], s["y1"], s["x2"], s["y2"]], dtype=torch.float32
        )

        # --- labels ---
        labels = s["labels"]                       # (80,) float32

        # --- metadata ---
        meta = {
            "video_id":      video_id,
            "timestamp":     s["timestamp"],
            "person_id":     s["person_id"],
            "center_frame":  center_frame,
            "sample_id":     s["sample_id"],
            "frame_indices": frame_indices,
        }

        return clip, box, labels, meta