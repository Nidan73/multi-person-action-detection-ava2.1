#!/usr/bin/env python
"""
temporal_track_dataset.py

PyTorch Dataset that joins the temporal track sidecar with the actor CSV.
Loads clip frames, temporal boxes, valid masks, and multi-hot labels
for downstream model training.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TemporalTrackDataset(Dataset):
    """
    Dataset for actor-centric temporal action recognition.

    Joins a temporal track sidecar (.pkl) with the processed actor CSV
    to provide per-sample clip frames, temporal bounding boxes,
    valid masks, and multi-hot action labels.

    Args:
        actor_csv_path:     Path to processed actor CSV.
        sidecar_pkl_path:   Path to merged temporal track sidecar .pkl.
        frame_root:         Root directory containing per-video frame folders.
        clip_len:           Expected clip length T (must match sidecar).
        min_track_quality:  Minimum track_quality to include a sample (0.0 = all).
        transform:          Optional transform applied to each frame tensor
                            AFTER scaling to [0, 1]. Receives (T, 3, H, W) tensor.

    Returns per __getitem__:
        frames:    (T, 3, 256, 256) float32 tensor, [0, 1] scaled
        boxes:     (T, 4) float32 tensor, normalized [x1, y1, x2, y2]
        valid_mask: (T,) int64 tensor, 1=observed 0=inferred
        labels:    (80,) float32 multi-hot tensor
        metadata:  dict with sample_id, video_id, timestamp, person_id,
                   center_frame, track_quality
    """

    def __init__(
        self,
        actor_csv_path: Union[str, Path],
        sidecar_pkl_path: Union[str, Path],
        frame_root: Union[str, Path],
        clip_len: int = 8,
        min_track_quality: float = 0.0,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.frame_root = Path(frame_root)
        self.clip_len = clip_len
        self.transform = transform

        # ── Load actor CSV ──
        self._label_lookup: Dict[str, np.ndarray] = {}
        df = self._load_actor_csv(str(actor_csv_path))
        label_cols = [f"label_{i}" for i in range(1, 81)]
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            labels = np.array(
                [int(row[lc]) for lc in label_cols], dtype=np.float32,
            )
            self._label_lookup[sid] = labels
        logger.info(f"Actor CSV: {len(self._label_lookup)} label entries loaded")

        # ── Load sidecar ──
        with open(str(sidecar_pkl_path), "rb") as f:
            full_sidecar: Dict[str, Dict[str, Any]] = pickle.load(f)
        logger.info(f"Sidecar: {len(full_sidecar)} total records loaded")

        # ── Filter by track quality ──
        self._sidecar: Dict[str, Dict[str, Any]] = {}
        for sid, rec in full_sidecar.items():
            if rec["track_quality"] >= min_track_quality:
                if sid in self._label_lookup:
                    self._sidecar[sid] = rec

        # ── Build ordered index ──
        self._sample_ids: List[str] = sorted(self._sidecar.keys())

        n_filtered = len(full_sidecar) - len(self._sample_ids)
        logger.info(
            f"Dataset: {len(self._sample_ids)} samples "
            f"(filtered {n_filtered} by min_track_quality={min_track_quality:.2f})"
        )

    @staticmethod
    def _load_actor_csv(csv_path: str) -> pd.DataFrame:
        """Load actor CSV with header auto-detection."""
        col_names = (
            ["sample_id", "video_id", "timestamp", "person_id",
             "x1", "y1", "x2", "y2", "center_frame"]
            + [f"label_{i}" for i in range(1, 81)]
        )
        df_peek = pd.read_csv(csv_path, nrows=1)
        if "video_id" in df_peek.columns and "center_frame" in df_peek.columns:
            return pd.read_csv(csv_path)
        return pd.read_csv(csv_path, header=None, names=col_names)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load one sample.

        Returns:
            frames:     (T, 3, 256, 256) float32
            boxes:      (T, 4) float32
            valid_mask: (T,) int64
            labels:     (80,) float32
            metadata:   dict
        """
        sid = self._sample_ids[idx]
        rec = self._sidecar[sid]

        # ── Load frames ──
        frames = self._load_clip_frames(
            rec["video_id"], rec["frame_indices"],
        )

        # ── Boxes ──
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32)  # (T, 4)

        # ── Valid mask ──
        valid_mask = torch.tensor(rec["valid_mask"], dtype=torch.int64)  # (T,)

        # ── Labels ──
        labels = torch.from_numpy(self._label_lookup[sid].copy())  # (80,)

        # ── Apply transform ──
        if self.transform is not None:
            frames = self.transform(frames)

        # ── Metadata ──
        metadata = {
            "sample_id":     sid,
            "video_id":      rec["video_id"],
            "timestamp":     rec["timestamp"],
            "person_id":     rec["person_id"],
            "center_frame":  rec["center_frame"],
            "track_quality": rec["track_quality"],
        }

        return frames, boxes, valid_mask, labels, metadata

    def _load_clip_frames(
        self, video_id: str, frame_indices: List[int],
    ) -> torch.Tensor:
        """
        Load T frames and return as (T, 3, 256, 256) float32 tensor in [0, 1].

        Missing frames are replaced with zero tensors.
        """
        tensors: List[torch.Tensor] = []

        for fi in frame_indices:
            path = self.frame_root / video_id / f"{fi:06d}.jpg"
            if path.exists():
                img = Image.open(path).convert("RGB")
                # Ensure 256x256
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.BILINEAR)
                # Convert to tensor: (3, H, W), float32, [0, 1]
                arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
                t = torch.from_numpy(arr).permute(2, 0, 1)     # (3, H, W)
            else:
                logger.warning(f"Missing frame: {path}")
                t = torch.zeros(3, 256, 256, dtype=torch.float32)

            tensors.append(t)

        return torch.stack(tensors, dim=0)  # (T, 3, 256, 256)

    def get_sample_ids(self) -> List[str]:
        """Return ordered list of sample_ids in this dataset."""
        return list(self._sample_ids)

    def get_record(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Return the raw sidecar record for a sample_id, or None."""
        return self._sidecar.get(sample_id)