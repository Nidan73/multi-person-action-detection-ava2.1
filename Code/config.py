"""
config.py — Centralized training configuration for AVA actor-centric experiments.

ModelConfig is imported from models.actor_model (single source of truth).
TrainConfig holds all training / data / logging settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.actor_model import ModelConfig  # noqa: F401 — re-export


@dataclass
class TrainConfig:
    """Training configuration with all paths, hyperparameters, and flags."""

    # ── data paths ────────────────────────────
    train_actor_csv: str = r"E:\Ava\New folder\annotation\ava_train_actor.csv"
    val_actor_csv: str = r"E:\Ava\New folder\annotation\ava_val_actor.csv"
    train_sidecar_pkl: str = r"E:\Ava\New folder\temporal_tracks\ava_train_temporal_tracks.pkl"
    val_sidecar_pkl: str = r"E:\Ava\New folder\temporal_tracks\ava_val_temporal_tracks.pkl"
    train_frame_root: str = r"E:\Ava\New folder\frame\train"
    val_frame_root: str = r"E:\Ava\New folder\frame\val"

    # ── training hyperparameters ──────────────
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 5

    # ── data filtering ────────────────────────
    min_track_quality: float = 0.0

    # ── output paths ──────────────────────────
    checkpoint_dir: str = r"E:\Ava\checkpoints"
    log_dir: str = r"E:\Ava\logs"
    experiment_name: str = "baseline0"

    # ── resume ────────────────────────────────
    resume: str = ""

    # ── debug ─────────────────────────────────
    debug: bool = False
    debug_samples: int = 1000