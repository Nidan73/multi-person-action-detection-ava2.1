"""
train.py — Training loop for actor-centric action model on AVA v2.1.

Usage:
    python train.py --mode baseline0 --experiment_name baseline0 --batch_size 8
    python train.py --mode ablation3 --experiment_name ablation3 --epochs 20
    python train.py --resume E:\\Ava\\checkpoints\\baseline0_last.pt
    python train.py --mode baseline0 --debug --batch_size 4 --epochs 2

All ModelConfig and TrainConfig fields are CLI-overridable.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from models.actor_model import ActorActionModel
from temporal_track_dataset import TemporalTrackDataset


# ──────────────────────────────────────────────────────────
# Custom collate — merges metadata dicts into dict-of-lists
# ──────────────────────────────────────────────────────────
def custom_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List]]:
    """Stack tensors normally; merge per-sample metadata dicts into
    a single dict whose values are Python lists.
    """
    frames_list: List[torch.Tensor] = []
    boxes_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    merged_meta: Dict[str, List[Any]] = {}

    for frames, boxes, valid_mask, labels, meta in batch:
        frames_list.append(frames)
        boxes_list.append(boxes)
        mask_list.append(valid_mask)
        labels_list.append(labels)

        for k, v in meta.items():
            merged_meta.setdefault(k, []).append(v)

    return (
        torch.stack(frames_list),
        torch.stack(boxes_list),
        torch.stack(mask_list),
        torch.stack(labels_list),
        merged_meta,
    )


# ──────────────────────────────────────────────────────────
# Argparse — every config field is overridable
# ──────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser from ModelConfig + TrainConfig fields."""
    parser = argparse.ArgumentParser(
        description="Train actor-centric action model on AVA v2.1",
    )

    # Safely handle 'from __future__ import annotations'
    for dc_cls in (ModelConfig, TrainConfig):
        for f in fields(dc_cls):
            name = f"--{f.name}"
            ft = f.type if isinstance(f.type, str) else f.type.__name__

            if ft == "bool":
                parser.add_argument(name, action="store_true", default=None)
            elif ft == "float":
                parser.add_argument(name, type=float, default=None)
            elif ft == "int":
                parser.add_argument(name, type=int, default=None)
            else:
                parser.add_argument(name, type=str, default=None)

    return parser


def apply_overrides(
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> None:
    """Mutate *model_cfg* and *train_cfg* in-place from CLI overrides."""
    args_dict = vars(args)

    for dc_obj, dc_cls in [(model_cfg, ModelConfig), (train_cfg, TrainConfig)]:
        for f in fields(dc_cls):
            val = args_dict.get(f.name)
            if val is not None:
                setattr(dc_obj, f.name, val)


# ──────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────
def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Construct train and val DataLoaders."""
    train_ds = TemporalTrackDataset(
        actor_csv_path=cfg.train_actor_csv,
        sidecar_pkl_path=cfg.train_sidecar_pkl,
        frame_root=cfg.train_frame_root,
        clip_len=8,
        min_track_quality=cfg.min_track_quality,
        transform=None,
    )
    val_ds = TemporalTrackDataset(
        actor_csv_path=cfg.val_actor_csv,
        sidecar_pkl_path=cfg.val_sidecar_pkl,
        frame_root=cfg.val_frame_root,
        clip_len=8,
        min_track_quality=0.0,          # always evaluate on ALL val samples
        transform=None,
    )

    if cfg.debug:
        n_tr = min(cfg.debug_samples, len(train_ds))
        n_va = min(cfg.debug_samples, len(val_ds))
        train_ds = Subset(train_ds, list(range(n_tr)))
        val_ds = Subset(val_ds, list(range(n_va)))
        print(f"[DEBUG] Subset — {n_tr} train / {n_va} val samples")

    common = dict(
        pin_memory=True,
        num_workers=cfg.num_workers,
        collate_fn=custom_collate,
        persistent_workers=cfg.num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────
# CSV logger
# ──────────────────────────────────────────────────────────
class CSVLogger:
    """Append-mode CSV logger for per-epoch metrics."""

    COLUMNS = ["epoch", "train_loss", "val_loss", "val_mAP", "lr", "time_seconds"]

    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            with open(self.path, "w", newline="") as fh:
                csv.writer(fh).writerow(self.COLUMNS)

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_mAP: float,
        lr: float,
        elapsed: float,
    ) -> None:
        with open(self.path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_mAP:.4f}",
                f"{lr:.2e}",
                f"{elapsed:.1f}",
            ])


# ──────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_mAP: float,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_mAP": best_val_mAP,
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


# ──────────────────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────────────────
def train_one_epoch(
    model: ActorActionModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
) -> float:
    """Returns average training loss for this epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Train {epoch:02d}/{total_epochs:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for frames, boxes, valid_mask, labels, _meta in pbar:
        frames = frames.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(frames, boxes, valid_mask)
        loss = criterion(logits, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────
# One validation epoch
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch(
    model: ActorActionModel,
    loader: DataLoader,
    criterion: nn.Module,
    metric: MultilabelAveragePrecision,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Returns (avg_val_loss, val_mAP)."""
    model.eval()
    metric.reset()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Val   {epoch:02d}/{total_epochs:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for frames, boxes, valid_mask, labels, _meta in pbar:
        frames = frames.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames, boxes, valid_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        preds = torch.sigmoid(logits)
        metric.update(preds, labels.int())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    val_mAP = metric.compute().item()
    return avg_loss, val_mAP


# ──────────────────────────────────────────────────────────
# Main training driver
# ──────────────────────────────────────────────────────────
def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    """Full training pipeline: data → train loop → checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── directories ───────────────────────────
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    log_dir = Path(train_cfg.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / f"{train_cfg.experiment_name}_best.pt"
    last_path = ckpt_dir / f"{train_cfg.experiment_name}_last.pt"
    csv_path = log_dir / f"{train_cfg.experiment_name}_log.csv"

    # ── model ─────────────────────────────────
    model = ActorActionModel(model_cfg).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model mode:       {model_cfg.mode}")
    print(f"Total params:     {total_p:,}")
    print(f"Trainable params: {train_p:,}")
    print(f"Frozen params:    {total_p - train_p:,}")

    # ── optimizer / scheduler ─────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs,
    )

    # ── loss / metric ─────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    metric = MultilabelAveragePrecision(
        num_labels=model_cfg.num_classes, average="macro",
    ).to(device)

    # ── resume ────────────────────────────────
    start_epoch = 1
    best_mAP = 0.0
    no_improve = 0

    if train_cfg.resume:
        rp = Path(train_cfg.resume)
        if rp.exists():
            print(f"Resuming from: {rp}")
            ckpt = load_checkpoint(rp, device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_mAP = ckpt["best_val_mAP"]
            print(f"  Resumed at epoch {start_epoch}, best mAP so far: {best_mAP:.4f}")
        else:
            print(f"WARNING: resume path not found: {rp} — training from scratch.")

    # ── data ──────────────────────────────────
    print("Loading datasets ...")
    train_loader, val_loader = build_dataloaders(train_cfg)
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val   batches: {len(val_loader):,}")

    # ── logging ───────────────────────────────
    tb_writer = SummaryWriter(log_dir=str(log_dir / train_cfg.experiment_name))
    csv_logger = CSVLogger(csv_path)

    print(f"\n{'=' * 70}")
    print(f"Experiment : {train_cfg.experiment_name}")
    print(f"Mode       : {model_cfg.mode}")
    print(f"Epochs     : {train_cfg.epochs}")
    print(f"Batch size : {train_cfg.batch_size}")
    print(f"LR         : {train_cfg.lr}")
    print(f"Debug      : {train_cfg.debug}")
    print(f"{'=' * 70}\n")

    # ── epoch loop ────────────────────────────
    experiment_start = time.time()
    epoch_times: list[float] = []

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, train_cfg.grad_clip, epoch, train_cfg.epochs,
        )

        val_loss, val_mAP = validate_one_epoch(
            model, val_loader, criterion, metric,
            device, epoch, train_cfg.epochs,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        # ── timing estimates ──────────────────
        total_elapsed = time.time() - experiment_start
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_remaining = train_cfg.epochs - epoch
        eta_seconds = avg_epoch_time * epochs_remaining

        total_h, total_m = divmod(int(total_elapsed), 3600)
        total_m = total_m // 60
        eta_h, eta_m = divmod(int(eta_seconds), 3600)
        eta_m = eta_m // 60

        # TensorBoard
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/mAP", val_mAP, epoch)
        tb_writer.add_scalar("lr", lr_now, epoch)

        # CSV
        csv_logger.log(epoch, train_loss, val_loss, val_mAP, lr_now, elapsed)

        # Console — with timing
        print(
            f"Epoch {epoch:02d}/{train_cfg.epochs:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_mAP: {val_mAP:.4f} | "
            f"lr: {lr_now:.2e} | "
            f"{elapsed:.0f}s | "
            f"total: {total_h}h{total_m:02d}m | "
            f"ETA: {eta_h}h{eta_m:02d}m"
        )

        # always save last
        save_checkpoint(
            last_path, model, optimizer, scheduler,
            epoch, best_mAP, model_cfg, train_cfg,
        )

        # best?
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            no_improve = 0
            save_checkpoint(
                best_path, model, optimizer, scheduler,
                epoch, best_mAP, model_cfg, train_cfg,
            )
            print(f"  * New best mAP: {val_mAP:.4f} — saved")
        else:
            no_improve += 1

        # early stopping
        if no_improve >= train_cfg.early_stop_patience:
            print(
                f"\nEarly stopping: no improvement for "
                f"{train_cfg.early_stop_patience} epochs."
            )
            break

    tb_writer.close()

    # ── final summary with total time ─────────
    total_time = time.time() - experiment_start
    final_h, final_rem = divmod(int(total_time), 3600)
    final_m = final_rem // 60

    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"Total time:      {final_h}h{final_m:02d}m")
    print(f"Best val mAP:    {best_mAP:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"CSV log:         {csv_path}")
    print(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    apply_overrides(args, model_cfg, train_cfg)

    train(model_cfg, train_cfg)


if __name__ == "__main__":
    print("[DEBUG] Successfully imported modules. Starting main()...")
    main()