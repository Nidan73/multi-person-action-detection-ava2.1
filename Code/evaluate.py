"""
evaluate.py — Standalone evaluation for actor-centric action model.

Usage:
    python evaluate.py --checkpoint E:\\Ava\\checkpoints\\baseline0_best.pt
    python evaluate.py --checkpoint E:\\Ava\\checkpoints\\ablation3_best.pt --output_json E:\\Ava\\outputs\\results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from models.actor_model import ActorActionModel
from temporal_track_dataset import TemporalTrackDataset


# ──────────────────────────────────────────────────────────
# Collate (identical to train.py)
# ──────────────────────────────────────────────────────────
def custom_collate(
    batch: List[Tuple],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List]]:
    """Stack tensors; merge metadata dicts into dict-of-lists."""
    frames_l, boxes_l, mask_l, labels_l = [], [], [], []
    merged_meta: Dict[str, List[Any]] = {}

    for frames, boxes, vmask, labels, meta in batch:
        frames_l.append(frames)
        boxes_l.append(boxes)
        mask_l.append(vmask)
        labels_l.append(labels)
        for k, v in meta.items():
            merged_meta.setdefault(k, []).append(v)

    return (
        torch.stack(frames_l),
        torch.stack(boxes_l),
        torch.stack(mask_l),
        torch.stack(labels_l),
        merged_meta,
    )


# ──────────────────────────────────────────────────────────
# Track-quality buckets
# ──────────────────────────────────────────────────────────
QUALITY_BUCKETS: List[Tuple[str, float, float]] = [
    ("tq < 0.50",            0.0,   0.50),
    ("0.50 <= tq < 0.75",    0.50,  0.75),
    ("0.75 <= tq < 1.00",    0.75,  1.00),
    ("tq == 1.00",           1.00,  1.001),   # tiny epsilon
]


def _bucket_mAP(
    preds: torch.Tensor,
    labels: torch.Tensor,
    tq_vals: List[float],
    num_labels: int,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Compute macro mAP per track-quality bucket."""
    tq = torch.tensor(tq_vals, dtype=torch.float32)
    out: Dict[str, Dict[str, Any]] = {}

    for name, lo, hi in QUALITY_BUCKETS:
        mask = (tq >= lo) & (tq < hi)
        cnt = int(mask.sum().item())

        if cnt < 10:
            out[name] = {"mAP": float("nan"), "count": cnt}
            continue

        m = MultilabelAveragePrecision(
            num_labels=num_labels, average="macro",
        ).to(device)
        m.update(preds[mask].to(device), labels[mask].to(device))

        try:
            val = m.compute().item()
        except Exception:
            val = float("nan")

        out[name] = {"mAP": val, "count": cnt}

    return out


# ──────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    val_actor_csv: str,
    val_sidecar_pkl: str,
    val_frame_root: str,
    batch_size: int,
    num_workers: int,
    output_json: Optional[Path],
) -> None:
    """Run full-val evaluation, print results, optionally save JSON."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ───────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = ModelConfig(**ckpt["model_config"])
    print(f"Mode:        {model_cfg.mode}")
    print(f"Trained for: {ckpt['epoch']} epoch(s)")
    print(f"Best mAP:    {ckpt['best_val_mAP']:.4f}")

    # ── model ─────────────────────────────────
    model = ActorActionModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tp = sum(p.numel() for p in model.parameters())
    trp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {tp:,}")
    print(f"Trainable params: {trp:,}")

    # ── dataset ───────────────────────────────
    val_ds = TemporalTrackDataset(
        actor_csv_path=val_actor_csv,
        sidecar_pkl_path=val_sidecar_pkl,
        frame_root=val_frame_root,
        clip_len=8,
        min_track_quality=0.0,
        transform=None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
        persistent_workers=num_workers > 0,
    )
    print(f"Val samples: {len(val_ds):,}")
    print(f"Val batches: {len(val_loader):,}")

    # ── inference ─────────────────────────────
    macro_metric = MultilabelAveragePrecision(
        num_labels=model_cfg.num_classes, average="macro",
    ).to(device)
    perclass_metric = MultilabelAveragePrecision(
        num_labels=model_cfg.num_classes, average=None,
    ).to(device)

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_tq: List[float] = []

    for frames, boxes, vmask, labels, meta in tqdm(
        val_loader, desc="Evaluating", dynamic_ncols=True,
    ):
        frames = frames.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        vmask = vmask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames, boxes, vmask)
        preds = torch.sigmoid(logits)
        labels_int = labels.int()

        macro_metric.update(preds, labels_int)
        perclass_metric.update(preds, labels_int)

        all_preds.append(preds.cpu())
        all_labels.append(labels_int.cpu())

        if "track_quality" in meta:
            for v in meta["track_quality"]:
                all_tq.append(v.item() if isinstance(v, torch.Tensor) else float(v))

    # ── overall ───────────────────────────────
    overall_mAP = macro_metric.compute().item()
    per_class_ap = perclass_metric.compute().cpu().tolist()      # length-80 list

    all_preds_t = torch.cat(all_preds)
    all_labels_t = torch.cat(all_labels)

    # ── print ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EVALUATION RESULTS — {model_cfg.mode}")
    print(f"{'=' * 70}")
    print(f"Overall mAP (macro): {overall_mAP:.4f}")
    print(f"Total samples:       {len(all_preds_t):,}")

    # per-class sorted
    indexed = [(i + 1, ap) for i, ap in enumerate(per_class_ap)]
    indexed.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'─' * 50}")
    print(f"Per-class AP  (sorted descending)")
    print(f"{'─' * 50}")
    print(f" {'Class':>5}   {'AP':>8}")
    print(f"{'─' * 50}")
    for cid, ap in indexed:
        ap_s = f"{ap:.4f}" if ap == ap else "   nan"     # NaN check
        print(f"  {cid:>4}    {ap_s:>8}")

    # bucketed
    if all_tq:
        print(f"\n{'─' * 50}")
        print(f"mAP by track-quality bucket")
        print(f"{'─' * 50}")
        bucketed = _bucket_mAP(
            all_preds_t, all_labels_t, all_tq,
            model_cfg.num_classes, device,
        )
        for bname, info in bucketed.items():
            mv = info["mAP"]
            ms = f"{mv:.4f}" if mv == mv else "   nan"
            print(f"  {bname:<25}  mAP: {ms}  (n={info['count']:,})")
    else:
        print("\n[WARNING] No track_quality in metadata — skipping bucketed eval.")

    # ── optional JSON ─────────────────────────
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        blob: Dict[str, Any] = {
            "model_mode": model_cfg.mode,
            "checkpoint": str(checkpoint_path),
            "epoch": ckpt["epoch"],
            "overall_mAP": overall_mAP,
            "total_samples": len(all_preds_t),
            "per_class_AP": {str(c): a for c, a in indexed},
        }
        if all_tq:
            blob["bucketed_mAP"] = bucketed
        with open(output_json, "w") as fh:
            json.dump(blob, fh, indent=2)
        print(f"\nResults saved → {output_json}")

    print(f"\n{'=' * 70}")
    print("Evaluation complete.")
    print(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
def main() -> None:
    defaults = TrainConfig()

    parser = argparse.ArgumentParser(
        description="Evaluate actor-centric action model on AVA v2.1 val",
    )
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--val_actor_csv",   type=str, default=defaults.val_actor_csv)
    parser.add_argument("--val_sidecar_pkl", type=str, default=defaults.val_sidecar_pkl)
    parser.add_argument("--val_frame_root",  type=str, default=defaults.val_frame_root)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--output_json",     type=str, default=None)

    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        val_actor_csv=args.val_actor_csv,
        val_sidecar_pkl=args.val_sidecar_pkl,
        val_frame_root=args.val_frame_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_json=Path(args.output_json) if args.output_json else None,
    )


if __name__ == "__main__":
    main()