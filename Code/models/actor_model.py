"""
actor_model.py — Actor-centric action model for AVA v2.1.

Supports 5 experimental modes in a single nn.Module:
    baseline0:  Keyframe RoI only
    baseline1:  Keyframe RoI + geometry gate
    ablation2:  Temporal pooling (feature-only scorer)
    ablation3:  Geometry-conditioned temporal attention (novelty)
    ablation4:  V2 Optimized System (res5 unfrozen + RoI upsample + A2 temporal base + mask + residual center gate)

Backbone: X3D-XS (frozen by default, from pytorchvideo via torch.hub)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

C_FEAT: int = 192
T_FEAT: int = 8
H_FEAT: int = 8
W_FEAT: int = 8
CENTER_IDX: int = 4
NUM_GEO_FEATURES: int = 6

VALID_MODES = {"baseline0", "baseline1", "ablation2", "ablation3", "ablation4"}

@dataclass
class ModelConfig:
    mode: str = "baseline0"  
    backbone_frozen: bool = True
    roi_output_size: int = 7
    geo_hidden: int = 32
    geo_out_dim: int = 32  
    temporal_hidden: int = 64
    head_hidden: int = 256
    head_dropout: float = 0.3
    mask_bias: float = -5.0
    learnable_mask_bias: bool = False
    num_classes: int = 80

class GeometryEncoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(NUM_GEO_FEATURES, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        return self.mlp(geo)


class TemporalScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  


class ActorActionModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if config.mode not in VALID_MODES:
            raise ValueError(f"Unknown mode '{config.mode}'. Must be one of {VALID_MODES}.")
        self.config = config
        self.mode = config.mode

        self._build_backbone()

        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        self.roi_pool = nn.AdaptiveAvgPool2d(1)

        if self.mode == "baseline1":
            self.geo_encoder = GeometryEncoder(hidden_dim=config.geo_hidden, output_dim=C_FEAT)
            self.temporal_scorer = None
            self.register_buffer("mask_bias", torch.tensor(0.0))
            
        elif self.mode == "ablation2":
            self.geo_encoder = None
            self.temporal_scorer = TemporalScorer(input_dim=C_FEAT, hidden_dim=config.temporal_hidden)
            self.register_buffer("mask_bias", torch.tensor(0.0))
            
        elif self.mode == "ablation3":
            self.geo_encoder = GeometryEncoder(hidden_dim=config.geo_hidden, output_dim=config.geo_out_dim)
            self.temporal_scorer = TemporalScorer(input_dim=C_FEAT + config.geo_out_dim, hidden_dim=config.temporal_hidden)
            self.register_buffer("mask_bias", torch.tensor(config.mask_bias))
            
        elif self.mode == "ablation4":
            self.geo_encoder = None 
            self.temporal_scorer = TemporalScorer(input_dim=C_FEAT, hidden_dim=config.temporal_hidden)

            self.geo_gate_encoder = GeometryEncoder(hidden_dim=config.geo_hidden, output_dim=C_FEAT)
            self.register_buffer("mask_bias", torch.tensor(config.mask_bias))
            
        else:
            self.geo_encoder = None
            self.temporal_scorer = None
            self.register_buffer("mask_bias", torch.tensor(0.0))

        self.head = nn.Sequential(
            nn.Linear(C_FEAT, config.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden, config.num_classes),
        )

    def _build_backbone(self) -> None:
        full_model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_xs", pretrained=True)
        self.backbone = nn.Sequential(*list(full_model.blocks[:5]))

        if self.config.backbone_frozen:
            self._freeze_backbone()

        if self.mode == "ablation4":
            self.unfreeze_res5()

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_res5(self) -> None:
        for param in self.backbone[4].parameters():
            param.requires_grad = True

    def train(self, mode: bool = True) -> "ActorActionModel":
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _compute_geometry(boxes: torch.Tensor) -> torch.Tensor:
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]

        width = x2 - x1
        height = y2 - y1
        log_area = torch.log(width * height + 1e-6)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        log_aspect = torch.log(width / (height + 1e-6) + 1e-6)

        return torch.stack([width, height, log_area, cx, cy, log_aspect], dim=-1)

    def _extract_roi_single(self, spatial_feat: torch.Tensor, boxes_t: torch.Tensor) -> torch.Tensor:
        B = spatial_feat.size(0)
        device = spatial_feat.device

        if self.mode == "ablation4":
            spatial_feat = F.interpolate(spatial_feat, scale_factor=2.0, mode='bilinear', align_corners=False)
            scale_w = float(W_FEAT) * 2.0
            scale_h = float(H_FEAT) * 2.0
        else:
            scale_w = float(W_FEAT)
            scale_h = float(H_FEAT)

        scaled_boxes = boxes_t.clone()
        scaled_boxes[:, 0] = scaled_boxes[:, 0] * scale_w  
        scaled_boxes[:, 1] = scaled_boxes[:, 1] * scale_h  
        scaled_boxes[:, 2] = scaled_boxes[:, 2] * scale_w  
        scaled_boxes[:, 3] = scaled_boxes[:, 3] * scale_h  

        batch_idx = torch.arange(B, device=device, dtype=torch.float32).unsqueeze(1)
        rois = torch.cat([batch_idx, scaled_boxes], dim=1)  

        roi_out = roi_align(
            spatial_feat,
            rois,
            output_size=(self.config.roi_output_size, self.config.roi_output_size),
            spatial_scale=1.0,
            aligned=True,
        )  

        pooled = self.roi_pool(roi_out).flatten(1)  
        return pooled

    def _extract_roi_center(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        spatial_feat = feat[:, :, CENTER_IDX, :, :]  
        boxes_center = boxes[:, CENTER_IDX, :]  
        return self._extract_roi_single(spatial_feat, boxes_center)

    def _extract_roi_all(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        roi_list = []
        for t in range(T_FEAT):
            spatial_feat = feat[:, :, t, :, :]  
            boxes_t = boxes[:, t, :]  
            pooled = self._extract_roi_single(spatial_feat, boxes_t)  
            roi_list.append(pooled)
        return torch.stack(roi_list, dim=1)  

    def forward(self, frames: torch.Tensor, boxes: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = frames.permute(0, 2, 1, 3, 4)
        x = (x - self.pixel_mean) / self.pixel_std
        feat = self.backbone(x)  

        if self.mode == "baseline0":
            actor_feat = self._extract_roi_center(feat, boxes)  
            logits = self.head(actor_feat)  

        elif self.mode == "baseline1":
            actor_feat = self._extract_roi_center(feat, boxes)  
            geo_6d = self._compute_geometry(boxes[:, CENTER_IDX, :])  
            geo_feat = self.geo_encoder(geo_6d)  
            geo_gate = torch.sigmoid(geo_feat)  
            actor_feat = actor_feat * geo_gate  
            logits = self.head(actor_feat)  

        elif self.mode == "ablation2":
            actor_feats = self._extract_roi_all(feat, boxes)  
            scores = self.temporal_scorer(actor_feats)  
            alpha = F.softmax(scores, dim=1)  
            pooled = (alpha.unsqueeze(-1) * actor_feats).sum(dim=1)  
            logits = self.head(pooled)  

        elif self.mode == "ablation3":
            actor_feats = self._extract_roi_all(feat, boxes)  
            geo_6d = self._compute_geometry(boxes)  
            geo_feats = self.geo_encoder(geo_6d)  
            
            combined = torch.cat([actor_feats, geo_feats], dim=-1)  
            scores = self.temporal_scorer(combined)  

            if valid_mask is not None:
                invalid = (valid_mask == 0).float()  
                scores = scores + invalid * self.mask_bias

            alpha = F.softmax(scores, dim=1)  
            pooled = (alpha.unsqueeze(-1) * actor_feats).sum(dim=1)  
            logits = self.head(pooled)  

        elif self.mode == "ablation4":
            actor_feats = self._extract_roi_all(feat, boxes)  
            
            scores = self.temporal_scorer(actor_feats) 

            if valid_mask is not None:
                invalid = (valid_mask == 0).float()  
                scores = scores + invalid * self.mask_bias

            alpha = F.softmax(scores, dim=1) 
            pooled = (alpha.unsqueeze(-1) * actor_feats).sum(dim=1)  

            center_geo_6d = self._compute_geometry(boxes[:, CENTER_IDX, :])  
            gate_feat = self.geo_gate_encoder(center_geo_6d) 
            geo_gate = torch.sigmoid(gate_feat) 
            
            pooled = pooled + (pooled * geo_gate)

            logits = self.head(pooled)  

        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mode in ["baseline0", "baseline1", "ablation2", "ablation3", "ablation4"]:
        print(f"\n{'=' * 60}")
        print(f"Testing mode: {mode}")
        print(f"{'=' * 60}")

        config = ModelConfig(mode=mode)
        model = ActorActionModel(config).to(device)

        B = 2
        frames = torch.randn(B, 8, 3, 256, 256, device=device)
        boxes = torch.rand(B, 8, 4, device=device)
        boxes[:, :, 2:] = boxes[:, :, :2] + 0.2
        boxes = boxes.clamp(0, 1)
        valid_mask = torch.ones(B, 8, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(frames, boxes, valid_mask)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        print(f"Output shape:     {logits.shape}")
        print(f"Total params:     {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Frozen params:    {frozen_params:,}")
        print("PASS")

    print(f"\n{'=' * 60}")
    print("ALL MODES PASSED")
    print(f"{'=' * 60}")