"""
Diagnostic: determine exact X3D-XS feature map shape
and backbone structure for actor_model.py
"""
import torch

# ---- Try loading ----
print("="*60)
print("Attempting torch.hub.load...")
print("="*60)

try:
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "x3d_xs",
        pretrained=True
    )
    load_method = "torch.hub"
    print("SUCCESS: loaded via torch.hub")
except Exception as e:
    print(f"torch.hub FAILED: {e}")
    print("\nAttempting create_x3d...")
    try:
        from pytorchvideo.models.x3d import create_x3d
        model = create_x3d(
            input_clip_length=8,
            input_crop_size=256,
            model_num_class=400,
        )
        load_method = "create_x3d"
        print("SUCCESS: loaded via create_x3d (NO pretrained weights)")
    except Exception as e2:
        print(f"create_x3d ALSO FAILED: {e2}")
        exit(1)

print(f"\nLoad method: {load_method}")

# ---- Print structure ----
print("\n" + "="*60)
print("MODEL BLOCKS")
print("="*60)

if hasattr(model, 'blocks'):
    for i, block in enumerate(model.blocks):
        print(f"  blocks[{i}]: {block.__class__.__name__}")
else:
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")

# ---- Build backbone (everything except last block) ----
print("\n" + "="*60)
print("BACKBONE EXTRACTION")
print("="*60)

if hasattr(model, 'blocks'):
    num_blocks = len(model.blocks)
    print(f"Total blocks: {num_blocks}")
    print(f"Using blocks[0:{num_blocks-1}] as backbone (dropping last block = head)")
    backbone = torch.nn.Sequential(*list(model.blocks[:-1]))
else:
    print("ERROR: model has no .blocks attribute")
    print("Named children:", [n for n, _ in model.named_children()])
    exit(1)

# ---- Dummy forward pass ----
print("\n" + "="*60)
print("FEATURE MAP SHAPE")
print("="*60)

backbone.eval()
dummy = torch.zeros(1, 3, 8, 256, 256)

with torch.no_grad():
    feat = backbone(dummy)

print(f"Input shape:       {dummy.shape}")
print(f"  → (B, C, T, H, W) = (1, 3, 8, 256, 256)")
print(f"")
print(f"Feature map shape: {feat.shape}")
print(f"  → (B, C_feat, T_feat, H_feat, W_feat) = {tuple(feat.shape)}")
print(f"")
print(f"C_feat  = {feat.shape[1]}")
print(f"T_feat  = {feat.shape[2]}")
print(f"H_feat  = {feat.shape[3]}")
print(f"W_feat  = {feat.shape[4]}")

# ---- Center frame mapping ----
print("\n" + "="*60)
print("TEMPORAL ALIGNMENT")
print("="*60)

T_input = 8
T_feat = feat.shape[2]
center_input = 4  # 0-based, verified from sidecar

if T_feat == T_input:
    center_feat = center_input
    subsample_indices = list(range(T_input))
    print(f"T_feat == T_input == {T_input}")
    print(f"No temporal subsampling needed")
    print(f"Center feature index = {center_feat}")
else:
    step = T_input / T_feat
    subsample_indices = [int(round(i * step)) for i in range(T_feat)]
    # find which feature index maps closest to center_input
    center_feat = min(range(T_feat),
                      key=lambda i: abs(subsample_indices[i] - center_input))
    print(f"T_feat ({T_feat}) < T_input ({T_input})")
    print(f"Subsample step = {step}")
    print(f"Subsample indices (input positions): {subsample_indices}")
    print(f"Center feature index = {center_feat}")
    print(f"  maps to input index {subsample_indices[center_feat]}")

print(f"\nSubsample indices: {subsample_indices}")
print(f"Center in feature space: {center_feat}")

# ---- Parameter count ----
print("\n" + "="*60)
print("PARAMETER COUNTS")
print("="*60)

total = sum(p.numel() for p in backbone.parameters())
print(f"Backbone parameters: {total:,}")

# ---- Summary for prompt ----
print("\n" + "="*60)
print("COPY THIS INTO YOUR PROMPT")
print("="*60)
print(f"""
EXACT X3D-XS BACKBONE FACTS (verified on my machine):
- Load method: {load_method}
- blocks structure: {num_blocks} blocks, backbone = blocks[0:{num_blocks-1}]
- Input:  (B, 3, 8, 256, 256)
- Output: (B, {feat.shape[1]}, {feat.shape[2]}, {feat.shape[3]}, {feat.shape[4]})
- C_feat  = {feat.shape[1]}
- T_feat  = {feat.shape[2]}
- H_feat  = {feat.shape[3]}
- W_feat  = {feat.shape[4]}
- Temporal subsampling needed: {T_feat != T_input}
- Subsample indices: {subsample_indices}
- Center index in feature space: {center_feat}
- Backbone parameters: {total:,}
""")