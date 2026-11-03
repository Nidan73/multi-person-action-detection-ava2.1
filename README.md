# Lightweight Actor-Centric Action Recognition on AVA

Undergraduate thesis code (AIUB, 2025 to 2026, grade A+).

**Thesis title:** Lightweight Actor-Centric Action Recognition on the AVA Dataset Using Temporal and Geometric Features

**Team of three:** Nidan Alam, Koushik Saha, Abdullah Al Maruf. Nidan designed the model architecture and wrote the training and ablation code.

## Problem

For each person in a short video clip, predict which of the 80 AVA 2.1 action classes apply (multi-label), with a model small enough for edge devices.

## Data

AVA 2.1 videos and annotations, using only the part that could be downloaded. Frames are indexed at 8 fps (`preprocess_ava_actor_csv.py`). Videos and annotations are not included; get them from the official AVA site.

## Method

All scripts are in `Code/`.

1. `preprocess_ava_actor_csv.py`: turns the AVA annotation CSV into one row per actor and keyframe, matched to the extracted frames.
2. `detect_and_cache.py`: YOLOv8s person detection (image size 256, confidence 0.3), cached to disk.
3. `assemble_temporal_tracks.py`: links detections to each annotated actor over 8 frames by IoU with Hungarian matching (threshold 0.3). The keyframe uses the ground-truth box; gaps are interpolated or repeated, and a validity mask records which frames are real.
4. `models/actor_model.py`: frozen X3D-XS backbone (pytorchvideo), RoIAlign on each frame, then one of five heads:

| Mode | What it adds |
| --- | --- |
| baseline0 | Keyframe RoI only |
| baseline1 | Box-geometry gate (width, height, log area, centre, log aspect) |
| ablation2 | Learned attention over the 8 per-frame RoI features |
| ablation3 | Attention conditioned on box geometry, with a validity mask |
| ablation4 | Attention plus mask plus residual geometry gate, last backbone stage unfrozen, 2x RoI upsampling |

5. `train.py`: AdamW (learning rate 1e-3, weight decay 1e-4), cosine schedule, binary cross-entropy loss, batch size 8, 20 epochs, gradient clipping at 1.0, early stopping with patience 5. Settings live in `config.py`.
6. `evaluate.py`: evaluates a checkpoint on the validation split.

## Results

The metric is macro average precision over the 80 classes (torchmetrics `MultilabelAveragePrecision`) on the validation split of the downloaded subset. Actors use ground-truth keyframe boxes, and the best epoch is selected on the same split. This is not the official AVA frame-mAP, so the numbers are not comparable to published full-benchmark results.

| Run | Best val mAP (epoch) | Last epoch | Log |
| --- | --- | --- | --- |
| baseline0 | 9.61% (4) | 9.04% | `logs/baseline0_log.csv` |
| baseline1 | 11.10% (17) | 11.03% | `logs/baseline1_log.csv` |
| ablation2 | 11.21% (20) | 11.21% | `logs/ablation2_log.csv` |
| ablation3 | 10.91% (20) | 10.91% | `logs/ablation3_log.csv` |
| ablation4 (final) | 11.80% (13) | 11.57% | `logs/ablation4_v2_final_log.csv` |

The thesis report gives about 3.15 GFLOPs and over 90 FPS on one consumer GPU for the final model; the measurement script is not in this repository.

## How to run

`Code/config.py` still contains the Windows paths of the machine used for training. Change them first, then run the scripts in this order:

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `preprocess_ava_actor_csv.py` | Raw AVA annotation CSV to actor-level rows matched to extracted frames |
| 2 | `detect_and_cache.py` | Detect people with YOLOv8s and cache the boxes |
| 3 | `assemble_temporal_tracks.py` | Link detections into short tracks over time |
| 4 | `train.py` | Train (settings in `config.py`, model in `models/actor_model.py`) |
| 5 | `evaluate.py` | Evaluate a checkpoint on the validation split |

Running `models/actor_model.py` directly builds each of the five modes on random input as a quick self-test. `verify_*.py` and `debug_*.py` are sanity checks on the processed data.

## Layout

- `Code/`: pipeline scripts, dataset classes and the model (`Code/models/actor_model.py`).
- `checkpoints/`: best and last weights for each run.
- `logs/`: per-epoch CSV logs and TensorBoard event files for each run.
