# Lightweight Actor-Centric Action Recognition on AVA

Undergraduate thesis code (AIUB, 2025-2026, grade A+). A small model that recognises what each person in a video clip is doing, built to be fast enough for edge devices.

**Thesis title:** Lightweight Actor-Centric Action Recognition on the AVA Dataset Using Temporal and Geometric Features

**Team of three:** Nidan Alam, Koushik Saha, Abdullah Al Maruf. Nidan designed the model architecture and wrote the training and ablation code.

## What it does

For every detected person (actor) in a clip, the model predicts the action labels from the AVA dataset.

- **Backbone:** X3D-XS video network.
- **Temporal RoIAlign:** pools features around each actor over 8 frames.
- **Box-geometry gate:** adds the actor's box position and size as a second signal.
- **Result:** validation mAP 11.8% at about 3.15 GFLOPs, over 90 frames per second on one consumer GPU. Each component was added one at a time and measured (see the ablation checkpoints).

The numbers are on the part of AVA 2.1 that could be downloaded, not the full benchmark, so they are not comparable to published full-dataset results.

## Code

The scripts in `Code/` follow this order:

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `preprocess_ava_actor_csv.py` | Turn the raw AVA annotation CSV into actor-level rows matched to extracted frames |
| 2 | `detect_and_cache.py` | Detect people with YOLOv8s and cache the boxes |
| 3 | `assemble_temporal_tracks.py` | Link detections into short tracks over time |
| 4 | `train.py` | Train (settings in `config.py`, model in `models/actor_model.py`) |
| 5 | `evaluate.py` | Evaluate a checkpoint on the validation split |

`verify_*.py` and `debug_*.py` are sanity checks on the processed data.

## Notes

- `config.py` still contains the Windows paths of the machine used for training. Change them before running.
- The AVA videos and annotations are not included; download them from the official AVA site.
- `checkpoints/` holds the trained and ablation weights.
