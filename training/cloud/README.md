# Renting a GPU for the heavy runs

The two large models do not fit an 8GB laptop card. Everything else in the training
rig runs locally; this directory covers the rented-box case.

## What needs a bigger card

| Model | Resolution | VRAM | Where |
|---|---|---|---|
| SegFormer-B5 crack segmentation | 1024 | ~20 GB | rented 24 GB |
| YOLO11x structural (CODEBRIM) | 1024 | ~18 GB | rented 24 GB |
| YOLO11l solar | 1024 | ~13 GB | free Kaggle P100 |
| YOLO11l corrosion | 960 | ~12 GB | free Kaggle P100 |
| SegFormer-B2 crack segmentation | 512 | ~5 GB | local laptop |

The B2 local config exists so real, registrable weights can be produced without
paid compute. It is a smaller backbone at half the resolution, not a substitute for
the B5 run.

## Cost

A 4090 interruptible on Vast.ai runs about $0.29/hr, so roughly 31 GPU-hours for $9.
SegFormer-B5 @1024 is about 10 h and YOLO11x @1024 about 9 h, leaving ~10 h of slack
for reruns and validation. That fits, without much room for mistakes.

## Interruptible instances get reclaimed

This is the constraint that shapes everything here. Both trainers checkpoint every
epoch and resume, `vast_bootstrap.sh` always passes `--resume`, and setting
`CHECKPOINT_REMOTE` starts a background sync so a reclaim costs one epoch rather
than the whole run. Without that variable the checkpoints live only on the rented
box and vanish with it.

## Running it

```bash
export KAGGLE_API_TOKEN=...       # never commit these
export ROBOFLOW_API_KEY=...
export REPO_URL=...               # or upload the tree to /workspace/OpenDroneKit
export CHECKPOINT_REMOTE=...      # rclone or rsync target, strongly recommended

bash training/cloud/vast_bootstrap.sh prepare   # deps + data, trains nothing
bash training/cloud/vast_bootstrap.sh crack     # SegFormer-B5
bash training/cloud/vast_bootstrap.sh all       # both heavy models
```

## Bringing the weights home

The rented box produces `training/runs/<name>/` containing `best.pt`, the ONNX
export, and `export_report.json`. Copy that directory back, then:

```bash
python -m training.register --run training/runs/crack_segformer_b5 --key crack_segmentation
python -m training.register --list
```

`register.py` refuses any export whose ONNX-vs-torch parity check failed, and records
the real sha256, source run, datasets, and licences. Nothing is trusted because it
came from a GPU box.

## Credentials

Keep both keys in the environment. They are read from `KAGGLE_API_TOKEN` /
`~/.kaggle/access_token` and `ROBOFLOW_API_KEY`, and `.gitignore` covers `.env`,
`*.key`, `kaggle.json`, `access_token`, and `secrets.json`. A rented box is shared
infrastructure: rotate both keys once the run is finished.
