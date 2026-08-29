#!/usr/bin/env bash
# Bootstrap a rented GPU box (Vast.ai / any Ubuntu + CUDA host) for OpenDroneKit training.
#
# Interruptible instances get reclaimed without warning, so everything here is
# idempotent and every training call resumes from the last checkpoint. Re-running
# this script on a fresh box after a reclaim continues where the previous one died,
# provided CHECKPOINT_REMOTE points at durable storage.
#
#   bash vast_bootstrap.sh prepare        # deps + data, no training
#   bash vast_bootstrap.sh crack          # SegFormer-B5 crack segmentation
#   bash vast_bootstrap.sh structural     # YOLO11x CODEBRIM
#   bash vast_bootstrap.sh semantic       # DINOv2/UPerNet shared land cover
#   bash vast_bootstrap.sh all
#
# Required in the environment (never commit these):
#   KAGGLE_API_TOKEN    for the crack corpora
#   ROBOFLOW_API_KEY    for the detection corpora
#
# The semantic target needs NEITHER: its corpus travels as one packed archive, set by
# SEMANTIC_CORPUS_URL or already unpacked at SEMANTIC_CORPUS, so no dataset credential
# ever reaches the rented box.

set -euo pipefail

REPO_URL="${REPO_URL:-}"
WORKDIR="${WORKDIR:-/workspace/OpenDroneKit}"
CHECKPOINT_REMOTE="${CHECKPOINT_REMOTE:-}"   # optional rclone/rsync target for runs/
SYNC_INTERVAL_S="${SYNC_INTERVAL_S:-600}"

log() { printf '\n=== %s ===\n' "$1"; }

require_env() {
  local missing=0
  for name in "$@"; do
    if [ -z "${!name:-}" ]; then
      echo "Missing required environment variable: $name" >&2
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || exit 2
}

setup_repo() {
  log "Repository"
  if [ -d "$WORKDIR/.git" ]; then
    git -C "$WORKDIR" pull --ff-only || true
  elif [ -n "$REPO_URL" ]; then
    git clone "$REPO_URL" "$WORKDIR"
  else
    echo "No repo at $WORKDIR and REPO_URL is unset. Upload the tree or set REPO_URL." >&2
    exit 2
  fi
  cd "$WORKDIR"
}

setup_python() {
  log "Python dependencies"
  python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    # Vast images normally ship CUDA torch already; only install if it is absent or CPU-only.
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
  }
  pip install --quiet -r training/requirements-train.txt
  pip install --quiet kagglehub roboflow
  python - <<'PY'
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu: {name} ({total:.0f} GB)")
PY
}

fetch_data() {
  log "Datasets"
  require_env KAGGLE_API_TOKEN
  python -m training.datasets.download crack_segmentation_kaggle deepcrack crackforest
  python -m training.datasets.prepare crack_seg
  if [ -n "${ROBOFLOW_API_KEY:-}" ]; then
    python -m training.datasets.download codebrim_structural
    python -m training.datasets.prepare structural_det
  else
    echo "ROBOFLOW_API_KEY unset: skipping the detection corpora."
  fi
}

# Push checkpoints off the box periodically. Without this a reclaim loses the run.
start_checkpoint_sync() {
  [ -n "$CHECKPOINT_REMOTE" ] || { echo "CHECKPOINT_REMOTE unset: checkpoints stay on this box only."; return; }
  log "Checkpoint sync every ${SYNC_INTERVAL_S}s -> $CHECKPOINT_REMOTE"
  (
    while true; do
      sleep "$SYNC_INTERVAL_S"
      rclone copy training/runs "$CHECKPOINT_REMOTE" --include '*.pt' --include '*.json' 2>/dev/null \
        || rsync -a --include='*.pt' --include='*.json' training/runs/ "$CHECKPOINT_REMOTE" 2>/dev/null \
        || true
    done
  ) &
  echo "sync pid $!"
}

fetch_semantic_corpus() {
  log "Semantic corpus"
  local root="${SEMANTIC_CORPUS:-/workspace/shared_semantic_v3}"
  if [ -f "$root/corpus.json" ]; then
    echo "corpus already present at $root"
  elif [ -n "${SEMANTIC_CORPUS_URL:-}" ]; then
    mkdir -p "$root"
    curl -fL "$SEMANTIC_CORPUS_URL" -o /tmp/semantic_corpus.zip
    unzip -q -o /tmp/semantic_corpus.zip -d "$root"
    rm -f /tmp/semantic_corpus.zip
  else
    echo "No semantic corpus. Set SEMANTIC_CORPUS_URL or upload it to $root." >&2
    exit 2
  fi
  # Refuses a corpus whose samples carry no ground sample distance. JPEG holds no
  # georeferencing, so without that field the loader crops a fixed pixel count from
  # sources spanning 0.20 to 4.78 m/px -- the defect this run exists to fix, reproduced
  # for hours on a machine billed by the hour, looking entirely normal throughout.
  python tools/check_packed_corpus.py "$root/corpus.json"
}

train_semantic() {
  log "DINOv2 ViT-B/14 + UPerNet shared land cover"
  local root="${SEMANTIC_CORPUS:-/workspace/shared_semantic_v3}"
  python -m training.train_shared_semantic     --config training/configs/shared_semantic_dinov2_vitb14.yaml     --corpus "$root/corpus.json"     --run-dir training/runs/shared_semantic_v3     --resume-if-available  # tolerant: the first run has nothing to resume, and plain --resume is strict
  # Scored on the four pinned India tiles, which were never in train or validation.
  python -m training.evaluate_holdout     --run training/runs/shared_semantic_v3     --corpus "$root/corpus.json"     --out training/runs/shared_semantic_v3/india_holdout.json || true
}

train_crack() {
  log "SegFormer-B5 crack segmentation @1024"
  # --resume is always passed: on a first run there is no checkpoint and it is a
  # no-op, and after a reclaim it is exactly what is needed.
  python -m training.train_seg \
    --config training/configs/crack_segformer_b5.yaml \
    --resume
  python -m training.export_onnx --run training/runs/crack_segformer_b5 --kind seg
}

train_structural() {
  log "YOLO11x structural detection @1024"
  python -m training.train_det \
    --config training/configs/structural_yolo11x.yaml \
    --resume
  python -m training.export_onnx --run training/runs/structural_yolo11x --kind yolo
}

main() {
  local target="${1:-prepare}"
  setup_repo
  setup_python
  # The semantic target brings its own corpus and needs no dataset credentials, so it
  # must not be blocked behind a Kaggle token it will never use.
  if [ "$target" = "semantic" ]; then
    fetch_semantic_corpus
  else
    fetch_data
  fi
  [ "$target" = "prepare" ] && { log "Prepared. Nothing trained."; return; }
  start_checkpoint_sync
  case "$target" in
    crack)      train_crack ;;
    structural) train_structural ;;
    semantic)   train_semantic ;;
    all)        train_crack; train_structural ;;
    *) echo "Unknown target: $target (prepare|crack|structural|semantic|all)" >&2; exit 2 ;;
  esac
  log "Done. Copy training/runs/*/ back to the workstation, then run training.register."
}

main "$@"
