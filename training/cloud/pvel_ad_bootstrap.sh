#!/usr/bin/env bash
# Bootstrap a Vast instance to train pvel_ad_yolo11l end to end.
#
# The corpus is 2.1 GB prepared and the source archive is 4.2 GB, both of which are
# faster to fetch on the rented box than to push from a home connection, so this pulls
# from the original sources rather than uploading. The repo is public, so the code comes
# from git and the run stays reproducible from a commit rather than from whatever
# happened to be on someone's laptop.
#
# Everything writes to /workspace/state so progress survives a dropped ssh session, and
# the final marker file is what the local monitor watches for.
set -euo pipefail

WORKDIR=/workspace
REPO="$WORKDIR/OpenDroneKit"
STATE="$WORKDIR/state"
mkdir -p "$STATE"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*" | tee -a "$STATE/bootstrap.log"; }
fail() { echo "BOOTSTRAP_FAILED: $*" | tee -a "$STATE/bootstrap.log"; echo "failed" > "$STATE/status"; exit 1; }

echo "running" > "$STATE/status"

log "GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$STATE/bootstrap.log"

log "system packages"
apt-get update -qq >/dev/null 2>&1 || true
# libarchive-tools provides bsdtar, which reads the RAR5 archive PVEL-AD ships as.
apt-get install -y -qq git libarchive-tools >/dev/null 2>&1 || fail "apt install"

log "python packages"
pip install -q --upgrade pip >/dev/null 2>&1 || true
pip install -q ultralytics gdown pillow numpy pyyaml >/dev/null 2>&1 || fail "pip install"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" | tee -a "$STATE/bootstrap.log"

log "clone repo at the pinned commit"
cd "$WORKDIR"
[ -d "$REPO" ] || git clone -q https://github.com/00PrabalK00/OpenDroneKit.git || fail "git clone"
cd "$REPO"
git fetch -q origin && git checkout -q "${ODK_COMMIT:-main}" || fail "git checkout"
log "at commit $(git rev-parse --short HEAD)"

log "fetch PVEL-AD from Drive (4.2 GB)"
mkdir -p training/data/_archives
if [ ! -s training/data/_archives/pvel_ad.rar ]; then
  gdown --id 1EtteKnLhSFQ3XMCRXt5wKY-lDkIP7299 \
        -O training/data/_archives/pvel_ad.rar || fail "gdown"
fi
# A RAR5 archive starts with these bytes. Google Drive serves an HTML interstitial when
# a download is refused, and that lands as a plausible-looking file of the wrong type,
# so the magic is checked rather than the exit code alone.
head -c 7 training/data/_archives/pvel_ad.rar | grep -q 'Rar!' || fail "downloaded file is not a RAR archive"
log "archive $(du -h training/data/_archives/pvel_ad.rar | cut -f1)"

log "extract"
mkdir -p training/data/pvel_ad
if [ ! -d training/data/pvel_ad/PVELAD ]; then
  bsdtar -xf training/data/_archives/pvel_ad.rar -C training/data/pvel_ad --strip-components=1 || fail "extract"
fi

log "prepare corpus"
python -m training.datasets.prepare pvel_ad_det 2>&1 | tee -a "$STATE/bootstrap.log" || fail "prepare"
test -s training/data/prepared/pvel_ad_det/data.yaml || fail "no data.yaml after prepare"
cat training/data/prepared/pvel_ad_det/data.yaml | tee -a "$STATE/bootstrap.log"

# The corpus must carry the eight trainable classes. Training on a corpus that silently
# built with a different class count wastes the whole rental.
NC=$(grep -E '^nc:' training/data/prepared/pvel_ad_det/data.yaml | awk '{print $2}')
[ "$NC" = "8" ] || fail "expected 8 classes in the prepared corpus, found ${NC:-none}"
log "corpus verified: nc=$NC"

echo "training" > "$STATE/status"
log "start training"
python -m training.train_det \
  --config training/configs/pvel_ad_yolo11l.yaml \
  --output-dir "$STATE/runs" \
  2>&1 | tee -a "$STATE/train.log"

RUN="$STATE/runs/pvel_ad_yolo11l/weights"
[ -s "$RUN/best.pt" ] || fail "training finished without best.pt"

log "training complete"
sha256sum "$RUN/best.pt" "$RUN/last.pt" 2>/dev/null | tee "$STATE/weights.sha256"
du -h "$RUN"/*.pt | tee -a "$STATE/bootstrap.log"
echo "complete" > "$STATE/status"
log "DONE"
