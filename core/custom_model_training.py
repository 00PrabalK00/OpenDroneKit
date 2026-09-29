"""Train, export and deploy a user's own detector from a corpus built in the app.

The trainers, the exporter and the registration step all existed as command-line tools,
and a user who had labelled boxes in the application had no way to turn them into a
model without leaving it. This is the path from the corpus the labelling canvas produced
to an installed registry key the rest of the application can run.

Three steps, kept separate because they answer different questions:

``train``    runs training/train_det.py in its own process on the corpus. Hours on a
             real corpus; a separate process so it can be cancelled and so a crash in
             the trainer cannot take the application down with it.
``deploy``   exports the best checkpoint to ONNX, refuses it if the export does not
             match the trained model (the parity gate in training/export_onnx.py), and
             registers it under a ``custom_`` key with the digest of the file installed.

Deployment is deliberately a second, explicit step: the metrics card is what a user
reads before deciding a model is good enough, and a pipeline that trained and installed
in one go would install a model nobody had looked at.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCH_LINE = re.compile(r"^\s*(\d+)/(\d+)\s")
# Ultralytics redraws its progress bar with terminal escapes ("[K") at the start of
# every line; they have to go before the epoch counter can be read.
ANSI = re.compile(r"\[[0-9;?]*[A-Za-z]")
CUSTOM_PREFIX = "custom_"


class TrainingRefused(ValueError):
    """The inputs cannot produce a model worth reading a metric from."""


def _base_weights() -> Path:
    """A local YOLO detection checkpoint to start from, so training needs no download."""
    from .models import extra_weights_root, models_root

    for root in (models_root(), extra_weights_root()):
        if root is None:
            continue
        for name in ("yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11n.pt"):
            candidate = root / "pretrained" / name
            if candidate.is_file():
                return candidate
    raise TrainingRefused(
        "No pretrained YOLO detection checkpoint in models/pretrained. Run "
        "`python -m training.pretrained` once with a network connection to fetch one.")


def custom_key(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not slug:
        raise TrainingRefused("A custom model needs a name.")
    return slug if slug.startswith(CUSTOM_PREFIX) else CUSTOM_PREFIX + slug


def train_detector(
    corpus_dir: str | Path,
    runs_dir: str | Path,
    *,
    name: str,
    epochs: int = 50,
    image_size: int = 640,
    batch_size: int = 8,
    progress: Callable[[int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Train a detector on a corpus built by core.label_sets, and return its metrics card."""
    corpus = Path(corpus_dir)
    if not (corpus / "data.yaml").is_file():
        raise TrainingRefused(
            f"{corpus} has no data.yaml. Build a corpus from labelled boxes first.")
    report_path = corpus / "corpus_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # counts[split] is {"images": n, <class>: boxes, ...}; a dict is truthy even
        # when the split holds no image, so the image count is what has to be read.
        if not ((report.get("counts") or {}).get("val") or {}).get("images"):
            raise TrainingRefused("The corpus has no validation images, so no metric could be read.")
    if int(epochs) < 1:
        raise TrainingRefused("Training needs at least one epoch.")

    key = custom_key(name)
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    config = {
        "name": key,
        "data_root": str(corpus.resolve()),
        "model_id": str(_base_weights()),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "patience": max(5, int(epochs) // 4),
        "workers": 0,
        "output_dir": str(runs.resolve()),
    }
    config_path = runs / f"{key}.config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = dict(os.environ)
    # Plots need a font ultralytics would otherwise try to download; an offline
    # machine should not fail a training run over a chart.
    env.setdefault("YOLO_OFFLINE", "1")
    command = [sys.executable, "-m", "training.train_det", "--config", str(config_path),
               "--workers", "0"]
    log_path = runs / f"{key}.train.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            match = EPOCH_LINE.match(ANSI.sub("", line))
            if match and progress:
                done, total = int(match.group(1)), max(1, int(match.group(2)))
                progress(min(95, 5 + int(90 * done / total)), f"epoch {done}/{total}")
            if should_cancel and should_cancel():
                process.terminate()
                process.wait(timeout=30)
                raise TrainingRefused("Training was cancelled.")
        code = process.wait()
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise RuntimeError(f"Training failed (exit {code}):\n" + "\n".join(tail))

    run_dir = runs / key
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Training finished without a metrics card at {summary_path}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"key": key, "run_dir": str(run_dir), "log_path": str(log_path),
            "metrics": summary.get("metrics", {}), "checkpoint": summary.get("checkpoint", "")}


def deploy_detector(run_dir: str | Path, key: str) -> dict[str, Any]:
    """Export a trained run to ONNX, gate it on parity, and install it under `key`."""
    from training.export_onnx import export_yolo
    from training.register import register

    if not str(key).startswith(CUSTOM_PREFIX):
        raise TrainingRefused(
            f"Deploying from the app installs under a {CUSTOM_PREFIX}* key only; "
            f"{key!r} would replace a shipped model and its published metrics.")
    run = Path(run_dir)
    try:
        report = export_yolo(run)
    except SystemExit as exc:
        raise TrainingRefused(str(exc)) from exc
    (run / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report.get("parity_ok"):
        raise TrainingRefused(
            "The ONNX export does not match the trained model (parity check failed); "
            "it was not installed.")
    try:
        registered = register(run, key)
    except SystemExit as exc:
        raise TrainingRefused(str(exc)) from exc
    return {"key": key, "export": report, "registered": registered}
