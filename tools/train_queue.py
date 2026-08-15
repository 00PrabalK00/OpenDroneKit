"""Run a list of training configs back to back on one machine.

The small models in this project each take between twenty minutes and ninety minutes on
a laptop GPU, which is too short to babysit and too long to run in a terminal someone
needs back. This runs them in sequence, writes one log per model, and keeps going when
one fails so a single bad config does not cost the whole night.

Failure is recorded rather than raised. A queue that stops at the first error is a queue
that produces nothing by morning, and the useful thing at 7am is six trained models and
one legible error, not one error and five untrained models.

    python -m tools.train_queue --list
    python -m tools.train_queue laptop
    python -m tools.train_queue solar_thermal_cls corrosion_yolo11l
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "training" / "configs"
LOGS = ROOT / "training" / "runs" / "_queue_logs"

TRAINERS = {
    "classification": "training.train_cls",
    "detection": "training.train_det",
    "segmentation": "training.train_seg",
}


@dataclass(frozen=True)
class QueueItem:
    config: str
    kind: str
    note: str = ""


# Ordered cheapest first. A queue that front-loads the short runs means an interrupted
# night still leaves several finished models rather than one half-trained large one.
QUEUES: dict[str, tuple[QueueItem, ...]] = {
    "laptop": (
        QueueItem("solar_cell_cls", "classification", "2,624 EL cells, ~20 min."),
        # agriculture_segformer_b2 is deliberately absent. Its corpus has three classes
        # -- soil, maize, weed -- and train_seg.py is binary, so running it here would
        # collapse maize and weed into one foreground class and report a plausible IoU
        # for a model that cannot tell a crop from a weed. It needs a multiclass
        # trainer first.
        QueueItem("solar_module_segformer_b2", "segmentation", "459 UAV frames, ~40 min."),
        QueueItem("rail_corridor_segformer_b2", "segmentation", "630 samples, ~45 min."),
        QueueItem("corrosion_yolo11l", "detection", "717 images, ~50 min."),
        QueueItem("solar_yolo11l", "detection", "1,483 images, ~90 min."),
    ),
}


def _detect_kind(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    if "model_id: yolo" in text:
        return "detection"
    if "model_id: nvidia/mit-" in text:
        return "segmentation"
    return "classification"


def run_one(item: QueueItem, *, dry_run: bool = False) -> dict[str, object]:
    config_path = CONFIGS / f"{item.config}.yaml"
    started = time.time()
    record: dict[str, object] = {
        "config": item.config,
        "kind": item.kind,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not config_path.is_file():
        record.update(status="missing_config", detail=str(config_path))
        return record

    command = [sys.executable, "-m", TRAINERS[item.kind], "--config", str(config_path)]
    record["command"] = " ".join(command)
    if dry_run:
        record["status"] = "dry_run"
        return record

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{item.config}.log"
    print(f"  running -> {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    record["log"] = str(log_path)
    record["minutes"] = round((time.time() - started) / 60.0, 1)
    if result.returncode == 0:
        record["status"] = "ok"
    else:
        record["status"] = "failed"
        record["exit_code"] = result.returncode
        # The last lines are what a person actually needs; the full log stays on disk.
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        record["tail"] = tail
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.train_queue")
    parser.add_argument("names", nargs="*", help="Queue name or individual config stems.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run.")
    args = parser.parse_args(argv)

    if args.list:
        for name, items in QUEUES.items():
            print(f"{name}:")
            for item in items:
                print(f"  {item.config:<32} {item.kind:<14} {item.note}")
        return 0

    requested = args.names or ["laptop"]
    items: list[QueueItem] = []
    for name in requested:
        if name in QUEUES:
            items.extend(QUEUES[name])
        else:
            path = CONFIGS / f"{name}.yaml"
            if not path.is_file():
                print(f"No queue or config named {name!r}.")
                return 2
            items.append(QueueItem(name, _detect_kind(path)))

    print(f"Queue of {len(items)}: {', '.join(i.config for i in items)}\n")
    results = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item.config}", flush=True)
        record = run_one(item, dry_run=args.dry_run)
        results.append(record)
        print(f"  {record['status']}"
              + (f" in {record['minutes']} min" if "minutes" in record else ""), flush=True)
        if record["status"] == "failed":
            for line in record.get("tail", []):  # type: ignore[union-attr]
                print(f"    | {line}")

    LOGS.mkdir(parents=True, exist_ok=True)
    summary = LOGS / "queue_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} finished. Summary: {summary}")
    for record in results:
        if record["status"] != "ok":
            print(f"  {record['status']}: {record['config']}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
