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
    # Generous by default: this is a deadlock guard, not a performance budget. A model
    # legitimately slower than this should get its own value rather than a shorter one
    # here, since a killed-but-healthy run wastes everything it had done.
    timeout_s: int = 6 * 60 * 60


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


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill a stuck run and the workers it spawned.

    Terminating the parent alone leaves ultralytics' dataloader workers alive holding
    the GPU, so the next model in the queue starts against a card that is not actually
    free. psutil is used when present because walking the child list is the only
    reliable way to catch them; the plain terminate is the fallback.
    """
    try:
        import psutil

        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:  # noqa: BLE001 - a child that already exited is fine
                pass
    except Exception:  # noqa: BLE001 - psutil absent or process already gone
        pass
    try:
        process.kill()
        process.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass


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
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=item.timeout_s)
        except subprocess.TimeoutExpired:
            # "Continue past failures" only works if the failing process actually dies.
            # A YOLO run whose spawned dataloader workers fail to load torch's CUDA DLLs
            # leaves those workers hung, the parent waiting on them, and the whole queue
            # blocked behind a model that will never finish. Killing the tree is what
            # lets the remaining models run.
            _kill_tree(process)
            record.update(status="timed_out", minutes=round((time.time() - started) / 60.0, 1),
                          log=str(log_path))
            return record

    result_code = returncode
    record["log"] = str(log_path)
    record["minutes"] = round((time.time() - started) / 60.0, 1)
    if result_code == 0:
        record["status"] = "ok"
    else:
        record["status"] = "failed"
        record["exit_code"] = result_code
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
