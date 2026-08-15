"""What can be trained right now, and what each missing piece is.

Four things have to line up before a model can be trained: the raw dataset downloaded,
the corpus prepared into a trainer layout, a config that names the run, and a trainer
that understands that kind of task. This reports all four per model, so the answer to
"is everything ready" is read off the disk rather than remembered.

It deliberately says what is missing rather than only that something is. "No config" and
"no data" are different jobs -- one is five minutes, the other is a download and a
licence check -- and a readiness report that flattens them into a red cross is no use
when deciding what to start.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "training" / "data" / "prepared"
CONFIGS = ROOT / "training" / "configs"
RUNS = ROOT / "training" / "runs"

TRAINERS = {
    "segmentation": "training/train_seg.py",
    "detection": "training/train_det.py",
    "classification": "training/train_cls.py",
    "semantic": "training/train_shared_semantic.py",
}


@dataclass
class ModelPlan:
    """One model we intend to have, and what it needs."""

    key: str
    kind: str
    corpus: str
    config: str
    purpose: str
    registry_key: str = ""
    blocked_by: str = ""
    notes: str = ""
    # Set when a trained artefact already exists somewhere.
    trained_run: str = ""


PLANS: list[ModelPlan] = [
    ModelPlan("crack_segmentation", "segmentation", "crack_seg", "crack_segformer_b5.yaml",
              "Pixel-level crack extent on structures.",
              registry_key="crack_segmentation", trained_run="crack_segformer_b5",
              notes="Shipping: test IoU 0.637 at threshold 0.85."),
    ModelPlan("structural_multiclass_detector", "detection", "structural_det",
              "structural_yolo11x.yaml",
              "Spalling, exposed bars, corrosion staining, efflorescence, cracks.",
              registry_key="structural_multiclass_detector", trained_run="structural_yolo11x",
              notes="Shipping: mAP50 0.417."),
    ModelPlan("solar_thermal_anomaly", "classification", "solar_thermal_cls",
              "solar_thermal_cls.yaml",
              "Per-module infrared anomaly class: hot spot, diode, offline module, soiling.",
              notes="Corpus is 20,000 crops with 10,000 normal against 249 hot spots; "
                    "balanced accuracy selects the checkpoint."),
    ModelPlan("solar_cell_severity", "classification", "solar_cls", "solar_cell_cls.yaml",
              "Electroluminescence cell health as ordinal severity tiers.",
              notes="ELPV carries no class names and no boxes, so this judges severity "
                    "and cannot name or localise a defect."),
    ModelPlan("solar_surface_condition", "detection", "solar_det", "solar_yolo11l.yaml",
              "Panel surface condition: clear, dusty, snow-covered, physical damage.",
              notes="Not an electrical-fault detector. Corpus rebuilt after dropping a "
                    "zero-instance class and an export artefact named '1'."),
    ModelPlan("metal_corrosion_detector", "detection", "corrosion_det",
              "corrosion_yolo11l.yaml", "Corrosion and rust on metal structures.",
              notes="Corpus rebuilt as a single Corrosion class after merging a duplicate "
                    "label and dropping a negative class."),
    ModelPlan("crack_screening_cls", "classification", "crack_cls", "crack_cls.yaml",
              "Cheap crack / no-crack screening and hard negatives for the segmenter.",
              notes="96,092 tiles, the largest prepared corpus and until now unused."),
    ModelPlan("shared_semantic", "semantic", "shared_semantic",
              "shared_semantic_dinov2_vitb14.yaml",
              "DINOv2 + UPerNet land-cover foundation for the India packs.",
              blocked_by="Awaiting the go-ahead; the India holdout condition is now met.",
              notes="2,879 indexed samples across 97 sites. The India holdout exists: 87 "
                    "samples across 4 Indian SpaceNet7 sites are pinned to test. It "
                    "measures the building class only, so it cannot certify road, "
                    "vegetation, water or bare_land transfer to India."),
    ModelPlan("agriculture_crop_weed", "segmentation", "agriculture_seg",
              "agriculture_segformer_b2.yaml",
              "Maize crop and weed separation from multispectral UAV imagery.",
              notes="WeedsGalore, 156 captures on the authors' own splits. Trains on an "
                    "RGB composite built from 3 of the 5 bands; RE and NIR, where the "
                    "separation actually lives, need a trainer change to use."),
    ModelPlan("solar_cell_defect_detector", "detection", "pvel_ad_det",
              "pvel_ad_yolo11l.yaml",
              "Named and localised electroluminescence cell defects.",
              notes="PVEL-AD. Eight trainable classes of the twelve declared; the other "
                    "four have under 35 boxes each and are dropped rather than guessed. "
                    "Complements solar_cell_severity rather than replacing it."),
    ModelPlan("road_damage_detector", "detection", "roads_det", "roads_yolo11l.yaml",
              "Road surface damage from RDD2022 India and drone imagery.",
              notes="Restricted to the four CRDDC2022 classes both subsets share. India "
                    "is ground imagery and China_Drone is aerial, so the two must be "
                    "scored separately or the headline number is not about drones."),
    ModelPlan("rail_corridor_segmentation", "segmentation", "rail_seg",
              "rail_corridor_segformer_b2.yaml",
              "Railway corridor segmentation from UAV-RSOD.",
              notes="630 samples over two independent mask aspects, so about 315 "
                    "distinct Indian UAV images. Binary by design."),
    ModelPlan("rail_obstacle_detector", "detection", "rail_obstacle_det",
              "rail_obstacle_yolo11l.yaml",
              "Obstacles in the rail corridor.",
              notes="2,002 India-collected UAV images, six classes. Person recall must "
                    "be reported separately rather than averaged into mAP."),
    ModelPlan("solar_module_inventory", "segmentation", "solar_module_seg",
              "solar_module_segformer_b2.yaml",
              "PV module masks for inventory and thermal association.",
              notes="Duke solar_pv_uav. Binary PV-surface extent: an annotation may cover "
                    "a whole array, so this cannot claim per-module identity."),
    ModelPlan("mining_change_semantics", "semantic", "mining_change", "",
              "Mine and quarry scene change.",
              blocked_by="No change-detection trainer exists.",
              notes="MineNetCD is downloaded and usable: ~100 sites of im1/im2/ref "
                    "triplets. What is missing is not data but a trainer -- every "
                    "trainer here takes one image per sample, and change detection "
                    "takes a pair. Writing one is the work, not preparing a corpus."),
    ModelPlan("construction_change_semantics", "semantic", "construction_change", "",
              "Construction progress change.",
              blocked_by="The IARPA SMART download contains annotations but no imagery.",
              notes="32,813 geojson and 1,540 csv, zero rasters: the imagery is fetched "
                    "separately from commercial archives per its own documentation. "
                    "Same shape of problem as OpenEarthMap's gorakhpur region."),
]


def _corpus_state(name: str) -> dict[str, Any]:
    root = PREPARED / name
    if not root.exists():
        return {"ready": False, "detail": "not prepared"}

    # Detection and classification corpora are folders of files; the semantic corpus is
    # an index of sample records pointing at rasters left where they were downloaded.
    index = root / "corpus.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        counts = payload.get("counts", {}).get("samples_by_split", {})
        return {"ready": bool(samples), "detail": f"{len(samples)} indexed samples",
                "splits": counts}

    images = sum(1 for _ in root.rglob("*.jpg")) + sum(1 for _ in root.rglob("*.png"))
    if not images:
        return {"ready": False, "detail": "prepared directory is empty"}
    splits = {split: sum(1 for _ in (root / split).rglob("*.jpg"))
                     + sum(1 for _ in (root / split).rglob("*.png"))
              for split in ("train", "val", "test") if (root / split).exists()}
    return {"ready": True, "detail": f"{images} images", "splits": splits}


def audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in PLANS:
        corpus = _corpus_state(plan.corpus)
        config_path = CONFIGS / plan.config if plan.config else None
        has_config = bool(config_path and config_path.exists())
        trainer = TRAINERS.get(plan.kind, "")
        has_trainer = bool(trainer and (ROOT / trainer).exists())
        trained = bool(plan.trained_run and (RUNS / plan.trained_run).exists())

        missing: list[str] = []
        if not corpus["ready"]:
            missing.append("corpus")
        if not has_config:
            missing.append("config")
        if not has_trainer:
            missing.append("trainer")

        if trained:
            state = "trained"
        elif plan.blocked_by:
            state = "blocked"
        elif not missing:
            state = "ready to train"
        else:
            state = "missing " + ", ".join(missing)

        rows.append({
            "model": plan.key, "kind": plan.kind, "state": state,
            "corpus": plan.corpus, "corpus_detail": corpus["detail"],
            "config": plan.config or "-", "trainer": trainer or "-",
            "blocked_by": plan.blocked_by, "notes": plan.notes,
            "registry_key": plan.registry_key,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.readiness")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable rows.")
    args = parser.parse_args(argv)

    rows = audit()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    order = {"ready to train": 0, "missing config": 1, "blocked": 2, "trained": 4}
    rows.sort(key=lambda r: (order.get(r["state"], 3), r["model"]))

    width = max(len(r["model"]) for r in rows)
    print(f"\nOpenDroneKit model readiness  ({len(rows)} models)")
    print("-" * (width + 62))
    for row in rows:
        print(f"  {row['model']:<{width}}  {row['state']:<22} {row['corpus_detail']}")
        if row["blocked_by"]:
            print(f"  {'':<{width}}  blocked: {row['blocked_by']}")

    counts: dict[str, int] = {}
    for row in rows:
        key = row["state"].split(" missing")[0] if row["state"].startswith("missing") else row["state"]
        counts[key] = counts.get(key, 0) + 1
    print("\n" + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
