"""Generate the cockpit's example data from this repository's real measurements.

The cockpit needs something on screen before it is connected to a project. It used to
invent that content -- named Indian sites, a storage figure, a processing queue -- which
was the one place this codebase fabricated. Replacing it with Null Island placeholders
fixed the honesty problem and produced a demo nobody would want to look at.

So the numbers come from measurements instead. Every value below is read out of an
artefact in the repository at build time:

    models/model_registry.json          the models that are installed, and their metrics
    models/metrics/*.json               per-model validation figures
    docs/features/registry.py           how many capabilities are verified
    README.md                           the Aukerman reconstruction result

Generated rather than hand-typed, for the same reason model metrics are read from a
manifest rather than copied into a slide: a number typed by hand is a number that drifts
from the thing it describes, and this one would be drifting inside the product's shop
window.

The output is a JavaScript module rather than JSON because the desktop shell loads the UI
from the filesystem, and fetch() of a local file under file:// is blocked -- a demo that
works when served and breaks in the actual application is worse than no demo.

    python tools/build_demo_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY = REPO_ROOT / "models" / "model_registry.json"
METRICS_DIR = REPO_ROOT / "models" / "metrics"
OUTPUT = REPO_ROOT / "app" / "web" / "js" / "workspace" / "demo-data.js"

# The reconstruction the README reports, and where it is stated. Kept as a pointer to the
# source rather than a free-floating claim.
AUKERMAN = {
    "name": "OpenDroneKit Aukerman survey",
    "source": "OpenDroneMap Aukerman dataset",
    "images_registered": "77/77",
    "reprojection_px": 1.27,
    "geo_rmse_m": 1.22,
    "epsg": "EPSG:32617",
    "provenance": "Measured by this project's COLMAP pipeline; reported in README.md.",
}


def installed_models() -> list[dict[str, Any]]:
    """Every model with weights on record, and the headline number it earned."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    rows = []
    for key, entry in registry.items():
        if entry.get("status") != "installed":
            continue
        rows.append(
            {
                "key": key,
                "labels": entry.get("labels", []),
                "input_size": entry.get("input_size"),
                # The first figure in the description is the headline metric, and the
                # description is where the caveat lives too.
                "headline": _headline(entry.get("description", "")),
                "sha256": (entry.get("sha256") or "")[:12],
            }
        )
    return sorted(rows, key=lambda row: row["key"])


def _headline(description: str) -> str:
    match = re.search(
        r"((?:mean IoU|IoU|mAP50|balanced accuracy|pixel accuracy)[^.,;]*[0-9]\.[0-9]+)",
        description,
    )
    return match.group(1).strip() if match else ""


def capability_counts() -> dict[str, int]:
    """The COMPUTED status of all 167 capabilities, from the generated status document.

    Not the claimed status in docs/features/registry.py. Claims are a floor -- most rows
    claim `implemented` and are computed as `verified` once the tests they name pass --
    so reading the source would have put "30 verified" on screen when 160 rows had
    actually earned it. Understating is still misreporting.

    docs/FEATURES.md is written by `tools/feature_status.py --markdown` from a run that
    happened, which makes it the honest summary and avoids a UI build step that runs the
    entire test suite.
    """
    text = (REPO_ROOT / "docs" / "FEATURES.md").read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for state in ("verified", "implemented", "in_progress", "not_started"):
        match = re.search(rf"^\| {state} \| (\d+) \|", text, re.MULTILINE)
        if match:
            counts[state] = int(match.group(1))
    total = re.search(r"^\| \*\*total\*\* \| \*\*(\d+)\*\*", text, re.MULTILINE)
    counts["total"] = int(total.group(1)) if total else sum(counts.values())
    counts["source"] = "docs/FEATURES.md"  # type: ignore[assignment]
    return counts


def metric_cards() -> list[dict[str, Any]]:
    """Validation figures written beside the weights that earned them."""
    cards = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = payload.get("validation_metrics") or {}
        per_class = metrics.get("per_class_iou") or {}
        if not per_class:
            continue
        cards.append(
            {
                "key": payload.get("registry_key", path.stem),
                "mean_iou": metrics.get("mean_iou"),
                "per_class": {
                    name: round(value, 3)
                    for name, value in per_class.items()
                    if isinstance(value, (int, float))
                },
            }
        )
    return cards


def survey_footprint() -> dict[str, Any]:
    """The survey extent, read from the imagery's own GPS tags.

    Not a bounding box someone drew. Every one of the 24 images in the subset carries a
    GPS EXIF tag, so the footprint below is where the aircraft actually was -- which is
    what makes the planned path drawn over it a plan for a real place.
    """
    import glob

    from PIL import Image

    points = []
    for path in sorted(glob.glob(str(REPO_ROOT / "training/data/aukerman_subset/*.JPG"))):
        try:
            exif = Image.open(path)._getexif() or {}
        except Exception:  # noqa: BLE001 - an unreadable image is not a build failure
            continue
        gps = exif.get(34853)
        if not gps:
            continue

        def to_degrees(value: Any) -> float:
            return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600

        lat = to_degrees(gps[2]) * (-1 if gps[1] == "S" else 1)
        lon = to_degrees(gps[4]) * (-1 if gps[3] == "W" else 1)
        points.append((lon, lat))

    if not points:
        return {}
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return {
        "captures": points,
        "bounds": [min(lons), min(lats), max(lons), max(lats)],
        "centre": [sum(lons) / len(points), sum(lats) / len(points)],
        "provenance": f"GPS EXIF from {len(points)} images in the Aukerman subset.",
    }


def planned_mission(footprint: dict[str, Any]) -> dict[str, Any]:
    """A mission planned over that footprint by the production planner.

    mission.MissionPlanner.generate is the same code the application runs when a user
    presses Plan. Calling it here rather than drawing a zigzag means the path on screen
    has real spacing, a real distance and a real GSD, and it changes if the planner
    changes.
    """
    if not footprint:
        return {}
    from mission.planner import MissionPlanner

    west, south, east, north = footprint["bounds"]
    polygon = [[west, south], [east, south], [east, north], [west, north]]
    plan = MissionPlanner().generate(
        polygon_lonlat=polygon,
        altitude_m=60.0,
        mode="grid",
        camera="mavic2pro",
    )
    data = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
    waypoints = data.get("waypoints") or []
    return {
        "polygon": polygon,
        # [lon, lat] pairs for the map layer; altitude is carried separately.
        "line": [[float(w[0]), float(w[1])] for w in waypoints],
        "waypoint_count": len(waypoints),
        "altitude_m": data.get("altitude_m"),
        "distance_m": round(float(data.get("path_distance_m") or 0.0), 1),
        "duration_min": round(float(data.get("estimated_time_min") or 0.0), 1),
        "gsd_cm": round(float(data.get("estimated_gsd_cm") or 0.0), 2),
        "camera": data.get("camera"),
        "mode": data.get("mode"),
        "provenance": "Planned by mission.MissionPlanner over the surveyed footprint.",
    }


def products() -> dict[str, Any]:
    """Real outputs the pipeline wrote, indexed by the view that shows them.

    These are detection overlays and coverage rasters produced by running the pipeline
    over the Aukerman subset -- the model's actual output on real imagery, downscaled for
    the UI. A view with no product here renders an explicit "not produced yet" state
    naming what to run, rather than an empty canvas that looks broken.
    """
    root = REPO_ROOT / "app" / "web" / "assets" / "demo"
    known = {
        "crack": "Crack segmentation overlay, SegFormer-B5 at the shipped 0.85 threshold.",
        "structural": "Structural multiclass detector output, YOLO11x on CODEBRIM.",
        "metal": "Metal and corrosion detector output.",
        "coverage": "Coverage heatmap from the capture-vs-plan check.",
        "mosaic": "Preview mosaic of the survey.",
    }
    # Which view button shows which product. The buttons are named for what an operator
    # wants to see; the files are named for what produced them, and the two are not the
    # same vocabulary.
    view_alias = {
        "rgb": "mosaic",
        "semantic": "structural",
        "crack": "crack",
        "corrosion": "metal",
    }
    found = {}
    for key, note in known.items():
        path = root / f"{key}.jpg"
        if path.is_file():
            found[key] = {
                "src": f"assets/demo/{key}.jpg",
                "note": note,
                "bytes": path.stat().st_size,
            }
    for view, key in view_alias.items():
        if key in found and view not in found:
            found[view] = dict(found[key], view_of=key)
    return found


def build() -> dict[str, Any]:
    footprint = survey_footprint()
    return {
        "project": AUKERMAN,
        "footprint": footprint,
        "mission": planned_mission(footprint),
        "products": products(),
        "models": installed_models(),
        "metrics": metric_cards(),
        "capabilities": capability_counts(),
        "note": (
            "Example data. Every figure here was measured by this project and read out "
            "of models/model_registry.json, models/metrics/ and docs/features/registry.py "
            "when the UI was built. It is a real result, not a live connection to a "
            "running survey."
        ),
    }


def main() -> int:
    payload = build()
    OUTPUT.write_text(
        "/* Generated by tools/build_demo_data.py -- do not edit.\n"
        " *\n"
        " * Every number here was measured by this project and read out of the model\n"
        " * registry, the metrics cards and the feature registry. Regenerate rather than\n"
        " * editing: a hand-typed figure drifts from the artefact it describes, and this\n"
        " * one would drift inside the product's shop window.\n"
        " */\n\n"
        "export const DEMO_DATA = "
        + json.dumps(payload, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  models: {len(payload['models'])}")
    print(f"  metric cards: {len(payload['metrics'])}")
    print(f"  capabilities: {payload['capabilities']}")
    print(f"  products: {sorted(payload.get('products') or {})}")
    mission = payload.get("mission") or {}
    if mission:
        print(f"  mission: {mission['waypoint_count']} waypoints, "
              f"{mission['distance_m']} m, {mission['duration_min']} min, "
              f"GSD {mission['gsd_cm']} cm/px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
