"""Per-defect risk scoring and prioritisation.

`health_scoring` answers "what condition is this asset in overall". This module
answers the operational question that follows: *which defects do I send someone to
look at first, and how urgently.* The `risk_scoring` pipeline stage previously wrote
only a note saying the work happened elsewhere, which was not true of either module.

Risk is scored as severity times exposure:

*Severity* comes from the defect class (an exposed reinforcing bar is not a stain),
scaled by measured extent where the reconstruction produced a georeferenced defect
layer with real square metres, and by detection confidence.

*Exposure* comes from the structure type, using the same class weighting
`health_scoring` applies, so the two modules cannot disagree about whether corrosion
matters more on a steel tower than on a concrete deck.

The output is deliberately auditable: every defect carries the components that
produced its score, so a number can be argued with rather than merely believed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .health_scoring import _grade_from_integrity, _weights_for_structure

# Intrinsic severity of each defect class, independent of size, on 0-1.
# Ordered by how directly the condition threatens load-bearing capacity.
CLASS_SEVERITY: dict[str, float] = {
    "rebar_exposure": 0.95,
    "exposed_bar": 0.95,
    "spalling": 0.85,
    "spallation": 0.85,
    "delamination": 0.80,
    "crack": 0.70,
    "corrosion": 0.65,
    "rust": 0.60,
    "efflorescence": 0.40,
    "moisture_intrusion": 0.45,
    "stain": 0.20,
    "soiling": 0.15,
    # Solar-specific classes.
    "hotspot": 0.75,
    "cell_crack": 0.60,
    "diode_failure": 0.70,
    "shading": 0.25,
    "unknown": 0.50,
}

# Which health-scoring weight group each defect class belongs to, so structure type
# shifts priority the same way in both modules.
CLASS_GROUP: dict[str, str] = {
    "crack": "crack",
    "cell_crack": "solar",
    "hotspot": "solar",
    "diode_failure": "solar",
    "shading": "solar",
    "soiling": "solar",
    "corrosion": "metal",
    "rust": "metal",
    "rebar_exposure": "structural",
    "exposed_bar": "structural",
    "spalling": "structural",
    "spallation": "structural",
    "delamination": "structural",
    "efflorescence": "structural",
    "moisture_intrusion": "structural",
    "stain": "structural",
}

# Area at which a defect is treated as fully extensive. Above this the extent factor
# saturates, so one enormous region cannot dominate the whole asset's score.
AREA_SATURATION_M2 = 2.0

ACTION_BANDS = (
    (0.75, "immediate", "Inspect on site before further loading or operation."),
    (0.55, "urgent", "Schedule a close-range inspection within days."),
    (0.35, "planned", "Include in the next scheduled maintenance cycle."),
    (0.15, "monitor", "Re-survey at the next routine interval and compare."),
    (0.00, "record", "No action beyond keeping the record."),
)


def _normalise_class(name: Any) -> str:
    return str(name or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


def _extent_factor(area_m2: float, length_m: float) -> tuple[float, str]:
    """Scale severity by measured size, when a real measurement exists.

    Returns the factor and the basis used, because a defect scored without
    georeferenced geometry is a materially weaker claim and the report should say so.
    """
    if area_m2 > 0.0:
        return min(1.0, 0.35 + 0.65 * (area_m2 / AREA_SATURATION_M2)), "area_m2"
    if length_m > 0.0:
        # A 2 m crack is treated as fully extensive on length alone.
        return min(1.0, 0.35 + 0.65 * (length_m / 2.0)), "length_m"
    return 0.6, "unmeasured"


def _action_for(score: float) -> tuple[str, str]:
    for threshold, action, guidance in ACTION_BANDS:
        if score >= threshold:
            return action, guidance
    return "record", ACTION_BANDS[-1][2]


def _iter_defects(
    defect_summary: str | Path | None,
    defects_geojson: str | Path | None,
) -> tuple[list[dict[str, Any]], str]:
    """Collect defect records, preferring the georeferenced layer.

    The GeoJSON produced by `defect_projection` carries real square metres measured
    on the reconstructed surface; `defects.json` carries only pixel geometry. When
    both exist the former wins, and the source is reported either way.
    """
    if defects_geojson and Path(defects_geojson).exists():
        payload = json.loads(Path(defects_geojson).read_text(encoding="utf-8"))
        records: list[dict[str, Any]] = []
        for feature in payload.get("features", []) or []:
            properties = dict(feature.get("properties", {}) or {})
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates")
            if geometry.get("type") == "Point" and isinstance(coordinates, list):
                properties["longitude"] = coordinates[0]
                properties["latitude"] = coordinates[1]
            records.append(properties)
        if records:
            return records, "georeferenced_defect_layer"

    if defect_summary and Path(defect_summary).exists():
        payload = json.loads(Path(defect_summary).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = payload.get("defects") or payload.get("detections") or []
        else:
            records = payload if isinstance(payload, list) else []
        flattened: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            # Per-image detection files nest hits under the image entry.
            hits = record.get("defects") or record.get("hits")
            if isinstance(hits, list):
                for hit in hits:
                    if isinstance(hit, dict):
                        merged = dict(hit)
                        merged.setdefault("source_image_name", record.get("image_name") or record.get("image"))
                        flattened.append(merged)
            else:
                flattened.append(record)
        if flattened:
            return flattened, "detection_output"

    return [], "none"


def score_run_risk(
    defect_summary: str | Path | None,
    *,
    defects_geojson: str | Path | None = None,
    structure_type: str = "generic",
    asset_id: str = "",
) -> dict[str, Any]:
    """Score every detected defect and produce a prioritised action list."""
    records, source = _iter_defects(defect_summary, defects_geojson)
    weights = _weights_for_structure(structure_type)

    if not records:
        return {
            "asset_id": asset_id,
            "structure_type": structure_type,
            "source": source,
            "defect_count": 0,
            "risk_index": 0.0,
            "integrity_score": 100.0,
            "grade": 1,
            "grade_label": "Excellent",
            "defects": [],
            "by_action": {},
            "note": (
                "No defect records were available to score. This is not evidence that "
                "the asset is sound -- run defect detection first."
            ),
        }

    scored: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        defect_class = _normalise_class(
            record.get("defect_type") or record.get("class") or record.get("label")
        )
        base_severity = CLASS_SEVERITY.get(defect_class, CLASS_SEVERITY["unknown"])
        group = CLASS_GROUP.get(defect_class, "structural")
        # Normalised against the largest group weight so structure type re-ranks
        # classes without collapsing every score toward zero.
        group_weight = weights.get(group, 0.15)
        exposure = group_weight / max(weights.values())

        area = float(record.get("area_m2") or 0.0)
        length = float(record.get("length_m") or 0.0)
        extent, extent_basis = _extent_factor(area, length)
        confidence = float(record.get("confidence") or record.get("score") or 0.7)
        confidence = min(1.0, max(0.05, confidence))

        score = base_severity * extent * exposure * confidence
        action, guidance = _action_for(score)

        scored.append(
            {
                "rank": 0,
                "defect_id": record.get("defect_id") or f"defect_{index:04d}",
                "defect_type": defect_class,
                "risk_score": round(score, 4),
                "action": action,
                "guidance": guidance,
                "components": {
                    "class_severity": base_severity,
                    "extent_factor": round(extent, 4),
                    "extent_basis": extent_basis,
                    "structure_exposure": round(exposure, 4),
                    "confidence": round(confidence, 4),
                },
                "area_m2": round(area, 5) if area else None,
                "length_m": round(length, 4) if length else None,
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "source_images": record.get("source_images") or record.get("source_image_name"),
            }
        )

    scored.sort(key=lambda item: item["risk_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    scores = [item["risk_score"] for item in scored]
    # The index is driven by the worst defects rather than the mean: an asset with one
    # critical spall and fifty stains is not in average condition.
    worst = scores[: max(1, len(scores) // 10)]
    risk_index = sum(worst) / len(worst)
    integrity_score = round(max(0.0, 100.0 * (1.0 - risk_index)), 2)
    grade, grade_label = _grade_from_integrity(integrity_score)

    by_action: dict[str, int] = {}
    for item in scored:
        by_action[item["action"]] = by_action.get(item["action"], 0) + 1

    return {
        "asset_id": asset_id,
        "structure_type": structure_type,
        "source": source,
        "defect_count": len(scored),
        "risk_index": round(risk_index, 4),
        "integrity_score": integrity_score,
        "grade": grade,
        "grade_label": grade_label,
        "highest_risk": scored[0] if scored else None,
        "by_action": by_action,
        "by_type": _counts_by_type(scored),
        "defects": scored,
        "method": (
            "risk = class_severity * extent * structure_exposure * confidence; "
            "the index is the mean of the worst decile."
        ),
        "measurement_note": (
            "Extent came from georeferenced areas."
            if source == "georeferenced_defect_layer"
            else "No georeferenced defect layer was available, so extent is estimated "
                 "from detection output rather than measured on the surface."
        ),
    }


def _counts_by_type(scored: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for item in scored:
        bucket = buckets.setdefault(
            item["defect_type"], {"count": 0, "max_risk": 0.0, "mean_risk": 0.0, "_total": 0.0}
        )
        bucket["count"] += 1
        bucket["_total"] += item["risk_score"]
        bucket["max_risk"] = max(bucket["max_risk"], item["risk_score"])
    for bucket in buckets.values():
        bucket["mean_risk"] = round(bucket.pop("_total") / bucket["count"], 4)
        bucket["max_risk"] = round(bucket["max_risk"], 4)
    return dict(sorted(buckets.items(), key=lambda kv: kv[1]["max_risk"], reverse=True))
