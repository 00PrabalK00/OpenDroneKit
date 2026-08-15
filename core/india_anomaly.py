"""Geospatial anomaly candidates against an explicitly validated normal baseline.

The engine deliberately emits ``deviation_candidate`` rather than guessing a
defect name.  It operates on real feature rasters, preserves their survey grid,
and refuses metric output when the target raster is not projected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .india_geospatial import (
    IndiaPackRefused,
    RasterEvidence,
    polygonize_mask,
    rasterize_features,
    read_raster_evidence,
    write_aligned_raster,
)


BASELINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidatedAnomalyBaseline:
    baseline_id: str
    feature_schema: str
    band_count: int
    center: tuple[float, ...]
    robust_scale: tuple[float, ...]
    validated_by: str
    validation_scope: str
    source_files: tuple[dict[str, Any], ...]
    valid_pixel_count: int
    method: str = "per_band_median_and_scaled_mad"
    schema_version: int = BASELINE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["center"] = list(self.center)
        payload["robust_scale"] = list(self.robust_scale)
        payload["source_files"] = list(self.source_files)
        return payload

    @classmethod
    def from_file(cls, path: str | Path) -> "ValidatedAnomalyBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != BASELINE_SCHEMA_VERSION:
            raise IndiaPackRefused("Unsupported anomaly baseline schema version.")
        return cls(
            baseline_id=str(payload["baseline_id"]),
            feature_schema=str(payload["feature_schema"]),
            band_count=int(payload["band_count"]),
            center=tuple(float(value) for value in payload["center"]),
            robust_scale=tuple(float(value) for value in payload["robust_scale"]),
            validated_by=str(payload["validated_by"]),
            validation_scope=str(payload["validation_scope"]),
            source_files=tuple(dict(item) for item in payload["source_files"]),
            valid_pixel_count=int(payload["valid_pixel_count"]),
            method=str(payload.get("method", "per_band_median_and_scaled_mad")),
            schema_version=int(payload["schema_version"]),
        )


@dataclass(frozen=True)
class AnomalyPackage:
    score_raster_path: str
    candidates_path: str
    summary_path: str
    baseline_path: str

    def artifact_paths(self) -> tuple[str, ...]:
        return (self.score_raster_path, self.candidates_path,
                self.summary_path, self.baseline_path)


@dataclass(frozen=True)
class _FeatureRaster:
    path: Path
    values: np.ndarray
    valid: np.ndarray
    reference: RasterEvidence
    feature_schema: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_feature_raster(path: str | Path) -> _FeatureRaster:
    import rasterio

    source_path = Path(path)
    reference = read_raster_evidence(source_path, band=1, require_projected=True)
    with rasterio.open(source_path) as source:
        values = source.read().astype(np.float64, copy=False)
        masks = source.read_masks() > 0
        tags = {str(key): str(value) for key, value in source.tags().items()}
    valid = np.all(masks, axis=0) & np.all(np.isfinite(values), axis=0)
    schema = tags.get("ODK_FEATURE_SCHEMA", "").strip()
    if not schema:
        raise IndiaPackRefused(
            f"{source_path.name} has no ODK_FEATURE_SCHEMA tag; bands cannot be "
            "compared to a baseline whose feature meaning is unknown."
        )
    if not np.any(valid):
        raise IndiaPackRefused(f"{source_path.name} contains no valid feature pixels.")
    return _FeatureRaster(source_path, values, valid, reference, schema)


def fit_validated_baseline(
    raster_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    validated_by: str,
    validation_scope: str,
    minimum_valid_pixels: int = 32,
) -> ValidatedAnomalyBaseline:
    """Fit robust feature statistics from sources confirmed to be normal."""

    if len(raster_paths) < 2:
        raise IndiaPackRefused(
            "An anomaly baseline needs at least two independently captured normal rasters."
        )
    if not validated_by.strip() or not validation_scope.strip():
        raise IndiaPackRefused(
            "A normal baseline must state validated_by and its validation_scope."
        )
    rasters = [_read_feature_raster(path) for path in raster_paths]
    schemas = {item.feature_schema for item in rasters}
    band_counts = {item.values.shape[0] for item in rasters}
    if len(schemas) != 1 or len(band_counts) != 1:
        raise IndiaPackRefused("Baseline rasters must share one feature schema and band count.")

    samples = [item.values[:, item.valid].T for item in rasters]
    matrix = np.concatenate(samples, axis=0)
    if matrix.shape[0] < int(minimum_valid_pixels):
        raise IndiaPackRefused(
            f"The baseline has {matrix.shape[0]} valid pixels; at least "
            f"{minimum_valid_pixels} are required."
        )
    center = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    unusable = (~np.isfinite(scale)) | (scale <= 1e-12)
    if np.any(unusable):
        bands = ", ".join(str(index + 1) for index in np.flatnonzero(unusable))
        raise IndiaPackRefused(
            f"Baseline band(s) {bands} have no measurable normal variation; "
            "a sigma anomaly threshold would be undefined."
        )

    source_files = tuple({
        "path": str(item.path),
        "sha256": _sha256(item.path),
        "valid_pixels": int(item.valid.sum()),
        "crs_epsg": item.reference.epsg,
    } for item in rasters)
    identity_payload = json.dumps({
        "schema": next(iter(schemas)), "center": center.tolist(),
        "scale": scale.tolist(), "sources": source_files,
        "validated_by": validated_by, "validation_scope": validation_scope,
    }, sort_keys=True).encode("utf-8")
    baseline = ValidatedAnomalyBaseline(
        baseline_id=hashlib.sha256(identity_payload).hexdigest(),
        feature_schema=next(iter(schemas)),
        band_count=int(next(iter(band_counts))),
        center=tuple(float(value) for value in center),
        robust_scale=tuple(float(value) for value in scale),
        validated_by=validated_by.strip(),
        validation_scope=validation_scope.strip(),
        source_files=source_files,
        valid_pixel_count=int(matrix.shape[0]),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(baseline.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )
    return baseline


def detect_anomaly_candidates(
    feature_raster_path: str | Path,
    baseline_path: str | Path,
    output_dir: str | Path,
    *,
    threshold_sigma: float = 3.5,
    minimum_area_m2: float = 1.0,
) -> AnomalyPackage:
    """Map deviations, preserving the target grid and declining defect labels."""

    if not math.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be finite and positive.")
    if not math.isfinite(minimum_area_m2) or minimum_area_m2 < 0:
        raise ValueError("minimum_area_m2 must be finite and non-negative.")
    baseline_file = Path(baseline_path)
    if not baseline_file.is_file():
        raise IndiaPackRefused(f"Anomaly baseline does not exist: {baseline_file}")
    baseline = ValidatedAnomalyBaseline.from_file(baseline_file)
    target = _read_feature_raster(feature_raster_path)
    if target.feature_schema != baseline.feature_schema:
        raise IndiaPackRefused(
            f"Target feature schema {target.feature_schema!r} does not match baseline "
            f"{baseline.feature_schema!r}."
        )
    if target.values.shape[0] != baseline.band_count:
        raise IndiaPackRefused("Target band count does not match the validated baseline.")

    center = np.asarray(baseline.center, dtype=np.float64)[:, None, None]
    scale = np.asarray(baseline.robust_scale, dtype=np.float64)[:, None, None]
    score = np.max(np.abs(target.values - center) / scale, axis=0)
    score[~target.valid] = np.nan
    candidate_mask = target.valid & (score >= threshold_sigma)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    score_path = write_aligned_raster(
        output / "anomaly_score.tif",
        np.where(target.valid, score, -9999.0),
        target.reference,
        dtype="float32",
        nodata=-9999.0,
        tags={
            "ODK_VALUE_KIND": "validated_baseline_deviation_score",
            "ODK_VALUE_UNIT": "robust_sigma",
            "ODK_BASELINE_ID": baseline.baseline_id,
            "ODK_FEATURE_SCHEMA": baseline.feature_schema,
            "ODK_NAMED_DEFECT_CLASS": "none",
        },
        categorical=False,
    )

    features = polygonize_mask(
        candidate_mask, target.reference,
        properties={
            "label": "deviation_candidate",
            "threshold_sigma": float(threshold_sigma),
            "baseline_id": baseline.baseline_id,
            "interpretation": "review_required_not_a_named_defect",
        },
        min_area_m2=float(minimum_area_m2),
    )
    for feature in features:
        component = rasterize_features([feature], target.reference) & target.valid
        values = score[component]
        feature["properties"]["max_score_sigma"] = round(float(np.max(values)), 6)
        feature["properties"]["mean_score_sigma"] = round(float(np.mean(values)), 6)

    candidates_path = output / "anomaly_candidates.geojson"
    candidates_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"EPSG:{target.reference.epsg}"}},
        "features": features,
    }, indent=2, allow_nan=False), encoding="utf-8")
    summary_path = output / "anomaly_summary.json"
    summary_path.write_text(json.dumps({
        "status": "complete",
        "result_kind": "deviation_candidates",
        "named_defect_class": None,
        "candidate_count": len(features),
        "candidate_area_m2": round(sum(float(item["properties"]["area_m2"])
                                          for item in features), 6),
        "threshold_sigma": float(threshold_sigma),
        "feature_schema": baseline.feature_schema,
        "crs_epsg": target.reference.epsg,
        "baseline": baseline.to_dict(),
        "source": str(target.path),
        "limits": [
            "Candidates are deviations from the stated validated-normal scope, not confirmed faults.",
            "No unsupported defect class or cause is assigned.",
            "Candidate area is measured only because the target raster has a projected CRS.",
        ],
    }, indent=2, allow_nan=False), encoding="utf-8")

    return AnomalyPackage(
        score_raster_path=str(score_path),
        candidates_path=str(candidates_path),
        summary_path=str(summary_path),
        baseline_path=str(baseline_file),
    )

