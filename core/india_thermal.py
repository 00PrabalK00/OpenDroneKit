"""Honest browser artifacts for radiometric thermal survey products.

``core.thermal`` owns the temperature measurement.  This module does not infer
temperature from a colour palette: it packages an already-calibrated
``ThermalImage`` as a georeferenced GeoTIFF plus a small JSON grid that the
local Hub can render without a server or a GeoTIFF JavaScript dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .india_geospatial import (
    IndiaPackRefused,
    IndiaPackResult,
    geometry_area_units2,
    read_geojson_features,
)
from .thermal import NotRadiometric, ThermalImage, load_radiometric_file, write_thermal_geotiff


@dataclass(frozen=True)
class ThermalRegistration:
    """Operator tie-point evidence for an RGB-to-thermal transform.

    The scored constructor below checks correspondence redundancy and geometric
    self-consistency. It does not discover image features or prove that the operator
    selected the same physical feature in both images; ``evidence_kind`` makes that
    limitation machine-readable.
    """

    method: str
    rgb_width: int
    rgb_height: int
    residual_px: float
    validated_by: str
    thermal_width: int | None = None
    thermal_height: int | None = None
    max_residual_px: float | None = None
    tie_point_count: int | None = None
    inlier_count: int | None = None
    inlier_fraction: float | None = None
    coverage_fraction: float | None = None
    ransac_reproj_threshold_px: float | None = None
    acceptance_rmse_px: float | None = None
    evidence_kind: str | None = None
    rgb_to_thermal_homography: tuple[float, ...] | None = None
    rgb_sha256: str | None = None
    thermal_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("Thermal registration method is required.")
        if self.rgb_width < 1 or self.rgb_height < 1:
            raise ValueError("Registered RGB dimensions must be positive.")
        if not math.isfinite(self.residual_px) or self.residual_px < 0:
            raise ValueError("Registration residual_px must be finite and non-negative.")
        if not self.validated_by.strip():
            raise ValueError("Registration must name who or what validated it.")
        if self.rgb_to_thermal_homography is not None:
            if len(self.rgb_to_thermal_homography) != 9 or not all(
                math.isfinite(float(value)) for value in self.rgb_to_thermal_homography
            ):
                raise ValueError("RGB-to-thermal homography must contain nine finite values.")
            if not self.thermal_width or not self.thermal_height:
                raise ValueError("A geometric registration needs thermal dimensions.")

    def to_dict(self) -> dict[str, Any]:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return {**payload, "validated": True, "target": "rgb"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_points(value: Sequence[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise IndiaPackRefused(f"RGB/thermal control-point file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("points", payload) if isinstance(payload, dict) else payload
    else:
        rows = value
    if not isinstance(rows, Sequence) or len(rows) < 6:
        raise IndiaPackRefused(
            "RGB/thermal registration needs at least six operator-supplied tie points. "
            "Four exactly determine a homography and cannot provide residual evidence."
        )
    points: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise IndiaPackRefused(f"Tie point {index + 1} is not an object.")
        rgb, thermal = row.get("rgb"), row.get("thermal")
        if not all(isinstance(point, Sequence) and len(point) == 2 for point in (rgb, thermal)):
            raise IndiaPackRefused(f"Tie point {index + 1} needs rgb and thermal [x, y] coordinates.")
        points.append({
            "rgb": [float(rgb[0]), float(rgb[1])],
            "thermal": [float(thermal[0]), float(thermal[1])],
        })
    if not all(math.isfinite(number) for row in points for key in ("rgb", "thermal")
               for number in row[key]):
        raise IndiaPackRefused("RGB/thermal tie-point coordinates must be finite.")
    return points


def score_rgb_thermal_registration(
    rgb_path: str | Path,
    thermal_path: str | Path,
    control_points: Sequence[Mapping[str, Any]] | str | Path,
    *,
    validated_by: str,
    maximum_rmse_px: float = 1.5,
    ransac_reproj_threshold_px: float = 3.0,
    minimum_inlier_fraction: float = 0.75,
    minimum_coverage_fraction: float = 0.08,
) -> ThermalRegistration:
    """Fit and check an operator-supplied RGB/thermal correspondence set.

    Dimensions matching is never treated as registration.  The result is issued
    only when observed correspondences pass residual, inlier and spatial-coverage
    gates, and the thermal source is genuinely radiometric. The RGB pixels are not
    feature-matched by this function: the score is tie-point self-consistency, not
    independent evidence of image-to-image alignment.
    """

    if not validated_by.strip():
        raise IndiaPackRefused("Registration must name its validator.")
    if maximum_rmse_px <= 0 or ransac_reproj_threshold_px <= maximum_rmse_px:
        raise ValueError(
            "RANSAC's reprojection threshold must be greater than the acceptance RMSE gate."
        )
    if not 0 < minimum_inlier_fraction <= 1:
        raise ValueError("Registration quality thresholds are invalid.")
    if not 0 < minimum_coverage_fraction <= 1:
        raise ValueError("minimum_coverage_fraction must be between zero and one.")

    rgb_file, thermal_file = Path(rgb_path), Path(thermal_path)
    if not rgb_file.is_file():
        raise IndiaPackRefused(f"RGB image does not exist: {rgb_file}")
    try:
        thermal_image = load_radiometric_file(thermal_file)
    except (NotRadiometric, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise IndiaPackRefused(
            "RGB/thermal registration requires radiometric counts and camera calibration; "
            "a rendered thermal palette is not a temperature source."
        ) from exc

    import cv2

    rgb_image = cv2.imread(str(rgb_file), cv2.IMREAD_UNCHANGED)
    if rgb_image is None or rgb_image.ndim < 2:
        raise IndiaPackRefused(f"RGB input is not a readable image: {rgb_file}")
    rgb_height, rgb_width = rgb_image.shape[:2]
    if thermal_image.celsius.ndim != 2 or min(thermal_image.celsius.shape) < 2:
        raise IndiaPackRefused("Radiometric registration requires a non-empty 2D thermal grid.")
    thermal_height, thermal_width = thermal_image.celsius.shape
    rows = _control_points(control_points)
    rgb_points = np.asarray([row["rgb"] for row in rows], dtype=np.float64)
    thermal_points = np.asarray([row["thermal"] for row in rows], dtype=np.float64)
    if np.any(rgb_points[:, 0] < 0) or np.any(rgb_points[:, 0] >= rgb_width) or np.any(
        rgb_points[:, 1] < 0) or np.any(rgb_points[:, 1] >= rgb_height
    ):
        raise IndiaPackRefused("At least one RGB tie point lies outside the RGB image.")
    if np.any(thermal_points[:, 0] < 0) or np.any(thermal_points[:, 0] >= thermal_width) or np.any(
        thermal_points[:, 1] < 0) or np.any(thermal_points[:, 1] >= thermal_height
    ):
        raise IndiaPackRefused("At least one thermal tie point lies outside the radiometric grid.")

    matrix, inliers = cv2.findHomography(
        rgb_points, thermal_points, cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold_px,
    )
    if matrix is None or inliers is None:
        raise IndiaPackRefused("RGB/thermal tie points do not define a usable homography.")
    projected = cv2.perspectiveTransform(rgb_points[None].astype(np.float32), matrix)[0]
    residuals = np.linalg.norm(projected - thermal_points, axis=1)
    selected = inliers.reshape(-1).astype(bool)
    if int(selected.sum()) < 5:
        raise IndiaPackRefused(
            "Fewer than five RGB/thermal tie points agree with one transform; four "
            "inliers are exactly determined and cannot support a residual check."
        )
    rmse = float(np.sqrt(np.mean(np.square(residuals[selected]))))
    maximum = float(np.max(residuals[selected]))
    inlier_fraction = float(selected.mean())
    rgb_hull = cv2.convexHull(rgb_points.astype(np.float32))
    thermal_hull = cv2.convexHull(thermal_points.astype(np.float32))
    coverage = min(
        float(cv2.contourArea(rgb_hull)) / float(rgb_width * rgb_height),
        float(cv2.contourArea(thermal_hull)) / float(thermal_width * thermal_height),
    )
    failures = []
    if rmse > maximum_rmse_px:
        failures.append(f"RMSE {rmse:.3f} px exceeds {maximum_rmse_px:.3f} px")
    if inlier_fraction < minimum_inlier_fraction:
        failures.append(
            f"inlier fraction {inlier_fraction:.3f} is below {minimum_inlier_fraction:.3f}"
        )
    if coverage < minimum_coverage_fraction:
        failures.append(f"spatial coverage {coverage:.3f} is below {minimum_coverage_fraction:.3f}")
    if failures:
        raise IndiaPackRefused("RGB/thermal registration quality failed: " + "; ".join(failures))

    return ThermalRegistration(
        method="homography_ransac_operator_tie_point_self_consistency",
        rgb_width=int(rgb_width), rgb_height=int(rgb_height),
        thermal_width=int(thermal_width), thermal_height=int(thermal_height),
        residual_px=round(rmse, 6), max_residual_px=round(maximum, 6),
        tie_point_count=len(rows), inlier_count=int(selected.sum()),
        inlier_fraction=round(inlier_fraction, 6), coverage_fraction=round(coverage, 6),
        ransac_reproj_threshold_px=float(ransac_reproj_threshold_px),
        acceptance_rmse_px=float(maximum_rmse_px),
        evidence_kind="operator_tie_point_self_consistency_not_image_feature_matching",
        validated_by=validated_by.strip(),
        rgb_to_thermal_homography=tuple(float(value) for value in matrix.reshape(-1)),
        rgb_sha256=_sha256(rgb_file), thermal_sha256=_sha256(thermal_file),
    )


def _json_temperatures(celsius: np.ndarray) -> list[float | None]:
    return [round(float(value), 6) if math.isfinite(float(value)) else None
            for value in celsius.reshape(-1)]


def write_radiometric_thermal_map(
    image: ThermalImage,
    output_dir: str | Path,
    *,
    epsg: int,
    west: float,
    north: float,
    pixel_size: float,
    registration: ThermalRegistration | None = None,
) -> IndiaPackResult:
    """Write one measured temperature map for GIS and the local Hub.

    The JSON values are copied from ``ThermalImage.celsius``.  They are not
    interpolated, recoloured values, or a visual proxy.  RGB comparison metadata
    is included only when an explicit validated registration is supplied.
    """

    field = np.asarray(image.celsius, dtype=np.float64)
    if field.ndim != 2 or not field.size:
        raise IndiaPackRefused("A thermal map requires a non-empty 2D temperature field.")
    finite = field[np.isfinite(field)]
    if not finite.size:
        raise IndiaPackRefused("The thermal image contains no valid temperature measurements.")
    if not all(math.isfinite(float(value)) for value in (west, north, pixel_size)):
        raise IndiaPackRefused("Thermal georeferencing values must be finite.")
    if pixel_size <= 0:
        raise IndiaPackRefused("Thermal pixel_size must be positive.")

    try:
        from rasterio.crs import CRS

        crs = CRS.from_epsg(int(epsg))
    except Exception as exc:  # pragma: no cover - rasterio supplies the detail
        raise IndiaPackRefused(f"Thermal EPSG:{epsg} is not a usable CRS.") from exc
    if not crs.is_projected:
        raise IndiaPackRefused(
            "Thermal survey maps require a projected CRS; degrees cannot support "
            "metric pixel size or 3D surface association."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geotiff = Path(write_thermal_geotiff(
        image, output / "thermal_temperature.tif", epsg=int(epsg),
        west=float(west), north=float(north), pixel_size=float(pixel_size),
    ))

    # Make the measurement origin explicit in the GIS artifact itself.
    import rasterio

    with rasterio.open(geotiff, "r+") as raster:
        raster.update_tags(
            ODK_VALUE_KIND="measured_surface_temperature",
            ODK_VALUE_UNIT="celsius",
            ODK_TEMPERATURE_SOURCE="radiometric_counts_via_core.thermal",
            ODK_INTERPOLATED="false",
        )

    height, width = field.shape
    payload: dict[str, Any] = {
        "type": "odk-thermal-map",
        "schema_version": 1,
        "width": int(width),
        "height": int(height),
        "crs_epsg": int(epsg),
        "transform": [float(pixel_size), 0.0, float(west),
                      0.0, -float(pixel_size), float(north)],
        "unit": "celsius",
        "values": _json_temperatures(field),
        "stats": image.stats,
        "source": image.source,
        "temperature_source": "radiometric_counts_via_core.thermal",
        "interpolated": False,
        "calibration": asdict(image.calibration),
        "geotiff": geotiff.name,
    }
    if registration is not None:
        payload["registration"] = registration.to_dict()

    manifest = output / "thermal_map.json"
    manifest.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = output / "thermal_summary.json"
    summary.write_text(json.dumps({
        "status": "complete",
        "measurement": "radiometric surface temperature",
        "unit": "celsius",
        "crs_epsg": int(epsg),
        "valid_pixels": int(finite.size),
        "stats": image.stats,
        "registration": registration.to_dict() if registration else {
            "available": False,
            "reason": "No validated RGB/thermal registration was supplied.",
        },
        "limits": [
            "Temperatures come from radiometric counts and camera calibration, not palette colours.",
            "No spatial interpolation was applied to the temperature grid.",
            "A statistical or visual hotspot is not a confirmed equipment fault.",
        ],
    }, indent=2, allow_nan=False), encoding="utf-8")

    return IndiaPackResult(
        summary_path=str(summary),
        artifact_paths=(str(geotiff), str(manifest)),
    )


def _radiometric_georeferencing(path: Path) -> tuple[ThermalImage, int, tuple[float, ...]]:
    try:
        image = load_radiometric_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (NotRadiometric, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise IndiaPackRefused(
            "Module temperature association requires a radiometric sidecar with raw "
            "counts and calibration; rendered thermal pixels are not measurements."
        ) from exc
    georeferencing = payload.get("georeferencing")
    if not isinstance(georeferencing, Mapping):
        raise IndiaPackRefused(
            "Radiometric thermal sidecar has no georeferencing; module temperatures "
            "cannot be placed on mapped assets."
        )
    epsg = int(georeferencing.get("epsg", 0))
    transform = georeferencing.get("transform")
    if not isinstance(transform, Sequence) or len(transform) != 6:
        raise IndiaPackRefused("Thermal georeferencing needs a six-value affine transform.")
    coefficients = tuple(float(value) for value in transform)
    if not all(math.isfinite(value) for value in coefficients):
        raise IndiaPackRefused("Thermal affine transform contains a non-finite value.")
    determinant = coefficients[0] * coefficients[4] - coefficients[1] * coefficients[3]
    if abs(determinant) < 1e-15:
        raise IndiaPackRefused("Thermal affine transform is degenerate.")
    from rasterio.crs import CRS

    crs = CRS.from_epsg(epsg)
    if not crs.is_projected:
        raise IndiaPackRefused(
            "Module-level thermal deliverables require a projected thermal CRS."
        )
    return image, epsg, coefficients


def _pixel_center_world(transform: Sequence[float], row: int, column: int) -> list[float]:
    a, b, c, d, e, f = transform
    x = a * (column + 0.5) + b * (row + 0.5) + c
    y = d * (column + 0.5) + e * (row + 0.5) + f
    return [round(float(x), 6), round(float(y), 6)]


def associate_module_temperatures(
    thermal_path: str | Path,
    registration: ThermalRegistration,
    modules_path: str | Path,
    output_dir: str | Path,
    *,
    warning_delta_c: float,
    critical_delta_c: float,
    hot_cell_delta_c: float,
    threshold_basis: str,
    minimum_valid_fraction: float = 0.8,
    minimum_polygon_iou: float = 0.5,
) -> IndiaPackResult:
    """Associate measured temperatures with geolocated module polygons.

    Every module must carry an observed RGB pixel polygon as
    ``properties.rgb_polygon_px``.  That polygon is transformed through the
    validated homography and intersected with the independently georeferenced
    module geometry.  Temperature is read only from the agreed cells.
    """

    quality_fields = (
        registration.max_residual_px,
        registration.tie_point_count,
        registration.inlier_count,
        registration.inlier_fraction,
        registration.coverage_fraction,
        registration.ransac_reproj_threshold_px,
        registration.acceptance_rmse_px,
        registration.evidence_kind,
        registration.rgb_sha256,
        registration.thermal_sha256,
    )
    if registration.rgb_to_thermal_homography is None or any(
        value is None for value in quality_fields
    ):
        raise IndiaPackRefused(
            "Module association requires a scored RGB-to-thermal geometric registration."
        )
    if not 0 < minimum_valid_fraction <= 1 or not 0 < minimum_polygon_iou <= 1:
        raise ValueError("Association fractions must be between zero and one.")
    if warning_delta_c <= 0 or critical_delta_c < warning_delta_c or hot_cell_delta_c <= 0:
        raise ValueError("Temperature-delta thresholds must be positive and ordered.")
    if not threshold_basis.strip():
        raise IndiaPackRefused(
            "Temperature severity thresholds need a named standard or configured convention."
        )

    thermal_file = Path(thermal_path)
    image, thermal_epsg, transform_values = _radiometric_georeferencing(thermal_file)
    if image.celsius.ndim != 2 or not image.celsius.size:
        raise IndiaPackRefused("Module association requires a non-empty 2D thermal grid.")
    height, width = image.celsius.shape
    if (registration.thermal_width, registration.thermal_height) != (width, height):
        raise IndiaPackRefused("Registration thermal dimensions do not match the radiometric grid.")
    if registration.thermal_sha256 != _sha256(thermal_file):
        raise IndiaPackRefused(
            "Registration was scored against a different thermal file; association is refused."
        )

    features, module_epsg = read_geojson_features(
        modules_path, allowed_geometry_types={"Polygon", "MultiPolygon"}
    )
    if not features:
        raise IndiaPackRefused("Module inventory contains no polygons.")
    if module_epsg != thermal_epsg:
        raise IndiaPackRefused(
            f"Module inventory EPSG:{module_epsg} does not match thermal EPSG:{thermal_epsg}."
        )
    from rasterio.crs import CRS
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    import cv2

    crs = CRS.from_epsg(module_epsg)
    unit_to_m = float(crs.linear_units_factor[1])
    transform = Affine(*transform_values)
    homography = np.asarray(registration.rgb_to_thermal_homography, dtype=np.float64).reshape(3, 3)
    modules: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, feature in enumerate(features):
        properties = dict(feature.get("properties") or {})
        module_id = str(properties.get("module_id") or properties.get("id") or "").strip()
        string_id = str(properties.get("string_id") or "").strip()
        rgb_polygon = properties.get("rgb_polygon_px")
        if not module_id or not string_id:
            raise IndiaPackRefused(
                f"Module feature {index + 1} must name module_id and string_id."
            )
        if module_id in seen_ids:
            raise IndiaPackRefused(f"Duplicate module_id {module_id!r}.")
        seen_ids.add(module_id)
        if not isinstance(rgb_polygon, Sequence) or len(rgb_polygon) < 3:
            raise IndiaPackRefused(
                f"Module {module_id} needs an observed rgb_polygon_px with at least three vertices."
            )
        rgb_points = np.asarray(rgb_polygon, dtype=np.float64)
        if rgb_points.ndim != 2 or rgb_points.shape[1] != 2 or not np.all(np.isfinite(rgb_points)):
            raise IndiaPackRefused(f"Module {module_id} rgb_polygon_px is malformed.")
        if np.any(rgb_points[:, 0] < 0) or np.any(rgb_points[:, 0] >= registration.rgb_width) or np.any(
            rgb_points[:, 1] < 0) or np.any(rgb_points[:, 1] >= registration.rgb_height
        ):
            raise IndiaPackRefused(f"Module {module_id} RGB polygon lies outside the registered image.")

        thermal_points = cv2.perspectiveTransform(
            rgb_points[None].astype(np.float32), homography
        )[0]
        registered_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(registered_mask, [np.rint(thermal_points).astype(np.int32)], 1)
        mapped_mask = rasterize(
            [(feature["geometry"], 1)], out_shape=(height, width), transform=transform,
            fill=0, all_touched=False, dtype="uint8",
        )
        intersection = (registered_mask > 0) & (mapped_mask > 0)
        union = (registered_mask > 0) | (mapped_mask > 0)
        iou = float(intersection.sum() / max(1, union.sum()))
        if iou < minimum_polygon_iou:
            raise IndiaPackRefused(
                f"Module {module_id} RGB/geospatial polygon IoU {iou:.3f} is below "
                f"{minimum_polygon_iou:.3f}; the temperature association is not trustworthy."
            )
        valid = intersection & np.isfinite(image.celsius)
        valid_fraction = float(valid.sum() / max(1, intersection.sum()))
        if valid_fraction < minimum_valid_fraction:
            raise IndiaPackRefused(
                f"Module {module_id} has {valid_fraction:.1%} valid temperature coverage, "
                f"below the required {minimum_valid_fraction:.1%}."
            )
        temperatures = image.celsius[valid]
        positions = np.argwhere(valid)
        hottest_index = int(np.argmax(temperatures))
        hottest_row, hottest_column = (int(value) for value in positions[hottest_index])
        modules.append({
            "feature": feature, "module_id": module_id, "string_id": string_id,
            "area_m2": float(geometry_area_units2(feature["geometry"]) * unit_to_m**2),
            "polygon_iou": iou, "valid_fraction": valid_fraction,
            "min_c": float(np.min(temperatures)), "max_c": float(np.max(temperatures)),
            "mean_c": float(np.mean(temperatures)), "median_c": float(np.median(temperatures)),
            "hottest_xy": _pixel_center_world(transform_values, hottest_row, hottest_column),
        })

    by_string: dict[str, list[float]] = {}
    for module in modules:
        by_string.setdefault(module["string_id"], []).append(module["median_c"])
    string_references = {
        string_id: float(np.median(values)) for string_id, values in by_string.items()
    }
    site_reference = float(np.median(list(string_references.values())))
    output_features: list[dict[str, Any]] = []
    candidate_points: list[dict[str, Any]] = []
    counts = {"hot_cell_candidate": 0, "module_deviation_candidate": 0,
              "string_deviation_candidate": 0, "review_required": 0,
              "warning": 0, "critical": 0}

    for module in modules:
        string_reference = string_references[module["string_id"]]
        deltas = {
            "hot_cell_delta_c": max(0.0, module["max_c"] - module["median_c"]),
            "module_delta_c": max(0.0, module["median_c"] - string_reference),
            "string_delta_c": max(0.0, string_reference - site_reference),
        }
        finding_types: list[str] = []
        if deltas["hot_cell_delta_c"] >= hot_cell_delta_c:
            finding_types.append("hot_cell_candidate")
        if deltas["module_delta_c"] >= warning_delta_c:
            finding_types.append("module_deviation_candidate")
        if deltas["string_delta_c"] >= warning_delta_c:
            finding_types.append("string_deviation_candidate")
        measured_delta = max(deltas.values())
        severity = "critical" if measured_delta >= critical_delta_c else (
            "warning" if finding_types else "normal"
        )
        for finding in finding_types:
            counts[finding] += 1
        if finding_types:
            counts["review_required"] += 1
            counts[severity] += 1

        properties = dict(module["feature"].get("properties") or {})
        properties.update({
            "module_id": module["module_id"], "string_id": module["string_id"],
            "temperature_unit": "celsius", "temperature_source": "radiometric_counts",
            "min_c": round(module["min_c"], 3), "max_c": round(module["max_c"], 3),
            "mean_c": round(module["mean_c"], 3), "median_c": round(module["median_c"], 3),
            "string_reference_c": round(string_reference, 3),
            "site_reference_c": round(site_reference, 3),
            **{key: round(value, 3) for key, value in deltas.items()},
            "measured_severity_delta_c": round(measured_delta, 3),
            "severity": severity,
            "review_state": "review_required" if finding_types else "within_peer_baseline",
            "finding_types": finding_types,
            "hottest_point_xy": module["hottest_xy"],
            "area_m2": round(module["area_m2"], 6),
            "association_polygon_iou": round(module["polygon_iou"], 6),
            "valid_temperature_fraction": round(module["valid_fraction"], 6),
        })
        output_features.append({
            "type": "Feature", "geometry": module["feature"]["geometry"],
            "properties": properties,
        })
        if finding_types:
            candidate_points.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": module["hottest_xy"]},
                "properties": {
                    "module_id": module["module_id"], "string_id": module["string_id"],
                    "finding_types": finding_types, "review_state": "review_required",
                    "severity": severity, "max_c": round(module["max_c"], 3),
                    "measured_severity_delta_c": round(measured_delta, 3),
                    "interpretation": "temperature_deviation_candidate_not_confirmed_defect",
                },
            })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    crs_payload = {"type": "name", "properties": {"name": f"EPSG:{module_epsg}"}}
    modules_output = output / "module_temperatures.geojson"
    modules_output.write_text(json.dumps({
        "type": "FeatureCollection", "crs": crs_payload, "features": output_features,
    }, indent=2, allow_nan=False), encoding="utf-8")
    candidates_output = output / "thermal_review_candidates.geojson"
    candidates_output.write_text(json.dumps({
        "type": "FeatureCollection", "crs": crs_payload, "features": candidate_points,
    }, indent=2, allow_nan=False), encoding="utf-8")
    summary = output / "solar_thermal_summary.json"
    summary.write_text(json.dumps({
        "status": "complete", "module_count": len(modules),
        "module_hotspot_count": counts["review_required"],
        "string_count": len(by_string), "site_reference_c": round(site_reference, 3),
        "counts": counts,
        "severity_thresholds": {
            "warning_delta_c": warning_delta_c, "critical_delta_c": critical_delta_c,
            "hot_cell_delta_c": hot_cell_delta_c,
        },
        "severity_threshold_basis": threshold_basis.strip(),
        "severity_classification": "configured_convention_not_equipment_diagnosis",
        "registration": registration.to_dict(),
        "thermal_source": str(thermal_file), "thermal_sha256": _sha256(thermal_file),
        "module_source": str(modules_path), "crs_epsg": module_epsg,
        "limits": [
            "All temperatures come from radiometric counts and camera calibration.",
            "Severity is backed by stated measured temperature deltas, not an inferred failure cause.",
            "Every candidate is review_required and is not a confirmed electrical defect.",
            "No output from the general anomaly engine is converted into a named defect.",
        ],
    }, indent=2, allow_nan=False), encoding="utf-8")
    return IndiaPackResult(
        summary_path=str(summary),
        artifact_paths=(str(modules_output), str(candidates_output)),
    )
