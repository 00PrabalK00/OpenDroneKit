"""Client-facing survey intelligence built from measured geospatial products.

The first mission-pack primitive is intentionally geometric, not an AI claim. Two
aligned DSMs can prove where the surface rose or fell and by how much. They cannot,
without semantic evidence, prove that the cause was a new building, an excavation,
or a stockpile delivery. The package produced here preserves that distinction while
still giving a construction or mining customer useful mapped quantities.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import geo
from .change_detection import SurfaceChange, compare_surfaces
from .dsm_analysis import RasterSurface, estimate_volume, load_surface


@dataclass(frozen=True)
class ChangeRegion:
    """One contiguous measured region of surface rise or fall."""

    region_id: str
    change: str
    area_m2: float
    volume_m3: float
    mean_delta_m: float
    extreme_delta_m: float
    cell_count: int
    geometry: dict[str, Any]

    def properties(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("geometry")
        return payload

    def feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": self.geometry,
            "properties": self.properties(),
        }


@dataclass(frozen=True)
class SurveyChangePackage:
    """Paths and measurements emitted by one T1/T2 surface comparison."""

    summary_path: str
    difference_raster_path: str
    regions_geojson_path: str
    report_path: str
    surface_change: SurfaceChange
    regions: tuple[ChangeRegion, ...]

    def artifact_paths(self) -> list[str]:
        return [
            self.summary_path,
            self.difference_raster_path,
            self.regions_geojson_path,
            self.report_path,
        ]


@dataclass(frozen=True)
class SelectedROIChangePackage:
    '''Measured T1/T2 change constrained to one operator-selected region.'''

    summary_path: str
    difference_raster_path: str
    selection_geojson_path: str
    regions_geojson_path: str
    report_path: str
    surface_change: SurfaceChange
    regions: tuple[ChangeRegion, ...]

    def artifact_paths(self) -> list[str]:
        return [
            self.summary_path,
            self.difference_raster_path,
            self.selection_geojson_path,
            self.regions_geojson_path,
            self.report_path,
        ]


@dataclass(frozen=True)
class StockpilePackage:
    """Mapped selection, measured volume and report for one stockpile."""

    summary_path: str
    selection_geojson_path: str
    report_path: str
    measurement: dict[str, Any]

    def artifact_paths(self) -> list[str]:
        return [self.summary_path, self.selection_geojson_path, self.report_path]


def _polygonize_regions(
    difference: RasterSurface,
    *,
    threshold_m: float,
    min_region_area_m2: float,
) -> list[ChangeRegion]:
    """Turn contiguous changed cells into exact georeferenced raster polygons."""

    geo.require("rasterio")
    try:
        from affine import Affine
        from rasterio.features import shapes
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Mapped change regions require rasterio, affine and scipy."
        ) from exc

    delta = difference.elevation
    valid = np.isfinite(delta)
    affine = Affine(*difference.transform[:6])
    regions: list[ChangeRegion] = []

    for change, mask in (
        ("surface_rise", valid & (delta >= threshold_m)),
        ("surface_fall", valid & (delta <= -threshold_m)),
    ):
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue

        geometries = {
            int(value): geometry
            for geometry, value in shapes(
                labels.astype(np.int32),
                mask=mask,
                transform=affine,
                connectivity=8,
            )
            if int(value) > 0
        }
        for label_id in range(1, count + 1):
            cells = labels == label_id
            cell_count = int(cells.sum())
            area_m2 = cell_count * difference.pixel_area_m2
            if area_m2 < min_region_area_m2:
                continue

            values = delta[cells]
            signed_volume = float(values.sum() * difference.pixel_area_m2)
            extreme = float(values.max() if change == "surface_rise" else values.min())
            regions.append(
                ChangeRegion(
                    region_id=f"{change}-{label_id}",
                    change=change,
                    area_m2=round(area_m2, 3),
                    volume_m3=round(abs(signed_volume), 3),
                    mean_delta_m=round(float(values.mean()), 4),
                    extreme_delta_m=round(extreme, 4),
                    cell_count=cell_count,
                    geometry=geometries[label_id],
                )
            )

    return sorted(regions, key=lambda region: (-region.volume_m3, region.region_id))


def _write_report(
    path: Path,
    *,
    earlier_dsm: str | Path,
    later_dsm: str | Path,
    change: SurfaceChange,
    regions: list[ChangeRegion],
    threshold_m: float,
    min_region_area_m2: float,
) -> str:
    lines = [
        "# Survey surface-change report",
        "",
        "## Result",
        "",
        f"- Surface rise volume: **{change.added_volume_m3:.3f} m3**",
        f"- Surface fall volume: **{change.removed_volume_m3:.3f} m3**",
        f"- Net volume change: **{change.net_volume_m3:.3f} m3**",
        f"- Changed area: **{change.changed_area_m2:.3f} m2**",
        f"- Maximum rise: **{change.max_rise_m:.3f} m**",
        f"- Maximum fall: **{change.max_fall_m:.3f} m**",
        f"- Coordinate reference system: **EPSG:{change.crs_epsg}**",
        "",
        "## Mapped regions",
        "",
    ]
    if regions:
        lines.extend([
            "| Region | Measured change | Area (m2) | Volume (m3) | Mean delta (m) |",
            "|---|---|---:|---:|---:|",
        ])
        for region in regions:
            lines.append(
                f"| {region.region_id} | {region.change.replace('_', ' ')} | "
                f"{region.area_m2:.3f} | {region.volume_m3:.3f} | "
                f"{region.mean_delta_m:.4f} |"
            )
    else:
        lines.append("No contiguous region met the configured thresholds.")

    lines.extend([
        "",
        "## Method and limits",
        "",
        f"Earlier DSM: `{earlier_dsm}`",
        "",
        f"Later DSM: `{later_dsm}`",
        "",
        (
            "The rasters were compared cell by cell only after their CRS, dimensions, "
            "resolution, origin, and orientation were confirmed to match."
        ),
        "",
        (
            f"A cell is mapped when its absolute elevation change is at least "
            f"{threshold_m:.3f} m. Regions smaller than {min_region_area_m2:.3f} m2 "
            "are omitted from the vector layer, while the headline volumes retain all "
            "finite elevation differences."
        ),
        "",
        (
            "A surface rise is measured added elevation and a surface fall is measured "
            "removed elevation. Geometry alone does not prove the cause: these results "
            "must not be relabelled as new construction, delivered material, excavation, "
            "or demolition without imagery, site records, or a validated semantic model."
        ),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def create_surface_change_package(
    earlier_dsm: str | Path,
    later_dsm: str | Path,
    output_dir: str | Path,
    *,
    change_threshold_m: float = 0.05,
    min_region_area_m2: float = 1.0,
) -> SurveyChangePackage:
    """Create mapped, measured, client-readable T1/T2 survey-change artifacts."""

    if change_threshold_m <= 0:
        raise ValueError("change_threshold_m must be greater than zero.")
    if min_region_area_m2 < 0:
        raise ValueError("min_region_area_m2 must not be negative.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    difference_path = out / "elevation_change.tif"
    surface_change = compare_surfaces(
        earlier_dsm,
        later_dsm,
        output_path=difference_path,
        change_threshold_m=change_threshold_m,
    )
    difference = load_surface(difference_path)
    regions = _polygonize_regions(
        difference,
        threshold_m=change_threshold_m,
        min_region_area_m2=min_region_area_m2,
    )

    regions_path = geo.write_geojson(
        out / "change_regions.geojson",
        [region.feature() for region in regions],
        epsg=int(surface_change.crs_epsg or 4326),
        properties={
            "source": "measured_dsm_difference",
            "change_threshold_m": change_threshold_m,
            "min_region_area_m2": min_region_area_m2,
            "interpretation": (
                "surface_rise/surface_fall are geometric measurements, not semantic causes"
            ),
        },
    )
    report_path = _write_report(
        out / "survey_change_report.md",
        earlier_dsm=earlier_dsm,
        later_dsm=later_dsm,
        change=surface_change,
        regions=regions,
        threshold_m=change_threshold_m,
        min_region_area_m2=min_region_area_m2,
    )

    summary_path = out / "survey_change.json"
    summary = {
        "status": "complete",
        "analysis": "measured_surface_change",
        "inputs": {"earlier_dsm": str(earlier_dsm), "later_dsm": str(later_dsm)},
        "crs_epsg": surface_change.crs_epsg,
        "units": {"elevation": "m", "area": "m2", "volume": "m3"},
        "thresholds": {
            "change_m": change_threshold_m,
            "minimum_mapped_region_area_m2": min_region_area_m2,
        },
        "totals": surface_change.to_dict(),
        "regions": [region.properties() for region in regions],
        "artifacts": {
            "difference_raster": str(difference_path),
            "regions_geojson": str(regions_path),
            "report": str(report_path),
        },
        "interpretation": (
            "Measured surface rise and fall only. Semantic causes require independent evidence."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return SurveyChangePackage(
        summary_path=str(summary_path),
        difference_raster_path=str(difference_path),
        regions_geojson_path=str(regions_path),
        report_path=str(report_path),
        surface_change=surface_change,
        regions=tuple(regions),
    )


def _closed_polygon(polygon_xy: Sequence[Sequence[float]]) -> list[list[float]]:
    vertices = [
        [float(vertex[0]), float(vertex[1])]
        for vertex in polygon_xy
        if len(vertex) >= 2
    ]
    if len(vertices) >= 2 and np.allclose(
        vertices[0], vertices[-1], rtol=0.0, atol=1e-9
    ):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ValueError("A stockpile selection needs at least three polygon vertices.")
    return [*vertices, list(vertices[0])]


def create_selected_roi_change_package(
    earlier_dsm: str | Path,
    later_dsm: str | Path,
    output_dir: str | Path,
    *,
    polygon_xy: Sequence[Sequence[float]],
    roi_type: str = 'stockpile',
    roi_name: str = '',
    change_threshold_m: float = 0.05,
    min_region_area_m2: float = 1.0,
) -> SelectedROIChangePackage:
    '''Measure surface rise/fall between surveys only inside a selected ROI.'''

    polygon = _closed_polygon(polygon_xy)
    roi_type = roi_type.strip().lower()
    if roi_type not in {'stockpile', 'pit'}:
        raise ValueError('roi_type must be either stockpile or pit.')
    if change_threshold_m <= 0:
        raise ValueError('change_threshold_m must be greater than zero.')
    if min_region_area_m2 < 0:
        raise ValueError('min_region_area_m2 must not be negative.')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    difference_path = out / 'roi_elevation_change.tif'
    surface_change = compare_surfaces(
        earlier_dsm,
        later_dsm,
        output_path=difference_path,
        change_threshold_m=change_threshold_m,
        roi_polygon_xy=polygon,
    )
    difference = load_surface(difference_path)
    regions = _polygonize_regions(
        difference,
        threshold_m=change_threshold_m,
        min_region_area_m2=min_region_area_m2,
    )
    compared_area_m2 = surface_change.compared_cells * difference.pixel_area_m2
    selection_path = geo.write_geojson(
        out / 'roi_selection.geojson',
        [{
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [polygon]},
            'properties': {
                'selection': 'operator_polygon',
                'roi_type': roi_type,
                'roi_name': roi_name or None,
                'compared_area_m2': round(compared_area_m2, 3),
                'surface_rise_volume_m3': round(surface_change.added_volume_m3, 3),
                'surface_fall_volume_m3': round(surface_change.removed_volume_m3, 3),
                'net_volume_change_m3': round(surface_change.net_volume_m3, 3),
            },
        }],
        epsg=int(surface_change.crs_epsg or 4326),
        properties={
            'source': 'operator_selected_roi',
            'units': {'area': 'm2', 'volume': 'm3'},
        },
    )
    regions_path = geo.write_geojson(
        out / 'roi_change_regions.geojson',
        [region.feature() for region in regions],
        epsg=int(surface_change.crs_epsg or 4326),
        properties={
            'source': 'measured_dsm_difference_inside_roi',
            'roi_type': roi_type,
            'change_threshold_m': change_threshold_m,
            'min_region_area_m2': min_region_area_m2,
        },
    )

    subject = roi_name or f'selected {roi_type}'
    report_path = out / 'roi_change_report.md'
    report_path.write_text(
        '\n'.join([
            f'# {roi_type.title()} change report',
            '',
            f'Selection: **{subject}**',
            '',
            '## Result',
            '',
            f'- Surface rise volume: **{surface_change.added_volume_m3:.3f} m3**',
            f'- Surface fall volume: **{surface_change.removed_volume_m3:.3f} m3**',
            f'- Net volume change: **{surface_change.net_volume_m3:.3f} m3**',
            f'- Changed area: **{surface_change.changed_area_m2:.3f} m2**',
            f'- Compared area: **{compared_area_m2:.3f} m2**',
            f'- Coordinate reference system: **EPSG:{surface_change.crs_epsg}**',
            '',
            '## Method and limits',
            '',
            'Only cells whose centres fall inside the saved operator polygon are compared.',
            '',
            (
                'Surface rise and fall are geometric evidence. For a stockpile, rise may '
                'indicate material added and fall may indicate material removed; for a pit '
                'the operational interpretation is often reversed. The report does not '
                'assert either cause without site records or validated semantic evidence.'
            ),
            '',
        ]),
        encoding='utf-8',
    )
    summary_path = out / 'roi_change.json'
    summary = {
        'status': 'complete',
        'analysis': 'selected_roi_surface_change',
        'selection': 'operator_polygon',
        'roi_type': roi_type,
        'roi_name': roi_name or None,
        'inputs': {'earlier_dsm': str(earlier_dsm), 'later_dsm': str(later_dsm)},
        'crs_epsg': surface_change.crs_epsg,
        'units': {'elevation': 'm', 'area': 'm2', 'volume': 'm3'},
        'compared_area_m2': round(compared_area_m2, 3),
        'thresholds': {
            'change_m': change_threshold_m,
            'minimum_mapped_region_area_m2': min_region_area_m2,
        },
        'totals': surface_change.to_dict(),
        'regions': [region.properties() for region in regions],
        'artifacts': {
            'difference_raster': str(difference_path),
            'selection_geojson': str(selection_path),
            'regions_geojson': str(regions_path),
            'report': str(report_path),
        },
        'interpretation': 'Measured surface change inside the selected ROI; cause is not inferred.',
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return SelectedROIChangePackage(
        summary_path=str(summary_path),
        difference_raster_path=str(difference_path),
        selection_geojson_path=str(selection_path),
        regions_geojson_path=str(regions_path),
        report_path=str(report_path),
        surface_change=surface_change,
        regions=tuple(regions),
    )


def create_stockpile_package(
    dsm_path: str | Path,
    output_dir: str | Path,
    *,
    polygon_xy: Sequence[Sequence[float]],
    dtm_path: str | Path | None = None,
    base_elevation_m: float | None = None,
) -> StockpilePackage:
    """Measure a selected stockpile against an explicit, defensible base surface.

    This is the useful pre-segmentation workflow: an operator selects the pile on the
    map, and geometry computes the volume. Automatic pile segmentation can replace
    that selection later without changing the measurement contract.
    """

    polygon = _closed_polygon(polygon_xy)
    if dtm_path is not None and not Path(dtm_path).is_file():
        raise ValueError(f"DTM does not exist: {dtm_path}")
    if dtm_path is None and base_elevation_m is None:
        raise ValueError(
            "Stockpile measurement requires either an aligned DTM or an "
            "operator-supplied base_elevation_m. The selection minimum is not a "
            "defensible base surface."
        )

    result = estimate_volume(
        dsm_path,
        dtm_path=dtm_path,
        polygon_xy=polygon,
        base_elevation_m=base_elevation_m,
    )
    if not result.get("ok") or not result.get("preferred"):
        raise ValueError(result.get("reason") or "Stockpile volume could not be measured.")

    preferred = result["preferred"]
    surface = load_surface(dsm_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    selection_path = geo.write_geojson(
        out / "stockpile_selection.geojson",
        [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": {
                "measurement": "stockpile_volume",
                "reference": preferred["reference"],
                "cut_volume_m3": preferred["cut_volume_m3"],
                "fill_volume_m3": preferred["fill_volume_m3"],
                "net_volume_m3": preferred["net_volume_m3"],
                "selected_area_m2": result["region_area_m2"],
            },
        }],
        epsg=int(surface.epsg or 4326),
        properties={
            "source": "operator_selected_polygon",
            "units": {"area": "m2", "volume": "m3"},
        },
    )

    reference_description = (
        f"aligned DTM {dtm_path}"
        if dtm_path is not None
        else f"operator-supplied horizontal plane at {float(base_elevation_m):.3f} m"
    )
    report_path = out / "stockpile_report.md"
    report_path.write_text(
        "\n".join([
            "# Stockpile measurement report",
            "",
            "## Result",
            "",
            f"- Material above reference: **{preferred['cut_volume_m3']:.3f} m3**",
            f"- Void below reference: **{preferred['fill_volume_m3']:.3f} m3**",
            f"- Net volume: **{preferred['net_volume_m3']:.3f} m3**",
            f"- Selected ground area: **{result['region_area_m2']:.3f} m2**",
            f"- Coordinate reference system: **EPSG:{result['crs_epsg']}**",
            f"- Base surface: {reference_description}",
            "",
            "## Method and limits",
            "",
            (
                "The volume is the cell-by-cell DSM height above the stated reference "
                "inside the operator-selected polygon. The polygon is stored as a "
                "georeferenced layer with this report."
            ),
            "",
            (
                "The result does not classify the material. Survey accuracy, occlusion, "
                "vegetation and the chosen base surface remain the dominant sources of "
                "measurement uncertainty."
            ),
            "",
        ]),
        encoding="utf-8",
    )

    summary_path = out / "stockpile_measurement.json"
    summary = {
        "status": "complete",
        "analysis": "selected_stockpile_volume",
        "dsm_path": str(dsm_path),
        "dtm_path": str(dtm_path) if dtm_path is not None else None,
        "base_elevation_m": base_elevation_m,
        "crs_epsg": result["crs_epsg"],
        "units": {"elevation": "m", "area": "m2", "volume": "m3"},
        "selection": "operator_polygon",
        "measurement": result,
        "artifacts": {
            "selection_geojson": str(selection_path),
            "report": str(report_path),
        },
        "interpretation": (
            "Geometry measures volume inside the selected region; no material class "
            "or automatic stockpile detection is claimed."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return StockpilePackage(
        summary_path=str(summary_path),
        selection_geojson_path=str(selection_path),
        report_path=str(report_path),
        measurement=result,
    )
