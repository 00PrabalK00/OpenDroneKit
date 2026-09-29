"""Reports for the project the desktop has open.

The report engine used to be reached through a mirror: the desktop project was
re-created by name in core.project's separate workspace, with no root folder, and the
engine read that empty mirror. A project with 24 geotagged images and a 144-waypoint
mission produced a report saying "No missions saved / No datasets imported", written to
a folder outside the project.

This module builds the report context from what the desktop actually holds -- its
SQLite store (datasets, mission versions, audit events) and the project folder
(annotations, measurements, reconstruction outputs, pipeline runs) -- and writes the
report into <project root>/reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.report_engine import (
    ReportConfig,
    ReportReadiness,
    ReportResult,
    find_pipeline_runs,
    find_reconstructions,
    findings_from_annotations,
    list_reports_in,
    readiness_from_context,
    render_report,
    select_photos,
    sections_for,
)


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def _latest_missions(store: Any, project_id: int) -> list[dict[str, Any]]:
    """The newest saved version of each mission."""
    latest: dict[str, dict[str, Any]] = {}
    for version in store.list_mission_versions(project_id):
        name = str(version.get("mission_name", ""))
        if name in latest and int(latest[name]["version"]) >= int(version.get("version_num", 0)):
            continue
        summary = _json(version.get("plan_summary_json"), {})
        latest[name] = {
            "name": name,
            "mode": version.get("template", ""),
            "version": int(version.get("version_num", 0)),
            "saved_at": version.get("created_at", ""),
            "waypoint_count": summary.get("waypoints"),
            "altitude_m": summary.get("altitude_m"),
            "path_distance_m": summary.get("path_distance_m"),
            "estimated_time_min": summary.get("estimated_time_min"),
            "estimated_gsd_cm": summary.get("estimated_gsd_cm"),
            "geojson": summary.get("geojson") or {},
        }
    return sorted(latest.values(), key=lambda m: m["name"])


def _audit_detail(payload: dict[str, Any]) -> str:
    """An audit payload as one line, with local folder paths reduced to their names."""
    parts = []
    for key, value in payload.items():
        text = str(value)
        if isinstance(value, str) and ("\\" in value or "/" in value) and len(value) > 3:
            text = Path(value).name
        parts.append(f"{key}={text}")
    return ", ".join(parts)[:200]


def desktop_context(session: Any, config: ReportConfig) -> dict[str, Any]:
    """Everything the open desktop project contains, in the report engine's shape."""
    from core.annotations import list_annotations
    from core.measurements import list_measurements

    project = session.ensure_active_project()
    project_id = int(project["id"])
    root = Path(session.project_root())
    store = session.store

    datasets: list[dict[str, Any]] = []
    image_dirs: list[Path] = []
    lons: list[float] = []
    lats: list[float] = []
    epsg = None
    for row in store.list_datasets(project_id):
        meta = _json(row.get("metadata_json"), {})
        datasets.append({
            "name": row.get("name", ""),
            "image_count": meta.get("image_count"),
            "geotagged_count": meta.get("geotagged_count"),
            "camera": meta.get("camera", ""),
            "imported": row.get("created_at", ""),
        })
        folder = Path(str(row.get("path", "")))
        if folder.is_dir() and folder not in image_dirs:
            image_dirs.append(folder)
        elif str(row.get("path", "")) and not folder.is_dir():
            # Imported, then moved or deleted: readiness says so rather than letting the
            # gallery and key photos come out silently empty.
            datasets[-1]["folder_missing"] = str(folder)
        bounds = meta.get("bounds_lonlat")
        if isinstance(bounds, list) and len(bounds) == 4:
            lons += [float(bounds[0]), float(bounds[2])]
            lats += [float(bounds[1]), float(bounds[3])]
        epsg = epsg or meta.get("suggested_epsg")
    site: dict[str, Any] = {}
    if lons:
        site["bounds_lonlat"] = [min(lons), min(lats), max(lons), max(lats)]
        site["center_lonlat"] = [(min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2]

    reconstructions = find_reconstructions(root)
    for recon in reconstructions:
        if recon.get("crs_epsg"):
            site["crs_epsg"] = recon["crs_epsg"]
            break
    else:
        if epsg:
            site["crs_epsg"] = epsg

    annotations = list_annotations(root) if config.include_annotations else []
    findings = findings_from_annotations(annotations, image_dirs)
    measurements = ([m.to_dict() for m in list_measurements(root)]
                    if config.include_measurements else [])

    audit = []
    if config.include_audit_trail:
        for event in store.list_audit_events(project_id, 200):
            audit.append({
                "timestamp": event.get("created_at", ""),
                "event_type": event.get("event_type", ""),
                "detail": _audit_detail(_json(event.get("payload_json"), {})),
            })

    cfg = config.to_dict()
    cfg["sections"] = sections_for(config.report_type, config.sections)
    return {
        "config": cfg,
        "project": {
            "id": str(project_id),
            "name": project.get("name", ""),
            "description": project.get("description", ""),
            "created_at": project.get("created_at", ""),
        },
        "project_root": str(root),
        "site": site,
        "missions": _latest_missions(store, project_id),
        "datasets": datasets,
        "image_dirs": [str(p) for p in image_dirs],
        "findings": findings,
        "defect_runs": [],
        "pipeline_runs": find_pipeline_runs(root) if config.include_defects else [],
        "crack_runs": [],
        "reconstructions": reconstructions,
        "measurements": measurements,
        "audit": audit,
        "photos": select_photos(image_dirs, findings) if config.include_images else [],
        "generated_at": _now(),
    }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


NOTHING_TO_REPORT = ("content: this project has no imported images, recorded findings, "
                     "saved missions or analysis runs, so a report would have nothing to be "
                     "about. Import a dataset, record a finding or save a mission first.")


def _judge(context: dict[str, Any], config: ReportConfig) -> ReportReadiness:
    """The engine's checklist, plus the one thing only the open project can say.

    A project with nothing in it used to be declared ready with "Nothing missing", and a
    report of empty sections was written -- a document indistinguishable from a survey
    that found nothing. It is refused instead.
    """
    ready = readiness_from_context(context, config)
    if not any(context.get(key) for key in ("datasets", "missions", "findings", "pipeline_runs",
                                            "reconstructions", "measurements")):
        ready.missing.insert(0, NOTHING_TO_REPORT)
        ready.ok = False
    return ready


def readiness(session: Any, config: ReportConfig) -> ReportReadiness:
    return _judge(desktop_context(session, config), config)


def generate(session: Any, config: ReportConfig) -> ReportResult:
    """Write a report of the open project into <project root>/reports."""
    context = desktop_context(session, config)
    ready = _judge(context, config)
    if not ready.ok:
        raise ValueError("Report not generated: " + "; ".join(ready.missing))
    root = Path(session.project_root())
    result = render_report(context, root / "reports", config.formats)
    result.warnings = list(ready.warnings)
    (Path(result.output_dir) / "report.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    session.store.save_report(
        session.project_id(), result.title, result.report_type, result.html_path,
        metadata={"pdf_path": result.pdf_path, "docx_path": result.docx_path,
                  "report_id": result.id, "findings": result.findings},
    )
    return result


def list_reports(session: Any) -> list[dict[str, Any]]:
    """Reports written into this project's folder, newest first."""
    root = Path(session.project_root())
    return [r.to_dict() for r in list_reports_in(root / "reports")]


def persist_pipeline_findings(root: Path | str, project_id: str | int,
                              summary_path: Path | str) -> dict[str, Any]:
    """Turn a pipeline run's MODEL detections into reviewable findings.

    Only detections from a trained model become findings, each created_by
    "model:<detector>" with status "open", so the report prints them as unreviewed model
    proposals until an inspector accepts them. Heuristic output (classical image
    processing, or a model that fell back to it) is left as screening data in the run
    summary: presenting a threshold hit as a finding is how an ordinary park photo came
    to be reported as critical corrosion.

    Severity is not something the detector grades, so it is recorded as "info" and the
    note says so, rather than inventing one from the confidence.
    """
    from core.annotations import add_tags, create_annotation, list_annotations

    root = Path(root)
    summary = _json(Path(summary_path).read_text(encoding="utf-8"), {})
    run_name = Path(summary_path).parent.name
    tag = f"pipeline:{run_name}"
    if any(tag in (a.tags or []) for a in list_annotations(root)):
        return {"created": 0, "skipped_heuristic": 0, "already_recorded": True}
    created, heuristic = 0, 0
    image_dir = Path(str(summary.get("input_image_dir", "")))
    new_ids: list[str] = []
    for row in summary.get("images", []) or []:
        for family in ("structural", "solar"):
            used = str(row.get(f"{family}_model_used", "") or "")
            detections = row.get(f"{family}_detections", []) or []
            if not used.startswith("onnx:") or "heuristic" in used:
                heuristic += len(detections)
                continue
            for hit in detections:
                box = hit.get("bbox_xyxy") or []
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                x0, y0, x1, y1 = (float(v) for v in box)
                annotation = create_annotation(
                    root, str(project_id), "image", str(image_dir / str(row.get("image", ""))),
                    "rectangle",
                    {"type": "Polygon",
                     "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
                    str(hit.get("label") or "defect"), "info", "open",
                    note=(f"Proposed by {used} with confidence "
                          f"{float(hit.get('confidence', 0)):.2f} in {run_name}. Severity "
                          "not graded by the detector."),
                    created_by=f"model:{used}",
                )
                new_ids.append(annotation.id)
                created += 1
    if new_ids:
        add_tags(root, new_ids, [tag])
    return {"created": created, "skipped_heuristic": heuristic, "already_recorded": False}
