"""Report engine — HTML, PDF and DOCX inspection reports built from real project data.

A report is assembled in three steps that every caller shares:

1. a *context*: what the project actually contains -- missions, datasets, findings,
   reconstruction outputs, measurements, automated detector runs and the audit trail.
   The desktop builds it from its own store (app/report_builder.py); the engine's own
   core.project workspace builds it with :func:`build_report_context`.
2. *blocks*: one per section, derived from the context and nothing else. Which
   sections appear depends on the report type, so an executive summary and an
   engineering report are different documents rather than one document with two labels.
3. *rendering*: the same blocks become HTML here and PDF/DOCX through
   core/report_formats.py (reportlab / python-docx), so the formats cannot disagree.

Nothing in a report is written by this module on the project's behalf. An empty section
says what was not done ("No reconstruction was run") rather than printing a default, and
the follow-up items are derived from the recorded findings rather than boilerplate.

Templates are stored next to the package under `core/report_templates/`; the HTML
template is a str.format() template with {title}, {report_type}, {generated_at},
{sections_html}, {report_id}, {project_name} and {section_count} placeholders.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotations import get_report_annotations
from .audit import list_audit_events
from .errors import AppError, ERR_REPORT_NOT_READY
from .events import REPORT_GENERATED, publish_event
from .measurements import list_measurements
from .project import get_manager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── Report types ──────────────────────────────────────────────────────────────

SECTION_TITLES = {
    "summary": "Executive Summary",
    "site_information": "Site Information",
    "mission_map": "Missions",
    "dataset_summary": "Datasets",
    "key_photos": "Key Photos",
    "findings": "Findings",
    "defect_summary": "Automated Detector Output",
    "crack_propagation": "Crack Propagation",
    "reconstruction": "3D Reconstruction",
    "provenance": "Provenance of Derived Files",
    "measurements": "Measurements",
    "recommendations": "Recommendations",
    "methodology": "Methodology",
    "audit_trail": "Audit Trail",
    "appendix": "Appendix",
}

# What each report type contains. The types are different documents: an executive
# summary leaves out the detector tables and the audit trail, an engineering report adds
# the provenance of every derived file, a mission summary is about the flight and not
# the findings. Before this every type rendered the same thirteen sections.
REPORT_TYPES: dict[str, list[str]] = {
    "standard": [
        "summary", "site_information", "mission_map", "dataset_summary", "key_photos",
        "findings", "defect_summary", "crack_propagation", "reconstruction",
        "measurements", "recommendations", "methodology", "audit_trail",
    ],
    "engineering": [
        "summary", "site_information", "methodology", "dataset_summary", "reconstruction",
        "provenance", "findings", "defect_summary", "crack_propagation", "measurements",
        "recommendations", "audit_trail", "appendix",
    ],
    "executive": ["summary", "key_photos", "findings", "recommendations"],
    "defect_only": ["findings", "defect_summary", "crack_propagation", "measurements"],
    "dataset_quality": [
        "dataset_summary", "key_photos", "reconstruction", "provenance", "methodology",
    ],
    "mission_summary": ["site_information", "mission_map", "dataset_summary", "audit_trail"],
}

DEFAULT_SECTIONS = list(REPORT_TYPES["standard"])


def sections_for(report_type: str, requested: list[str] | None = None) -> list[str]:
    """The section list for a report: an explicit selection wins over the type's."""
    if requested:
        return [str(s) for s in requested]
    return list(REPORT_TYPES.get(str(report_type), DEFAULT_SECTIONS))


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    project_id: str
    title: str = "Inspection Report"
    report_type: str = "standard"      # standard | engineering | executive | defect_only | dataset_quality | mission_summary
    # Empty means "the sections of this report type".
    sections: list[str] = field(default_factory=list)
    include_images: bool = True
    include_measurements: bool = True
    include_defects: bool = True
    include_audit_trail: bool = True
    include_annotations: bool = True
    author: str = ""
    notes: str = ""
    organization: str = ""
    client: str = ""
    logo_path: str = ""
    # Written by the author. Printed as written, never generated.
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    # Which files to write. HTML is always written; it is the preview.
    formats: list[str] = field(default_factory=lambda: ["html", "pdf"])
    # Where the findings (annotations) live, when that is not this engine's own
    # project folder. The desktop app keeps its project (and every finding a model
    # proposed and an inspector accepted) under its own root and id; without these the
    # report read the engine's empty folder and printed "No annotations".
    findings_root: str = ""
    findings_project_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportResult:
    id: str
    project_id: str
    title: str
    html_path: str
    pdf_path: str | None
    created_at: str = field(default_factory=_now_iso)
    docx_path: str | None = None
    report_type: str = "standard"
    output_dir: str = ""
    findings: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportReadiness:
    ok: bool
    missing: list[str]
    warnings: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Readiness ─────────────────────────────────────────────────────────────────

def _failed_reconstructions(context: dict[str, Any]) -> list[tuple[str, str]]:
    """(run name, error) for pipeline runs whose reconstruction was attempted and failed.

    "No reconstruction was run" is false for these: one was run, and it failed.
    """
    return [(str(run.get("name", "")), str(run["reconstruction_error"]))
            for run in context.get("pipeline_runs") or [] if run.get("reconstruction_error")]


def _no_reconstruction_text(context: dict[str, Any]) -> str:
    failed = _failed_reconstructions(context)
    if not failed:
        return "No reconstruction was run."
    runs = "; ".join(f"pipeline run {name}: {error}" for name, error in failed)
    return (f"Reconstruction was attempted {len(failed)} time(s) and failed ({runs}). "
            "No 3D output was produced.")


def readiness_from_context(context: dict[str, Any], config: ReportConfig) -> ReportReadiness:
    """What a report of this type still needs, judged from what the project contains.

    `missing` blocks generation: the report type has nothing to be about. `warnings` are
    sections that will state that something was not done. Both are shown to the user
    before they press Generate.
    """
    missing: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    datasets = context.get("datasets") or []
    missions = context.get("missions") or []
    findings = context.get("findings") or []
    defect_runs = context.get("defect_runs") or []
    pipeline_runs = context.get("pipeline_runs") or []
    reconstructions = context.get("reconstructions") or []
    report_type = str(config.report_type)
    sections = sections_for(report_type, config.sections)

    if report_type == "mission_summary" and not missions:
        missing.append("mission: a mission summary needs a saved mission (Mission > Save).")
    if report_type == "dataset_quality" and not datasets:
        missing.append("dataset: a dataset quality report needs an imported image folder.")
    if report_type == "defect_only" and not (findings or defect_runs or pipeline_runs):
        missing.append("findings: a defect report needs recorded findings or a detector run.")

    if not datasets and config.include_images and (
            "dataset_summary" in sections or "key_photos" in sections):
        warnings.append("No dataset imported — report will not include image gallery.")
    for dataset in datasets:
        if dataset.get("folder_missing") and config.include_images:
            warnings.append(f"Dataset {dataset.get('name', '')!r}: its image folder "
                            f"{dataset['folder_missing']} no longer exists, so the report "
                            "cannot show or read any of its images.")
    if config.include_defects and not defect_runs and not pipeline_runs and (
            "defect_summary" in sections or "findings" in sections):
        warnings.append("No defect run found — the automated detection section will say "
                        "that no detector was run.")
    if not findings and "findings" in sections:
        warnings.append("No findings recorded — the report will state that no findings "
                        "were recorded, which is not the same as a clean inspection.")
    unreviewed = [f for f in findings if f.get("source") == "model"
                  and f.get("review_state") not in ("accepted", "rejected")]
    if unreviewed:
        warnings.append(f"{len(unreviewed)} model-proposed finding(s) are not reviewed; "
                        "the report will say so.")
    if not reconstructions and "reconstruction" in sections:
        if _failed_reconstructions(context):
            notes.append("No reconstruction output — the 3D section will say the pipeline's "
                         "reconstruction was attempted and failed, and why.")
        else:
            notes.append("No reconstruction output — the 3D section will say none was run.")
    if not missions and "mission_map" in sections:
        notes.append("No mission saved — site map section will be sparse.")

    formats = [str(f).lower() for f in (config.formats or [])]
    for fmt, module in (("pdf", "reportlab"), ("docx", "docx")):
        if fmt in formats and not _importable(module):
            missing.append(f"{fmt}: this installation cannot write {fmt.upper()} "
                           f"({module} is not installed; pip install -r requirements.txt).")

    return ReportReadiness(ok=not missing, missing=missing, warnings=warnings, notes=notes)


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def validate_report_readiness(config: ReportConfig) -> ReportReadiness:
    """Readiness for a project in the engine's own core.project workspace."""
    mgr = get_manager()
    try:
        meta = mgr.load_project(config.project_id)
    except Exception:
        # A bare "project" code reached the checklist as the whole explanation.
        return ReportReadiness(False, [f"project: no project {config.project_id!r} exists "
                                       "in this workspace."], [], [])
    root = Path(str(meta.get("root_dir", "")))
    if not root.exists():
        return ReportReadiness(False, [f"project_folder: the project folder {root} does not "
                                       "exist; it was moved or deleted."], [], [])
    return readiness_from_context(build_report_context(config), config)


# ── Context ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


REVIEW_STATES = {
    "resolved": "accepted", "dismissed": "rejected", "in_review": "flagged",
    "open": "unreviewed",
}


def _pixel_bbox(geometry: dict[str, Any]) -> list[float] | None:
    """Bounding box of an annotation's coordinates, in its own (pixel) space."""
    points: list[list[float]] = []

    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
                points.append([float(value[0]), float(value[1])])
            else:
                for item in value:
                    walk(item)

    walk((geometry or {}).get("coordinates"))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    radius = float((geometry or {}).get("radius_px", 0) or 0)
    return [min(xs) - radius, min(ys) - radius, max(xs) + radius, max(ys) + radius]


def resolve_image(source_id: str, image_dirs: list[Path]) -> Path | None:
    """Find the file an image annotation refers to: a path, or a name in a dataset."""
    if not source_id:
        return None
    candidate = Path(str(source_id))
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for folder in image_dirs:
        found = folder / candidate.name
        if found.is_file():
            return found
    return None


def findings_from_annotations(annotations: list[Any], image_dirs: list[Path]) -> list[dict[str, Any]]:
    """Report findings from the project's annotations, with their real review state."""
    from . import geo

    out: list[dict[str, Any]] = []
    for annotation in annotations:
        a = annotation.to_dict() if hasattr(annotation, "to_dict") else dict(annotation)
        # A rejected finding is carried with review_state "rejected" so the report can
        # say how many were rejected; every renderer then leaves it out of the findings.
        # Anything else taken out of the report stays out entirely.
        if not a.get("include_in_report", True) and str(a.get("status")) != "dismissed":
            continue
        created_by = str(a.get("created_by") or "user")
        finding: dict[str, Any] = {
            "id": a.get("id"),
            "category": a.get("label"),
            "severity": a.get("severity"),
            "status": a.get("status"),
            "review_state": REVIEW_STATES.get(str(a.get("status")), "unreviewed"),
            # A finding a person drew and one a model proposed are different kinds of
            # evidence, and the report says which.
            "source": "human" if created_by == "user" else "model",
            "created_by": created_by,
            "note": a.get("note") or "",
            "tags": list(a.get("tags") or []),
            "annotation_type": a.get("annotation_type"),
            "source_type": a.get("source_type"),
            "created_at": a.get("created_at"),
        }
        geometry = a.get("geometry") or {}
        if a.get("source_type") == "image":
            finding["image"] = Path(str(a.get("source_id") or "")).name
            source = resolve_image(str(a.get("source_id") or ""), image_dirs)
            if source is not None:
                finding["image_source_path"] = str(source)
                finding["pixel_bbox"] = _pixel_bbox(geometry)
                fix = geo.read_exif_gps(source)
                if fix is not None:
                    # Where the camera was, not where the defect is. Labelled as such.
                    finding["location_lonlat"] = [fix.longitude, fix.latitude]
                    finding["location_kind"] = "photo position"
        elif a.get("source_type") == "map" and geometry.get("type") == "Point":
            coords = geometry.get("coordinates") or []
            if len(coords) >= 2:
                finding["location_lonlat"] = [float(coords[0]), float(coords[1])]
                finding["location_kind"] = "marked on map"
        for key in ("area_m2", "length_m"):
            if a.get(key):
                finding[key] = a[key]
        # Which model proposed it, which file, and how sure it was: a model finding
        # carries its provenance into every deliverable.
        for key in ("model_key", "model_sha256", "confidence"):
            if a.get(key) not in (None, ""):
                finding[key] = a[key]
        out.append(finding)
    return out


def reconstruction_entry(summary_path: Path, name: str) -> dict[str, Any] | None:
    """A reconstruction as the report shows it, from its digital_twin.json manifest."""
    data = _load_json(summary_path)
    if not isinstance(data, dict):
        return None
    folder = summary_path.parent
    qm = data.get("quality_metrics", {}) if isinstance(data.get("quality_metrics"), dict) else {}
    entry = {
        "name": name,
        "engine": data.get("engine") or qm.get("execution_mode_used") or "",
        "profile": data.get("profile") or qm.get("processing_profile") or "",
        "generated_utc": data.get("generated_utc", ""),
        "crs_epsg": data.get("crs_epsg"),
        "georeferenced": data.get("georeferenced"),
        "frame_count": data.get("frame_count", qm.get("frame_count")),
        "registered_images": data.get("registered_images"),
        "point_count": data.get("point_count", qm.get("total_points")),
        "dense": data.get("dense"),
        "mean_reprojection_error_px": data.get("mean_reprojection_error_px"),
        "geo_rmse_m": data.get("geo_rmse_m"),
        "ground_sample_distance_m": data.get("ground_sample_distance_m"),
        "folder": str(folder),
    }
    ortho = folder / "orthomosaic.png"
    if ortho.is_file():
        entry["orthomosaic_png"] = str(ortho)
    try:
        from .provenance import audit as provenance_audit

        audit = provenance_audit(folder)
        entry["provenance"] = {
            "total": audit["total"], "verified": len(audit["verified"]),
            "unattributed": len(audit["unattributed"]), "modified": len(audit["modified"]),
            "results": {k: v.get("status") for k, v in audit["results"].items()},
        }
    except Exception:  # noqa: BLE001 - provenance is reported when it can be read
        pass
    return entry


def find_reconstructions(root: Path) -> list[dict[str, Any]]:
    """Every reconstruction output under a project folder the desktop or engine wrote."""
    found: list[dict[str, Any]] = []
    candidates: list[tuple[Path, str]] = []
    if (root / "reconstruction" / "digital_twin.json").is_file():
        candidates.append((root / "reconstruction" / "digital_twin.json", "reconstruction"))
    analysis = root / "analysis"
    if analysis.is_dir():
        for run in sorted(analysis.glob("run_*")):
            twin = run / "reconstruction" / "digital_twin.json"
            if twin.is_file():
                candidates.append((twin, f"{run.name}/reconstruction"))
        recon_dir = analysis / "reconstruction"
        if recon_dir.is_dir():
            for sub in sorted(p for p in recon_dir.iterdir() if p.is_dir()):
                for name in ("digital_twin.json", "reconstruction_summary.json"):
                    if (sub / name).is_file():
                        candidates.append((sub / name, sub.name))
                        break
    for path, name in candidates:
        entry = reconstruction_entry(path, name)
        if entry:
            found.append(entry)
    return found


def find_pipeline_runs(root: Path) -> list[dict[str, Any]]:
    """Runs of the full inspection pipeline (core/pipeline.py) under <root>/analysis."""
    runs: list[dict[str, Any]] = []
    analysis = root / "analysis"
    if not analysis.is_dir():
        return runs
    for run in sorted(analysis.glob("run_*")):
        summary = _load_json(run / "summary.json")
        if not isinstance(summary, dict):
            continue
        images = summary.get("images") or []
        models = summary.get("defect_models") or {}
        structural = (models.get("structural_model") or {})
        model_used = sorted({str(r.get("structural_model_used", "")) for r in images} - {""})
        runs.append({
            "name": run.name,
            "run_utc": summary.get("run_utc", ""),
            "image_count": len(images),
            "model_used": model_used,
            "model_available": bool(structural.get("exists") or structural.get("available")),
            "images": [
                {
                    "image": r.get("image"),
                    "crack_ratio": r.get("crack_ratio"),
                    "metal_ratio": r.get("metal_ratio"),
                    "structural_detection_count": r.get("structural_detection_count"),
                    "structural_model_used": r.get("structural_model_used"),
                    "crack_model_used": r.get("crack_model_used"),
                    "overlay": r.get("crack_overlay_path"),
                    "quality_ok": r.get("quality_ok"),
                    "quality_flags": r.get("quality_flags") or [],
                }
                for r in images
            ],
            "reconstruction_error": (summary.get("reconstruction") or {}).get("error", ""),
            # Only the screening index: the run's grade and failure probabilities are
            # derived from heuristic ratios and are not a structural assessment.
            "screening_severity_index": (summary.get("health_scoring") or {}).get(
                "severity_index"),
            "report_markdown": str(run / "inspection_report.md")
            if (run / "inspection_report.md").is_file() else "",
            "coverage": summary.get("coverage_validation") or {},
        })
    return runs


def _engine_defect_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    defects_dir = root / "analysis" / "defects"
    if defects_dir.exists():
        for sub in sorted(p for p in defects_dir.iterdir() if p.is_dir()):
            data = _load_json(sub / "defects.json")
            if isinstance(data, dict):
                runs.append(data)
    return runs


def _engine_crack_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    crack_dir = root / "analysis" / "crack_growth"
    if crack_dir.exists():
        for sub in sorted(p for p in crack_dir.iterdir() if p.is_dir()):
            data = _load_json(sub / "crack_propagation.json")
            if isinstance(data, dict):
                runs.append(data)
    return runs


def select_photos(image_dirs: list[Path], findings: list[dict[str, Any]],
                  limit: int = 6) -> list[dict[str, Any]]:
    """Key photos: images with findings first, then evenly spaced dataset frames."""
    photos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        path = finding.get("image_source_path")
        if path and path not in seen and len(photos) < limit:
            seen.add(path)
            photos.append({"path": path, "caption": f"{Path(path).name} (has findings)"})
    for folder in image_dirs:
        if len(photos) >= limit:
            break
        try:
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        except OSError:
            continue
        if not images:
            continue
        want = max(0, limit - len(photos))
        step = max(1, len(images) // max(want, 1))
        for image in images[::step][:want]:
            if str(image) not in seen:
                seen.add(str(image))
                photos.append({"path": str(image), "caption": image.name})
    return photos


def build_report_context(config: ReportConfig) -> dict[str, Any]:
    """Context for a project in the engine's own core.project workspace."""
    mgr = get_manager()
    meta = mgr.load_project(config.project_id)
    root = Path(str(meta.get("root_dir", "")))

    missions: list[dict[str, Any]] = []
    missions_dir = root / "missions"
    if missions_dir.exists():
        for f in sorted(missions_dir.glob("*.json")):
            data = _load_json(f)
            if isinstance(data, dict):
                missions.append({
                    "name": data.get("name", f.stem),
                    "mode": data.get("mode", ""),
                    "waypoint_count": data.get("waypoint_count", len(data.get("waypoints", []) or [])),
                    "altitude_m": data.get("altitude_m"),
                })

    datasets: list[dict[str, Any]] = []
    image_dirs: list[Path] = []
    datasets_dir = root / "datasets"
    if datasets_dir.exists():
        for sub in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
            d = _load_json(sub / "metadata.json")
            if isinstance(d, dict):
                image_count = int(d.get("image_count", 0) or 0)
                datasets.append({
                    "name": d.get("name", sub.name),
                    "image_count": image_count,
                    "geotagged_count": d.get("geotagged_count",
                                             image_count if d.get("has_gps_metadata") else None),
                    "camera": d.get("camera", ""),
                    "imported": d.get("created_at", ""),
                })
                for candidate in (sub / "images", sub):
                    if candidate.is_dir():
                        image_dirs.append(candidate)

    # Findings may live outside this engine's project folder: the desktop app keeps its
    # project, and every finding an inspector accepted, under its own root and id.
    findings_root = Path(config.findings_root) if config.findings_root else root
    findings_pid = str(config.findings_project_id or config.project_id)
    annotations = (get_report_annotations(findings_root, findings_pid)
                   if config.include_annotations else [])
    findings = findings_from_annotations(annotations, image_dirs)
    measurements = ([m.to_dict() for m in list_measurements(root, config.project_id)]
                    if config.include_measurements else [])
    audit = []
    if config.include_audit_trail:
        for e in list_audit_events(config.project_id, limit=200):
            audit.append({"timestamp": e.get("timestamp", ""),
                          "event_type": e.get("event_type", ""),
                          "detail": e.get("detail", "")})

    return {
        "config": config.to_dict(),
        "project": {
            "id": meta.get("id", ""),
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "created_at": meta.get("created_at", ""),
        },
        "project_root": str(root),
        "site": {},
        "missions": missions,
        "datasets": datasets,
        "image_dirs": [str(p) for p in image_dirs],
        "findings": findings,
        "defect_runs": _engine_defect_runs(root) if config.include_defects else [],
        "pipeline_runs": find_pipeline_runs(root) if config.include_defects else [],
        "crack_runs": _engine_crack_runs(root),
        "reconstructions": find_reconstructions(root),
        "measurements": measurements,
        "audit": audit,
        "photos": select_photos(image_dirs, findings) if config.include_images else [],
        "generated_at": _now_iso(),
    }


# ── Assets ────────────────────────────────────────────────────────────────────

def _thumbnail(source: str | Path, target: Path, max_px: int = 1200) -> Path | None:
    try:
        from PIL import Image

        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((max_px, max_px))
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, "JPEG", quality=85)
        return target
    except Exception:  # noqa: BLE001 - an unreadable photo is left out, not faked
        return None


def _finding_crop(finding: dict[str, Any], target: Path) -> Path | None:
    """The part of the source photo the finding marks, outlined."""
    source = finding.get("image_source_path")
    if not source:
        return None
    try:
        from PIL import Image, ImageDraw

        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            bbox = finding.get("pixel_bbox")
            if bbox and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.9 * width * height:
                bbox = None  # the finding is about the whole photo: show it, unmarked
            if bbox:
                x0, y0, x1, y1 = bbox
                pad_x = max(0.05 * width, 0.35 * (x1 - x0))
                pad_y = max(0.05 * height, 0.35 * (y1 - y0))
                crop = [max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
                        min(width, int(x1 + pad_x)), min(height, int(y1 + pad_y))]
                if crop[2] - crop[0] > 4 and crop[3] - crop[1] > 4:
                    draw = ImageDraw.Draw(image)
                    line = max(2, int(0.004 * max(width, height)))
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=line)
                    image = image.crop(crop)
            image.thumbnail((900, 900))
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, "JPEG", quality=85)
        return target
    except Exception:  # noqa: BLE001
        return None


def _mission_map(missions: list[dict[str, Any]], target: Path) -> Path | None:
    """Draw the saved flight lines from their real coordinates. No basemap is invented."""
    lines: list[list[tuple[float, float]]] = []

    def collect(geometry: dict[str, Any]) -> None:
        kind = geometry.get("type")
        coords = geometry.get("coordinates") or []
        if kind == "LineString":
            lines.append([(float(c[0]), float(c[1])) for c in coords if len(c) >= 2])
        elif kind == "MultiLineString":
            for part in coords:
                lines.append([(float(c[0]), float(c[1])) for c in part if len(c) >= 2])
        elif kind == "Polygon" and coords:
            lines.append([(float(c[0]), float(c[1])) for c in coords[0] if len(c) >= 2])

    for mission in missions:
        geojson = mission.get("geojson") or {}
        for feature in geojson.get("features", []) or []:
            collect((feature or {}).get("geometry") or {})
    lines = [line for line in lines if len(line) >= 2]
    if not lines:
        return None
    try:
        import math

        from PIL import Image, ImageDraw

        xs = [p[0] for line in lines for p in line]
        ys = [p[1] for line in lines for p in line]
        lat0 = math.radians(sum(ys) / len(ys))
        span_x = max((max(xs) - min(xs)) * math.cos(lat0), 1e-9)
        span_y = max(max(ys) - min(ys), 1e-9)
        size, margin = 900, 40
        scale = (size - 2 * margin) / max(span_x, span_y)
        w = int(span_x * scale) + 2 * margin
        h = int(span_y * scale) + 2 * margin
        image = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        def px(p: tuple[float, float]) -> tuple[float, float]:
            return (margin + (p[0] - min(xs)) * math.cos(lat0) * scale,
                    h - margin - (p[1] - min(ys)) * scale)

        for line in lines:
            draw.line([px(p) for p in line], fill=(30, 90, 200), width=2)
        draw.text((margin, 10), "Saved flight lines (WGS84, north up, no basemap)",
                  fill=(80, 80, 80))
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "PNG")
        return target
    except Exception:  # noqa: BLE001
        return None


def prepare_assets(context: dict[str, Any], assets_dir: Path) -> dict[str, Any]:
    """Write every image the report embeds into the report's own folder."""
    assets: dict[str, Any] = {"photos": [], "findings": {}, "orthomosaics": {}, "mission_map": ""}
    for index, photo in enumerate(context.get("photos") or []):
        written = _thumbnail(photo["path"], assets_dir / f"photo_{index + 1:02d}.jpg")
        if written:
            assets["photos"].append({"path": str(written), "caption": photo.get("caption", "")})
    for index, finding in enumerate(context.get("findings") or []):
        written = _finding_crop(finding, assets_dir / f"finding_{index + 1:03d}.jpg")
        if written:
            assets["findings"][str(finding.get("id"))] = str(written)
    for index, recon in enumerate(context.get("reconstructions") or []):
        if recon.get("orthomosaic_png"):
            written = _thumbnail(recon["orthomosaic_png"], assets_dir / f"orthomosaic_{index + 1}.jpg",
                                 max_px=1600)
            if written:
                assets["orthomosaics"][recon["name"]] = str(written)
    mission_map = _mission_map(context.get("missions") or [], assets_dir / "mission_map.png")
    if mission_map:
        assets["mission_map"] = str(mission_map)
    return assets


# ── Blocks ────────────────────────────────────────────────────────────────────

SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None or value == "":
        return "not reported"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"
    return f"{value}{(' ' + unit) if unit else ''}"


def _reportable_findings(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Findings that belong in the deliverable, numbered worst first.

    A finding the reviewer rejected is a false positive, not a finding.
    """
    findings = [dict(f) for f in context.get("findings") or [] if f.get("review_state") != "rejected"]
    findings.sort(key=lambda f: SEVERITY_RANK.get(str(f.get("severity", "")).lower(), 9))
    for index, finding in enumerate(findings, start=1):
        finding["number"] = index
    return findings


def derive_summary(context: dict[str, Any]) -> str:
    """A factual summary of what the project contains. Counts, not conclusions."""
    datasets = context.get("datasets") or []
    images = sum(int(d.get("image_count") or 0) for d in datasets)
    geotagged = sum(int(d.get("geotagged_count") or 0) for d in datasets)
    findings = _reportable_findings(context)
    rejected = sum(1 for f in context.get("findings") or [] if f.get("review_state") == "rejected")
    by_severity: dict[str, int] = {}
    for f in findings:
        by_severity[str(f.get("severity"))] = by_severity.get(str(f.get("severity")), 0) + 1
    parts = [f"Project {context.get('project', {}).get('name', '')}."]
    if datasets:
        parts.append(f"{len(datasets)} dataset(s) with {images} image(s), {geotagged} geotagged.")
    else:
        parts.append("No imagery has been imported.")
    missions = context.get("missions") or []
    parts.append(f"{len(missions)} saved mission(s)." if missions else "No mission was saved.")
    recons = context.get("reconstructions") or []
    parts.append(f"{len(recons)} reconstruction(s)." if recons else _no_reconstruction_text(context))
    if findings:
        severity_text = ", ".join(
            f"{count} {sev}" for sev, count in sorted(
                by_severity.items(), key=lambda kv: SEVERITY_RANK.get(kv[0], 9)))
        accepted = sum(1 for f in findings if f.get("review_state") == "accepted")
        parts.append(f"{len(findings)} finding(s) recorded ({severity_text}); "
                     f"{accepted} confirmed by a reviewer.")
    else:
        parts.append("No findings were recorded. This is not evidence that the asset is "
                     "free of defects; it means none were recorded in this project.")
    if rejected:
        parts.append(f"{rejected} proposed finding(s) were rejected on review and are excluded.")
    return " ".join(parts)


def derive_follow_up(context: dict[str, Any]) -> list[str]:
    """Follow-up items that follow from the recorded data, and nothing else."""
    items: list[str] = []
    findings = _reportable_findings(context)
    for f in findings:
        if str(f.get("severity")) in ("critical", "high"):
            state = ("confirmed by a reviewer" if f.get("review_state") == "accepted"
                     else "not yet confirmed by a reviewer")
            where = f" on {f['image']}" if f.get("image") else ""
            items.append(f"Finding {f['number']} ({f.get('category')}, {f.get('severity')}){where} "
                         f"is {state}; it needs a follow-up decision.")
    unreviewed = [f for f in findings if f.get("source") == "model"
                  and f.get("review_state") != "accepted"]
    if unreviewed:
        items.append(f"Review the {len(unreviewed)} model-proposed finding(s) that no "
                     "inspector has confirmed.")
    for d in context.get("datasets") or []:
        total = int(d.get("image_count") or 0)
        tagged = d.get("geotagged_count")
        if tagged is not None and total and int(tagged) < total:
            items.append(f"Dataset {d.get('name')}: {total - int(tagged)} of {total} images "
                         "have no GPS tag, so findings on them cannot be placed on the map.")
    for run in context.get("pipeline_runs") or []:
        bad = [i["image"] for i in run.get("images", []) if i.get("quality_ok") is False]
        if bad:
            items.append(f"Pipeline run {run['name']}: {len(bad)} image(s) failed the quality "
                         "check and may need re-flying.")
        if run.get("reconstruction_error"):
            items.append(f"Pipeline run {run['name']}: reconstruction failed "
                         f"({run['reconstruction_error']}).")
    for recon in context.get("reconstructions") or []:
        prov = recon.get("provenance") or {}
        if prov.get("modified"):
            items.append(f"Reconstruction {recon['name']}: {prov['modified']} derived file(s) "
                         "changed after their provenance was recorded.")
    return items


def derive_methodology(context: dict[str, Any]) -> str:
    lines: list[str] = []
    cameras = sorted({str(d.get("camera") or "").strip() for d in context.get("datasets") or []} - {""})
    if cameras:
        lines.append(f"Imagery: captured with {', '.join(cameras)} (from EXIF).")
    for recon in context.get("reconstructions") or []:
        lines.append(
            f"Reconstruction {recon['name']}: engine {recon.get('engine') or 'not recorded'}, "
            f"profile {recon.get('profile') or 'not recorded'}, "
            f"CRS {('EPSG:' + str(recon['crs_epsg'])) if recon.get('crs_epsg') else 'not georeferenced'}.")
    findings = context.get("findings") or []
    human = sum(1 for f in findings if f.get("source") == "human")
    models = sorted({str(f.get("created_by")) for f in findings if f.get("source") == "model"})
    if human:
        lines.append(f"Findings: {human} recorded by an inspector.")
    if models:
        lines.append(f"Findings proposed by automated models: {', '.join(models)}. Each carries "
                     "its review state.")
    for run in context.get("pipeline_runs") or []:
        used = ", ".join(run.get("model_used") or []) or "not recorded"
        lines.append(f"Inspection pipeline {run['name']}: {run['image_count']} image(s); "
                     f"structural detector {used}.")
    if not lines:
        lines.append("No processing has been recorded for this project.")
    return "\n".join(lines)


def build_blocks(context: dict[str, Any], assets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One block per section: title, paragraphs, an optional table and figures."""
    blocks: dict[str, dict[str, Any]] = {}
    project = context.get("project", {}) or {}
    cfg = context.get("config", {}) or {}

    summary = []
    if cfg.get("summary"):
        summary.append(str(cfg["summary"]))
    summary.append(derive_summary(context))
    blocks["summary"] = {"paragraphs": summary}

    site = context.get("site", {}) or {}
    rows = [["Project", project.get("name", "")]]
    if project.get("description"):
        rows.append(["Description", project["description"]])
    if site.get("center_lonlat"):
        lon, lat = site["center_lonlat"]
        rows.append(["Image centre (lat, lon)", f"{lat:.6f}, {lon:.6f}"])
    if site.get("bounds_lonlat"):
        b = site["bounds_lonlat"]
        rows.append(["Image extent (lon/lat)", f"{b[0]:.6f}, {b[1]:.6f} to {b[2]:.6f}, {b[3]:.6f}"])
    if site.get("crs_epsg"):
        rows.append(["Coordinate system", f"EPSG:{site['crs_epsg']}"])
    if project.get("created_at"):
        rows.append(["Project created", project["created_at"]])
    blocks["site_information"] = {"table": {"header": [], "rows": rows}}

    missions = context.get("missions") or []
    if missions:
        blocks["mission_map"] = {
            "table": {"header": ["Name", "Template", "Version", "Waypoints", "Altitude (m)",
                                 "Distance (m)", "Est. time (min)"],
                      "rows": [[m.get("name", ""), m.get("mode", ""), m.get("version", ""),
                                _fmt(m.get("waypoint_count")), _fmt(m.get("altitude_m"), 1),
                                _fmt(m.get("path_distance_m"), 0),
                                _fmt(m.get("estimated_time_min"), 1)] for m in missions]},
            "images": ([{"path": assets["mission_map"],
                         "caption": "Saved flight lines drawn from the mission's coordinates."}]
                       if assets.get("mission_map") else []),
        }
    else:
        blocks["mission_map"] = {"paragraphs": ["No mission was saved in this project."]}

    datasets = context.get("datasets") or []
    if datasets:
        blocks["dataset_summary"] = {"table": {
            "header": ["Name", "Images", "Geotagged", "Camera", "Imported"],
            "rows": [[d.get("name", ""), _fmt(d.get("image_count")),
                      _fmt(d.get("geotagged_count")), d.get("camera") or "not recorded",
                      d.get("imported", "")] for d in datasets]}}
    else:
        blocks["dataset_summary"] = {"paragraphs": ["No imagery has been imported."]}

    if assets.get("photos"):
        blocks["key_photos"] = {"images": list(assets["photos"])}
    else:
        blocks["key_photos"] = {"paragraphs": ["No photos are available to show."]}

    findings = _reportable_findings(context)
    if findings:
        blocks["findings"] = {"table": {
            "header": ["#", "Category", "Severity", "Where", "Source", "Review", "Note"],
            "rows": [[f["number"], f.get("category", ""), f.get("severity", ""),
                      _where(f), f.get("created_by") if f.get("source") == "model" else "inspector",
                      f.get("review_state", ""), (f.get("note") or "").split("\n")[0]]
                     for f in findings]},
            "images": [{"path": assets["findings"][str(f.get("id"))],
                        "caption": f"Finding {f['number']}: {f.get('category')} "
                                   f"({f.get('severity')}) on {f.get('image', '')}"}
                       for f in findings if str(f.get("id")) in assets.get("findings", {})]}
    else:
        blocks["findings"] = {"paragraphs": [
            "No findings were recorded. This is not the same as an inspection that found "
            "nothing: no finding has been drawn or accepted in this project."]}

    rows = []
    heuristic = False
    for run in context.get("defect_runs") or []:
        for d in run.get("defects", []) or []:
            method = str(d.get("method") or run.get("method") or "")
            is_heuristic = (d.get("source") == "classical" or method == "heuristic"
                            or "heuristic" in str(d.get("model_used") or ""))
            heuristic = heuristic or is_heuristic
            score = d.get("confidence")
            detector = d.get("model_used") or run.get("model_used") or d.get("source") or "not recorded"
            rows.append([Path(str(d.get("image_path", ""))).name, d.get("defect_type", ""),
                         "heuristic" if is_heuristic else detector,
                         ("n/a (heuristic)" if is_heuristic or score is None
                          else f"{float(score):.2f}"),
                         d.get("severity", ""), _fmt(d.get("area_px"))])
    for run in context.get("pipeline_runs") or []:
        for image in run.get("images", []):
            used = image.get("structural_model_used") or "heuristic"
            # The row's crack and metal ratios are screening output unless a trained
            # crack model produced them, whatever ran for the structural hits.
            crack_used = str(image.get("crack_model_used") or "heuristic")
            heuristic = (heuristic or used in ("", "heuristic", "none")
                         or not crack_used.startswith("onnx:") or image.get("metal_ratio") is not None)
            rows.append([image.get("image", ""), "crack / metal / structural screening",
                         f"{run['name']}: {used}",
                         "n/a", f"crack ratio {_fmt(image.get('crack_ratio'), 4)}",
                         _fmt(image.get("structural_detection_count"))])
    if rows:
        paragraphs = []
        if heuristic:
            paragraphs.append(
                "Rows marked heuristic come from classical image processing, not a trained "
                "model. Their scores are screening indicators, not probabilities, and they "
                "are not findings unless an inspector records them as such.")
        blocks["defect_summary"] = {"paragraphs": paragraphs, "table": {
            "header": ["Image", "Type", "Detector", "Confidence", "Severity / ratio", "Area / count"],
            "rows": rows[:300]}}
    else:
        blocks["defect_summary"] = {"paragraphs": [
            "No automated detector was run on this project."]}

    crack_runs = context.get("crack_runs") or []
    if crack_runs:
        blocks["crack_propagation"] = {"paragraphs": [
            f"Run {r.get('id', '')}: {r.get('summary', '')} Illustrative risk level (from the "
            f"assumed stresses and toughness, not measured): {r.get('risk_level', '')}. "
            f"Assumptions: {'; '.join(str(a) for a in r.get('assumptions', []))}"
            for r in crack_runs]}
    else:
        blocks["crack_propagation"] = {"paragraphs": ["No crack propagation analysis was run."]}

    recons = context.get("reconstructions") or []
    if recons:
        blocks["reconstruction"] = {
            "table": {"header": ["Output", "Engine", "Registered / frames", "Points",
                                 "Reprojection (px)", "Geo RMSE (m)", "GSD (m)", "CRS"],
                      "rows": [[r["name"], f"{r.get('engine', '')} {r.get('profile', '')}".strip(),
                                f"{_fmt(r.get('registered_images'))} / {_fmt(r.get('frame_count'))}",
                                _fmt(r.get("point_count")), _fmt(r.get("mean_reprojection_error_px")),
                                _fmt(r.get("geo_rmse_m")), _fmt(r.get("ground_sample_distance_m"), 3),
                                (f"EPSG:{r['crs_epsg']}" if r.get("crs_epsg") else "none")]
                               for r in recons]},
            "images": [{"path": path, "caption": f"Orthomosaic from {name}."}
                       for name, path in (assets.get("orthomosaics") or {}).items()],
        }
    else:
        blocks["reconstruction"] = {"paragraphs": [_no_reconstruction_text(context)]}

    prov_rows = []
    for recon in recons:
        prov = recon.get("provenance")
        if prov:
            prov_rows.append([recon["name"], prov["total"], prov["verified"],
                              prov["unattributed"], prov["modified"]])
    if prov_rows:
        blocks["provenance"] = {
            "paragraphs": ["Each derived file is checked against the sha256 recorded when it "
                           "was produced. Unattributed files have no record; modified files "
                           "changed after their record was written."],
            "table": {"header": ["Output", "Files", "Verified", "Unattributed", "Modified"],
                      "rows": prov_rows}}
    else:
        blocks["provenance"] = {"paragraphs": ["No derived files to verify."]}

    measurements = context.get("measurements") or []
    if measurements:
        blocks["measurements"] = {"table": {
            "header": ["Label", "Type", "Value", "Unit", "Source"],
            "rows": [[m.get("label", ""), m.get("measurement_type", ""),
                      _fmt(m.get("value"), 4), m.get("unit", ""),
                      Path(str(m.get("source_id", ""))).name] for m in measurements]}}
    else:
        blocks["measurements"] = {"paragraphs": ["No measurements were recorded."]}

    recommendations = [str(r) for r in cfg.get("recommendations") or [] if str(r).strip()]
    follow_up = derive_follow_up(context)
    paragraphs = []
    if recommendations:
        paragraphs.append("Inspector's recommendations:")
        paragraphs.extend(f"- {r}" for r in recommendations)
    else:
        paragraphs.append("The author entered no recommendations.")
    if follow_up:
        paragraphs.append("Follow-up items arising from the recorded data:")
        paragraphs.extend(f"- {r}" for r in follow_up)
    else:
        paragraphs.append("No follow-up items arise from the recorded data.")
    blocks["recommendations"] = {"paragraphs": paragraphs,
                                 "_author": recommendations, "_derived": follow_up}

    blocks["methodology"] = {"paragraphs": derive_methodology(context).split("\n")}

    audit = context.get("audit") or []
    if audit:
        blocks["audit_trail"] = {"table": {
            "header": ["Time", "Event", "Detail"],
            "rows": [[e.get("timestamp", ""), e.get("event_type", ""), e.get("detail", "")]
                     for e in audit[:100]]}}
    else:
        blocks["audit_trail"] = {"paragraphs": ["No audit events were recorded."]}

    inputs = [["Report type", cfg.get("report_type", "standard")],
              ["Sections", ", ".join(context.get("sections") or [])],
              ["Generated", context.get("generated_at", "")],
              ["Report id", context.get("report_id", "")]]
    for d in datasets:
        inputs.append([f"Dataset {d.get('name')}", f"{_fmt(d.get('image_count'))} images"])
    for r in recons:
        inputs.append([f"Reconstruction {r['name']}", r.get("generated_utc") or "time not recorded"])
    try:
        from .provenance import _software_versions

        for name, version in sorted(_software_versions().items()):
            inputs.append([f"Software: {name}", version])
    except Exception:  # noqa: BLE001
        pass
    blocks["appendix"] = {"table": {"header": [], "rows": inputs}}

    for key, block in blocks.items():
        block.setdefault("title", SECTION_TITLES.get(key, key.replace("_", " ").title()))
    return blocks


def _where(finding: dict[str, Any]) -> str:
    parts = []
    if finding.get("image"):
        parts.append(str(finding["image"]))
    location = finding.get("location_lonlat")
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        kind = finding.get("location_kind") or "position"
        parts.append(f"{kind} {float(location[1]):.6f}, {float(location[0]):.6f}")
    return "; ".join(parts) or "-"


# ── HTML ──────────────────────────────────────────────────────────────────────

_BUILTIN_TEMPLATE_DIR = Path(__file__).resolve().parent / "report_templates"


_DEFAULT_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
:root {{
    --text: #1d2430;
    --muted: #5d6878;
    --border: #d5dbe3;
    --surface: #f6f8fa;
    --accent: #1f5fae;
    --warn-bg: #fff4e5;
    --warn: #8a4b00;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0; padding: 0; background: #ffffff; color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 24px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
h2 {{ font-size: 19px; margin: 28px 0 10px; border-bottom: 2px solid var(--accent); padding-bottom: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th, td {{ text-align: left; padding: 6px 8px; border: 1px solid var(--border); font-size: 12.5px; vertical-align: top; }}
th {{ background: var(--surface); }}
.muted {{ color: var(--muted); }}
.caveat {{ background: var(--warn-bg); color: var(--warn); padding: 8px 12px; border-radius: 4px; margin: 8px 0; }}
figure {{ margin: 12px 0; break-inside: avoid; }}
figure img {{ max-width: 100%; max-height: 520px; border: 1px solid var(--border); }}
figcaption {{ color: var(--muted); font-size: 12px; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
.logo {{ max-height: 60px; margin-bottom: 8px; }}
.footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; }}
@media print {{ .wrap {{ padding: 0; }} h2 {{ break-after: avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
    <header>
        <div class="muted">OpenDroneKit inspection report · {report_type}</div>
        <h1>{title}</h1>
        <div class="muted">Project {project_name} · generated {generated_at}</div>
    </header>

    {sections_html}

    <div class="footer">
        Report ID: {report_id} · Sections: {section_count}
    </div>
</div>
</body>
</html>
"""


def _builtin_template_path(name: str) -> Path:
    return _BUILTIN_TEMPLATE_DIR / name


def _ensure_builtin_template() -> Path:
    """The default template on disk, so reports can be customised."""
    _BUILTIN_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _builtin_template_path("standard.html")
    if not path.exists():
        path.write_text(_DEFAULT_HTML_TEMPLATE, encoding="utf-8")
    return path


def _rel(path: str, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).as_uri()


def _block_html(block: dict[str, Any], base: Path) -> str:
    out = [f"<section><h2>{html.escape(str(block.get('title', '')))}</h2>"]
    for line in block.get("paragraphs", []) or []:
        out.append(f"<p>{html.escape(str(line))}</p>")
    table = block.get("table")
    if table and table.get("rows"):
        out.append("<table>")
        if table.get("header"):
            out.append("<thead><tr>" + "".join(
                f"<th>{html.escape(str(h))}</th>" for h in table["header"]) + "</tr></thead>")
        out.append("<tbody>")
        for row in table["rows"]:
            cells = list(row)
            if not table.get("header") and cells:
                out.append(f"<tr><th>{html.escape(str(cells[0]))}</th>" + "".join(
                    f"<td>{html.escape(str(c))}</td>" for c in cells[1:]) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
        out.append("</tbody></table>")
    images = block.get("images") or []
    if images:
        out.append("<div class='gallery'>")
        for image in images:
            out.append(f"<figure><img src='{html.escape(_rel(image['path'], base))}' alt='' />"
                       f"<figcaption>{html.escape(str(image.get('caption', '')))}</figcaption></figure>")
        out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def caveats(context: dict[str, Any]) -> list[str]:
    """Caveats about the findings, printed only when the report lists findings."""
    from .report_formats import _measurement_caveat, _rejected_caveat, _unreviewed_caveat

    if "findings" not in (context.get("sections") or []):
        return []
    findings = _reportable_findings(context)
    return [c for c in (_unreviewed_caveat(findings), _measurement_caveat(findings),
                        _rejected_caveat(context.get("findings") or [])) if c]


def render_report_html(context: dict[str, Any], template_path: Path | None, output_path: Path | str,
                       assets: dict[str, Any] | None = None,
                       blocks: dict[str, dict[str, Any]] | None = None) -> Path:
    """Render the report HTML from the context's blocks."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template_path = template_path or _ensure_builtin_template()
    cfg = context.get("config", {}) or {}
    sections = context.get("sections") or sections_for(cfg.get("report_type", "standard"),
                                                        cfg.get("sections"))
    blocks = blocks if blocks is not None else build_blocks(context, assets or {})

    parts: list[str] = []
    logo = str(cfg.get("logo_path") or "")
    if logo and Path(logo).is_file():
        parts.append(f"<img class='logo' src='{html.escape(Path(logo).resolve().as_uri())}' alt='' />")
    meta_rows = [("Organization", cfg.get("organization")), ("Client", cfg.get("client")),
                 ("Prepared by", cfg.get("author"))]
    meta_rows = [(k, v) for k, v in meta_rows if v]
    if meta_rows:
        parts.append("<table>" + "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in meta_rows) + "</table>")
    for caveat in caveats(context):
        parts.append(f"<div class='caveat'>{html.escape(caveat)}</div>")
    if cfg.get("notes"):
        parts.append(f"<p>{html.escape(str(cfg['notes']))}</p>")
    for section in sections:
        if section in blocks:
            parts.append(_block_html(blocks[section], out.parent))

    tmpl_text = template_path.read_text(encoding="utf-8")
    html_str = tmpl_text.format(
        title=html.escape(str(cfg.get("title", "Inspection Report"))),
        report_type=html.escape(str(cfg.get("report_type", "standard"))),
        generated_at=html.escape(str(context.get("generated_at", _now_iso()))),
        sections_html="\n".join(parts),
        report_id=html.escape(str(context.get("report_id", ""))),
        project_id=html.escape(str(context.get("project", {}).get("id", ""))),
        project_name=html.escape(str(context.get("project", {}).get("name", ""))),
        section_count=len(sections),
    )
    out.write_text(html_str, encoding="utf-8")
    return out


# ── PDF / DOCX ────────────────────────────────────────────────────────────────

def deliverable_payload(context: dict[str, Any], blocks: dict[str, dict[str, Any]],
                        assets: dict[str, Any]) -> dict[str, Any]:
    """The context in the shape core/report_formats.py writes."""
    findings = []
    for f in context.get("findings") or []:
        entry = dict(f)
        entry["image_path"] = (assets.get("findings") or {}).get(str(f.get("id")), "")
        findings.append(entry)
    crs = (context.get("site") or {}).get("crs_epsg")
    if not crs:
        for recon in context.get("reconstructions") or []:
            if recon.get("crs_epsg"):
                crs = recon["crs_epsg"]
                break
    recommendations = list(blocks["recommendations"].get("_author") or [])
    recommendations += list(blocks["recommendations"].get("_derived") or [])
    measurements = [[m.get("label") or m.get("measurement_type") or "-",
                     f"{_fmt(m.get('value'), 4)} {m.get('unit', '')}".strip()]
                    for m in context.get("measurements") or []]
    passthrough = {k: v for k, v in blocks.items()
                   if k not in ("summary", "methodology", "findings", "measurements",
                                "recommendations")}
    return {
        "project_name": context.get("project", {}).get("name", ""),
        "client": (context.get("config") or {}).get("client", ""),
        "report_type": (context.get("config") or {}).get("report_type", ""),
        "crs_epsg": crs,
        "summary": " ".join(blocks["summary"]["paragraphs"]),
        "methodology": "\n".join(blocks["methodology"]["paragraphs"]),
        "findings": findings,
        "measurements": measurements,
        "recommendations": recommendations or ["No recommendations were entered and no "
                                               "follow-up items arise from the recorded data."],
        "blocks": passthrough,
    }


def export_report_pdf(html_path: Path | str, output_path: Path | str,
                      context: dict[str, Any] | None = None) -> Path | None:
    """Write the PDF for a report.

    The PDF is drawn by reportlab from the same blocks as the HTML. Converting the HTML
    needed weasyprint or wkhtmltopdf, neither of which this project installs, so every
    report came back with pdf_path None and nobody was told. With no context to draw
    from, the report's own manifest context is used.
    """
    html_path = Path(html_path)
    if context is None:
        saved = _load_json(html_path.parent / "context.json")
        if not isinstance(saved, dict):
            return None
        context = saved
    written = _write_deliverables(context, Path(output_path).parent, ["pdf"],
                                  pdf_name=Path(output_path).name)
    return Path(written["pdf"]) if written.get("pdf") else None


def _write_deliverables(context: dict[str, Any], out_dir: Path, formats: list[str],
                        pdf_name: str = "report.pdf") -> dict[str, str]:
    from .report_formats import ReportTemplate, write_docx, write_pdf

    assets = context.get("_assets") or prepare_assets(context, out_dir / "assets")
    blocks = build_blocks(context, assets)
    payload = deliverable_payload(context, blocks, assets)
    cfg = context.get("config") or {}
    sections = context.get("sections") or sections_for(cfg.get("report_type", "standard"),
                                                        cfg.get("sections"))
    template = ReportTemplate(
        title=str(cfg.get("title") or "Inspection Report"),
        organization=str(cfg.get("organization") or ""),
        client=str(cfg.get("client") or ""),
        logo_path=str(cfg.get("logo_path") or ""),
        author=str(cfg.get("author") or ""),
        sections=list(sections),
    )
    written: dict[str, str] = {}
    if "pdf" in formats:
        written["pdf"] = write_pdf(out_dir / pdf_name, payload, template)
    if "docx" in formats:
        written["docx"] = write_docx(out_dir / "report.docx", payload, template)
    return written


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return slug[:40] or "report"


def render_report(context: dict[str, Any], reports_root: Path | str,
                  formats: list[str] | None = None) -> ReportResult:
    """Write a report into its own folder under `reports_root`.

    The folder is named by time and title, so reports sort chronologically and never
    overwrite one another. A PDF or DOCX that was asked for and could not be written
    fails the call rather than returning quietly without it.
    """
    cfg = context.get("config") or {}
    formats = [str(f).lower() for f in (formats or cfg.get("formats") or ["html", "pdf"])]
    report_id = str(uuid.uuid4())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(reports_root) / f"{stamp}_{_slug(cfg.get('title', 'report'))}_{report_id[:8]}"
    assets = prepare_assets(context, out_dir / "assets")
    context = dict(context)
    context["report_id"] = report_id
    context["sections"] = sections_for(cfg.get("report_type", "standard"), cfg.get("sections"))
    context["_assets"] = assets
    blocks = build_blocks(context, assets)

    html_path = render_report_html(context, _ensure_builtin_template(), out_dir / "report.html",
                                   assets=assets, blocks=blocks)
    written = _write_deliverables(context, out_dir, [f for f in formats if f in ("pdf", "docx")])

    findings = _reportable_findings(context)
    result = ReportResult(
        id=report_id,
        project_id=str(context.get("project", {}).get("id", "")),
        title=str(cfg.get("title", "Inspection Report")),
        html_path=str(html_path),
        pdf_path=written.get("pdf"),
        docx_path=written.get("docx"),
        report_type=str(cfg.get("report_type", "standard")),
        output_dir=str(out_dir),
        findings=len(findings),
    )
    (out_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    saved_context = {k: v for k, v in context.items() if not k.startswith("_")}
    (out_dir / "context.json").write_text(json.dumps(saved_context, indent=2, default=str),
                                          encoding="utf-8")

    # A report is a derived file like any other: record what it was built from, so a
    # copy that reaches a client can be checked against what this project produced.
    try:
        from .provenance import record

        sources = [p["path"] for p in context.get("photos") or []]
        sources += [f.get("image_source_path") for f in context.get("findings") or []
                    if f.get("image_source_path")]
        for artifact in [html_path] + [Path(p) for p in written.values()]:
            record(artifact, engine="opendronekit.report_engine", sources=sources,
                   parameters={"report_type": result.report_type, "report_id": report_id,
                               "findings": len(findings)})
    except Exception:  # noqa: BLE001 - provenance must never lose a written report
        pass
    return result


# ── Engine workspace entry points ─────────────────────────────────────────────

def generate_report(config: ReportConfig) -> ReportResult:
    """End-to-end for the engine's own workspace: readiness, context, render."""
    readiness = validate_report_readiness(config)
    if not readiness.ok:
        raise AppError(
            ERR_REPORT_NOT_READY,
            "Report cannot be generated — required data missing: " + "; ".join(readiness.missing),
            technical_message=f"Missing: {readiness.missing}",
            recovery_action="Fix missing items in readiness checklist.",
        )

    mgr = get_manager()
    meta = mgr.load_project(config.project_id)
    root = Path(str(meta.get("root_dir", "")))
    context = build_report_context(config)
    result = render_report(context, root / "reports", config.formats)
    result.project_id = config.project_id
    result.warnings = list(readiness.warnings)
    (Path(result.output_dir) / "report.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    try:
        mgr.write_project_audit(config.project_id, "report_generated", f"Generated report: {config.title}",
                                {"report_id": result.id, "html_path": result.html_path})
    except Exception:
        pass
    publish_event(REPORT_GENERATED, {"report_id": result.id, "project_id": config.project_id})
    return result


def list_reports_in(reports_dir: Path | str) -> list[ReportResult]:
    """Every report manifest under a reports folder, newest first."""
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        return []
    out: list[ReportResult] = []
    for sub in sorted(reports_dir.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        manifest = sub / "report.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            allowed = {k: v for k, v in data.items() if k in ReportResult.__dataclass_fields__}
            out.append(ReportResult(**allowed))
        except Exception:
            pass
    return out


def list_reports(project_id: str) -> list[ReportResult]:
    """Return generated reports for a project in the engine workspace."""
    mgr = get_manager()
    try:
        meta = mgr.load_project(project_id)
    except Exception:
        return []
    return list_reports_in(Path(str(meta.get("root_dir", ""))) / "reports")
