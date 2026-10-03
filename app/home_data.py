"""What the Home workspace reads: the projects on this machine and the open one's summary.

The Home artboard shows figures the Api had no way to answer -- the survey's capture
window, what a project occupies on disk, which files have been exported, what the
reconstruction measured -- so a builder would have had to invent them. Each is read
here from the files themselves:

  * the survey date is the photographs' own EXIF DateTimeOriginal (camera clock, which
    carries no zone unless the camera wrote OffsetTimeOriginal), not the import time;
  * storage is the bytes on disk under the project folder, plus the dataset folder when
    it lives elsewhere, and the free space on that drive;
  * exports are the files in the folders the exporters write to, and in any folder the
    audit log records an export into;
  * reconstruction figures come from the files the engine wrote (digital_twin.json,
    camera_poses.json, geo_anchor.json, the GeoTIFFs' own pixel size) and the run record
    the Api writes when a reconstruction finishes. A figure no file holds is None, which
    the interface renders "not reported".

Nothing here is a default or an estimate presented as a reading.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: Where a project's own state lives (caches this module writes). Excluded from nothing:
#: it is counted in storage like any other folder.
CACHE_DIR = ".odk"

#: Folders the exporters write to by default, and what they hold.
EXPORT_FOLDERS = (
    ("missions", "mission"),
    ("flights", "flight log"),
    ("exports", "export"),
)

#: Report-folder files that are exports rather than generated reports.
REPORT_EXPORT_PATTERNS = (
    (re.compile(r"^audit_.*\.(csv|json)$", re.I), "audit log"),
    (re.compile(r"^findings_.*\.(pdf|docx|html|csv|json|geojson)$", re.I), "findings"),
)

FORMAT_LABELS = {
    ".plan": "QGC plan", ".kmz": "DJI WPML", ".waypoints": "QGC WPL", ".kml": "KML",
    ".geojson": "GeoJSON", ".csv": "CSV", ".json": "JSON", ".pdf": "PDF", ".docx": "Word",
    ".html": "HTML", ".gpx": "GPX", ".tlog": "MAVLink log", ".bin": "DataFlash log",
    ".zip": "ZIP", ".obj": "OBJ", ".tif": "GeoTIFF", ".tiff": "GeoTIFF", ".las": "LAS",
    ".laz": "LAZ", ".ply": "PLY",
}


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _images(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    except OSError:
        return []


# ---------------------------------------------------------------- capture window

def _exif_time(path: Path) -> tuple[str | None, str | None]:
    """(DateTimeOriginal as ISO without zone, OffsetTimeOriginal) or (None, None)."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow is a hard requirement
        return None, None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None, None
            sub = {}
            try:
                sub = dict(exif.get_ifd(0x8769))
            except Exception:  # noqa: BLE001 - an older Pillow: IFD0 only
                sub = {}
            raw = sub.get(36867) or sub.get(36868) or exif.get(306)  # Original, Digitized, DateTime
            offset = sub.get(36881) or sub.get(36880)  # OffsetTimeOriginal, OffsetTime
    except Exception:  # noqa: BLE001 - an unreadable file is simply undated
        return None, None
    if not raw:
        return None, None
    text = str(raw).strip().replace("\x00", "")
    try:
        stamp = datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None, None
    return stamp, (str(offset).strip().replace("\x00", "") or None) if offset else None


def capture_window(folder: str | Path, cache_root: str | Path | None = None) -> dict[str, Any]:
    """When the photographs in `folder` were taken, from their EXIF clocks.

    Cached under `<cache_root>/.odk/capture_windows.json`, keyed by the folder and its
    image count and newest modification time, so a changed folder is read again.
    """
    root = Path(folder)
    images = _images(root)
    signature = [len(images), max((p.stat().st_mtime for p in images), default=0.0)]
    cache_file = Path(cache_root) / CACHE_DIR / "capture_windows.json" if cache_root else None
    cache = _json(cache_file) if cache_file else {}
    hit = cache.get(str(root))
    if isinstance(hit, dict) and hit.get("signature") == signature:
        return hit["window"]

    stamps: list[str] = []
    offsets: set[str] = set()
    for path in images:
        stamp, offset = _exif_time(path)
        if stamp:
            stamps.append(stamp)
        if offset:
            offsets.add(offset)
    stamps.sort()
    window: dict[str, Any] = {
        "images": len(images),
        "dated": len(stamps),
        "first": stamps[0] if stamps else None,
        "last": stamps[-1] if stamps else None,
        # One offset for every photograph is a zone; none or several is camera-local time.
        "utc_offset": next(iter(offsets)) if len(offsets) == 1 else None,
        "clock": "camera local time" if len(offsets) != 1 else "camera time with UTC offset",
        "source": "EXIF DateTimeOriginal",
    }
    if stamps:
        first = datetime.fromisoformat(stamps[0])
        last = datetime.fromisoformat(stamps[-1])
        window["duration_s"] = round((last - first).total_seconds(), 1)
    else:
        window["duration_s"] = None
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache[str(root)] = {"signature": signature, "window": window}
            cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except OSError:
            pass  # a read-only project still answers, it just reads again next time
    return window


# --------------------------------------------------------------------- datasets

def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("metadata_json") or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _folder_bytes(folder: Path, *, skip: Path | None = None) -> tuple[int, int]:
    """(bytes, files) under `folder`, not following links, optionally skipping one subtree."""
    total = files = 0
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        here = Path(dirpath)
        if skip is not None:
            dirnames[:] = [d for d in dirnames if (here / d).resolve() != skip]
        for name in filenames:
            try:
                total += (here / name).stat().st_size
                files += 1
            except OSError:
                continue
    return total, files


def dataset_summary(store: Any, project_id: int, project_root: Path,
                    active_folder: str) -> dict[str, Any]:
    """The open project's datasets and the selected one's photographs."""
    rows = store.list_datasets(project_id)
    datasets = []
    for row in rows:
        meta = _metadata(row)
        path = str(row.get("path") or "")
        datasets.append({
            "id": row.get("id"),
            "name": row.get("name") or Path(path).name,
            "path": path,
            "exists": bool(path) and Path(path).is_dir(),
            "active": bool(active_folder) and path == str(active_folder),
            "image_count": meta.get("image_count"),
            "geotagged_count": meta.get("geotagged_count"),
            "camera": meta.get("camera") or None,
            "imported_utc": meta.get("imported_utc") or row.get("created_at"),
        })
    summary: dict[str, Any] = {"datasets": datasets, "count": len(datasets), "active": None}
    if not active_folder:
        return summary
    folder = Path(active_folder)
    row = next((d for d in datasets if d["path"] == str(folder)), None)
    meta = _metadata(next((r for r in rows if str(r.get("path")) == str(folder)), {}))
    images = _images(folder)
    size, _files = (0, 0)
    if folder.is_dir():
        size = sum(p.stat().st_size for p in images)
    summary["active"] = {
        "folder": str(folder),
        "name": (row or {}).get("name") or folder.name,
        "exists": folder.is_dir(),
        "image_count": len(images),
        "recorded_image_count": meta.get("image_count"),
        "geotagged_count": meta.get("geotagged_count"),
        "camera": meta.get("camera") or None,
        "center_lonlat": meta.get("center_lonlat"),
        "bounds_lonlat": meta.get("bounds_lonlat"),
        "suggested_epsg": meta.get("suggested_epsg"),
        "image_bytes": size,
        "capture": capture_window(folder, project_root) if folder.is_dir() else None,
    }
    return summary


# --------------------------------------------------------------------- projects

def _reconstruction_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found = [root / "reconstruction"] if (root / "reconstruction").is_dir() else []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and child.name not in {"reconstruction", CACHE_DIR}:
            candidate = child / "reconstruction"
            if candidate.is_dir():
                found.append(candidate)
            # run folders one level further down (analysis/run_*/reconstruction)
            try:
                found.extend(sorted(p for p in child.glob("*/reconstruction") if p.is_dir()))
            except OSError:
                pass
    return found


def _report_count(root: Path) -> int:
    from core.report_engine import list_reports_in

    try:
        return len(list_reports_in(root / "reports"))
    except Exception:  # noqa: BLE001 - an unreadable manifest is not a report
        return 0


def projects_overview(store: Any, active_id: int | None) -> list[dict[str, Any]]:
    """Every project on this machine with what it holds, newest change first.

    Status words are facts about the folder: products on disk, a report written,
    photographs imported. A project with none of them says "empty".
    """
    out = []
    for project in store.list_projects():
        root = Path(str(project.get("root_dir") or "")) if project.get("root_dir") else None
        datasets = store.list_datasets(int(project["id"]))
        latest = datasets[0] if datasets else None
        meta = _metadata(latest) if latest else {}
        capture = None
        if latest and Path(str(latest.get("path") or "")).is_dir():
            capture = capture_window(latest["path"], root if root and root.is_dir() else None)
        recon = [d for d in _reconstruction_dirs(root)] if root else []
        processed = any((d / "orthomosaic.tif").is_file() or (d / "dsm.tif").is_file() for d in recon)
        reports = _report_count(root) if root and root.is_dir() else 0
        state = []
        if processed:
            state.append("processed")
        if reports:
            state.append("report ready" if reports == 1 else f"{reports} reports")
        if not state and meta.get("image_count"):
            state.append("photos imported")
        if not state and int(project.get("mission_versions_count") or 0):
            state.append("mission saved")
        out.append({
            "id": project["id"],
            "name": project.get("name"),
            "root_dir": str(root) if root else "",
            "folder_exists": bool(root and root.is_dir()),
            "created_at": project.get("created_at"),
            "updated_at": project.get("updated_at"),
            "sync_status": project.get("sync_status"),
            "offline_enabled": bool(project.get("offline_enabled")),
            "active": active_id is not None and int(project["id"]) == int(active_id),
            "datasets_count": project.get("datasets_count"),
            "mission_versions_count": project.get("mission_versions_count"),
            "reports_count": reports,
            "photos": meta.get("image_count"),
            "geotagged": meta.get("geotagged_count"),
            "survey_first": capture.get("first") if capture else None,
            "survey_last": capture.get("last") if capture else None,
            "processed": processed,
            "status": state or ["empty"],
        })
    return out


# ------------------------------------------------------------- reconstruction

def _raster_facts(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    facts: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size}
    try:
        import rasterio

        with rasterio.open(path) as src:
            facts.update({
                "width": src.width, "height": src.height,
                "pixel_size": abs(float(src.res[0])),
                "epsg": src.crs.to_epsg() if src.crs else None,
                "units": (src.crs.linear_units if src.crs else None),
            })
    except Exception as exc:  # noqa: BLE001 - the file is there; say what could not be read
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _ply_points(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                if line.startswith(b"element vertex"):
                    return int(line.split()[-1])
                if line.strip() == b"end_header":
                    break
    except (OSError, ValueError):
        return None
    return None


def reconstruction_run(directory: Path, jobs: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """One reconstruction folder's figures, each with the file it was read from."""
    twin = _json(directory / "digital_twin.json")
    record = _json(directory / "run_record.json")
    meta = _json(directory / "run_meta.json")
    result = record.get("result") if isinstance(record.get("result"), dict) else (
        meta.get("result") if isinstance(meta.get("result"), dict) else {})
    poses = _json(directory / "camera_poses.json")
    anchor = _json(directory / "geo_anchor.json")
    sources: dict[str, str] = {}

    def pick(key: str, *candidates: tuple[str, Any]) -> Any:
        for source, value in candidates:
            if value is not None and value != "":
                sources[key] = source
                return value
        return None

    track_count = None
    track = directory / "camera_track.geojson"
    if track.is_file():
        track_count = len(_json(track).get("features") or [])

    registered = pick("registered_images",
                      ("digital_twin.json", twin.get("registered_images")),
                      ("camera_poses.json", poses.get("registered")),
                      ("run record", result.get("registered_images")),
                      ("camera_track.geojson", track_count))
    total = pick("total_images",
                 ("digital_twin.json", twin.get("frame_count")),
                 ("camera_poses.json", poses.get("total_images")),
                 ("run record", result.get("frame_count")))
    points = pick("sparse_points",
                  ("digital_twin.json", twin.get("point_count")),
                  ("run record", result.get("total_points")),
                  ("reconstruction.ply", _ply_points(directory / "reconstruction.ply")))
    reproj = pick("reprojection_error_px",
                  ("digital_twin.json", twin.get("mean_reprojection_error_px")),
                  ("camera_poses.json", poses.get("mean_reprojection_error_px")),
                  ("run record", result.get("reprojection_error_px")))
    rmse = pick("georef_rmse_m",
                ("geo_anchor.json", anchor.get("rmse_m")),
                ("digital_twin.json", twin.get("geo_rmse_m")),
                ("run record", result.get("geo_rmse_m")))
    elapsed = pick("runtime_s",
                   ("run_record.json", record.get("elapsed_s")),
                   ("run_meta.json", meta.get("elapsed_s")))
    if elapsed is None:
        out_path = str(directory.resolve())
        for job in jobs:
            payload = job.get("result") if isinstance(job.get("result"), dict) else {}
            if job.get("name") != "reconstruction" or job.get("status") != "done":
                continue
            ortho = str(payload.get("orthomosaic_path") or payload.get("orthomosaic_cog_path") or "")
            if ortho and Path(ortho).resolve().parent == Path(out_path) and job.get("started_utc") \
                    and job.get("finished_utc"):
                try:
                    a = datetime.fromisoformat(str(job["started_utc"]).replace("Z", "+00:00"))
                    b = datetime.fromisoformat(str(job["finished_utc"]).replace("Z", "+00:00"))
                    elapsed = round((b - a).total_seconds(), 1)
                    sources["runtime_s"] = "job record"
                except ValueError:
                    pass
    profile = pick("profile",
                   ("run_record.json", record.get("profile")),
                   ("digital_twin.json", twin.get("profile")),
                   ("run record", result.get("processing_profile")))
    engine = pick("engine",
                  ("run_record.json", record.get("engine")),
                  ("digital_twin.json", twin.get("engine")),
                  ("run record", result.get("engine")),
                  ("geo_anchor.json", "colmap" if anchor.get("method") else None))
    dense_points = result.get("dense_point_count")
    dense = twin.get("dense") if "dense" in twin else (
        bool(dense_points) if dense_points is not None else None)
    caps = meta.get("capabilities") if isinstance(meta.get("capabilities"), dict) else {}
    dense_reason = (record.get("dense_reason") or caps.get("dense_stereo_reason")
                    or next((w for w in (twin.get("warnings") or result.get("warnings") or [])
                             if "dense" in str(w).lower()), None))

    ortho = _raster_facts(directory / "orthomosaic.tif")
    dsm = _raster_facts(directory / "dsm.tif")
    products = [name for name in ("orthomosaic.tif", "dsm.tif", "dtm.tif", "dsm_hillshade.tif",
                                  "camera_track.geojson", "reconstruction.ply", "mesh.obj")
                if (directory / name).is_file()]
    finished = record.get("finished_utc") or meta.get("finished")
    if not finished and products:
        finished = _utc(max((directory / p).stat().st_mtime for p in products))
        sources["finished"] = "file modification time"
    return {
        "directory": str(directory),
        "name": directory.parent.name if directory.name == "reconstruction" else directory.name,
        "complete": bool(ortho or dsm),
        "engine": engine,
        "profile": profile,
        "registered_images": registered,
        "total_images": total,
        "sparse_points": points,
        "reprojection_error_px": reproj,
        "georef_rmse_m": rmse,
        "georef_method": anchor.get("method"),
        "ortho_gsd_m": ortho.get("pixel_size") if ortho else None,
        "dsm_gsd_m": dsm.get("pixel_size") if dsm else None,
        "orthomosaic": ortho,
        "dsm": dsm,
        "dense": dense,
        "dense_reason": dense_reason,
        "runtime_s": elapsed,
        "finished": finished,
        "products": products,
        "sources": sources,
    }


def reconstruction_summary(project_root: Path, jobs: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    jobs = list(jobs)
    runs = []
    for directory in _reconstruction_dirs(project_root):
        run = reconstruction_run(directory, jobs)
        try:
            run["name"] = directory.relative_to(project_root).as_posix()
        except ValueError:
            pass
        runs.append(run)
    runs = [r for r in runs if r["products"] or r["registered_images"] is not None]
    runs.sort(key=lambda r: str(r.get("finished") or ""), reverse=True)
    running = [j for j in jobs if j.get("name") in {"reconstruction", "pipeline"}
               and j.get("status") in {"queued", "running"}]
    return {"runs": runs, "count": len(runs), "running": running}


# ---------------------------------------------------------------------- storage

def project_storage(project_root: Path, dataset_folders: Iterable[str]) -> dict[str, Any]:
    """Bytes on disk: the project folder by top-level folder, outside datasets, free space."""
    root = Path(project_root)
    groups: list[dict[str, Any]] = []
    total = files = 0
    if root.is_dir():
        loose = loose_files = 0
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.is_symlink():
                size, count = _folder_bytes(child)
                groups.append({"name": child.name, "bytes": size, "files": count})
                total += size
                files += count
            elif child.is_file():
                try:
                    loose += child.stat().st_size
                    loose_files += 1
                except OSError:
                    pass
        if loose_files:
            groups.append({"name": "(project folder)", "bytes": loose, "files": loose_files})
            total += loose
            files += loose_files
    groups.sort(key=lambda g: g["bytes"], reverse=True)
    outside = []
    for folder in dict.fromkeys(str(f) for f in dataset_folders if f):
        path = Path(folder)
        if not path.is_dir():
            continue
        try:
            path.resolve().relative_to(root.resolve())
            continue  # already counted inside the project
        except ValueError:
            pass
        images = _images(path)
        outside.append({"path": str(path), "bytes": sum(p.stat().st_size for p in images),
                        "files": len(images)})
    disk = None
    probe = root if root.is_dir() else root.parent
    try:
        usage = shutil.disk_usage(probe)
        disk = {"path": str(probe.anchor or probe), "total": usage.total, "free": usage.free,
                "used": usage.used}
    except OSError:
        disk = None
    return {
        "root": str(root),
        "project_bytes": total,
        "project_files": files,
        "groups": groups,
        "datasets_outside": outside,
        "datasets_outside_bytes": sum(d["bytes"] for d in outside),
        "disk": disk,
    }


# ---------------------------------------------------------------------- exports

def _export_dirs_from_audit(events: Iterable[dict[str, Any]]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for event in events:
        kind = str(event.get("event_type") or "")
        if "export" not in kind:
            continue
        try:
            payload = json.loads(event.get("payload_json") or "{}")
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("directory", "output_dir", "folder", "path", "file", "output"):
            value = payload.get(key)
            if not value or not isinstance(value, str):
                continue
            path = Path(value)
            folder = path if path.is_dir() else path.parent
            if folder.is_dir():
                found.append((folder, kind.replace("_exported", "").replace("_", " ")))
    return found


def list_exports(project_root: Path, audit_events: Iterable[dict[str, Any]] = (),
                 limit: int = 200) -> dict[str, Any]:
    """Every exported file this project knows of, newest first, grouped by folder."""
    root = Path(project_root)
    candidates: list[tuple[Path, str, bool]] = []
    for name, kind in EXPORT_FOLDERS:
        candidates.append((root / name, kind, True))
    for folder, kind in _export_dirs_from_audit(audit_events):
        candidates.append((folder, kind or "export", False))
    seen: set[str] = set()
    files: list[dict[str, Any]] = []
    folders: list[dict[str, Any]] = []

    def add(path: Path, kind: str, folder: Path) -> None:
        key = str(path.resolve())
        if key in seen or not path.is_file() or path.name.endswith(".prov.json"):
            return
        seen.add(key)
        stat = path.stat()
        files.append({
            "name": path.name, "path": str(path), "folder": str(folder), "kind": kind,
            "extension": path.suffix.lower(),
            "format": ("Litchi CSV" if kind == "mission" and path.suffix.lower() == ".csv"
                       else FORMAT_LABELS.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "file")),
            "bytes": stat.st_size, "modified_utc": _utc(stat.st_mtime),
        })

    for folder, kind, recursive in candidates:
        if not folder.is_dir():
            continue
        before = len(files)
        walker = folder.rglob("*") if recursive else folder.iterdir()
        for path in walker:
            if path.is_file():
                # A folder the audit log names (reports/, a chosen directory) holds other
                # things too: only files written at or after an export are exports, so
                # those are matched by the report-folder patterns or taken whole only
                # from the exporters' own folders.
                if recursive:
                    add(path, kind, folder)
                else:
                    for pattern, label in REPORT_EXPORT_PATTERNS:
                        if pattern.match(path.name):
                            add(path, label, folder)
                            break
                    else:
                        if kind not in {"audit log", "report", "findings"} and \
                                path.suffix.lower() in FORMAT_LABELS:
                            add(path, kind, folder)
        if len(files) > before:
            folders.append({"path": str(folder), "kind": kind, "files": len(files) - before})
    reports_dir = root / "reports"
    if reports_dir.is_dir():
        before = len(files)
        for path in reports_dir.iterdir():
            for pattern, label in REPORT_EXPORT_PATTERNS:
                if path.is_file() and pattern.match(path.name):
                    add(path, label, reports_dir)
                    break
        if len(files) > before:
            folders.append({"path": str(reports_dir), "kind": "reports folder", "files": len(files) - before})
    files.sort(key=lambda f: (f["modified_utc"], f["name"]), reverse=True)
    return {"files": files[: max(1, int(limit))], "count": len(files),
            "total_bytes": sum(f["bytes"] for f in files), "folders": folders}


# ---------------------------------------------------------------------- reports

def _pdf_pages(path: Path) -> int | None:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001 - no pypdf, or not a PDF it can read
        return None


def report_files(report: dict[str, Any]) -> dict[str, Any]:
    """A generated report's files on disk: size of each format, PDF page count."""
    formats = []
    for kind, key in (("pdf", "pdf_path"), ("docx", "docx_path"), ("html", "html_path")):
        value = report.get(key)
        if not value:
            continue
        path = Path(str(value))
        entry: dict[str, Any] = {"kind": kind, "path": str(path), "exists": path.is_file(),
                                 "bytes": path.stat().st_size if path.is_file() else None}
        if kind == "pdf" and path.is_file():
            entry["pages"] = _pdf_pages(path)
        formats.append(entry)
    pages = next((f.get("pages") for f in formats if f.get("pages")), None)
    return {"report": report, "formats": formats, "pages": pages}
