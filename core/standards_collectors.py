"""Facts for standards-structured reports, read from real files.

Every function here returns facts in the shape core.standards_reports.resolve() takes:
``{"fields": {item_id: {"value", "source"}}, "tables": {...}, "observations": [...],
"evidence": [...]}`` plus the list of files it read (``inputs``), which go into the
signed manifest. A value is set only when a file states it; the ``source`` says which
file and which tag, column or computation. Anything the files do not state is left out,
so the report prints it as MISSING.

Deliberately not filled from data:
* official identifiers and locations (an NBIS bridge number or B.L.05 latitude is the
  agency's record, not the centroid of some photos -- photo positions are reported as
  an observation instead);
* inspection-program records (an NBIS inspection date is not the date a research
  dataset was flown);
* anything a qualified person assigns: IEC CoA classes, SNBI condition ratings and
  states, emissivity when the camera did not record it.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def _set(facts: dict[str, Any], item_id: str, value: Any, source: str) -> None:
    facts.setdefault("fields", {})[item_id] = {"value": value, "source": source}


def _observe(facts: dict[str, Any], text: str, source: str) -> None:
    facts.setdefault("observations", []).append({"text": text, "source": source})


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Operator-entered facts on top of data-derived ones (the operator's value wins)."""
    out = {"fields": dict(base.get("fields") or {}), "tables": dict(base.get("tables") or {}),
           "observations": list(base.get("observations") or []),
           "evidence": list(base.get("evidence") or [])}
    out["fields"].update(override.get("fields") or {})
    for key, rows in (override.get("tables") or {}).items():
        if rows:
            out["tables"][key] = list(rows)
    out["observations"] += list(override.get("observations") or [])
    out["evidence"] += list(override.get("evidence") or [])
    return out


# ── Image metadata ────────────────────────────────────────────────────────────

def _exif_datetime(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _xmp_text(path: Path) -> str:
    """The XMP packet of a JPEG or TIFF, or ''."""
    data = path.read_bytes()[:400_000] if path.suffix.lower() in (".jpg", ".jpeg") else b""
    if data:
        start = data.find(b"<x:xmpmeta")
        end = data.find(b"</x:xmpmeta>")
        return data[start:end].decode("utf-8", "replace") if start >= 0 and end > start else ""
    from PIL import Image

    try:
        with Image.open(path) as image:
            raw = getattr(image, "tag_v2", {}).get(700)
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw or "")


def _xmp_value(xmp: str, name: str) -> str | None:
    match = re.search(rf"{name}\s*=\s*\"([^\"]*)\"|<{name}>\s*([^<]*)<", xmp)
    if not match:
        return None
    return (match.group(1) if match.group(1) is not None else match.group(2)).strip()


def image_metadata(path: str | Path) -> dict[str, Any]:
    """Capture metadata of one image, from its own tags only."""
    from .geo import _raw_exif, read_exif_gps

    target = Path(path)
    tags, _gps, width, height = _raw_exif(target)
    fix = read_exif_gps(target)
    when = _exif_datetime(tags.get("DateTimeOriginal") or tags.get("DateTime"))
    xmp = _xmp_text(target)
    pitch = _xmp_value(xmp, "drone-dji:GimbalPitchDegree")
    serial = None
    try:
        from PIL import Image

        with Image.open(target) as image:
            serial = (getattr(image, "tag_v2", {}) or {}).get(50735)  # CameraSerialNumber
    except Exception:  # noqa: BLE001
        serial = None
    return {
        "path": str(target), "name": target.name,
        "make": str(tags.get("Make", "") or "").strip("\x00 ").strip(),
        "model": str(tags.get("Model", "") or "").strip("\x00 ").strip(),
        "width": int(width), "height": int(height),
        "datetime": when.isoformat(sep=" ") if when else None,
        "focal_length_mm": _num(tags.get("FocalLength")),
        "f_number": _num(tags.get("FNumber")),
        "serial": str(serial).strip() if serial else (
            str(tags.get("BodySerialNumber") or "").strip() or None),
        # (0, 0) is the "no fix" value some cameras write, not a position.
        "gps": ([fix.latitude, fix.longitude] if fix and (abs(fix.latitude) > 1e-9 or
                                                          abs(fix.longitude) > 1e-9) else None),
        "gimbal_pitch_deg": float(pitch) if pitch not in (None, "") else None,
    }


def _num(value: Any) -> float | None:
    from .geo import _rational

    return _rational(value)


def list_images(folder: str | Path, pattern: str = "*") -> list[Path]:
    return sorted(p for p in Path(folder).glob(pattern)
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def capture_summary(paths: Iterable[str | Path]) -> dict[str, Any]:
    rows = [image_metadata(p) for p in paths]
    times = sorted(r["datetime"] for r in rows if r["datetime"])
    cameras = sorted({f"{r['make']} {r['model']}".strip() for r in rows if r["make"] or r["model"]})
    gps = [r["gps"] for r in rows if r["gps"]]
    summary: dict[str, Any] = {
        "count": len(rows), "rows": rows, "cameras": cameras,
        "first": times[0] if times else None, "last": times[-1] if times else None,
        "resolutions": sorted({f"{r['width']}x{r['height']}" for r in rows}),
        "geotagged": len(gps),
    }
    if gps:
        lats, lons = [g[0] for g in gps], [g[1] for g in gps]
        summary["bounds"] = [min(lons), min(lats), max(lons), max(lats)]
        summary["centroid"] = [sum(lats) / len(lats), sum(lons) / len(lons)]
    return summary


def describe_capture(facts: dict[str, Any], summary: dict[str, Any], label: str) -> None:
    """Observations every template shares: what was captured, when, with what, where."""
    if not summary["count"]:
        return
    source = f"EXIF of {summary['count']} {label} file(s)"
    when = (f"{summary['first']} to {summary['last']}" if summary["first"] else
            "no capture time recorded")
    _observe(facts, f"{summary['count']} {label} image(s), {when}, camera(s): "
             f"{', '.join(summary['cameras']) or 'not recorded'}, "
             f"resolution(s): {', '.join(summary['resolutions'])}.", source)
    if summary.get("bounds"):
        lon0, lat0, lon1, lat1 = summary["bounds"]
        _observe(facts, f"{summary['geotagged']} geotagged; image GPS extent lat "
                 f"{lat0:.6f}..{lat1:.6f}, lon {lon0:.6f}..{lon1:.6f} (camera positions, not "
                 "the asset's surveyed location).", source)
    else:
        _observe(facts, "No image carries a usable GPS position.", source)


# ── Environmental logger readings ─────────────────────────────────────────────

def _read_csv_text(source: str) -> tuple[str, str]:
    """(text, label) from ``path`` or ``archive.zip::member``."""
    if "::" in source:
        archive, member = source.split("::", 1)
        with zipfile.ZipFile(archive) as zf:
            raw = zf.read(member)
        label = f"{Path(archive).name}::{member}"
    else:
        raw = Path(source).read_bytes()
        label = Path(source).name
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), label
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{label} is not text")


def environment_at(source: str, when: datetime, columns: dict[str, str],
                   *, timestamp_column: str = "Timestamp",
                   timestamp_format: str = "%Y-%m-%d %H:%M:%S",
                   max_gap: timedelta = timedelta(minutes=5)) -> dict[str, Any]:
    """The logger row nearest ``when`` and the named columns' values.

    ``columns`` maps our name -> the exact header prefix in the file (no guessing which
    column is irradiance). Refuses (ValueError) when no row lies within ``max_gap`` or a
    column is absent, rather than borrowing a reading from another time.
    """
    text, label = _read_csv_text(source)
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    index = {}
    for name, prefix in columns.items():
        hits = [i for i, h in enumerate(header) if h.startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"{label}: column {prefix!r} matched {len(hits)} headers.")
        index[name] = hits[0]
    ts_col = header.index(timestamp_column)
    best, best_gap = None, None
    for row in reader:
        if not row or len(row) <= ts_col:
            continue
        try:
            stamp = datetime.strptime(row[ts_col].strip(), timestamp_format)
        except ValueError:
            continue
        gap = abs(stamp - when)
        if best_gap is None or gap < best_gap:
            best, best_gap = (stamp, row), gap
    if best is None or best_gap > max_gap:
        raise ValueError(f"{label}: no reading within {max_gap} of {when}.")
    stamp, row = best
    values = {}
    for name, i in index.items():
        try:
            values[name] = float(row[i])
        except (ValueError, IndexError):
            raise ValueError(f"{label}: {header[i]!r} at {stamp} is not a number.") from None
    return {"values": values, "row_time": stamp.isoformat(sep=" "), "label": label,
            "headers": {name: header[i] for name, i in index.items()}}


# ── IEC TS 62446-3 from radiometric frames ────────────────────────────────────

_NAME_TIME = re.compile(r"(\d{8})_(\d{6})")


def _frame_time(meta: dict[str, Any], path: Path) -> tuple[datetime | None, str]:
    if meta.get("datetime"):
        return datetime.fromisoformat(meta["datetime"]), "EXIF DateTimeOriginal"
    match = _NAME_TIME.search(path.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S"), "file name timestamp"
    return None, ""


def iec_facts_from_frames(frames: list[str | Path], regions_file: str | Path,
                          work_dir: str | Path, *, environment: dict[str, Any] | None = None,
                          rgb_suffix: tuple[str, str] | None = None) -> dict[str, Any]:
    """IEC TS 62446-3 facts from decoded radiometric frames and operator module outlines.

    ``regions_file``: ``{"region_source", "frames": {name: {"regions": [...]}}}`` (the
    shape of tests/fixtures/thedpv_module_regions.json). ``environment``:
    ``{"source", "columns": {"poa", "air", "wind", "wind_dir"} -> header prefixes}``.
    Module temperatures come from core.india_thermal.module_temperatures_in_image; the
    report's ΔT is the module median minus the median of the frame's module medians
    (the regular-spot comparison of clause 8 s). No CoA is assigned.
    """
    import json

    from .india_thermal import module_temperatures_in_image
    from .thermal import load_radiometric_file

    facts: dict[str, Any] = {"fields": {}, "tables": {"anomalies": []}, "observations": [],
                             "evidence": []}
    inputs: list[dict[str, Any]] = [{"path": str(regions_file), "role": "module outlines"}]
    regions = json.loads(Path(regions_file).read_text(encoding="utf-8"))
    frame_paths = [Path(f) for f in frames]
    metas = [image_metadata(p) for p in frame_paths]
    times = []
    for path, meta in zip(frame_paths, metas):
        inputs.append({"path": str(path), "role": "radiometric thermal frame"})
        when, how = _frame_time(meta, path)
        if when:
            times.append((when, how))
    if times:
        first, last = min(t for t, _ in times), max(t for t, _ in times)
        _set(facts, "inspection.datetime",
             first.isoformat(sep=" ") if first == last else
             f"{first.isoformat(sep=' ')} to {last.isoformat(sep=' ')}",
             f"{times[0][1]} of the thermal frames (camera clock; time zone not recorded)")

    cams = sorted({f"{m['make']} {m['model']}".strip() for m in metas if m["make"] or m["model"]})
    if cams:
        _set(facts, "camera.make_model", ", ".join(cams), "TIFF Make/Model tags of the frames")
    serials = sorted({m["serial"] for m in metas if m["serial"]})
    lens = sorted({f"{m['focal_length_mm']:g} mm f/{m['f_number']:g}" for m in metas
                   if m["focal_length_mm"] and m["f_number"]})
    if serials or lens:
        _set(facts, "camera.serial_lens",
             f"serial {', '.join(serials) or 'not recorded'}; lens {', '.join(lens) or 'not recorded'}",
             "TIFF CameraSerialNumber (50735), EXIF FocalLength / FNumber")
    sizes = sorted({f"{m['width']} x {m['height']}" for m in metas})
    if sizes:
        _set(facts, "camera.ir_resolution_px", ", ".join(sizes), "pixel dimensions of the frames")
    xmp = _xmp_text(frame_paths[0]) if frame_paths else ""
    centre = re.search(r"CentralWavelength>\s*<rdf:Seq>\s*<rdf:li>([\d.]+)<", xmp)
    fwhm = re.search(r"WavelengthFWHM>\s*<rdf:Seq>\s*<rdf:li>([\d.]+)<", xmp)
    if centre and fwhm:
        c, w = float(centre.group(1)) / 1000, float(fwhm.group(1)) / 1000
        _set(facts, "camera.spectral_response",
             f"{c - w / 2:.2f}-{c + w / 2:.2f} µm (centre {c:g} µm, FWHM {w:g} µm)",
             "XMP Camera:CentralWavelength / Camera:WavelengthFWHM")

    decoder_notes: set[str] = set()
    for path, meta in zip(frame_paths, metas):
        entry = (regions.get("frames") or {}).get(path.name)
        if not entry:
            _observe(facts, f"{path.name}: no module outlines supplied, so no module was "
                     "evaluated in this frame.", "module outlines file")
            continue
        image = load_radiometric_file(path)
        decoder_notes.add(str(image.provenance.get("validation") or ""))
        result = module_temperatures_in_image(
            path, entry["regions"], Path(work_dir) / path.stem,
            region_source=str(regions.get("region_source") or ""),
            # Only so the function runs; its severity labels are not used in this report.
            warning_delta_c=1e6, critical_delta_c=1e6, hot_cell_delta_c=1e6,
            threshold_basis="not used: IEC TS 62446-3 CoA is assigned by the thermographer")
        rows = json.loads(Path(result.artifact_paths[0]).read_text(encoding="utf-8"))["modules"]
        when, how = _frame_time(meta, path)
        # The label below promises the median of every module median in the frame, so
        # compute exactly that (a per-string reference would differ with several strings).
        medians = sorted(float(r["median_c"]) for r in rows)
        reference = ((medians[(len(medians) - 1) // 2] + medians[len(medians) // 2]) / 2
                     if medians else None)
        for row in rows:
            delta = row["median_c"] - reference
            facts["tables"]["anomalies"].append({
                "object": f"{row['module_id']} in {path.name} (positional; plant module "
                          "identity not recorded)",
                "image_file": f"{path.name}, {when.isoformat(sep=' ') if when else 'time not recorded'}",
                "temperature_c": f"median {row['median_c']:.2f} °C (max {row['max_c']:.2f} °C)",
                "reference_c": f"{reference:.2f} °C (median of {len(rows)} module medians "
                               "in the frame)",
                "delta_t_k": f"{delta:+.2f} K",
                "source": f"{image.provenance.get('decoder', 'decoder')} decode of {path.name}; "
                          f"outline by {regions.get('region_source', 'unnamed')[:80]}",
            })
        _observe(facts, f"{path.name}: frame median {image.stats.get('median_c', float('nan')):.2f} °C, "
                 f"range {image.stats.get('min_c', float('nan')):.2f}-"
                 f"{image.stats.get('max_c', float('nan')):.2f} °C over {len(rows)} outlined module(s).",
                 f"{image.provenance.get('decoder')} ({image.provenance.get('temperature_kind')})")
        if rgb_suffix:
            rgb = path.with_name(path.name.replace(*rgb_suffix))
            if rgb.is_file():
                facts["evidence"].append({"path": str(rgb), "caption":
                                          f"Visual photo taken with {path.name}"})
    for note in sorted(n for n in decoder_notes if n):
        _observe(facts, note, "thermal decoder provenance")
    for limit in regions.get("limits") or []:
        _observe(facts, limit, "module outlines file")

    if environment:
        if not times:
            _observe(facts, "No frame time, so no logger reading could be matched.",
                     environment["source"])
        else:
            mid = min(t for t, _ in times) + (max(t for t, _ in times) - min(t for t, _ in times)) / 2
            env = environment_at(environment["source"], mid, environment["columns"])
            src = (f"{env['label']} row {env['row_time']} (nearest to the frames' mid-time "
                   f"{mid.isoformat(sep=' ')}; assumes the camera clock and the logger "
                   "share one time base, which neither file records)")
            names = {"poa": "env.irradiance_poa_w_m2", "air": "env.air_temperature_c",
                     "wind": "env.wind_speed_m_s", "wind_dir": "env.wind_direction_deg"}
            for name, item_id in names.items():
                if name in env["values"]:
                    _set(facts, item_id, env["values"][name],
                         f"{src}, column {env['headers'][name]!r}")
            for name, value in env["values"].items():
                if name not in names:
                    _observe(facts, f"{env['headers'][name]} = {value:g}", src)
            inputs.append({"path": environment["source"].split("::")[0],
                           "role": "environmental logger data"})
    return {"facts": facts, "inputs": inputs}


# ── Other templates ───────────────────────────────────────────────────────────

def bridge_facts(images: list[str | Path], thermal_raster: str | Path | None = None) -> dict[str, Any]:
    """Bridge facts from the inspection imagery: equipment, plus observations."""
    facts: dict[str, Any] = {"fields": {}, "tables": {}, "observations": [], "evidence": []}
    summary = capture_summary(images)
    inputs = [{"path": str(p), "role": "inspection image"} for p in images]
    if summary["cameras"]:
        _set(facts, "ie.equipment", f"UAV imagery, camera {', '.join(summary['cameras'])}",
             f"EXIF Make/Model of {summary['count']} images")
    describe_capture(facts, summary, "bridge")
    if thermal_raster:
        import rasterio

        with rasterio.open(thermal_raster) as src:
            band = src.read(1, masked=True)
            _observe(facts, f"Thermal orthomosaic {Path(thermal_raster).name}: "
                     f"{src.width}x{src.height} px, {abs(src.transform.a):g} m pixels, "
                     f"{src.crs}, values {float(band.min()):.2f}..{float(band.max()):.2f} "
                     "(dataset states °C apparent temperature, no emissivity retained).",
                     "rasterio read of the raster")
        inputs.append({"path": str(thermal_raster), "role": "thermal orthomosaic"})
    if images:
        facts["evidence"].append({"path": str(images[len(images) // 2]),
                                  "caption": "Representative inspection frame"})
    return {"facts": facts, "inputs": inputs}


def roof_facts(images: list[str | Path]) -> dict[str, Any]:
    """Roof facts from UAV imagery: date, method (nadir vs oblique from the gimbal)."""
    facts: dict[str, Any] = {"fields": {}, "tables": {}, "observations": [], "evidence": []}
    summary = capture_summary(images)
    inputs = [{"path": str(p), "role": "inspection image"} for p in images]
    rows = summary["rows"]
    if summary["first"]:
        _set(facts, "general.datetime", f"{summary['first']} to {summary['last']}",
             f"EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
    pitched = [r for r in rows if r["gimbal_pitch_deg"] is not None]
    nadir = [r for r in pitched if r["gimbal_pitch_deg"] <= -80]
    if pitched:
        _set(facts, "general.method",
             f"UAV imagery ({', '.join(summary['cameras'])}): {len(nadir)} near-nadir frame(s) "
             f"(gimbal pitch <= -80°) and {len(pitched) - len(nadir)} oblique frame(s); "
             "nothing was walked or touched",
             "DJI XMP drone-dji:GimbalPitchDegree of each image")
        if nadir:
            facts["evidence"].append({"path": nadir[len(nadir) // 2]["path"],
                                      "caption": "Near-nadir roof frame"})
    describe_capture(facts, summary, "roof")
    return {"facts": facts, "inputs": inputs}


def tower_facts(images: list[str | Path]) -> dict[str, Any]:
    facts: dict[str, Any] = {"fields": {}, "tables": {}, "observations": [], "evidence": []}
    summary = capture_summary(images)
    if summary["first"]:
        _set(facts, "site.datetime", f"{summary['first']} to {summary['last']}",
             f"EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
    describe_capture(facts, summary, "tower")
    return {"facts": facts, "inputs": [{"path": str(p), "role": "inspection image"}
                                       for p in images]}


def capture_facts_for(template_key: str, images: list[str | Path]) -> dict[str, Any]:
    """The data-derived facts a project's imagery gives each template."""
    if template_key == "bridge_snbi":
        return bridge_facts(images)
    if template_key == "roof":
        return roof_facts(images)
    if template_key == "telecom_tower":
        return tower_facts(images)
    if template_key == "iec62446_3":
        facts: dict[str, Any] = {"fields": {}, "tables": {}, "observations": [], "evidence": []}
        summary = capture_summary(images)
        if summary["first"]:
            _set(facts, "inspection.datetime", f"{summary['first']} to {summary['last']}",
                 f"EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
        if summary["cameras"]:
            _set(facts, "camera.make_model", ", ".join(summary["cameras"]),
                 f"EXIF Make/Model of {summary['count']} images")
        describe_capture(facts, summary, "inspection")
        return {"facts": facts, "inputs": [{"path": str(p), "role": "inspection image"}
                                           for p in images]}
    raise ValueError(f"Unknown standards template {template_key!r}.")
