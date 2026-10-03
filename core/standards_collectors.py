"""Facts for standards-structured reports, read or computed from real files.

Every function here returns facts in the shape core.standards_reports.resolve() takes:
``{"fields": {item_id: {"value", "source", "origin"}}, "tables": {...},
"observations": [...], "evidence": [...], "figures": [...], "findings": [...],
"measurements": [...], "summary": {...}, "details": {...}}`` plus the list of files it
read (``inputs``), which go into the signed manifest.

A value is set only when a file states it or a documented computation on stated values
gives it; ``source`` says which file and which tag, column or formula. What the files do
not state is left out, so the report lists it on its "Before this report can be issued"
page. Software proposals (a CoA class from ΔT, a roof facet segmented from a DSM, a
detector box) are labelled ``suggested`` and count for nothing until a person confirms
them.
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def _empty() -> dict[str, Any]:
    return {"fields": {}, "tables": {}, "observations": [], "evidence": [], "figures": [],
            "findings": [], "measurements": [], "summary": {}, "details": {}}


def _set(facts: dict[str, Any], item_id: str, value: Any, source: str,
         origin: str = "data") -> None:
    facts.setdefault("fields", {})[item_id] = {"value": value, "source": source,
                                               "origin": origin}


def _observe(facts: dict[str, Any], text: str, source: str) -> None:
    facts.setdefault("observations", []).append({"text": text, "source": source})


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Operator-entered facts on top of data-derived ones (the operator's value wins)."""
    out = _empty()
    for key in out:
        value = base.get(key)
        if value is not None:
            out[key] = (dict(value) if isinstance(out[key], dict) else list(value))
    out["fields"].update(override.get("fields") or {})
    for key, rows in (override.get("tables") or {}).items():
        if rows:
            out["tables"][key] = list(rows)
    for key in ("observations", "evidence", "figures", "findings", "measurements"):
        out[key] += list(override.get(key) or [])
    out["summary"].update(override.get("summary") or {})
    out["details"].update(override.get("details") or {})
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


def _xmp_float(xmp: str, name: str) -> float | None:
    raw = _xmp_value(xmp, name)
    if raw in (None, ""):
        return None
    try:
        if "/" in raw:
            num, den = raw.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(raw)
    except ValueError:
        return None


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
        "relative_altitude_m": (_xmp_float(xmp, "drone-dji:RelativeAltitude")
                                if "drone-dji:RelativeAltitude" in xmp else
                                _xmp_float(xmp, "FLIR:MAVRelativeAltitude")),
        "lrf_distance_m": (_xmp_float(xmp, "drone-dji:LRFTargetDistance")
                           if _xmp_value(xmp, "drone-dji:LRFStatus") == "Normal" else None),
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


# ── Location ──────────────────────────────────────────────────────────────────

def format_latlon(lat: float, lon: float) -> str:
    return (f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.5f}° {'E' if lon >= 0 else 'W'}")


def nearest_saved_place(lat: float, lon: float, gazetteer_path: str | Path | None,
                        max_km: float = 2.0) -> dict[str, Any] | None:
    """The closest place in the offline gazetteer (``places.json``) within ``max_km``."""
    if not gazetteer_path or not Path(gazetteer_path).is_file():
        return None
    from .geocoding import OfflineProvider

    best, best_km = None, None
    for place in OfflineProvider(gazetteer_path=Path(gazetteer_path))._load():
        dlat = math.radians(place.latitude - lat)
        dlon = math.radians(place.longitude - lon)
        a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) *
             math.cos(math.radians(place.latitude)) * math.sin(dlon / 2) ** 2)
        km = 6371.0088 * 2 * math.asin(math.sqrt(a))
        if best_km is None or km < best_km:
            best, best_km = place, km
    if best is None or best_km > max_km:
        return None
    return {"name": best.name, "distance_km": round(best_km, 2),
            "source": f"offline gazetteer {Path(gazetteer_path).name}"}


def location_fact(summary: dict[str, Any], gazetteer_path: str | Path | None = None,
                  georeference: dict[str, Any] | None = None) -> tuple[str, str] | None:
    """(text, source) for a site location: georeference first, else the GPS centroid."""
    if georeference and georeference.get("lat") is not None:
        lat, lon = float(georeference["lat"]), float(georeference["lon"])
        how = georeference.get("source") or "project georeference"
    elif summary.get("centroid"):
        lat, lon = summary["centroid"]
        how = f"centroid of {summary['geotagged']} camera GPS positions (EXIF)"
    else:
        return None
    text = format_latlon(lat, lon)
    place = nearest_saved_place(lat, lon, gazetteer_path)
    if place:
        text = f"{place['name']} ({text}; {place['distance_km']:g} km from the saved place)"
        how += f"; place name from the {place['source']}"
    return text, how


def raster_centre_lonlat(path: str | Path) -> tuple[float, float, str] | None:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        b = src.bounds
        x, y = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)
        return lon, lat, str(src.crs)


# ── Camera registry, radiometric parameters, geometric resolution ─────────────

def match_camera_profile(meta: dict[str, Any], *, thermal: bool) -> tuple[Any, str]:
    """The registry profile a frame was taken with: focal length, width and the name."""
    from mission.cameras import all_profiles

    focal, width = meta.get("focal_length_mm"), meta.get("width")
    if not focal or not width:
        return None, "the frame records no focal length"
    tokens = {t for t in re.split(r"[^a-z0-9]+", f"{meta.get('make', '')} {meta.get('model', '')}"
                                   .lower()) if len(t) > 1}
    candidates = [p for p in all_profiles().values()
                  if p.thermal == thermal and abs(p.focal_mm - float(focal)) <= 0.05
                  and int(p.image_w_px) == int(width)]
    named = [p for p in candidates
             if tokens and all(t in re.sub(r"[^a-z0-9 ]", " ", p.name.lower()).split()
                               or t in p.key for t in tokens - {"dji", "flir"})]
    chosen = named or (candidates if len(candidates) == 1 else [])
    if len(chosen) != 1:
        return None, (f"no single registry profile for {meta.get('make')} {meta.get('model')} "
                      f"at {focal:g} mm, {width} px")
    return chosen[0], (f"camera registry profile '{chosen[0].key}' (matched on focal "
                       f"{focal:g} mm, {width} px, EXIF {meta.get('make')} {meta.get('model')})")


def geometric_resolution_cm(distance_m: float, pitch_um: float, focal_mm: float) -> float:
    """Size of one pixel on a surface ``distance_m`` away, in cm (pinhole model)."""
    return distance_m * pitch_um / focal_mm / 10.0


def radiometric_parameters(image: Any) -> dict[str, Any] | None:
    """Emissivity / reflected temperature the file itself records, with the reader named."""
    prov = getattr(image, "provenance", {}) or {}
    params = prov.get("measurement_params")
    if prov.get("decoder", "").startswith("dji_thermal_sdk") and params:
        return {"emissivity": params["emissivity"], "reflected_temp_c": params["reflected_temp_c"],
                "distance_m": params.get("distance_m"),
                "source": "measurement parameters stored in the R-JPEG, read by the DJI Thermal SDK"}
    cal = getattr(image, "calibration", None)
    if prov.get("decoder") == "flir_fff_cleanroom_planck" and cal is not None:
        return {"emissivity": round(float(cal.emissivity), 4),
                "reflected_temp_c": round(float(cal.reflected_temp_c), 2),
                "source": "FLIR FFF camera-info record of the radiometric JPEG"}
    return None


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


# ── IEC TS 62446-3 CoA suggestion ─────────────────────────────────────────────

# Annex C of IEC TS 62446-3:2017 gives *informative* examples of patterns with typical ΔT
# at ~1000 W/m²; it sets no fixed ΔT bands. Public summaries of those examples (e.g. the
# MIT-licensed pvdelta project, https://github.com/yusufzb/pvdelta) give: single hot cell
# 10-40 K -> CoA 2, > 40 K -> CoA 3; open-circuit module or sub-string (uniformly warm,
# +2..+7 K) -> CoA 2; broken glass -> CoA 3. These rules only ever produce a SUGGESTION.
COA_RULE_SOURCE = ("OpenDroneKit suggestion from the informative examples of IEC TS 62446-3 "
                   "Annex C as publicly summarised (pvdelta); not a classification")
PATTERN_ALIASES = {"hotspot": "single_hot_cell", "hot_spot": "single_hot_cell",
                   "single_hot_cell": "single_hot_cell", "string_fault": "open_substring",
                   "open_substring": "open_substring", "open_module": "open_substring",
                   "broken_glass": "broken_glass"}


def suggest_coa(delta_max_k: float | None, pattern: str | None = None) -> dict[str, Any] | None:
    """A labelled CoA suggestion, or None when no rule applies (then a person classifies)."""
    key = PATTERN_ALIASES.get(str(pattern or "").lower().strip())
    if key == "broken_glass":
        return {"coa": 3, "basis": "pattern 'broken glass' (Annex C example: any ΔT)"}
    if key == "open_substring":
        return {"coa": 2, "basis": "pattern 'open-circuit module / sub-string' "
                                   "(Annex C example, typically +2..+7 K)"}
    if delta_max_k is None:
        return None
    if key == "single_hot_cell":
        if delta_max_k > 40:
            return {"coa": 3, "basis": f"single hot cell, ΔTmax {delta_max_k:.1f} K > 40 K"}
        if delta_max_k >= 10:
            return {"coa": 2, "basis": f"single hot cell, ΔTmax {delta_max_k:.1f} K in 10-40 K"}
    if pattern in (None, "") and delta_max_k < 10:
        return {"coa": 1, "basis": f"no pattern identified and ΔTmax {delta_max_k:.1f} K is "
                                   "below the 10 K lower bound of the hot-cell example"}
    return None


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
                          rgb_suffix: tuple[str, str] | None = None,
                          camera_distance_m: float | None = None,
                          gazetteer_path: str | Path | None = None) -> dict[str, Any]:
    """IEC TS 62446-3 facts from decoded radiometric frames and operator module outlines.

    ``regions_file``: ``{"region_source", "frames": {name: {"regions": [...]}}}`` (the
    shape of tests/fixtures/thedpv_module_regions.json). ``environment``:
    ``{"source", "columns": {"poa", "air", "wind", "wind_dir", "cloud_okta", "soiling"}
    -> header prefixes}``. Module temperatures come from
    core.india_thermal.module_temperatures_in_image; the report's ΔT is the module median
    minus the median of the frame's module medians (the regular-spot comparison of clause
    8 s). A CoA is only *suggested* (see suggest_coa); the thermographer confirms it.
    """
    import json

    from .india_thermal import module_temperatures_in_image
    from .thermal import load_radiometric_file

    facts = _empty()
    facts["tables"]["anomalies"] = []
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
        facts["details"]["date"] = first.date().isoformat()

    summary = capture_summary(frame_paths)
    located = location_fact(summary, gazetteer_path)
    if located:
        _set(facts, "inspection.location", located[0], located[1])
        facts["details"]["site"] = located[0]

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

    # NETD and geometric resolution from the camera registry.
    profile, why = match_camera_profile(metas[0], thermal=True) if metas else (None, "")
    if profile is not None and profile.netd_mk:
        _set(facts, "camera.netd_k", profile.netd_mk / 1000.0,
             f"{why}; NETD < {profile.netd_mk:g} mK, upper bound from {profile.netd_source}")
    distance, distance_src = None, ""
    alts = [m["lrf_distance_m"] or m["relative_altitude_m"] for m in metas]
    alts = [a for a in alts if a and a > 1.0]
    if alts:
        distance = statistics.median(alts)
        distance_src = (f"median camera-to-target distance {distance:.1f} m from the frames' "
                        "laser range finder / relative altitude XMP")
    elif camera_distance_m:
        distance = float(camera_distance_m)
        distance_src = f"camera-to-module distance {distance:g} m entered for this inspection"
    if profile is not None:
        facts["details"]["camera_profile"] = profile.key
        if distance:
            gsd = geometric_resolution_cm(distance, profile.pixel_pitch_um, profile.focal_mm)
            _set(facts, "camera.geometric_resolution_cm_px", round(gsd, 2),
                 f"computed: distance × pixel pitch / focal length = {distance:g} m × "
                 f"{profile.pixel_pitch_um:.1f} µm / {profile.focal_mm:g} mm; {distance_src}; "
                 f"pitch and focal from {why}",
                 origin="data" if alts else "operator")
        else:
            facts["details"]["geometric_formula"] = (
                f"distance × {profile.pixel_pitch_um:.1f} µm / {profile.focal_mm:g} mm "
                f"(IFOV {profile.pixel_pitch_um / profile.focal_mm:.2f} mrad)")

    decoder_notes: set[str] = set()
    figures_rows: list[dict[str, Any]] = []
    listed: list[str] = []
    params_found: dict[str, Any] | None = None
    for path, meta in zip(frame_paths, metas):
        entry = (regions.get("frames") or {}).get(path.name)
        if not entry:
            _observe(facts, f"{path.name}: no module outlines supplied, so no module was "
                     "evaluated in this frame.", "module outlines file")
            continue
        image = load_radiometric_file(path)
        params_found = params_found or radiometric_parameters(image)
        decoder_notes.add(str(image.provenance.get("validation") or ""))
        result = module_temperatures_in_image(
            path, entry["regions"], Path(work_dir) / path.stem,
            region_source=str(regions.get("region_source") or ""),
            # Only so the function runs; its severity labels are not used in this report.
            warning_delta_c=1e6, critical_delta_c=1e6, hot_cell_delta_c=1e6,
            threshold_basis="not used: IEC TS 62446-3 CoA is assigned by the thermographer")
        rows = json.loads(Path(result.artifact_paths[0]).read_text(encoding="utf-8"))["modules"]
        when, how = _frame_time(meta, path)
        medians = sorted(float(r["median_c"]) for r in rows)
        reference = ((medians[(len(medians) - 1) // 2] + medians[len(medians) // 2]) / 2
                     if medians else None)
        regions_by_id = {r["module_id"]: r for r in entry["regions"]}
        frame_regions = []
        for row in rows:
            delta = row["median_c"] - reference
            delta_max = row["max_c"] - reference
            key = f"{path.name}:{row['module_id']}"
            suggestion = suggest_coa(delta_max, None)
            listed.append(f"{row['module_id']} ({path.name})")
            facts["tables"]["anomalies"].append({
                "key": key,
                "object": f"{row['module_id']} in {path.name} (positional; plant module "
                          "identity not recorded)",
                "image_file": f"{path.name}, {when.isoformat(sep=' ') if when else 'time not recorded'}",
                "temperature_c": f"median {row['median_c']:.2f} °C (max {row['max_c']:.2f} °C)",
                "reference_c": f"{reference:.2f} °C (median of {len(rows)} module medians "
                               "in the frame)",
                "delta_t_k": f"{delta:+.2f} K",
                "source": f"{image.provenance.get('decoder', 'decoder')} decode of {path.name}; "
                          f"outline by {str(regions.get('region_source') or 'unnamed').split(':')[0]}",
                "suggestions": ({"coa": {"value": f"CoA {suggestion['coa']}",
                                         "basis": suggestion["basis"],
                                         "source": COA_RULE_SOURCE}} if suggestion else {}),
                "extra": {"median_c": row["median_c"], "max_c": row["max_c"],
                          "reference_c": reference, "delta_k": delta, "delta_max_k": delta_max,
                          "module_id": row["module_id"], "frame": path.name},
            })
            region = regions_by_id.get(row["module_id"]) or {}
            frame_regions.append({"polygon_px": region.get("polygon_px"),
                                  "label": f"{row['module_id']}  ΔT {delta:+.1f} K"})
        figures_rows.append({"path": str(path), "regions": frame_regions, "frame": path.name})
        _observe(facts, f"{path.name}: frame median {image.stats.get('median_c', float('nan')):.2f} °C, "
                 f"range {image.stats.get('min_c', float('nan')):.2f}-"
                 f"{image.stats.get('max_c', float('nan')):.2f} °C over {len(rows)} outlined module(s).",
                 f"{image.provenance.get('decoder')} ({image.provenance.get('temperature_kind')})")
        if rgb_suffix:
            rgb = path.with_name(path.name.replace(*rgb_suffix))
            if rgb.is_file():
                facts["evidence"].append({"path": str(rgb), "caption":
                                          f"Visual photo taken with {path.name}"})
    if listed:
        _set(facts, "scope.inspected_components",
             f"{len(listed)} module region(s): " + "; ".join(listed)
             + " (positional IDs; plant identity not recorded)",
             f"module outlines file {Path(regions_file).name}")
    if params_found:
        _set(facts, "camera.emissivity", params_found["emissivity"], params_found["source"])
        _set(facts, "camera.reflected_temperature_c", params_found["reflected_temp_c"],
             params_found["source"])
    for note in sorted(n for n in decoder_notes if n):
        _observe(facts, note, "thermal decoder provenance")
    for limit in regions.get("limits") or []:
        _observe(facts, limit, "module outlines file")

    env_values: dict[str, Any] = {}
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
                     "wind": "env.wind_speed_m_s", "wind_dir": "env.wind_direction_deg",
                     "cloud_okta": "env.cloud_okta"}
            for name, item_id in names.items():
                if name in env["values"]:
                    _set(facts, item_id, env["values"][name],
                         f"{src}, column {env['headers'][name]!r}")
            env_values = env["values"]
            if "soiling" in env["values"]:
                photos = len(facts["evidence"])
                _set(facts, "env.soiling",
                     f"soiling ratio {env['values']['soiling']:g} % as logged"
                     + (f"; {photos} paired visual photo(s) attached as evidence" if photos else ""),
                     f"{src}, column {env['headers']['soiling']!r}"
                     + ("; photos: the *_8b.JPG frames paired with the thermal frames" if photos else ""))
            for name, value in env["values"].items():
                if name not in names and name != "soiling":
                    _observe(facts, f"{env['headers'][name]} = {value:g}", src)
            inputs.append({"path": environment["source"].split("::")[0],
                           "role": "environmental logger data"})
    elif facts["evidence"]:
        _set(facts, "env.soiling", f"{len(facts['evidence'])} visual photo(s) of the inspected "
             "modules attached as soiling evidence (no soiling sensor reading)",
             "the visual frames paired with the thermal frames", origin="suggested")

    _iec_visuals(facts, figures_rows, env_values)
    return {"facts": facts, "inputs": inputs}


def _iec_visuals(facts: dict[str, Any], frames: list[dict[str, Any]],
                 env: dict[str, Any]) -> None:
    rows = facts["tables"].get("anomalies") or []
    if facts["evidence"]:
        facts["figures"].append({"role": "hero", "spec": {"type": "photo",
                                                          "path": facts["evidence"][0]["path"]},
                                 "caption": facts["evidence"][0]["caption"]})
    elif frames:
        facts["figures"].append({"role": "hero", "spec": {"type": "thermal",
                                                          "path": frames[0]["path"]},
                                 "caption": f"Thermal frame {frames[0]['frame']}"})
    for frame in frames:
        facts["figures"].append({
            "role": "overview",
            "spec": {"type": "thermal", "path": frame["path"], "regions": frame["regions"]},
            "caption": f"{frame['frame']}: decoded apparent temperature with the outlined "
                       "module regions and their ΔT against the frame reference."})
    for row in rows:
        extra = row["extra"]
        sug = (row.get("suggestions") or {}).get("coa")
        facts["findings"].append({
            "key": row["key"],
            "title": f"{extra['module_id'].replace('_', ' ')} — {extra['frame']}",
            "severity": ({"label": f"{sug['value']} suggested", "level": int(sug["value"][-1]),
                          "confirmed": False} if sug else
                         {"label": "Not classified", "level": None, "confirmed": False}),
            "figure": {"type": "thermal", "path": next(f["path"] for f in frames
                                                       if f["frame"] == extra["frame"]),
                       "regions": [r for f in frames if f["frame"] == extra["frame"]
                                   for r in f["regions"]],
                       "focus": extra["module_id"]},
            "facts": [("Module median", f"{extra['median_c']:.2f} °C"),
                      ("Module max", f"{extra['max_c']:.2f} °C"),
                      ("Reference (frame)", f"{extra['reference_c']:.2f} °C"),
                      ("ΔT median", f"{extra['delta_k']:+.2f} K"),
                      ("ΔT max", f"{extra['delta_max_k']:+.2f} K")],
            "note": (f"Suggestion basis: {sug['basis']}. Confirm or change in review."
                     if sug else "No rule applies; the thermographer classifies it."),
            "source": row["source"],
        })
    deltas = [r["extra"]["delta_k"] for r in rows]
    kpis = [{"label": "Thermal frames analysed", "value": str(len(frames))},
            {"label": "Module regions evaluated", "value": str(len(rows))}]
    if deltas:
        kpis.append({"label": "Largest |ΔT| (median)", "value": f"{max(abs(d) for d in deltas):.1f} K"})
    if "poa" in env:
        kpis.append({"label": "In-plane irradiance", "value": f"{env['poa']:g} W/m²"})
    facts["summary"]["kpis"] = kpis
    counts: dict[str, int] = {}
    for f in facts["findings"]:
        counts[f["severity"]["label"]] = counts.get(f["severity"]["label"], 0) + 1
    facts["summary"]["severity_counts"] = counts
    if rows:
        top = max(rows, key=lambda r: abs(r["extra"]["delta_k"]))
        facts["fields"]["results.summary"] = {
            "value": (f"{len(rows)} module region(s) in {len(frames)} frame(s) evaluated; "
                      f"module-median ΔT ranges {min(deltas):+.2f} to {max(deltas):+.2f} K, the "
                      f"largest at {top['extra']['module_id']} in {top['extra']['frame']}."),
            "source": "drafted by OpenDroneKit from the ΔT table; the thermographer confirms",
            "origin": "suggested"}


# ── Other templates ───────────────────────────────────────────────────────────

def _positions(summary: dict[str, Any], highlight: set[str] | None = None) -> list[dict[str, Any]]:
    return [{"lat": r["gps"][0], "lon": r["gps"][1], "name": r["name"],
             "highlight": bool(highlight and r["name"] in highlight)}
            for r in summary["rows"] if r["gps"]]


def _central_frame(summary: dict[str, Any]) -> Path | None:
    """The geotagged frame nearest the centroid of all camera positions."""
    if not summary.get("centroid"):
        return None
    lat, lon = summary["centroid"]
    rows = [r for r in summary["rows"] if r["gps"]]
    best = min(rows, key=lambda r: (r["gps"][0] - lat) ** 2 +
               ((r["gps"][1] - lon) * math.cos(math.radians(lat))) ** 2)
    return Path(best["path"])


def bridge_facts(images: list[str | Path], thermal_raster: str | Path | None = None, *,
                 detections: dict[str, Any] | None = None,
                 gazetteer_path: str | Path | None = None) -> dict[str, Any]:
    """Bridge facts from the inspection imagery, the thermal orthomosaic and detector runs.

    ``detections``: ``{"per_frame": path, "image_dir": folder, "mode": key in each row,
    "models": [{"name", "sha256", "digest_source"}]}`` -- the structural detector's
    per-frame boxes. They become *proposals* in the report, never findings.
    """
    import json

    facts = _empty()
    summary = capture_summary(images)
    inputs = [{"path": str(p), "role": "inspection image"} for p in images]
    models: list[dict[str, Any]] = []
    if summary["cameras"]:
        _set(facts, "ie.equipment", f"UAV imagery, camera {', '.join(summary['cameras'])}",
             f"EXIF Make/Model of {summary['count']} images")
    if summary["first"]:
        _set(facts, "ie.begin_date", summary["first"][:10],
             f"earliest EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
        _set(facts, "ie.completion_date", summary["last"][:10],
             f"latest EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
        facts["details"]["date"] = summary["first"][:10]
    describe_capture(facts, summary, "bridge")
    centre = None
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
        centre = raster_centre_lonlat(thermal_raster)
    if centre:
        lon, lat, crs = centre
        how = (f"centre of the orthomosaic footprint {Path(thermal_raster).name} ({crs} -> "
               "WGS84); not the agency's surveyed B.L.05/06 record — confirm")
    elif summary.get("centroid"):
        lat, lon = summary["centroid"]
        how = f"centroid of {summary['geotagged']} camera GPS positions (EXIF)"
    if centre or summary.get("centroid"):
        _set(facts, "loc.latitude", f"{lat:.6f}", how)
        _set(facts, "loc.longitude", f"{lon:.6f}", how)
        located = location_fact({"centroid": [lat, lon], "geotagged": 1}, gazetteer_path)
        facts["details"]["site"] = located[0] if located else format_latlon(lat, lon)

    # Ground sample distance on the structure from the laser range finder.
    rows = summary["rows"]
    ranges = [r["lrf_distance_m"] for r in rows if r["lrf_distance_m"]]
    if ranges and rows:
        from .geo import SENSOR_WIDTH_MM

        sensor = next((w for k, w in SENSOR_WIDTH_MM.items()
                       if rows[0]["model"].lower().lstrip("z") in k.replace(" ", "")), None)
        if sensor and rows[0]["focal_length_mm"]:
            pitch = sensor * 1000.0 / rows[0]["width"]
            gsd = geometric_resolution_cm(statistics.median(ranges), pitch, rows[0]["focal_length_mm"])
            facts["measurements"].append({
                "title": "Image resolution on the structure",
                "columns": ["Quantity", "Value", "Source"],
                "rows": [["Median laser range to target", f"{statistics.median(ranges):.1f} m",
                          f"XMP drone-dji:LRFTargetDistance of {len(ranges)} frames"],
                         ["Pixel pitch", f"{pitch:.2f} µm",
                          f"sensor width {sensor:g} mm (core/geo.py SENSOR_WIDTH_MM) / "
                          f"{rows[0]['width']} px"],
                         ["Focal length", f"{rows[0]['focal_length_mm']:g} mm", "EXIF FocalLength"],
                         ["Ground sample distance", f"{gsd:.2f} cm/px",
                          "range × pitch / focal length"]],
                "status": "derived"})

    proposals: dict[str, int] = {}
    if detections and Path(detections["per_frame"]).is_file():
        per_frame = json.loads(Path(detections["per_frame"]).read_text(encoding="utf-8"))
        inputs.append({"path": str(detections["per_frame"]), "role": "detector output (boxes)"})
        models += list(detections.get("models") or [])
        mode = detections.get("mode", "structural_letterbox")
        image_dir = Path(detections["image_dir"])
        boxes = []
        for frame in per_frame:
            for det in frame.get(mode) or []:
                proposals[det["label"]] = proposals.get(det["label"], 0) + 1
                boxes.append({**det, "frame": frame["frame"]})
        flagged = {b["frame"] for b in boxes}
        _observe(facts, f"Structural detector ({mode}): {len(boxes)} proposal(s) in "
                 f"{len(flagged)} of {len(per_frame)} frames: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(proposals.items())) + ". The "
                 "dataset carries no defect labels, so the detector's precision on it is unknown.",
                 f"{Path(detections['per_frame']).name}")
        best: dict[str, list[dict[str, Any]]] = {}
        for box in sorted(boxes, key=lambda b: -b["confidence"]):
            best.setdefault(box["label"], [])
            if len(best[box["label"]]) < 2:
                best[box["label"]].append(box)
        for label, picks in sorted(best.items()):
            for box in picks:
                frame_path = image_dir / box["frame"]
                facts["findings"].append({
                    "key": f"{box['frame']}:{label}:{box['bbox_xyxy']}",
                    "title": f"{label} — {box['frame']}",
                    "severity": {"label": "Model proposal", "level": None, "confirmed": False},
                    "figure": {"type": "boxes", "path": str(frame_path),
                               "boxes": [{"bbox_xyxy": box["bbox_xyxy"], "label":
                                          f"{label} {box['confidence']:.2f}"}],
                               "crop_to_boxes": True},
                    "facts": [("Class", label), ("Detector confidence", f"{box['confidence']:.2f}"),
                              ("Box (px)", "x {0}-{2}, y {1}-{3}".format(*box["bbox_xyxy"]))],
                    "note": "A model proposal, not a finding: the inspector accepts or rejects "
                            "it and records accepted ones in the deficiencies table.",
                    "source": f"structural detector ({mode}) output "
                              f"{Path(detections['per_frame']).name}"})
        facts["summary"]["severity_counts"] = {f"{k} (proposals)": v
                                               for k, v in sorted(proposals.items())}
        facts["summary"]["highlight_frames"] = sorted(flagged)
    positions = _positions(summary, set(facts["summary"].get("highlight_frames") or []))
    if images:
        hero = _central_frame(summary) or Path(images[len(images) // 2])
        facts["figures"].append({"role": "hero", "spec": {"type": "photo", "path": str(hero)},
                                 "caption": f"Frame {Path(hero).name}, the one taken nearest the "
                                            "centre of the surveyed area"})
    if thermal_raster:
        facts["figures"].append({
            "role": "map", "spec": {"type": "raster_map", "path": str(thermal_raster),
                                    "points": positions, "value_label": "Apparent temperature (°C)"},
            "caption": f"Thermal orthomosaic {Path(thermal_raster).name} with the camera positions "
                       "(dots; ringed where the detector proposed a defect)."})
    elif positions:
        facts["figures"].append({"role": "map", "spec": {"type": "points_map", "points": positions},
                                 "caption": "Camera positions from EXIF GPS."})
    kpis = [{"label": "Frames", "value": str(summary["count"])}]
    if summary["first"]:
        kpis.append({"label": "Flown", "value": summary["first"][:10]})
    if proposals:
        kpis.append({"label": "Detector proposals", "value": str(sum(proposals.values()))})
    if facts["measurements"]:
        kpis.append({"label": "GSD on structure", "value": facts["measurements"][0]["rows"][-1][1]})
    facts["summary"]["kpis"] = kpis
    return {"facts": facts, "inputs": inputs, "models": models}


def roof_facts(images: list[str | Path], *, dsm: str | Path | None = None,
               dtm: str | Path | None = None, dsm_note: str = "",
               gazetteer_path: str | Path | None = None) -> dict[str, Any]:
    """Roof facts from UAV imagery (date, method, site) and, given a DSM, roof facets."""
    facts = _empty()
    summary = capture_summary(images)
    inputs = [{"path": str(p), "role": "inspection image"} for p in images]
    rows = summary["rows"]
    if summary["first"]:
        _set(facts, "general.datetime", f"{summary['first']} to {summary['last']}",
             f"EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
        facts["details"]["date"] = summary["first"][:10]
    located = location_fact(summary, gazetteer_path)
    if located:
        _set(facts, "general.address", located[0], located[1])
        facts["details"]["site"] = located[0]
    pitched = [r for r in rows if r["gimbal_pitch_deg"] is not None]
    nadir = [r for r in pitched if r["gimbal_pitch_deg"] <= -80]
    if pitched:
        _set(facts, "general.method",
             f"UAV imagery ({', '.join(summary['cameras'])}): {len(nadir)} near-nadir frame(s) "
             f"(gimbal pitch <= -80°) and {len(pitched) - len(nadir)} oblique frame(s); "
             "nothing was walked or touched",
             "DJI XMP drone-dji:GimbalPitchDegree of each image")
    describe_capture(facts, summary, "roof")
    if pitched and not nadir:
        steepest = min(r["gimbal_pitch_deg"] for r in pitched)
        line = (f"None of the {len(pitched)} frames looks down on the roof: the steepest gimbal "
                f"pitch is {steepest:.1f}°, so the imagery shows facades. A near-nadir roof pass "
                "is needed before the roof surfaces can be assessed.")
        _observe(facts, line, "DJI XMP drone-dji:GimbalPitchDegree of each image")
        facts["summary"].setdefault("lines", []).append(line)
    # The steepest-looking frames show the most roof: nadir ones first, else the most
    # downward-pitched obliques.
    steep = sorted(pitched, key=lambda r: r["gimbal_pitch_deg"])
    if steep:
        hero = steep[0]
        kind = "Near-nadir" if hero["gimbal_pitch_deg"] <= -80 else "Steepest oblique"
        facts["figures"].append({"role": "hero", "spec": {"type": "photo", "path": hero["path"]},
                                 "caption": f"{kind} frame {hero['name']} (gimbal pitch "
                                            f"{hero['gimbal_pitch_deg']:.1f}°)"})
        picks = steep[:: max(1, len(steep) // 4)][:4]
        for r in picks:
            facts["evidence"].append({"path": r["path"], "caption": f"{r['name']}, gimbal pitch "
                                                                    f"{r['gimbal_pitch_deg']:.1f}°"})
    positions = _positions(summary)
    if dsm and dtm:
        from .surface_planes import PlaneFitError, segment_roof_facets

        inputs += [{"path": str(dsm), "role": "digital surface model"},
                   {"path": str(dtm), "role": "digital terrain model"}]
        try:
            planes = segment_roof_facets(dsm, dtm)
        except PlaneFitError as exc:
            _observe(facts, f"Roof facets not measured: {exc}", Path(dsm).name)
            planes = None
        if planes:
            rows_out = [[s["label"], f"{s['tilt_deg']:.1f}°",
                         f"{s['azimuth_deg']:.0f}° ({s['facing']})" if s["azimuth_deg"] is not None
                         else s["facing"], f"{s['area_m2']:.0f} m²", f"{s['rmse_m']:.2f} m",
                         "yes" if s["accepted"] else f"no: {s['reason']}"]
                        for s in planes["surfaces"]]
            facts["measurements"].append({
                "title": "Roof facets measured from the DSM (suggested)",
                "columns": ["Facet", "Pitch", "Facing (true N)", "Area", "Fit RMSE", "Plane fit OK"],
                "rows": rows_out, "status": "suggested",
                "method": planes["method"] + (f" {dsm_note}" if dsm_note else ""),
                "source": f"{Path(dsm).name} / {Path(dtm).name}, {planes['cell_size_m']:g} m cells"})
        marks = [{"x": s["centre_xy"][0], "y": s["centre_xy"][1], "label": s["label"]}
                 for s in (planes or {}).get("surfaces", []) if s.get("centre_xy")]
        facts["figures"].append({"role": "map", "spec": {"type": "points_map", "points": positions,
                                                         "background": str(dsm), "marks": marks},
                                 "caption": "Camera positions (EXIF GPS) over the DSM "
                                            f"{Path(dsm).name}."})
    elif positions:
        facts["figures"].append({"role": "map", "spec": {"type": "points_map", "points": positions},
                                 "caption": "Camera positions from EXIF GPS."})
    facts["summary"]["kpis"] = [{"label": "Frames", "value": str(summary["count"])},
                                {"label": "Near-nadir / oblique",
                                 "value": f"{len(nadir)} / {len(pitched) - len(nadir)}"}]
    if summary["first"]:
        facts["summary"]["kpis"].append({"label": "Flown", "value": summary["first"][:10]})
    return {"facts": facts, "inputs": inputs}


def tower_facts(images: list[str | Path], *, gazetteer_path: str | Path | None = None) -> dict[str, Any]:
    facts = _empty()
    summary = capture_summary(images)
    if summary["first"]:
        _set(facts, "site.datetime", f"{summary['first']} to {summary['last']}",
             f"EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
        facts["details"]["date"] = summary["first"][:10]
    located = location_fact(summary, gazetteer_path)
    if located:
        _set(facts, "site.location", located[0], located[1])
        facts["details"]["site"] = located[0]
    describe_capture(facts, summary, "tower")
    if images:
        facts["figures"].append({"role": "hero", "spec": {"type": "photo",
                                                          "path": str(images[len(images) // 2])},
                                 "caption": "Representative frame"})
    return {"facts": facts, "inputs": [{"path": str(p), "role": "inspection image"}
                                       for p in images]}


def capture_facts_for(template_key: str, images: list[str | Path], *,
                      gazetteer_path: str | Path | None = None,
                      dsm: str | Path | None = None, dtm: str | Path | None = None) -> dict[str, Any]:
    """The data-derived facts a project's imagery gives each template."""
    if template_key == "bridge_snbi":
        return bridge_facts(images, gazetteer_path=gazetteer_path)
    if template_key == "roof":
        return roof_facts(images, dsm=dsm, dtm=dtm, gazetteer_path=gazetteer_path)
    if template_key == "telecom_tower":
        return tower_facts(images, gazetteer_path=gazetteer_path)
    if template_key == "iec62446_3":
        facts = _empty()
        summary = capture_summary(images)
        if summary["first"]:
            _set(facts, "inspection.datetime", f"{summary['first']} to {summary['last']}",
                 f"EXIF DateTimeOriginal of {summary['count']} images (camera clock)")
            facts["details"]["date"] = summary["first"][:10]
        if summary["cameras"]:
            _set(facts, "camera.make_model", ", ".join(summary["cameras"]),
                 f"EXIF Make/Model of {summary['count']} images")
        located = location_fact(summary, gazetteer_path)
        if located:
            _set(facts, "inspection.location", located[0], located[1])
            facts["details"]["site"] = located[0]
        describe_capture(facts, summary, "inspection")
        if images:
            facts["figures"].append({"role": "hero", "spec": {"type": "photo",
                                                              "path": str(images[len(images) // 2])},
                                     "caption": "Representative frame"})
        return {"facts": facts, "inputs": [{"path": str(p), "role": "inspection image"}
                                           for p in images]}
    raise ValueError(f"Unknown standards template {template_key!r}.")
