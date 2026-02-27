"""Multi-sensor capture bundle loading, synchronization, and QA utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_LIDAR_EXTENSIONS = {".xyz", ".txt", ".csv", ".npy", ".npz"}


@dataclass
class MultiSensorConfig:
    """Configuration for bundle validation and sensor fusion checks."""

    max_frame_delta_ms: float = 120.0
    max_lidar_delta_ms: float = 220.0
    max_imu_delta_ms: float = 50.0
    max_gnss_delta_ms: float = 250.0
    thermal_hotspot_threshold_c: float = 65.0
    thermal_min_hotspot_area_px: int = 60
    lidar_voxel_size_m: float = 0.08
    lidar_preview_max_points: int = 150_000


@dataclass
class CalibrationProfile:
    """Calibration profile per payload (intrinsics, extrinsics, thermal params)."""

    payload_id: str
    version: int
    created_utc: str
    camera_intrinsics: dict[str, dict[str, float]]
    extrinsics: dict[str, dict[str, Any]]
    thermal_calibration: dict[str, float]
    notes: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_path(root: Path, value: str) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _ensure_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    return []


def _normalize_timestamp(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_frame_stream(items: Iterable[Any], root: Path) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            rows.append(
                {
                    "path": _resolve_path(root, item),
                    "timestamp_ms": float(idx * 100.0),
                    "id": Path(item).stem,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            continue
        rows.append(
            {
                "path": _resolve_path(root, raw_path),
                "timestamp_ms": _normalize_timestamp(item.get("timestamp_ms"), idx * 100.0),
                "id": str(item.get("id") or Path(raw_path).stem),
            }
        )
    rows.sort(key=lambda r: float(r["timestamp_ms"]))
    return rows


def _normalize_multispectral_stream(items: Iterable[Any], root: Path) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            continue
        rows.append(
            {
                "path": _resolve_path(root, raw_path),
                "timestamp_ms": _normalize_timestamp(item.get("timestamp_ms"), idx * 100.0),
                "band": str(item.get("band") or "unknown").lower(),
                "id": str(item.get("id") or Path(raw_path).stem),
            }
        )
    rows.sort(key=lambda r: float(r["timestamp_ms"]))
    return rows


def _normalize_imu(items: Iterable[Any]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        accel = item.get("accel_m_s2") or item.get("accel") or [0.0, 0.0, 0.0]
        gyro = item.get("gyro_rad_s") or item.get("gyro") or [0.0, 0.0, 0.0]
        if not isinstance(accel, list) or len(accel) < 3:
            accel = [0.0, 0.0, 0.0]
        if not isinstance(gyro, list) or len(gyro) < 3:
            gyro = [0.0, 0.0, 0.0]
        rows.append(
            {
                "timestamp_ms": _normalize_timestamp(item.get("timestamp_ms"), idx * 10.0),
                "accel_m_s2": [float(accel[0]), float(accel[1]), float(accel[2])],
                "gyro_rad_s": [float(gyro[0]), float(gyro[1]), float(gyro[2])],
            }
        )
    rows.sort(key=lambda r: float(r["timestamp_ms"]))
    return rows


def _normalize_gnss(items: Iterable[Any]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "timestamp_ms": _normalize_timestamp(item.get("timestamp_ms"), idx * 100.0),
                "lat": float(item.get("lat", 0.0)),
                "lon": float(item.get("lon", 0.0)),
                "alt_m": float(item.get("alt_m", item.get("alt", 0.0))),
                "fix": str(item.get("fix", "unknown")),
            }
        )
    rows.sort(key=lambda r: float(r["timestamp_ms"]))
    return rows


def _normalize_lidar_packets(items: Iterable[Any], root: Path) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            p = _resolve_path(root, item)
            rows.append(
                {
                    "path": p,
                    "timestamp_ms": float(idx * 120.0),
                    "id": Path(item).stem,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            continue
        rows.append(
            {
                "path": _resolve_path(root, raw_path),
                "timestamp_ms": _normalize_timestamp(item.get("timestamp_ms"), idx * 120.0),
                "id": str(item.get("id") or Path(raw_path).stem),
            }
        )
    rows.sort(key=lambda r: float(r["timestamp_ms"]))
    return rows


def load_capture_bundle(path: str | Path) -> dict:
    """Load a capture bundle manifest and resolve relative paths."""
    source = Path(path)
    payload = _read_json(source)
    root = source.parent

    bundle_id = str(payload.get("bundle_id") or source.stem)
    created = str(payload.get("created_utc") or datetime.now(timezone.utc).isoformat())

    rgb_frames = _normalize_frame_stream(_ensure_list(payload.get("rgb_frames")), root)
    thermal_frames = _normalize_frame_stream(_ensure_list(payload.get("thermal_frames")), root)
    multispectral_frames = _normalize_multispectral_stream(_ensure_list(payload.get("multispectral_frames")), root)
    lidar_packets = _normalize_lidar_packets(_ensure_list(payload.get("lidar_packets")), root)
    imu = _normalize_imu(_ensure_list(payload.get("imu")))
    gnss = _normalize_gnss(_ensure_list(payload.get("gnss")))

    camera_intrinsics = payload.get("camera_intrinsics", {})
    if not isinstance(camera_intrinsics, dict):
        camera_intrinsics = {}

    thermal_calibration = payload.get("thermal_calibration", {})
    if not isinstance(thermal_calibration, dict):
        thermal_calibration = {}

    calibration_profile_path = str(payload.get("calibration_profile_path", "")).strip()
    if calibration_profile_path:
        calibration_profile_path = _resolve_path(root, calibration_profile_path)

    return {
        "bundle_id": bundle_id,
        "payload_id": str(payload.get("payload_id") or "payload"),
        "created_utc": created,
        "root_dir": str(root.resolve()),
        "rgb_frames": rgb_frames,
        "thermal_frames": thermal_frames,
        "multispectral_frames": multispectral_frames,
        "lidar_packets": lidar_packets,
        "imu": imu,
        "gnss": gnss,
        "camera_intrinsics": camera_intrinsics,
        "thermal_calibration": thermal_calibration,
        "calibration_profile_path": calibration_profile_path,
        "meta": payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {},
    }


def write_capture_bundle(path: str | Path, bundle: dict) -> str:
    """Write capture bundle manifest JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    return str(target)


def build_capture_bundle_from_folders(
    root_dir: str | Path,
    output_manifest_path: str | Path | None = None,
    payload_id: str = "payload",
    start_timestamp_ms: float = 0.0,
    frame_interval_ms: float = 200.0,
) -> dict:
    """
    Build a capture bundle from a conventional folder structure.

    Expected subfolders (optional except rgb):
    - rgb/
    - thermal/
    - multispectral/
    - lidar/
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Bundle root does not exist: {root}")

    def _collect_images(folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        rows = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
        rows.sort()
        return rows

    def _collect_lidar(folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        rows = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_LIDAR_EXTENSIONS]
        rows.sort()
        return rows

    rgb_files = _collect_images(root / "rgb")
    thermal_files = _collect_images(root / "thermal")
    ms_files = _collect_images(root / "multispectral")
    lidar_files = _collect_lidar(root / "lidar")

    if not rgb_files:
        raise ValueError("No RGB files found under <root>/rgb")

    def _make_frames(paths: list[Path]) -> list[dict]:
        frames = []
        for idx, p in enumerate(paths):
            frames.append(
                {
                    "path": str(p.resolve()),
                    "timestamp_ms": float(start_timestamp_ms + idx * frame_interval_ms),
                    "id": p.stem,
                }
            )
        return frames

    multispectral_frames = []
    for idx, p in enumerate(ms_files):
        name = p.stem.lower()
        band = "unknown"
        for candidate in ("nir", "red", "green", "blue", "rededge", "swir"):
            if candidate in name:
                band = candidate
                break
        multispectral_frames.append(
            {
                "path": str(p.resolve()),
                "timestamp_ms": float(start_timestamp_ms + idx * frame_interval_ms),
                "band": band,
                "id": p.stem,
            }
        )

    lidar_packets = []
    for idx, p in enumerate(lidar_files):
        lidar_packets.append(
            {
                "path": str(p.resolve()),
                "timestamp_ms": float(start_timestamp_ms + idx * frame_interval_ms),
                "id": p.stem,
            }
        )

    bundle = {
        "bundle_id": f"{root.name}_bundle",
        "payload_id": str(payload_id),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rgb_frames": _make_frames(rgb_files),
        "thermal_frames": _make_frames(thermal_files),
        "multispectral_frames": multispectral_frames,
        "lidar_packets": lidar_packets,
        "imu": [],
        "gnss": [],
        "camera_intrinsics": {},
        "thermal_calibration": {"temp_scale_c_per_dn": 0.1, "temp_offset_c": -40.0},
        "meta": {"source_root": str(root.resolve())},
    }

    if output_manifest_path is not None:
        write_capture_bundle(output_manifest_path, bundle)
    return bundle


def create_calibration_profile(
    path: str | Path,
    payload_id: str,
    camera_intrinsics: dict[str, dict[str, float]] | None = None,
    extrinsics: dict[str, dict[str, Any]] | None = None,
    thermal_calibration: dict[str, float] | None = None,
    notes: dict[str, Any] | None = None,
    version: int = 1,
) -> str:
    """Create and save a calibration profile for a payload."""
    profile = CalibrationProfile(
        payload_id=str(payload_id or "payload"),
        version=max(1, int(version)),
        created_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        camera_intrinsics=camera_intrinsics or {},
        extrinsics=extrinsics or {},
        thermal_calibration=thermal_calibration or {},
        notes=notes or {},
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(profile.to_dict(), handle, indent=2)
    return str(target)


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    payload = _read_json(Path(path))
    return CalibrationProfile(
        payload_id=str(payload.get("payload_id") or "payload"),
        version=max(1, int(payload.get("version", 1))),
        created_utc=str(payload.get("created_utc") or ""),
        camera_intrinsics=payload.get("camera_intrinsics", {}) if isinstance(payload.get("camera_intrinsics", {}), dict) else {},
        extrinsics=payload.get("extrinsics", {}) if isinstance(payload.get("extrinsics", {}), dict) else {},
        thermal_calibration=payload.get("thermal_calibration", {}) if isinstance(payload.get("thermal_calibration", {}), dict) else {},
        notes=payload.get("notes", {}) if isinstance(payload.get("notes", {}), dict) else {},
    )

def _is_monotonic(rows: list[dict], key: str = "timestamp_ms") -> bool:
    if len(rows) < 2:
        return True
    values = [float(r.get(key, 0.0)) for r in rows]
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def _nearest_by_timestamp(rows: list[dict], timestamp_ms: float) -> tuple[dict | None, float]:
    if not rows:
        return None, float("inf")
    ts = np.asarray([float(r.get("timestamp_ms", 0.0)) for r in rows], dtype=np.float64)
    idx = int(np.argmin(np.abs(ts - float(timestamp_ms))))
    row = rows[idx]
    delta = abs(float(row.get("timestamp_ms", 0.0)) - float(timestamp_ms))
    return row, float(delta)


def _interp_vector(a: list[float], b: list[float], t: float) -> list[float]:
    if len(a) < 3 or len(b) < 3:
        return [0.0, 0.0, 0.0]
    return [
        float(a[0] + (b[0] - a[0]) * t),
        float(a[1] + (b[1] - a[1]) * t),
        float(a[2] + (b[2] - a[2]) * t),
    ]


def _interpolate_imu(samples: list[dict], timestamp_ms: float) -> tuple[dict | None, float]:
    if not samples:
        return None, float("inf")
    ts = [float(s.get("timestamp_ms", 0.0)) for s in samples]
    if timestamp_ms <= ts[0]:
        return samples[0], abs(timestamp_ms - ts[0])
    if timestamp_ms >= ts[-1]:
        return samples[-1], abs(timestamp_ms - ts[-1])

    hi = int(np.searchsorted(ts, timestamp_ms, side="right"))
    lo = max(0, hi - 1)
    s0 = samples[lo]
    s1 = samples[hi]

    t0 = float(s0.get("timestamp_ms", 0.0))
    t1 = float(s1.get("timestamp_ms", 0.0))
    if abs(t1 - t0) < 1e-9:
        return s0, abs(timestamp_ms - t0)

    alpha = float((timestamp_ms - t0) / (t1 - t0))
    interp = {
        "timestamp_ms": float(timestamp_ms),
        "accel_m_s2": _interp_vector(
            s0.get("accel_m_s2", [0.0, 0.0, 0.0]),
            s1.get("accel_m_s2", [0.0, 0.0, 0.0]),
            alpha,
        ),
        "gyro_rad_s": _interp_vector(
            s0.get("gyro_rad_s", [0.0, 0.0, 0.0]),
            s1.get("gyro_rad_s", [0.0, 0.0, 0.0]),
            alpha,
        ),
    }
    return interp, float(min(abs(timestamp_ms - t0), abs(timestamp_ms - t1)))


def _interpolate_gnss(samples: list[dict], timestamp_ms: float) -> tuple[dict | None, float]:
    if not samples:
        return None, float("inf")
    ts = [float(s.get("timestamp_ms", 0.0)) for s in samples]
    if timestamp_ms <= ts[0]:
        return samples[0], abs(timestamp_ms - ts[0])
    if timestamp_ms >= ts[-1]:
        return samples[-1], abs(timestamp_ms - ts[-1])

    hi = int(np.searchsorted(ts, timestamp_ms, side="right"))
    lo = max(0, hi - 1)
    s0 = samples[lo]
    s1 = samples[hi]

    t0 = float(s0.get("timestamp_ms", 0.0))
    t1 = float(s1.get("timestamp_ms", 0.0))
    if abs(t1 - t0) < 1e-9:
        return s0, abs(timestamp_ms - t0)

    alpha = float((timestamp_ms - t0) / (t1 - t0))
    interp = {
        "timestamp_ms": float(timestamp_ms),
        "lat": float(s0.get("lat", 0.0) + (s1.get("lat", 0.0) - s0.get("lat", 0.0)) * alpha),
        "lon": float(s0.get("lon", 0.0) + (s1.get("lon", 0.0) - s0.get("lon", 0.0)) * alpha),
        "alt_m": float(s0.get("alt_m", 0.0) + (s1.get("alt_m", 0.0) - s0.get("alt_m", 0.0)) * alpha),
        "fix": str(s0.get("fix", "unknown")),
    }
    return interp, float(min(abs(timestamp_ms - t0), abs(timestamp_ms - t1)))


def _group_multispectral_by_band(stream: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in stream:
        band = str(row.get("band") or "unknown").lower()
        grouped.setdefault(band, []).append(row)
    for band in grouped:
        grouped[band].sort(key=lambda r: float(r.get("timestamp_ms", 0.0)))
    return grouped


def _validate_path_rows(rows: list[dict], sensor_name: str, expected_exts: set[str]) -> tuple[int, list[str]]:
    missing = 0
    bad_ext: list[str] = []
    for row in rows:
        p = Path(str(row.get("path", "")))
        if not p.exists():
            missing += 1
        elif p.suffix.lower() not in expected_exts:
            bad_ext.append(str(p))
    errors = []
    if missing > 0:
        errors.append(f"{sensor_name}: {missing} file(s) missing")
    if bad_ext:
        errors.append(f"{sensor_name}: unsupported extension for {len(bad_ext)} file(s)")
    return missing, errors


def _load_lidar_points(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path), allow_pickle=False)
    elif ext == ".npz":
        data = np.load(str(path), allow_pickle=False)
        key = next(iter(data.files), "")
        if not key:
            return np.zeros((0, 3), dtype=np.float32)
        arr = data[key]
    else:
        delimiter = "," if ext == ".csv" else None
        arr = np.loadtxt(str(path), delimiter=delimiter)

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return arr[:, :3].astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size_m: float, max_points: int) -> np.ndarray:
    if len(points) == 0:
        return points
    voxel = max(1e-4, float(voxel_size_m))
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    sampled = points[np.sort(idx)]
    if len(sampled) > max_points:
        step = max(1, len(sampled) // max_points)
        sampled = sampled[::step]
    return sampled


def _list_sensor_names(bundle: dict) -> list[str]:
    sensors = []
    if bundle.get("rgb_frames"):
        sensors.append("rgb")
    if bundle.get("thermal_frames"):
        sensors.append("thermal")
    if bundle.get("multispectral_frames"):
        sensors.append("multispectral")
    if bundle.get("lidar_packets"):
        sensors.append("lidar")
    if bundle.get("imu"):
        sensors.append("imu")
    if bundle.get("gnss"):
        sensors.append("gnss")
    return sensors


def _serialize_timeline_entry(entry: dict) -> dict:
    out = dict(entry)
    imu = out.get("imu")
    gnss = out.get("gnss")
    if isinstance(imu, dict):
        out["imu"] = {
            "timestamp_ms": float(imu.get("timestamp_ms", 0.0)),
            "accel_m_s2": [float(v) for v in imu.get("accel_m_s2", [0.0, 0.0, 0.0])],
            "gyro_rad_s": [float(v) for v in imu.get("gyro_rad_s", [0.0, 0.0, 0.0])],
        }
    if isinstance(gnss, dict):
        out["gnss"] = {
            "timestamp_ms": float(gnss.get("timestamp_ms", 0.0)),
            "lat": float(gnss.get("lat", 0.0)),
            "lon": float(gnss.get("lon", 0.0)),
            "alt_m": float(gnss.get("alt_m", 0.0)),
            "fix": str(gnss.get("fix", "unknown")),
        }
    return out

class MultiSensorProcessor:
    """Validate and fuse a capture bundle into aligned per-frame records."""

    def __init__(self, config: MultiSensorConfig | None = None):
        self.config = config or MultiSensorConfig()

    def validate_bundle(self, bundle: dict, calibration: CalibrationProfile | None = None) -> dict:
        errors: list[str] = []
        warnings: list[str] = []

        rgb = bundle.get("rgb_frames", [])
        thermal = bundle.get("thermal_frames", [])
        multispectral = bundle.get("multispectral_frames", [])
        lidar = bundle.get("lidar_packets", [])
        imu = bundle.get("imu", [])
        gnss = bundle.get("gnss", [])

        if not rgb:
            errors.append("No RGB frames found in bundle.")

        if rgb and not _is_monotonic(rgb):
            warnings.append("RGB frame timestamps are not monotonic.")
        if thermal and not _is_monotonic(thermal):
            warnings.append("Thermal frame timestamps are not monotonic.")
        if multispectral and not _is_monotonic(multispectral):
            warnings.append("Multispectral frame timestamps are not monotonic.")
        if lidar and not _is_monotonic(lidar):
            warnings.append("LiDAR packet timestamps are not monotonic.")
        if imu and not _is_monotonic(imu):
            warnings.append("IMU timestamps are not monotonic.")
        if gnss and not _is_monotonic(gnss):
            warnings.append("GNSS timestamps are not monotonic.")

        _, file_errors = _validate_path_rows(rgb, "rgb", SUPPORTED_IMAGE_EXTENSIONS)
        errors.extend(file_errors)
        _, file_errors = _validate_path_rows(thermal, "thermal", SUPPORTED_IMAGE_EXTENSIONS)
        errors.extend(file_errors)
        _, file_errors = _validate_path_rows(multispectral, "multispectral", SUPPORTED_IMAGE_EXTENSIONS)
        errors.extend(file_errors)
        _, file_errors = _validate_path_rows(lidar, "lidar", SUPPORTED_LIDAR_EXTENSIONS)
        errors.extend(file_errors)

        intrinsics = bundle.get("camera_intrinsics", {})
        if not isinstance(intrinsics, dict):
            intrinsics = {}

        required_intrinsics = []
        if rgb:
            required_intrinsics.append("rgb")
        if thermal:
            required_intrinsics.append("thermal")
        if multispectral:
            required_intrinsics.append("multispectral")

        missing_intrinsics = [s for s in required_intrinsics if s not in intrinsics]
        if missing_intrinsics:
            warnings.append(f"Missing camera intrinsics for: {', '.join(sorted(missing_intrinsics))}")

        thermal_cal = bundle.get("thermal_calibration", {})
        if thermal and not isinstance(thermal_cal, dict):
            warnings.append("Thermal calibration object missing or invalid.")

        if calibration is None:
            warnings.append("No external calibration profile loaded; using bundle-local calibration only.")
        else:
            if calibration.payload_id != bundle.get("payload_id"):
                warnings.append(
                    f"Calibration payload_id ({calibration.payload_id}) does not match bundle payload_id ({bundle.get('payload_id')})."
                )

        return {
            "bundle_id": bundle.get("bundle_id"),
            "payload_id": bundle.get("payload_id"),
            "sensors": _list_sensor_names(bundle),
            "counts": {
                "rgb_frames": len(rgb),
                "thermal_frames": len(thermal),
                "multispectral_frames": len(multispectral),
                "lidar_packets": len(lidar),
                "imu_samples": len(imu),
                "gnss_samples": len(gnss),
            },
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0,
        }

    def align_bundle(self, bundle: dict) -> dict:
        rgb = bundle.get("rgb_frames", [])
        thermal = bundle.get("thermal_frames", [])
        multispectral = bundle.get("multispectral_frames", [])
        lidar = bundle.get("lidar_packets", [])
        imu = bundle.get("imu", [])
        gnss = bundle.get("gnss", [])

        reference = rgb if rgb else thermal
        if not reference:
            return {
                "timeline": [],
                "stats": {
                    "reference_sensor": "none",
                    "frame_count": 0,
                },
            }

        reference_sensor = "rgb" if rgb else "thermal"
        ms_by_band = _group_multispectral_by_band(multispectral)

        timeline: list[dict] = []
        thermal_deltas: list[float] = []
        lidar_deltas: list[float] = []
        imu_deltas: list[float] = []
        gnss_deltas: list[float] = []
        ms_deltas: dict[str, list[float]] = {band: [] for band in ms_by_band}

        for ref in reference:
            t_ref = float(ref.get("timestamp_ms", 0.0))
            entry: dict[str, Any] = {
                "timestamp_ms": t_ref,
                "rgb_path": str(ref.get("path", "")) if reference_sensor == "rgb" else "",
                "thermal_path": "",
                "lidar_path": "",
                "multispectral": {},
                "imu": None,
                "gnss": None,
                "sync_error_ms": {},
            }

            if reference_sensor == "thermal":
                entry["thermal_path"] = str(ref.get("path", ""))

            if thermal:
                row, dt = _nearest_by_timestamp(thermal, t_ref)
                if row is not None and dt <= self.config.max_frame_delta_ms:
                    entry["thermal_path"] = str(row.get("path", ""))
                entry["sync_error_ms"]["thermal"] = float(dt)
                thermal_deltas.append(float(dt))

            for band, rows in ms_by_band.items():
                row, dt = _nearest_by_timestamp(rows, t_ref)
                if row is not None and dt <= self.config.max_frame_delta_ms:
                    entry["multispectral"][band] = str(row.get("path", ""))
                entry["sync_error_ms"][f"multispectral_{band}"] = float(dt)
                ms_deltas.setdefault(band, []).append(float(dt))

            if lidar:
                row, dt = _nearest_by_timestamp(lidar, t_ref)
                if row is not None and dt <= self.config.max_lidar_delta_ms:
                    entry["lidar_path"] = str(row.get("path", ""))
                entry["sync_error_ms"]["lidar"] = float(dt)
                lidar_deltas.append(float(dt))

            imu_interp, dt_imu = _interpolate_imu(imu, t_ref)
            if imu_interp is not None and dt_imu <= self.config.max_imu_delta_ms:
                entry["imu"] = imu_interp
            entry["sync_error_ms"]["imu"] = float(dt_imu)
            if imu_interp is not None:
                imu_deltas.append(float(dt_imu))

            gnss_interp, dt_gnss = _interpolate_gnss(gnss, t_ref)
            if gnss_interp is not None and dt_gnss <= self.config.max_gnss_delta_ms:
                entry["gnss"] = gnss_interp
            entry["sync_error_ms"]["gnss"] = float(dt_gnss)
            if gnss_interp is not None:
                gnss_deltas.append(float(dt_gnss))

            timeline.append(entry)

        def _coverage(field: str) -> float:
            if not timeline:
                return 0.0
            filled = sum(1 for row in timeline if row.get(field))
            return float(100.0 * filled / len(timeline))

        ms_coverage = {
            band: float(100.0 * sum(1 for row in timeline if row.get("multispectral", {}).get(band)) / max(1, len(timeline)))
            for band in ms_by_band
        }

        stats = {
            "reference_sensor": reference_sensor,
            "frame_count": len(timeline),
            "thermal_match_pct": _coverage("thermal_path"),
            "lidar_match_pct": _coverage("lidar_path"),
            "imu_match_pct": float(100.0 * sum(1 for row in timeline if row.get("imu") is not None) / max(1, len(timeline))),
            "gnss_match_pct": float(100.0 * sum(1 for row in timeline if row.get("gnss") is not None) / max(1, len(timeline))),
            "multispectral_match_pct": ms_coverage,
            "mean_abs_sync_error_ms": {
                "thermal": float(np.mean(thermal_deltas)) if thermal_deltas else None,
                "lidar": float(np.mean(lidar_deltas)) if lidar_deltas else None,
                "imu": float(np.mean(imu_deltas)) if imu_deltas else None,
                "gnss": float(np.mean(gnss_deltas)) if gnss_deltas else None,
            },
        }
        for band, deltas in ms_deltas.items():
            key = f"multispectral_{band}"
            stats["mean_abs_sync_error_ms"][key] = float(np.mean(deltas)) if deltas else None

        return {
            "timeline": [_serialize_timeline_entry(row) for row in timeline],
            "stats": stats,
        }

    def _thermal_processing(
        self,
        timeline: list[dict],
        thermal_calibration: dict,
        output_dir: Path,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        scale = float(thermal_calibration.get("temp_scale_c_per_dn", thermal_calibration.get("scale", 0.1)))
        offset = float(thermal_calibration.get("temp_offset_c", thermal_calibration.get("offset", -40.0)))

        frame_rows: list[dict] = []
        hotspot_max = None
        overlay_paths: list[str] = []

        for idx, row in enumerate(timeline):
            thermal_path = str(row.get("thermal_path", ""))
            if not thermal_path:
                continue
            p = Path(thermal_path)
            if not p.exists():
                continue

            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            temp_c = img.astype(np.float32) * scale + offset
            dynamic_thr = float(np.percentile(temp_c, 99.2))
            threshold_c = float(max(self.config.thermal_hotspot_threshold_c, dynamic_thr))
            mask = (temp_c >= threshold_c).astype(np.uint8) * 255

            comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            filtered = np.zeros_like(mask)
            hotspot_count = 0
            hotspot_area = 0
            for cid in range(1, comp):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < self.config.thermal_min_hotspot_area_px:
                    continue
                filtered[labels == cid] = 255
                hotspot_count += 1
                hotspot_area += area

            vis = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            vis[filtered > 0] = (0, 0, 255)

            out_path = output_dir / f"thermal_hotspot_{idx:05d}.png"
            cv2.imwrite(str(out_path), vis)
            overlay_paths.append(str(out_path))

            max_temp = float(np.max(temp_c))
            hotspot_max = max_temp if hotspot_max is None else max(float(hotspot_max), max_temp)

            frame_rows.append(
                {
                    "timestamp_ms": float(row.get("timestamp_ms", 0.0)),
                    "thermal_path": thermal_path,
                    "overlay_path": str(out_path),
                    "threshold_c": threshold_c,
                    "max_temp_c": max_temp,
                    "mean_temp_c": float(np.mean(temp_c)),
                    "hotspot_count": hotspot_count,
                    "hotspot_area_px": hotspot_area,
                }
            )

        return {
            "frame_count": len(frame_rows),
            "frames_with_hotspots": int(sum(1 for r in frame_rows if int(r.get("hotspot_count", 0)) > 0)),
            "global_max_temp_c": float(hotspot_max) if hotspot_max is not None else None,
            "overlays": overlay_paths,
            "frames": frame_rows,
            "thermal_calibration_used": {
                "temp_scale_c_per_dn": scale,
                "temp_offset_c": offset,
            },
        }

    def _lidar_processing(self, timeline: list[dict], output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        packet_paths = []
        for row in timeline:
            path = str(row.get("lidar_path", ""))
            if path:
                packet_paths.append(path)

        unique_paths = sorted(set(packet_paths))
        all_points: list[np.ndarray] = []
        invalid_packets: list[str] = []
        per_packet: list[dict] = []

        for path in unique_paths:
            p = Path(path)
            if not p.exists():
                invalid_packets.append(path)
                continue
            try:
                points = _load_lidar_points(p)
            except Exception:
                points = np.zeros((0, 3), dtype=np.float32)
            if len(points) == 0:
                invalid_packets.append(path)
                continue
            all_points.append(points)
            per_packet.append(
                {
                    "path": path,
                    "point_count": int(len(points)),
                    "min_xyz": [float(v) for v in points.min(axis=0)],
                    "max_xyz": [float(v) for v in points.max(axis=0)],
                }
            )

        if all_points:
            merged = np.concatenate(all_points, axis=0)
        else:
            merged = np.zeros((0, 3), dtype=np.float32)

        down = _voxel_downsample(
            merged,
            voxel_size_m=self.config.lidar_voxel_size_m,
            max_points=self.config.lidar_preview_max_points,
        )

        preview_path = output_dir / "lidar_preview.xyz"
        if len(down) > 0:
            np.savetxt(str(preview_path), down, fmt="%.6f")
            bbox_min = [float(v) for v in down.min(axis=0)]
            bbox_max = [float(v) for v in down.max(axis=0)]
        else:
            bbox_min = [0.0, 0.0, 0.0]
            bbox_max = [0.0, 0.0, 0.0]

        return {
            "packet_count": len(unique_paths),
            "valid_packet_count": len(per_packet),
            "invalid_packet_count": len(invalid_packets),
            "invalid_packets": invalid_packets,
            "total_points": int(len(merged)),
            "preview_points": int(len(down)),
            "bbox_min_xyz": bbox_min,
            "bbox_max_xyz": bbox_max,
            "preview_cloud_path": str(preview_path) if len(down) > 0 else "",
            "packets": per_packet,
        }

    def _multispectral_processing(self, timeline: list[dict], output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        ndvi_rows: list[dict] = []
        preview_paths: list[str] = []

        for idx, row in enumerate(timeline):
            bands = row.get("multispectral", {})
            if not isinstance(bands, dict):
                continue
            red_path = str(bands.get("red", ""))
            nir_path = str(bands.get("nir", ""))
            if not red_path or not nir_path:
                continue

            red = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)
            nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
            if red is None or nir is None:
                continue
            if red.ndim == 3:
                red = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
            if nir.ndim == 3:
                nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)

            if red.shape != nir.shape:
                nir = cv2.resize(nir, (red.shape[1], red.shape[0]), interpolation=cv2.INTER_LINEAR)

            red_f = red.astype(np.float32)
            nir_f = nir.astype(np.float32)
            denom = np.maximum(nir_f + red_f, 1e-6)
            ndvi = (nir_f - red_f) / denom

            ndvi_u8 = np.clip((ndvi + 1.0) * 127.5, 0, 255).astype(np.uint8)
            ndvi_col = cv2.applyColorMap(ndvi_u8, cv2.COLORMAP_VIRIDIS)
            out_path = output_dir / f"ndvi_{idx:05d}.png"
            cv2.imwrite(str(out_path), ndvi_col)
            preview_paths.append(str(out_path))

            ndvi_rows.append(
                {
                    "timestamp_ms": float(row.get("timestamp_ms", 0.0)),
                    "red_path": red_path,
                    "nir_path": nir_path,
                    "ndvi_mean": float(np.mean(ndvi)),
                    "ndvi_min": float(np.min(ndvi)),
                    "ndvi_max": float(np.max(ndvi)),
                    "preview_path": str(out_path),
                }
            )

        return {
            "frame_count": len(ndvi_rows),
            "ndvi_previews": preview_paths,
            "ndvi_frames": ndvi_rows,
            "ndvi_mean_global": float(np.mean([r["ndvi_mean"] for r in ndvi_rows])) if ndvi_rows else None,
        }

    def process(
        self,
        bundle_path: str | Path,
        output_dir: str | Path,
        calibration_profile_path: str = "",
    ) -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = load_capture_bundle(bundle_path)

        profile_path = calibration_profile_path or bundle.get("calibration_profile_path", "")
        calibration = None
        if profile_path:
            p = Path(profile_path)
            if p.exists():
                try:
                    calibration = load_calibration_profile(p)
                except Exception:
                    calibration = None

        qa = self.validate_bundle(bundle, calibration=calibration)
        aligned = self.align_bundle(bundle)
        timeline = aligned.get("timeline", []) if isinstance(aligned, dict) else []

        timeline_path = out_dir / "aligned_timeline.json"
        with timeline_path.open("w", encoding="utf-8") as handle:
            json.dump(timeline, handle, indent=2)

        thermal_cal = bundle.get("thermal_calibration", {})
        if calibration is not None and calibration.thermal_calibration:
            merged_thermal_cal = dict(thermal_cal)
            merged_thermal_cal.update(calibration.thermal_calibration)
            thermal_cal = merged_thermal_cal

        thermal_payload = self._thermal_processing(timeline, thermal_calibration=thermal_cal, output_dir=out_dir / "thermal")
        lidar_payload = self._lidar_processing(timeline, output_dir=out_dir / "lidar")
        multispectral_payload = self._multispectral_processing(timeline, output_dir=out_dir / "multispectral")

        payload = {
            "bundle_manifest_path": str(Path(bundle_path).resolve()),
            "bundle_id": bundle.get("bundle_id", ""),
            "payload_id": bundle.get("payload_id", ""),
            "sensors": _list_sensor_names(bundle),
            "qa": qa,
            "alignment": aligned.get("stats", {}),
            "artifacts": {
                "aligned_timeline_path": str(timeline_path),
                "thermal_dir": str((out_dir / "thermal").resolve()),
                "lidar_dir": str((out_dir / "lidar").resolve()),
                "multispectral_dir": str((out_dir / "multispectral").resolve()),
            },
            "thermal": thermal_payload,
            "lidar": lidar_payload,
            "multispectral": multispectral_payload,
        }

        if calibration is not None:
            payload["calibration_profile"] = calibration.to_dict()
            payload["calibration_profile_path"] = str(Path(profile_path).resolve())
        else:
            payload["calibration_profile"] = {}
            payload["calibration_profile_path"] = ""

        summary_path = out_dir / "multisensor_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        payload["summary_path"] = str(summary_path)
        return payload
