"""Data Library — dataset import, metadata extraction, thumbnails, QA, tagging."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .errors import AppError, ERR_DATASET_MISSING, ERR_INVALID_INPUT
from .validation import ValidationMessage, SEVERITY_ERROR, SEVERITY_WARNING


SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ImageAsset:
    id: str
    dataset_id: str
    file_path: str
    thumbnail_path: str | None = None
    width: int = 0
    height: int = 0
    gps_lat: float | None = None
    gps_lon: float | None = None
    altitude_m: float | None = None
    captured_at: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    iso: int | None = None
    exposure_s: float | None = None
    focal_length_mm: float | None = None
    file_size_bytes: int = 0
    sha1: str | None = None
    qa_flags: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Dataset:
    id: str
    project_id: str
    name: str
    dataset_type: str           # rgb | thermal | mask | video | bundle | calibration | reconstruction | analysis
    root_dir: str
    image_count: int = 0
    has_gps_metadata: bool = False
    has_camera_metadata: bool = False
    linked_mission_id: str | None = None
    qa_status: str = "unchecked"   # unchecked | passing | warnings | failing
    notes: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateGroup:
    hash_value: str
    files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityMetric:
    metric: str
    value: float
    pass_: bool
    threshold: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": float(self.value),
            "pass": bool(self.pass_),
            "threshold": float(self.threshold),
            "note": self.note,
        }


@dataclass
class MetadataCoverage:
    total: int
    with_gps: int
    with_timestamp: int
    with_camera: int

    @property
    def gps_pct(self) -> float:
        return 100.0 * self.with_gps / max(1, self.total)

    @property
    def timestamp_pct(self) -> float:
        return 100.0 * self.with_timestamp / max(1, self.total)

    @property
    def camera_pct(self) -> float:
        return 100.0 * self.with_camera / max(1, self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "with_gps": self.with_gps,
            "with_timestamp": self.with_timestamp,
            "with_camera": self.with_camera,
            "gps_pct": round(self.gps_pct, 2),
            "timestamp_pct": round(self.timestamp_pct, 2),
            "camera_pct": round(self.camera_pct, 2),
        }


@dataclass
class DatasetValidationReport:
    dataset_id: str
    total_images: int
    corrupt_count: int
    blurry_count: int
    duplicate_count: int
    missing_gps_count: int
    metadata_coverage: MetadataCoverage
    messages: list[dict[str, Any]] = field(default_factory=list)
    overall_status: str = "passing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "total_images": self.total_images,
            "corrupt_count": self.corrupt_count,
            "blurry_count": self.blurry_count,
            "duplicate_count": self.duplicate_count,
            "missing_gps_count": self.missing_gps_count,
            "metadata_coverage": self.metadata_coverage.to_dict(),
            "messages": self.messages,
            "overall_status": self.overall_status,
        }


# ── Storage ───────────────────────────────────────────────────────────────────

def _dataset_index_path(project_root: Path) -> Path:
    return project_root / "datasets" / "dataset_index.db"


def _open_index(project_root: Path) -> sqlite3.Connection:
    db_path = _dataset_index_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            root_dir TEXT NOT NULL,
            image_count INTEGER DEFAULT 0,
            has_gps_metadata INTEGER DEFAULT 0,
            has_camera_metadata INTEGER DEFAULT 0,
            linked_mission_id TEXT,
            qa_status TEXT DEFAULT 'unchecked',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            thumbnail_path TEXT,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            gps_lat REAL,
            gps_lon REAL,
            altitude_m REAL,
            captured_at TEXT,
            camera_make TEXT,
            camera_model TEXT,
            iso INTEGER,
            exposure_s REAL,
            focal_length_mm REAL,
            file_size_bytes INTEGER DEFAULT 0,
            sha1 TEXT,
            qa_flags TEXT DEFAULT '[]',
            tags TEXT DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_id);
    """)
    conn.commit()
    return conn


def _dataset_to_row(d: Dataset) -> tuple:
    return (
        d.id, d.project_id, d.name, d.dataset_type, d.root_dir,
        d.image_count, int(d.has_gps_metadata), int(d.has_camera_metadata),
        d.linked_mission_id, d.qa_status, d.notes, d.created_at, d.updated_at,
    )


def _row_to_dataset(row: sqlite3.Row) -> Dataset:
    return Dataset(
        id=row["id"],
        project_id=row["project_id"],
        name=row["name"],
        dataset_type=row["dataset_type"],
        root_dir=row["root_dir"],
        image_count=int(row["image_count"] or 0),
        has_gps_metadata=bool(row["has_gps_metadata"]),
        has_camera_metadata=bool(row["has_camera_metadata"]),
        linked_mission_id=row["linked_mission_id"],
        qa_status=row["qa_status"] or "unchecked",
        notes=row["notes"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_image(row: sqlite3.Row) -> ImageAsset:
    qa_flags = []
    tags = []
    try:
        qa_flags = json.loads(row["qa_flags"] or "[]")
    except Exception:
        pass
    try:
        tags = json.loads(row["tags"] or "[]")
    except Exception:
        pass
    return ImageAsset(
        id=row["id"],
        dataset_id=row["dataset_id"],
        file_path=row["file_path"],
        thumbnail_path=row["thumbnail_path"],
        width=int(row["width"] or 0),
        height=int(row["height"] or 0),
        gps_lat=row["gps_lat"],
        gps_lon=row["gps_lon"],
        altitude_m=row["altitude_m"],
        captured_at=row["captured_at"],
        camera_make=row["camera_make"],
        camera_model=row["camera_model"],
        iso=row["iso"],
        exposure_s=row["exposure_s"],
        focal_length_mm=row["focal_length_mm"],
        file_size_bytes=int(row["file_size_bytes"] or 0),
        sha1=row["sha1"],
        qa_flags=qa_flags,
        tags=tags,
    )


# ── Metadata extraction ───────────────────────────────────────────────────────

def scan_dataset_folder(folder_path: Path | str) -> list[Path]:
    """Find supported image files (ignores hidden + unsupported)."""
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        raise AppError(
            ERR_DATASET_MISSING,
            f"Dataset folder not found: {root}",
            recovery_action="Choose a folder that contains images.",
        )
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in SUPPORTED_IMAGE_EXT:
            files.append(p)
    files.sort()
    return files


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return ""


def _exif_to_decimal(value, ref) -> float | None:
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        dec = d + (m / 60.0) + (s / 3600.0)
        if str(ref).upper() in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def extract_image_metadata(image_path: Path | str) -> dict[str, Any]:
    """Extract dims, EXIF, GPS, timestamp, camera info. Returns plain dict."""
    p = Path(image_path)
    meta: dict[str, Any] = {
        "file_path": str(p),
        "file_size_bytes": p.stat().st_size if p.exists() else 0,
        "width": 0,
        "height": 0,
        "gps_lat": None,
        "gps_lon": None,
        "altitude_m": None,
        "captured_at": None,
        "camera_make": None,
        "camera_model": None,
        "iso": None,
        "exposure_s": None,
        "focal_length_mm": None,
    }
    try:
        from PIL import Image, ExifTags
        with Image.open(str(p)) as img:
            meta["width"], meta["height"] = img.size
            exif_raw = img.getexif() if hasattr(img, "getexif") else None
            if exif_raw:
                tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
                meta["camera_make"] = str(tags.get("Make")) if tags.get("Make") else None
                meta["camera_model"] = str(tags.get("Model")) if tags.get("Model") else None
                iso = tags.get("ISOSpeedRatings") or tags.get("PhotographicSensitivity")
                if iso is not None:
                    try:
                        meta["iso"] = int(iso)
                    except Exception:
                        pass
                exp = tags.get("ExposureTime")
                if exp is not None:
                    try:
                        meta["exposure_s"] = float(exp)
                    except Exception:
                        pass
                focal = tags.get("FocalLength")
                if focal is not None:
                    try:
                        meta["focal_length_mm"] = float(focal)
                    except Exception:
                        pass
                dt = tags.get("DateTimeOriginal") or tags.get("DateTime")
                if dt:
                    try:
                        meta["captured_at"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S").isoformat()
                    except Exception:
                        meta["captured_at"] = str(dt)
                gps_info = tags.get("GPSInfo")
                if isinstance(gps_info, dict):
                    gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
                    lat = _exif_to_decimal(gps_tags.get("GPSLatitude"), gps_tags.get("GPSLatitudeRef", "N"))
                    lon = _exif_to_decimal(gps_tags.get("GPSLongitude"), gps_tags.get("GPSLongitudeRef", "E"))
                    alt = gps_tags.get("GPSAltitude")
                    if lat is not None:
                        meta["gps_lat"] = lat
                    if lon is not None:
                        meta["gps_lon"] = lon
                    if alt is not None:
                        try:
                            meta["altitude_m"] = float(alt)
                        except Exception:
                            pass
    except Exception:
        # Fall back to OpenCV for dimensions only
        try:
            import cv2
            img = cv2.imread(str(p))
            if img is not None:
                meta["height"], meta["width"] = img.shape[:2]
        except Exception:
            pass
    return meta


def generate_thumbnail(
    image_path: Path | str,
    output_dir: Path | str,
    size: tuple[int, int] = (320, 320),
) -> Path | None:
    """Create thumbnail; returns output path or None on failure."""
    src = Path(image_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / f"{src.stem}.jpg"
    try:
        from PIL import Image
        with Image.open(str(src)) as img:
            img.thumbnail(size)
            img.convert("RGB").save(str(target), "JPEG", quality=82)
        return target
    except Exception:
        try:
            import cv2
            img = cv2.imread(str(src))
            if img is None:
                return None
            h, w = img.shape[:2]
            scale = min(size[0] / max(1, w), size[1] / max(1, h), 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return target
        except Exception:
            return None


# ── Quality checks ────────────────────────────────────────────────────────────

def check_blur(image_path: Path | str, threshold: float = 100.0) -> QualityMetric:
    """Variance of Laplacian. Higher = sharper."""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return QualityMetric("blur", 0.0, False, threshold, "unreadable")
        var = float(cv2.Laplacian(img, cv2.CV_64F).var())
        return QualityMetric("blur", var, var >= threshold, threshold)
    except Exception as exc:
        return QualityMetric("blur", 0.0, False, threshold, f"error: {exc}")


def check_exposure(
    image_path: Path | str,
    clip_threshold: float = 0.10,
) -> QualityMetric:
    """Estimate over/under exposure from clipped-pixel ratio."""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return QualityMetric("exposure", 1.0, False, clip_threshold, "unreadable")
        total = img.size
        dark = float((img < 5).sum()) / max(1, total)
        bright = float((img > 250).sum()) / max(1, total)
        clip = max(dark, bright)
        ok = clip <= clip_threshold
        note = ("ok" if ok else ("underexposed" if dark > bright else "overexposed"))
        return QualityMetric("exposure", clip, ok, clip_threshold, note)
    except Exception as exc:
        return QualityMetric("exposure", 1.0, False, clip_threshold, f"error: {exc}")


def _perceptual_hash_fallback(path: Path) -> str:
    """8x8 average-hash fallback when imagehash is missing."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        avg = float(small.mean())
        bits = (small >= avg).astype(np.uint8).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return f"{h:016x}"
    except Exception:
        return ""


def check_duplicate_images(image_paths: Iterable[Path | str]) -> list[DuplicateGroup]:
    """Perceptual-hash near-duplicate detection."""
    hashes: dict[str, list[str]] = {}
    use_lib = False
    try:
        import imagehash
        from PIL import Image
        use_lib = True
    except Exception:
        pass

    for p in image_paths:
        p = Path(p)
        if not p.exists():
            continue
        if use_lib:
            try:
                with Image.open(str(p)) as img:
                    h = str(imagehash.phash(img))
            except Exception:
                h = _perceptual_hash_fallback(p)
        else:
            h = _perceptual_hash_fallback(p)
        if not h:
            continue
        hashes.setdefault(h, []).append(str(p))

    return [DuplicateGroup(hash_value=k, files=v) for k, v in hashes.items() if len(v) > 1]


def check_metadata_coverage(project_root: Path | str, dataset_id: str) -> MetadataCoverage:
    """Calculate % of images with GPS, timestamp and camera metadata."""
    conn = _open_index(Path(project_root))
    try:
        rows = conn.execute(
            "SELECT gps_lat, gps_lon, captured_at, camera_model FROM images WHERE dataset_id=?",
            (dataset_id,),
        ).fetchall()
        total = len(rows)
        with_gps = sum(1 for r in rows if r["gps_lat"] is not None and r["gps_lon"] is not None)
        with_ts = sum(1 for r in rows if r["captured_at"])
        with_cam = sum(1 for r in rows if r["camera_model"])
        return MetadataCoverage(total=total, with_gps=with_gps, with_timestamp=with_ts, with_camera=with_cam)
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────────

def import_image_dataset(
    project_root: Path | str,
    project_id: str,
    folder_path: Path | str,
    dataset_name: str | None = None,
    dataset_type: str = "rgb",
    generate_thumbnails: bool = True,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Dataset:
    """Scan folder, extract metadata, generate thumbnails, save dataset record."""
    pr = Path(project_root)
    fp = Path(folder_path)
    files = scan_dataset_folder(fp)
    if not files:
        raise AppError(
            ERR_DATASET_MISSING,
            f"No supported images in folder: {fp}",
            recovery_action="Add image files (jpg, png, tif, bmp, webp) and try again.",
        )

    name = (dataset_name or fp.name).strip() or fp.name
    dataset_id = str(uuid.uuid4())
    root = pr / "datasets" / dataset_id
    thumbs_root = root / "thumbnails"
    root.mkdir(parents=True, exist_ok=True)
    thumbs_root.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(
        id=dataset_id,
        project_id=project_id,
        name=name,
        dataset_type=dataset_type,
        root_dir=str(root),
    )

    conn = _open_index(pr)
    try:
        with conn:
            conn.execute(
                "INSERT INTO datasets (id,project_id,name,dataset_type,root_dir,image_count,has_gps_metadata,has_camera_metadata,linked_mission_id,qa_status,notes,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                _dataset_to_row(dataset),
            )

        gps_count = 0
        cam_count = 0
        total = len(files)
        for idx, src in enumerate(files):
            if progress_callback:
                try:
                    progress_callback(int(100.0 * idx / max(1, total)), f"{src.name}")
                except Exception:
                    pass
            meta = extract_image_metadata(src)
            sha1 = _sha1(src)
            thumb_path: Path | None = None
            if generate_thumbnails:
                thumb_path = generate_thumbnail(src, thumbs_root)
            asset = ImageAsset(
                id=str(uuid.uuid4()),
                dataset_id=dataset_id,
                file_path=str(src),
                thumbnail_path=str(thumb_path) if thumb_path else None,
                width=int(meta.get("width", 0) or 0),
                height=int(meta.get("height", 0) or 0),
                gps_lat=meta.get("gps_lat"),
                gps_lon=meta.get("gps_lon"),
                altitude_m=meta.get("altitude_m"),
                captured_at=meta.get("captured_at"),
                camera_make=meta.get("camera_make"),
                camera_model=meta.get("camera_model"),
                iso=meta.get("iso"),
                exposure_s=meta.get("exposure_s"),
                focal_length_mm=meta.get("focal_length_mm"),
                file_size_bytes=int(meta.get("file_size_bytes", 0) or 0),
                sha1=sha1,
            )
            if asset.gps_lat is not None and asset.gps_lon is not None:
                gps_count += 1
            if asset.camera_model:
                cam_count += 1
            with conn:
                conn.execute(
                    "INSERT INTO images (id,dataset_id,file_path,thumbnail_path,width,height,gps_lat,gps_lon,altitude_m,captured_at,camera_make,camera_model,iso,exposure_s,focal_length_mm,file_size_bytes,sha1,qa_flags,tags) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        asset.id, asset.dataset_id, asset.file_path, asset.thumbnail_path,
                        asset.width, asset.height, asset.gps_lat, asset.gps_lon, asset.altitude_m,
                        asset.captured_at, asset.camera_make, asset.camera_model,
                        asset.iso, asset.exposure_s, asset.focal_length_mm,
                        asset.file_size_bytes, asset.sha1,
                        json.dumps(asset.qa_flags), json.dumps(asset.tags),
                    ),
                )

        dataset.image_count = total
        dataset.has_gps_metadata = gps_count > 0
        dataset.has_camera_metadata = cam_count > 0
        dataset.updated_at = _now_iso()
        with conn:
            conn.execute(
                "UPDATE datasets SET image_count=?, has_gps_metadata=?, has_camera_metadata=?, updated_at=? WHERE id=?",
                (dataset.image_count, int(dataset.has_gps_metadata), int(dataset.has_camera_metadata),
                 dataset.updated_at, dataset_id),
            )
    finally:
        conn.close()

    # Persist dataset metadata to disk
    meta_path = root / "metadata.json"
    meta_path.write_text(json.dumps(dataset.to_dict(), indent=2), encoding="utf-8")
    if progress_callback:
        try:
            progress_callback(100, "import complete")
        except Exception:
            pass
    return dataset


def list_datasets(
    project_root: Path | str,
    project_id: str | None = None,
    dataset_type: str | None = None,
    mission_id: str | None = None,
) -> list[Dataset]:
    conn = _open_index(Path(project_root))
    try:
        q = "SELECT * FROM datasets WHERE 1=1"
        args: list[Any] = []
        if project_id:
            q += " AND project_id=?"
            args.append(project_id)
        if dataset_type:
            q += " AND dataset_type=?"
            args.append(dataset_type)
        if mission_id:
            q += " AND linked_mission_id=?"
            args.append(mission_id)
        q += " ORDER BY updated_at DESC"
        rows = conn.execute(q, args).fetchall()
        return [_row_to_dataset(r) for r in rows]
    finally:
        conn.close()


def get_dataset(project_root: Path | str, dataset_id: str) -> Dataset | None:
    conn = _open_index(Path(project_root))
    try:
        row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return _row_to_dataset(row) if row else None
    finally:
        conn.close()


def get_image_assets(
    project_root: Path | str,
    dataset_id: str,
    page: int = 0,
    page_size: int = 100,
) -> list[ImageAsset]:
    conn = _open_index(Path(project_root))
    try:
        offset = max(0, page) * max(1, page_size)
        rows = conn.execute(
            "SELECT * FROM images WHERE dataset_id=? ORDER BY file_path LIMIT ? OFFSET ?",
            (dataset_id, int(page_size), int(offset)),
        ).fetchall()
        return [_row_to_image(r) for r in rows]
    finally:
        conn.close()


def link_dataset_to_mission(project_root: Path | str, dataset_id: str, mission_id: str) -> None:
    conn = _open_index(Path(project_root))
    try:
        with conn:
            conn.execute(
                "UPDATE datasets SET linked_mission_id=?, updated_at=? WHERE id=?",
                (mission_id, _now_iso(), dataset_id),
            )
    finally:
        conn.close()


def tag_image_asset(project_root: Path | str, image_id: str, tag: str, remove: bool = False) -> None:
    conn = _open_index(Path(project_root))
    try:
        row = conn.execute("SELECT tags FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            raise AppError(ERR_INVALID_INPUT, f"Image not found: {image_id}")
        try:
            tags = json.loads(row["tags"] or "[]")
        except Exception:
            tags = []
        if remove:
            tags = [t for t in tags if t != tag]
        elif tag not in tags:
            tags.append(tag)
        with conn:
            conn.execute("UPDATE images SET tags=? WHERE id=?", (json.dumps(tags), image_id))
    finally:
        conn.close()


def set_image_qa_flags(project_root: Path | str, image_id: str, flags: list[str]) -> None:
    conn = _open_index(Path(project_root))
    try:
        with conn:
            conn.execute("UPDATE images SET qa_flags=? WHERE id=?", (json.dumps(list(flags)), image_id))
    finally:
        conn.close()


def delete_dataset(project_root: Path | str, dataset_id: str, delete_files: bool = False) -> bool:
    """Remove dataset from index. delete_files=True also removes thumbnails folder."""
    conn = _open_index(Path(project_root))
    try:
        row = conn.execute("SELECT root_dir FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not row:
            return False
        with conn:
            conn.execute("DELETE FROM images WHERE dataset_id=?", (dataset_id,))
            conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        if delete_files:
            try:
                import shutil
                shutil.rmtree(row["root_dir"], ignore_errors=True)
            except Exception:
                pass
        return True
    finally:
        conn.close()


def validate_dataset(
    project_root: Path | str,
    dataset_id: str,
    blur_threshold: float = 100.0,
    exposure_clip_threshold: float = 0.10,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DatasetValidationReport:
    """Check corrupt, blurry, duplicates, missing GPS, metadata coverage."""
    ds = get_dataset(project_root, dataset_id)
    if ds is None:
        raise AppError(ERR_DATASET_MISSING, f"Dataset not found: {dataset_id}")
    assets = get_image_assets(project_root, dataset_id, page=0, page_size=10**6)

    corrupt = 0
    blurry = 0
    missing_gps = 0
    messages: list[dict[str, Any]] = []

    conn = _open_index(Path(project_root))
    try:
        total = len(assets)
        for idx, asset in enumerate(assets):
            if progress_callback:
                try:
                    progress_callback(int(100.0 * idx / max(1, total)), Path(asset.file_path).name)
                except Exception:
                    pass
            flags = list(asset.qa_flags)
            if not Path(asset.file_path).exists():
                corrupt += 1
                flags.append("missing_file")
            else:
                blur_m = check_blur(asset.file_path, blur_threshold)
                if not blur_m.pass_:
                    blurry += 1
                    flags.append("blurry")
                exp_m = check_exposure(asset.file_path, exposure_clip_threshold)
                if not exp_m.pass_:
                    flags.append(f"{exp_m.note}")
            if asset.gps_lat is None or asset.gps_lon is None:
                missing_gps += 1
                flags.append("missing_gps")
            with conn:
                conn.execute("UPDATE images SET qa_flags=? WHERE id=?", (json.dumps(flags), asset.id))

        # Duplicates
        dup_groups = check_duplicate_images([a.file_path for a in assets])
        duplicate_count = sum(len(g.files) - 1 for g in dup_groups)
        if dup_groups:
            messages.append({"severity": "warning", "message": f"{duplicate_count} near-duplicate images detected."})

        coverage = check_metadata_coverage(project_root, dataset_id)
        if coverage.gps_pct < 50.0 and coverage.total > 0:
            messages.append({"severity": "warning", "message": f"GPS coverage only {coverage.gps_pct:.0f}%."})

        if corrupt:
            messages.append({"severity": "error", "message": f"{corrupt} files are missing or corrupt."})
        if blurry > 0:
            messages.append({"severity": "warning", "message": f"{blurry} images appear blurry."})

        overall = "passing"
        if corrupt > 0 or blurry / max(1, total) > 0.4:
            overall = "failing"
        elif blurry > 0 or duplicate_count > 0 or coverage.gps_pct < 80.0:
            overall = "warnings"

        with conn:
            conn.execute(
                "UPDATE datasets SET qa_status=?, updated_at=? WHERE id=?",
                (overall, _now_iso(), dataset_id),
            )

        report = DatasetValidationReport(
            dataset_id=dataset_id,
            total_images=total,
            corrupt_count=corrupt,
            blurry_count=blurry,
            duplicate_count=duplicate_count,
            missing_gps_count=missing_gps,
            metadata_coverage=coverage,
            messages=messages,
            overall_status=overall,
        )
        # Persist QA result file
        qa_path = Path(ds.root_dir) / "qa_results.json"
        qa_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return report
    finally:
        conn.close()


def export_dataset_manifest(project_root: Path | str, dataset_id: str, output_format: str = "csv") -> Path:
    """Export the dataset image manifest as CSV or JSON."""
    ds = get_dataset(project_root, dataset_id)
    if ds is None:
        raise AppError(ERR_DATASET_MISSING, f"Dataset not found: {dataset_id}")
    assets = get_image_assets(project_root, dataset_id, page=0, page_size=10**6)
    out_dir = Path(ds.root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        path = out_dir / "manifest.json"
        path.write_text(
            json.dumps({
                "dataset": ds.to_dict(),
                "images": [a.to_dict() for a in assets],
            }, indent=2),
            encoding="utf-8",
        )
        return path
    path = out_dir / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "id", "file_path", "width", "height", "gps_lat", "gps_lon", "altitude_m",
            "captured_at", "camera_make", "camera_model", "iso", "exposure_s",
            "focal_length_mm", "file_size_bytes", "sha1", "qa_flags", "tags",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for a in assets:
            d = a.to_dict()
            d["qa_flags"] = ",".join(a.qa_flags)
            d["tags"] = ",".join(a.tags)
            writer.writerow({k: d.get(k, "") for k in cols})
    return path
