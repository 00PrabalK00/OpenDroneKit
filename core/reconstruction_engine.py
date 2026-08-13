"""3D reconstruction façade — spec-named API on top of reconstruction.py.

Supports import-from-folder (no compute) and full reconstruction via the
existing CustomDroneReconstructor.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AppError, ERR_INVALID_INPUT
from .reconstruction import (
    CustomDroneReconstructor,
    ReconstructionResult as _RawReconstructionResult,
    available_reconstruction_profiles,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ReconstructionConfig:
    image_folder: str
    output_folder: str
    mask_folder: str | None = None
    profile: str = "standard"
    execution_mode: str = "local"
    reuse_cache: bool = True
    cloud_endpoint: str = ""
    max_points: int = 150_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReconstructionResult:
    id: str
    output_folder: str
    point_cloud_path: str | None
    mesh_path: str | None
    camera_poses_path: str | None
    orthomosaic_path: str | None
    dsm_path: str | None
    dtm_path: str | None
    textured_mesh_obj_path: str | None
    texture_image_path: str | None
    digital_twin_path: str | None
    defect_projection_path: str | None
    quality_metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PointCloudData:
    points_path: str
    point_count: int
    has_colors: bool
    bounds: dict[str, list[float]]


@dataclass
class MeshData:
    mesh_path: str
    vertex_count: int
    face_count: int
    has_texture: bool


@dataclass
class DefectProjectionResult:
    id: str
    defect_count: int
    projected_count: int
    output_path: str


@dataclass
class ReconstructionQuality:
    image_count: int
    sparse_point_count: int
    dense_point_count: int
    coverage_pct: float
    failed_image_count: int


# ── Public API ────────────────────────────────────────────────────────────────

def _convert(raw: _RawReconstructionResult) -> ReconstructionResult:
    d = raw.to_dict()
    return ReconstructionResult(
        id=str(uuid.uuid4()),
        output_folder=str(Path(d.get("point_cloud_path", "")).parent or ""),
        point_cloud_path=d.get("point_cloud_path") or None,
        mesh_path=d.get("mesh_path") or None,
        camera_poses_path=d.get("camera_pose_path") or None,
        orthomosaic_path=d.get("orthomosaic_path") or None,
        dsm_path=d.get("dsm_path") or None,
        dtm_path=d.get("dtm_path") or None,
        textured_mesh_obj_path=d.get("textured_mesh_obj_path") or None,
        texture_image_path=d.get("texture_image_path") or None,
        digital_twin_path=d.get("digital_twin_path") or None,
        defect_projection_path=d.get("crack_cloud_path") or None,
        quality_metrics={
            "frame_count": d.get("frame_count"),
            "processed_pairs": d.get("processed_pairs"),
            "failed_pairs": d.get("failed_pairs"),
            "total_points": d.get("total_points"),
            "crack_points": d.get("crack_points"),
            "cache_hits": d.get("cache_hits"),
            "cache_misses": d.get("cache_misses"),
            "execution_mode_used": d.get("execution_mode_used"),
            "processing_profile": d.get("processing_profile"),
        },
        warnings=list(d.get("warnings") or []),
        raw=d,
    )


def run_reconstruction(config: ReconstructionConfig) -> ReconstructionResult:
    """Run reconstruction pipeline with the configured backend."""
    image_folder = Path(config.image_folder)
    if not image_folder.exists():
        raise AppError(ERR_INVALID_INPUT, f"Image folder not found: {image_folder}")
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if config.profile not in available_reconstruction_profiles():
        raise AppError(
            ERR_INVALID_INPUT,
            f"Unknown reconstruction profile: {config.profile!r}",
            recovery_action=f"Available profiles: {', '.join(available_reconstruction_profiles())}",
        )

    reconstructor = CustomDroneReconstructor(
        max_points=int(config.max_points),
        profile=config.profile,
        execution_mode=config.execution_mode,
        use_cache=bool(config.reuse_cache),
        cloud_endpoint=config.cloud_endpoint or "",
    )
    raw = reconstructor.reconstruct(
        image_dir=image_folder,
        crack_mask_dir=Path(config.mask_folder) if config.mask_folder else None,
        output_dir=output_folder,
        profile=config.profile,
        execution_mode=config.execution_mode,
        use_cache=bool(config.reuse_cache),
        cloud_endpoint=config.cloud_endpoint or "",
    )
    result = _convert(raw)
    # Persist
    (output_folder / "reconstruction_summary.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return result


def import_reconstruction_folder(project_id: str, folder_path: Path | str) -> ReconstructionResult:
    """Scan a folder for point clouds / meshes / camera poses and register it."""
    fp = Path(folder_path)
    if not fp.exists() or not fp.is_dir():
        raise AppError(ERR_INVALID_INPUT, f"Reconstruction folder not found: {fp}")

    def _find(suffixes: tuple[str, ...]) -> str | None:
        for ext in suffixes:
            for p in fp.rglob(f"*{ext}"):
                if p.is_file():
                    return str(p)
        return None

    pc = _find((".ply", ".pcd", ".las", ".laz", ".xyz"))
    mesh = _find((".obj", ".stl", ".glb", ".gltf", ".ply"))
    poses = _find(("camera_poses.json", "poses.json", "cameras.json"))

    quality_path = fp / "quality_metrics.json"
    qm: dict[str, Any] = {}
    if quality_path.exists():
        try:
            qm = json.loads(quality_path.read_text(encoding="utf-8"))
        except Exception:
            qm = {}

    result = ReconstructionResult(
        id=str(uuid.uuid4()),
        output_folder=str(fp),
        point_cloud_path=pc,
        mesh_path=mesh,
        camera_poses_path=poses,
        orthomosaic_path=_find(("ortho.tif", "orthomosaic.tif", ".tif")),
        dsm_path=_find(("dsm.tif",)),
        dtm_path=_find(("dtm.tif",)),
        textured_mesh_obj_path=_find(("textured_mesh.obj",)),
        texture_image_path=_find(("texture.png", "texture.jpg")),
        digital_twin_path=_find(("digital_twin.json",)),
        defect_projection_path=_find(("crack_cloud.ply", "defect_cloud.ply")),
        quality_metrics=qm,
    )
    (fp / "imported_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def load_point_cloud(path: Path | str) -> PointCloudData:
    """Load point cloud using open3d when available; fall back to PLY header parse."""
    p = Path(path)
    if not p.exists():
        raise AppError(ERR_INVALID_INPUT, f"Point cloud not found: {p}")
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(p))
        pts = pcd.points
        colors = pcd.colors
        count = len(pts)
        import numpy as np
        arr = np.asarray(pts)
        if arr.size == 0:
            bounds = {"min": [0, 0, 0], "max": [0, 0, 0]}
        else:
            bounds = {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
        return PointCloudData(points_path=str(p), point_count=count,
                              has_colors=bool(len(colors)), bounds=bounds)
    except Exception:
        # Minimal PLY header parse
        count = 0
        has_colors = False
        try:
            with p.open("rb") as f:
                in_header = True
                while in_header:
                    line = f.readline().decode("utf-8", errors="replace").strip()
                    if line.startswith("element vertex"):
                        count = int(line.split()[-1])
                    if line in ("red", "property uchar red"):
                        has_colors = True
                    if line == "end_header":
                        in_header = False
        except Exception:
            pass
        return PointCloudData(points_path=str(p), point_count=count, has_colors=has_colors,
                              bounds={"min": [0, 0, 0], "max": [0, 0, 0]})


def load_mesh(path: Path | str) -> MeshData:
    p = Path(path)
    if not p.exists():
        raise AppError(ERR_INVALID_INPUT, f"Mesh not found: {p}")
    try:
        import trimesh
        m = trimesh.load(str(p), force="mesh")
        return MeshData(
            mesh_path=str(p),
            vertex_count=int(len(m.vertices)),
            face_count=int(len(m.faces)),
            has_texture=bool(getattr(m.visual, "material", None)),
        )
    except Exception:
        try:
            import open3d as o3d
            m = o3d.io.read_triangle_mesh(str(p))
            return MeshData(
                mesh_path=str(p),
                vertex_count=int(len(m.vertices)),
                face_count=int(len(m.triangles)),
                has_texture=bool(m.textures and len(m.textures) > 0),
            )
        except Exception:
            return MeshData(mesh_path=str(p), vertex_count=0, face_count=0, has_texture=False)


def project_defects_to_3d(
    defect_summary_path: Path | str,
    reconstruction_summary_path: Path | str,
    output_path: Path | str,
) -> DefectProjectionResult:
    """Locate 2-D defects on the reconstructed surface and write a georeferenced layer.

    When the reconstruction came from the COLMAP engine and is georeferenced, each
    defect is ray-cast from its recovering camera onto the DSM, merged across views,
    and emitted as GeoJSON with real coordinates and square-metre areas alongside the
    JSON side-car.

    For a reconstruction that is not georeferenced (no geotags, or the custom engine)
    this falls back to the image-name point linking that was previously the only
    behaviour, and records why in the payload rather than implying a 3-D fix.
    """
    defects_path = Path(defect_summary_path)
    recon_path = Path(reconstruction_summary_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not defects_path.exists() or not recon_path.exists():
        raise AppError(ERR_INVALID_INPUT,
                       "Defect summary or reconstruction summary not found.")
    defects = json.loads(defects_path.read_text(encoding="utf-8")).get("defects", [])
    recon = json.loads(recon_path.read_text(encoding="utf-8"))

    raw = recon.get("raw", {}) if isinstance(recon.get("raw"), dict) else {}
    reconstruction_dir = (
        raw.get("reconstruction_dir")
        or recon.get("reconstruction_dir")
        or (Path(raw.get("point_cloud_path") or recon.get("point_cloud_path") or ".").parent)
    )

    geo_result: dict[str, Any] = {"status": "skipped", "reason": "Reconstruction directory not resolvable."}
    if reconstruction_dir:
        from .defect_projection import project_run_defects

        detections = [
            {
                "image": d.get("image_path") or d.get("image") or "",
                "mask": d.get("mask_path") or d.get("mask") or "",
                "bbox": d.get("bbox"),
                "defect_type": d.get("defect_type") or d.get("type") or "defect",
                "severity": d.get("severity"),
                "confidence": d.get("confidence") or d.get("score"),
            }
            for d in defects
        ]
        try:
            geo_result = project_run_defects(
                reconstruction_dir=reconstruction_dir,
                detections=detections,
                output_path=out_path.with_name(f"{out_path.stem}.geojson"),
            )
        except Exception as exc:  # noqa: BLE001
            geo_result = {"status": "error", "reason": f"3-D projection failed: {exc}"}

    point_link_path = raw.get("point_link_path") or recon.get("point_link_path")
    point_index: dict[str, list[dict[str, Any]]] = {}
    if point_link_path and Path(point_link_path).exists():
        try:
            links = json.loads(Path(point_link_path).read_text(encoding="utf-8"))
            for entry in (links if isinstance(links, list) else []):
                # The reconstructor writes `source_image_name`; older payloads used
                # `image`. Accepting both is what makes this linking work at all.
                raw_name = entry.get("source_image_name") or entry.get("image") or entry.get("image_path") or ""
                image_name = Path(str(raw_name)).name
                if not image_name:
                    continue
                point_index.setdefault(image_name, []).append(entry)
        except Exception:
            pass

    projected: list[dict[str, Any]] = []
    for d in defects:
        image_name = Path(d.get("image_path", "")).name
        linked = point_index.get(image_name, [])
        if not linked:
            continue
        projected.append({
            "defect_id": d.get("id"),
            "defect_type": d.get("defect_type"),
            "image": image_name,
            "linked_points": linked[:50],
        })

    georeferenced = geo_result.get("status") == "ok"
    payload = {
        "defect_count": len(defects),
        "projected_count": int(geo_result.get("defect_count", 0)) if georeferenced else len(projected),
        "projections": projected,
        "georeferenced": georeferenced,
        "geo_projection": geo_result,
        "created_at": _now_iso(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return DefectProjectionResult(
        id=str(uuid.uuid4()),
        defect_count=len(defects),
        projected_count=int(payload["projected_count"]),
        output_path=str(geo_result["path"]) if georeferenced else str(out_path),
    )


def calculate_reconstruction_quality(result: ReconstructionResult) -> ReconstructionQuality:
    qm = result.quality_metrics or {}
    return ReconstructionQuality(
        image_count=int(qm.get("frame_count", 0) or 0),
        sparse_point_count=int(qm.get("processed_pairs", 0) or 0),
        dense_point_count=int(qm.get("total_points", 0) or 0),
        coverage_pct=100.0 * (1.0 - float(qm.get("failed_pairs", 0) or 0) / max(1, int(qm.get("processed_pairs", 1) or 1))),
        failed_image_count=int(qm.get("failed_pairs", 0) or 0),
    )
