"""Processing run orchestration — stage-by-stage execution with progress + status.

A `ProcessingRun` is a JSON-backed record under
`<project>/processing/<run_id>/`. Each stage has a status, log file, and
output artifact list. Stages may delegate to existing pipeline modules
(`core.pipeline`, `core.defect_engine`, `core.crack_engine`,
`core.reconstruction_engine`).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .data_library import get_dataset
from .defect_engine import DefectDetectionConfig, run_defect_detection
from .errors import AppError, ERR_DATASET_MISSING, ERR_PIPELINE_INPUTS
from .events import (
    PROCESSING_CANCELLED,
    PROCESSING_COMPLETED,
    PROCESSING_FAILED,
    PROCESSING_PROGRESS,
    PROCESSING_STAGE_COMPLETED,
    PROCESSING_STARTED,
    publish_event,
)
from .pipeline import PipelineConfig, StructuralFaultPipeline
from .reconstruction_engine import ReconstructionConfig, run_reconstruction
from .workflows import get_workflow_template


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


STATUS_PENDING = "pending"
STATUS_READY = "ready"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CANCELLED = "cancelled"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PipelineStage:
    id: str
    name: str
    status: str = STATUS_PENDING
    required_inputs: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    progress_percent: float = 0.0
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingRun:
    id: str
    project_id: str
    dataset_id: str
    workflow_id: str
    stages: list[PipelineStage]
    status: str
    output_dir: str
    config: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    progress_percent: float = 0.0
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "output_dir": self.output_dir,
            "config": self.config,
            "stages": [s.to_dict() for s in self.stages],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "progress_percent": self.progress_percent,
            "created_at": self.created_at,
        }


@dataclass
class PipelineReadiness:
    ok: bool
    issues: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    stage_id: str
    status: str
    artifacts: list[str]
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStatus:
    run_id: str
    status: str
    progress_percent: float
    active_stage: str | None
    completed_stages: list[str]
    failed_stages: list[str]
    error: str | None = None


# ── Storage ───────────────────────────────────────────────────────────────────

def _run_dir(project_root: Path, run_id: str) -> Path:
    p = Path(project_root) / "processing" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_run(project_root: Path, run: ProcessingRun) -> None:
    path = _run_dir(project_root, run.id) / "run.json"
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")


def _load_run(project_root: Path, run_id: str) -> ProcessingRun | None:
    path = _run_dir(project_root, run_id) / "run.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    stages = [
        PipelineStage(**{k: v for k, v in s.items() if k in PipelineStage.__dataclass_fields__})
        for s in data.get("stages", [])
    ]
    return ProcessingRun(
        id=data.get("id", run_id),
        project_id=data.get("project_id", ""),
        dataset_id=data.get("dataset_id", ""),
        workflow_id=data.get("workflow_id", ""),
        stages=stages,
        status=data.get("status", STATUS_PENDING),
        output_dir=data.get("output_dir", str(_run_dir(project_root, run_id))),
        config=data.get("config", {}),
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        error=data.get("error"),
        progress_percent=float(data.get("progress_percent", 0.0) or 0.0),
        created_at=data.get("created_at", _now_iso()),
    )


# ── Stage definitions ─────────────────────────────────────────────────────────

DEFAULT_STAGE_INPUTS = {
    "dataset_validation": ["dataset"],
    "image_quality_check": ["dataset"],
    "image_alignment": ["dataset"],
    "reconstruction": ["dataset"],
    "defect_detection": ["dataset", "model_registry"],
    "structural_detection": ["dataset", "model_registry"],
    "solar_defect_detection": ["dataset", "model_registry"],
    "crack_detection": ["dataset"],
    "crack_propagation": ["crack_mask"],
    "measurement_extraction": ["dataset"],
    "risk_scoring": ["defect_results"],
    "defect_projection": ["reconstruction", "defect_results"],
    "report_generation": ["project"],
    "volume_estimation": ["reconstruction"],
    "survey_change": ["earlier_dsm", "later_dsm"],
    "selected_roi_change": ["earlier_dsm", "later_dsm", "roi_polygon"],
    "semantic_segmentation": ["orthomosaic", "semantic_model", "semantic_model_manifest"],
}


def _stages_for_workflow(workflow_id: str) -> list[PipelineStage]:
    try:
        template = get_workflow_template(workflow_id)
        stage_ids = list(template.processing_stages) or ["dataset_validation", "defect_detection", "report_generation"]
    except Exception:
        stage_ids = ["dataset_validation", "defect_detection", "report_generation"]
    return [
        PipelineStage(
            id=sid,
            name=sid.replace("_", " ").title(),
            status=STATUS_PENDING,
            required_inputs=list(DEFAULT_STAGE_INPUTS.get(sid, [])),
        )
        for sid in stage_ids
    ]


# ── Cancellation ──────────────────────────────────────────────────────────────

_cancel_flags: dict[str, threading.Event] = {}
_cancel_lock = threading.RLock()


def _cancel_event(run_id: str) -> threading.Event:
    with _cancel_lock:
        if run_id not in _cancel_flags:
            _cancel_flags[run_id] = threading.Event()
        return _cancel_flags[run_id]


def _is_cancelled(run_id: str) -> bool:
    return _cancel_event(run_id).is_set()


# ── Public API ────────────────────────────────────────────────────────────────

def create_processing_run(
    project_root: Path | str,
    project_id: str,
    dataset_id: str,
    workflow_id: str,
    config: dict[str, Any] | None = None,
) -> ProcessingRun:
    """Create new processing run folder, build stages from workflow, save config."""
    run_id = str(uuid.uuid4())
    project_root = Path(project_root)
    out_dir = _run_dir(project_root, run_id)
    stages = _stages_for_workflow(workflow_id)
    run = ProcessingRun(
        id=run_id,
        project_id=project_id,
        dataset_id=dataset_id,
        workflow_id=workflow_id,
        stages=stages,
        status=STATUS_PENDING,
        output_dir=str(out_dir),
        config=dict(config or {}),
    )
    _save_run(project_root, run)
    return run


def validate_pipeline_inputs(project_root: Path | str, run_id: str) -> PipelineReadiness:
    """Verify required dataset, model files, output paths and processing settings."""
    pr = Path(project_root)
    run = _load_run(pr, run_id)
    if run is None:
        return PipelineReadiness(ok=False, issues=[f"Run not found: {run_id}"])
    issues: list[str] = []
    notes: list[str] = []

    needs_dataset = any("dataset" in stage.required_inputs for stage in run.stages)
    ds = get_dataset(pr, run.dataset_id) if run.dataset_id else None
    if needs_dataset:
        if ds is None:
            issues.append("Dataset missing.")
        elif ds.image_count == 0:
            issues.append("Dataset has no images.")

    if any(stage.id in {"survey_change", "selected_roi_change"} for stage in run.stages):
        for label, key in (
            ("Earlier DSM", "earlier_dsm_path"),
            ("Later DSM", "later_dsm_path"),
        ):
            value = run.config.get(key)
            if not value:
                issues.append(f"{label} missing ({key}).")
            elif not Path(str(value)).is_file():
                issues.append(f"{label} does not exist: {value}")

    if any(stage.id == "selected_roi_change" for stage in run.stages):
        polygon = run.config.get("roi_polygon_xy")
        if not polygon or len(polygon) < 3:
            issues.append("ROI polygon missing (roi_polygon_xy).")

    if any(stage.id == 'semantic_segmentation' for stage in run.stages):
        semantic_paths = (
            ('Orthomosaic', 'orthomosaic_path'),
            ('Semantic model', 'semantic_model_path'),
            ('Semantic model manifest', 'semantic_model_manifest_path'),
        )
        for label, key in semantic_paths:
            value = run.config.get(key)
            if not value:
                issues.append(f'{label} missing ({key}).')
            elif not Path(str(value)).is_file():
                issues.append(f'{label} does not exist: {value}')
        manifest_path = run.config.get('semantic_model_manifest_path')
        if manifest_path and Path(str(manifest_path)).is_file():
            try:
                from .semantic_engine import load_semantic_manifest

                _, semantic_model, _ = load_semantic_manifest(manifest_path)
                if not semantic_model.task_trained:
                    issues.append('Semantic model manifest describes an untrained foundation initializer.')
            except Exception as exc:
                issues.append(f'Semantic model manifest is invalid: {exc}')

    needs_model = any("model_registry" in s.required_inputs for s in run.stages)
    if needs_model:
        from .models import model_status
        key = str(run.config.get("structural_model_key") or "structural_multiclass_detector")
        info = model_status(key)
        if not info.get("exists"):
            notes.append(f"AI model {key!r} missing — AI stages will fall back to classical detectors.")

    if not Path(run.output_dir).exists():
        try:
            Path(run.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            issues.append(f"Cannot create output folder: {exc}")

    return PipelineReadiness(ok=not issues, issues=issues, notes=notes)


def _emit_progress(run: ProcessingRun, project_root: Path, percent: float, message: str = "") -> None:
    run.progress_percent = float(max(0.0, min(100.0, percent)))
    _save_run(project_root, run)
    publish_event(PROCESSING_PROGRESS, {
        "run_id": run.id,
        "percent": run.progress_percent,
        "message": message,
    })


def _mark_stage(
    run: ProcessingRun,
    project_root: Path,
    stage: PipelineStage,
    status: str,
    artifacts: list[str] | None = None,
    error: str | None = None,
) -> None:
    stage.status = status
    if status == STATUS_RUNNING:
        stage.started_at = _now_iso()
    if status in (STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED, STATUS_CANCELLED):
        stage.completed_at = _now_iso()
        stage.progress_percent = 100.0
    if artifacts:
        stage.output_artifacts = list(artifacts)
    if error:
        stage.error_message = error
    _save_run(project_root, run)
    publish_event(PROCESSING_STAGE_COMPLETED, {
        "run_id": run.id,
        "stage_id": stage.id,
        "status": status,
        "error": error,
    })


def run_pipeline_stage(
    project_root: Path | str,
    run_id: str,
    stage_id: str,
) -> StageResult:
    """Execute a single stage."""
    pr = Path(project_root)
    run = _load_run(pr, run_id)
    if run is None:
        raise AppError(ERR_PIPELINE_INPUTS, f"Run not found: {run_id}")
    stage = next((s for s in run.stages if s.id == stage_id), None)
    if stage is None:
        raise AppError(ERR_PIPELINE_INPUTS, f"Stage not found: {stage_id}")

    _mark_stage(run, pr, stage, STATUS_RUNNING)
    try:
        artifacts = _execute_stage(pr, run, stage)
        _mark_stage(run, pr, stage, STATUS_COMPLETED, artifacts=artifacts)
        return StageResult(stage_id=stage_id, status=STATUS_COMPLETED, artifacts=artifacts)
    except AppError as exc:
        _mark_stage(run, pr, stage, STATUS_FAILED, error=exc.user_message)
        return StageResult(stage_id=stage_id, status=STATUS_FAILED, artifacts=[], error=exc.user_message)
    except Exception as exc:
        _mark_stage(run, pr, stage, STATUS_FAILED, error=str(exc))
        return StageResult(stage_id=stage_id, status=STATUS_FAILED, artifacts=[], error=str(exc))


def run_pipeline(
    project_root: Path | str,
    run_id: str,
    selected_stages: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> ProcessingRun:
    """Execute selected pipeline stages in order. Stops on blocking failure."""
    pr = Path(project_root)
    run = _load_run(pr, run_id)
    if run is None:
        raise AppError(ERR_PIPELINE_INPUTS, f"Run not found: {run_id}")
    readiness = validate_pipeline_inputs(pr, run_id)
    if not readiness.ok:
        run.status = STATUS_FAILED
        run.error = "; ".join(readiness.issues)
        run.completed_at = _now_iso()
        _save_run(pr, run)
        publish_event(PROCESSING_FAILED, {"run_id": run.id, "error": run.error})
        raise AppError(ERR_PIPELINE_INPUTS, run.error, recovery_action="Fix listed inputs and retry.")

    run.status = STATUS_RUNNING
    run.started_at = _now_iso()
    _save_run(pr, run)
    publish_event(PROCESSING_STARTED, {"run_id": run.id, "workflow_id": run.workflow_id})

    selected = set(selected_stages) if selected_stages else None
    cancel = _cancel_event(run.id)
    cancel.clear()

    total = max(1, sum(1 for s in run.stages if selected is None or s.id in selected))
    done = 0
    for stage in run.stages:
        if selected is not None and stage.id not in selected:
            _mark_stage(run, pr, stage, STATUS_SKIPPED)
            continue
        if cancel.is_set():
            _mark_stage(run, pr, stage, STATUS_CANCELLED)
            continue
        result = run_pipeline_stage(pr, run.id, stage.id)
        # run_pipeline_stage reloads and persists its own ProcessingRun instance.
        # Synchronise before _emit_progress saves this outer instance, otherwise it
        # overwrites the completed stage with the stale pending copy held here.
        persisted = _load_run(pr, run.id)
        if persisted is not None:
            run.stages = persisted.stages
        done += 1
        if progress_callback:
            try:
                progress_callback(100.0 * done / total, f"{stage.name} → {result.status}")
            except Exception:
                pass
        _emit_progress(run, pr, 100.0 * done / total, f"{stage.name} {result.status}")
        if result.status == STATUS_FAILED:
            run.status = STATUS_FAILED
            run.error = result.error
            run.completed_at = _now_iso()
            _save_run(pr, run)
            publish_event(PROCESSING_FAILED, {"run_id": run.id, "stage": stage.id, "error": result.error})
            return run

    if cancel.is_set():
        run.status = STATUS_CANCELLED
        publish_event(PROCESSING_CANCELLED, {"run_id": run.id})
    else:
        run.status = STATUS_COMPLETED
        publish_event(PROCESSING_COMPLETED, {"run_id": run.id})
    run.completed_at = _now_iso()
    _save_run(pr, run)
    return run


def stop_processing_run(project_root: Path | str, run_id: str) -> None:
    """Request cancellation; allow current stage to stop safely."""
    _cancel_event(run_id).set()
    pr = Path(project_root)
    run = _load_run(pr, run_id)
    if run and run.status == STATUS_RUNNING:
        run.status = STATUS_CANCELLED
        _save_run(pr, run)


def get_processing_status(project_root: Path | str, run_id: str) -> ProcessingStatus:
    run = _load_run(Path(project_root), run_id)
    if run is None:
        return ProcessingStatus(run_id=run_id, status="missing", progress_percent=0.0,
                                active_stage=None, completed_stages=[], failed_stages=[],
                                error="Run not found")
    active = next((s.id for s in run.stages if s.status == STATUS_RUNNING), None)
    completed = [s.id for s in run.stages if s.status == STATUS_COMPLETED]
    failed = [s.id for s in run.stages if s.status == STATUS_FAILED]
    return ProcessingStatus(
        run_id=run.id,
        status=run.status,
        progress_percent=run.progress_percent,
        active_stage=active,
        completed_stages=completed,
        failed_stages=failed,
        error=run.error,
    )


def list_processing_runs(project_root: Path | str, project_id: str | None = None) -> list[ProcessingRun]:
    pr = Path(project_root)
    root = pr / "processing"
    if not root.exists():
        return []
    out: list[ProcessingRun] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        rid = run_dir.name
        run = _load_run(pr, rid)
        if run is None:
            continue
        if project_id and run.project_id != project_id:
            continue
        out.append(run)
    return out


# ── Stage execution ──────────────────────────────────────────────────────────-

def _reconstruction_artifacts(run: ProcessingRun) -> dict[str, str]:
    """Locate the rasters and defect layer a prior reconstruction stage produced.

    Later stages need the DSM/DTM by path, and the reconstruction stage records only
    its summary JSON as an artifact. The summary names the rest, so it is read when
    present and the output directory is scanned as a fallback for runs whose summary
    predates those fields.
    """
    found: dict[str, str] = {}

    summary_path = next(
        (Path(p) for s in run.stages if s.id in {"reconstruction", "image_alignment"}
         for p in s.output_artifacts if str(p).endswith(".json")),
        None,
    )
    if summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        for key, field_names in (
            ("dsm", ("dsm_cog_path", "dsm_path")),
            ("dtm", ("dtm_cog_path", "dtm_path")),
            ("orthomosaic", ("orthomosaic_cog_path", "orthomosaic_path")),
            ("defects_geojson", ("defects_geojson_path",)),
        ):
            for name in field_names:
                value = summary.get(name)
                if value and Path(value).exists():
                    found[key] = str(value)
                    break

    search_root = Path(run.output_dir)
    for key, filename in (
        ("dsm", "dsm.tif"),
        ("dtm", "dtm.tif"),
        ("orthomosaic", "orthomosaic.tif"),
    ):
        if key not in found:
            match = next(iter(sorted(search_root.rglob(filename))), None)
            if match:
                found[key] = str(match)
    if "defects_geojson" not in found:
        match = next(iter(sorted(search_root.rglob("*defect*.geojson"))), None)
        if match:
            found["defects_geojson"] = str(match)

    return found


def _execute_stage(project_root: Path, run: ProcessingRun, stage: PipelineStage) -> list[str]:
    """Dispatch known stages to underlying engines. Returns artifact paths."""
    sid = stage.id.lower()
    cancel = _cancel_event(run.id)
    cfg = run.config or {}
    ds_id = run.dataset_id

    if sid == "dataset_validation":
        from .data_library import validate_dataset
        report = validate_dataset(project_root, ds_id)
        artifact = Path(run.output_dir) / "dataset_validation.json"
        artifact.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return [str(artifact)]

    if sid in ("defect_detection", "structural_detection", "crack_detection", "solar_defect_detection"):
        d_cfg = DefectDetectionConfig(
            mode=str(cfg.get("defect_mode", "hybrid")),
            model_key=str(cfg.get("structural_model_key", "structural_multiclass_detector")),
            threshold=float(cfg.get("defect_threshold", 0.25)),
            min_area_px=int(cfg.get("min_area_px", 30)),
            classes=list(cfg.get("classes", []) or []),
        )
        result = run_defect_detection(project_root, ds_id, config=d_cfg)
        return [str(Path(result.output_dir) / "defects.json")]

    if sid in ("reconstruction", "image_alignment"):
        ds = get_dataset(project_root, ds_id)
        if ds is None:
            raise AppError(ERR_DATASET_MISSING, "Dataset missing for reconstruction stage.")
        image_folder = cfg.get("image_folder") or ds.root_dir
        recon_cfg = ReconstructionConfig(
            image_folder=str(image_folder),
            output_folder=str(Path(run.output_dir) / "reconstruction"),
            mask_folder=cfg.get("mask_folder") or None,
            profile=str(cfg.get("reconstruction_profile", "standard")),
            execution_mode=str(cfg.get("reconstruction_execution_mode", "local")),
            reuse_cache=bool(cfg.get("reconstruction_use_cache", True)),
            cloud_endpoint=str(cfg.get("reconstruction_cloud_endpoint", "")),
            max_points=int(cfg.get("reconstruction_max_points", 150_000)),
        )
        result = run_reconstruction(recon_cfg)
        return [str(Path(result.output_folder) / "reconstruction_summary.json")]

    if sid == "crack_propagation":
        from .crack_engine import CrackPropagationConfig, run_crack_propagation
        cp_cfg = CrackPropagationConfig(
            image_path=str(cfg.get("crack_image_path", "")),
            mask_path=cfg.get("crack_mask_path"),
            pixel_size_mm_per_px=float(cfg.get("pixel_size_mm_per_px", 0.5)),
            sigma_nominal_mpa=float(cfg.get("sigma_nominal_mpa", 35.0)),
            delta_sigma_mpa=float(cfg.get("delta_sigma_mpa", 12.0)),
            cycles_per_year=float(cfg.get("cycles_per_year", 1_000_000.0)),
            horizon_years=float(cfg.get("horizon_years", 1.0)),
            steps=int(cfg.get("forecast_steps", 6)),
            structure_type=str(cfg.get("structure_type", "generic")),
            material_profile=str(cfg.get("material_type", "concrete")),
            mode=str(cfg.get("propagation_mode", "physics_informed")),
        )
        if not cp_cfg.image_path:
            raise AppError(ERR_PIPELINE_INPUTS, "crack_image_path required for crack_propagation stage.")
        result = run_crack_propagation(cp_cfg, output_dir=Path(run.output_dir) / "crack_propagation")
        return [str(Path(result.output_dir) / "crack_propagation.json")]

    if sid == "report_generation":
        from .report_engine import ReportConfig, generate_report
        rcfg = ReportConfig(
            project_id=run.project_id,
            title=str(cfg.get("report_title", "Inspection Report")),
            report_type=str(cfg.get("report_type", "standard")),
        )
        result = generate_report(rcfg)
        return [str(result.html_path)] + ([str(result.pdf_path)] if result.pdf_path else [])

    if sid == "measurement_extraction":
        from .dsm_analysis import extract_measurements
        artifacts = _reconstruction_artifacts(run)
        report = extract_measurements(
            defects_geojson=cfg.get("defects_geojson") or artifacts.get("defects_geojson"),
            dsm_path=cfg.get("dsm_path") or artifacts.get("dsm"),
            dtm_path=cfg.get("dtm_path") or artifacts.get("dtm"),
        )
        artifact = Path(run.output_dir) / "measurements_summary.json"
        artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return [str(artifact)]

    if sid == "risk_scoring":
        from .risk_scoring import score_run_risk
        defect_summary = next(
            (Path(p) for s in run.stages if "defect" in s.id and "projection" not in s.id
             for p in s.output_artifacts),
            None,
        )
        report = score_run_risk(
            defect_summary,
            defects_geojson=_reconstruction_artifacts(run).get("defects_geojson"),
            structure_type=str(cfg.get("structure_type", "generic")),
            asset_id=run.project_id,
        )
        artifact = Path(run.output_dir) / "risk_summary.json"
        artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return [str(artifact)]

    if sid == "image_quality_check":
        from .data_library import validate_dataset
        report = validate_dataset(project_root, ds_id)
        artifact = Path(run.output_dir) / "image_quality.json"
        artifact.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return [str(artifact)]

    if sid == "volume_estimation":
        from .dsm_analysis import estimate_volume
        artifacts = _reconstruction_artifacts(run)
        dsm_path = cfg.get("dsm_path") or artifacts.get("dsm")
        if not dsm_path or not Path(dsm_path).exists():
            raise AppError(
                ERR_PIPELINE_INPUTS,
                "Volume estimation needs a DSM from the reconstruction stage.",
                recovery_action="Run reconstruction with the COLMAP engine first, or set dsm_path in the run config.",
            )
        dtm_path = cfg.get("dtm_path") or artifacts.get("dtm")
        polygon_xy = cfg.get("volume_polygon_xy") or None
        base_elevation_m = cfg.get("volume_base_elevation_m")
        if run.workflow_id == "stockpile_measurement":
            if not polygon_xy:
                raise AppError(
                    ERR_PIPELINE_INPUTS,
                    "Stockpile measurement needs a selected pile polygon.",
                    recovery_action="Draw the stockpile boundary on the DSM and retry.",
                )
            from .survey_intelligence import create_stockpile_package

            package = create_stockpile_package(
                dsm_path,
                Path(run.output_dir) / "stockpile",
                polygon_xy=polygon_xy,
                dtm_path=dtm_path,
                base_elevation_m=base_elevation_m,
            )
            return package.artifact_paths()
        report = estimate_volume(
            dsm_path,
            dtm_path=dtm_path,
            polygon_xy=polygon_xy,
            base_elevation_m=base_elevation_m,
        )
        artifact = Path(run.output_dir) / "volume_estimation.json"
        artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return [str(artifact)]

    if sid == "survey_change":
        from .survey_intelligence import create_surface_change_package

        earlier_dsm = cfg.get("earlier_dsm_path")
        later_dsm = cfg.get("later_dsm_path")
        if not earlier_dsm or not later_dsm:
            raise AppError(
                ERR_PIPELINE_INPUTS,
                "Survey change requires earlier_dsm_path and later_dsm_path.",
                recovery_action="Select two aligned, georeferenced DSM GeoTIFFs.",
            )
        package = create_surface_change_package(
            earlier_dsm,
            later_dsm,
            Path(run.output_dir) / "survey_change",
            change_threshold_m=float(cfg.get("change_threshold_m", 0.05)),
            min_region_area_m2=float(cfg.get("min_region_area_m2", 1.0)),
        )
        return package.artifact_paths()

    if sid == "selected_roi_change":
        from .survey_intelligence import create_selected_roi_change_package

        earlier_dsm = cfg.get("earlier_dsm_path")
        later_dsm = cfg.get("later_dsm_path")
        polygon_xy = cfg.get("roi_polygon_xy")
        if not earlier_dsm or not later_dsm or not polygon_xy:
            raise AppError(
                ERR_PIPELINE_INPUTS,
                "Selected ROI change requires two DSMs and roi_polygon_xy.",
                recovery_action="Select aligned DSMs and draw a stockpile or pit boundary.",
            )
        package = create_selected_roi_change_package(
            earlier_dsm,
            later_dsm,
            Path(run.output_dir) / "selected_roi_change",
            polygon_xy=polygon_xy,
            roi_type=str(cfg.get("roi_type", "stockpile")),
            roi_name=str(cfg.get("roi_name", "")),
            change_threshold_m=float(cfg.get("change_threshold_m", 0.05)),
            min_region_area_m2=float(cfg.get("min_region_area_m2", 1.0)),
        )
        return package.artifact_paths()

    if sid == 'semantic_segmentation':
        from .semantic_engine import (
            ONNXSemanticPredictor,
            SemanticInferenceConfig,
            load_semantic_manifest,
            run_semantic_inference,
        )

        orthomosaic = cfg.get('orthomosaic_path')
        model_path = cfg.get('semantic_model_path')
        manifest_path = cfg.get('semantic_model_manifest_path')
        if not orthomosaic or not model_path or not manifest_path:
            raise AppError(
                ERR_PIPELINE_INPUTS,
                'Semantic segmentation requires orthomosaic, model and manifest paths.',
                recovery_action='Select a georeferenced orthomosaic and a trained semantic ONNX package.',
            )
        schema, model, defaults = load_semantic_manifest(manifest_path)
        device = str(cfg.get('semantic_device', defaults.get('device', 'cuda')))
        predictor = ONNXSemanticPredictor(
            model_path,
            device=device,
            mean=defaults.get('mean', (0.485, 0.456, 0.406)),
            std=defaults.get('std', (0.229, 0.224, 0.225)),
        )
        package = run_semantic_inference(
            orthomosaic,
            Path(run.output_dir) / 'semantic_segmentation',
            schema=schema,
            model=model,
            predictor=predictor,
            config=SemanticInferenceConfig(
                tile_size=int(cfg.get('semantic_tile_size', defaults.get('tile_size', 518))),
                overlap=int(cfg.get('semantic_overlap', defaults.get('overlap', 126))),
                device=device,
                allow_cpu=bool(cfg.get('semantic_allow_cpu', defaults.get('allow_cpu', False))),
                max_cpu_pixels=int(cfg.get(
                    'semantic_max_cpu_pixels', defaults.get('max_cpu_pixels', 4_000_000)
                )),
                min_polygon_area_m2=float(cfg.get(
                    'semantic_min_polygon_area_m2', defaults.get('min_polygon_area_m2', 1.0)
                )),
                polygonize_background=bool(cfg.get(
                    'semantic_polygonize_background', defaults.get('polygonize_background', False)
                )),
                input_bands=tuple(cfg.get(
                    'semantic_input_bands', defaults.get('input_bands', (1, 2, 3))
                )),
            ),
        )
        return package.artifact_paths()

    if sid == "defect_projection":
        from .reconstruction_engine import project_defects_to_3d
        # Look up artifacts from prior stages
        recon_summary = next(
            (Path(p) for s in run.stages if s.id == "reconstruction" for p in s.output_artifacts),
            None,
        )
        defect_summary = next(
            (Path(p) for s in run.stages if "defect" in s.id for p in s.output_artifacts),
            None,
        )
        if not recon_summary or not defect_summary:
            raise AppError(ERR_PIPELINE_INPUTS, "Defect projection requires reconstruction + defect stages.")
        out = Path(run.output_dir) / "defect_projection.json"
        project_defects_to_3d(defect_summary, recon_summary, out)
        return [str(out)]

    if sid == "structural_health":
        # Delegate to original full pipeline
        ds = get_dataset(project_root, ds_id)
        if ds is None:
            raise AppError(ERR_DATASET_MISSING, "Dataset missing for structural_health stage.")
        pipeline = StructuralFaultPipeline(config=PipelineConfig(
            structural_model_key=str(cfg.get("structural_model_key", "structural_multiclass_detector")),
            asset_id=run.project_id,
            structure_type=str(cfg.get("structure_type", "generic")),
            material_type=str(cfg.get("material_type", "concrete")),
        ))
        result = pipeline.run(
            image_dir=ds.root_dir,
            output_dir=Path(run.output_dir) / "structural_health",
            progress_callback=None,
        )
        return [result.summary_json_path, result.report_markdown_path]

    # Unknown stage — mark skipped with note
    stage.notes = f"Unknown stage id {sid!r} — skipped."
    return []
