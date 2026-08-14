"""Workflow template manager — built-in and user-defined inspection workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class WorkflowTemplate:
    id: str
    name: str
    description: str
    asset_type: str
    required_inputs: list[str]
    optional_inputs: list[str]
    mission_modes: list[str]
    processing_stages: list[str]
    report_sections: list[str]
    default_parameters: dict[str, Any] = field(default_factory=dict)
    user_defined: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowReadiness:
    ready: bool
    missing_required: list[str]
    optional_improvements: list[str]
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Built-in templates ────────────────────────────────────────────────────────

_BUILTIN_TEMPLATES: list[WorkflowTemplate] = [
    WorkflowTemplate(
        id="general_inspection",
        name="General Inspection",
        description="Standard visual inspection workflow for generic structures. Captures overview and detail imagery.",
        asset_type="generic",
        required_inputs=["image_dataset"],
        optional_inputs=["calibration_profile", "capture_bundle"],
        mission_modes=["waypoint", "polygon_survey"],
        processing_stages=["dataset_validation", "defect_detection", "report_generation"],
        report_sections=["summary", "defect_table", "image_gallery"],
        default_parameters={"forecast_steps": 6, "propagation_mode": "geometric"},
    ),
    WorkflowTemplate(
        id="3d_mapping",
        name="3D Mapping",
        description="Full photogrammetric mapping workflow — generates point cloud, mesh and orthomosaic.",
        asset_type="generic",
        required_inputs=["image_dataset"],
        optional_inputs=["crack_mask_dir", "calibration_profile"],
        mission_modes=["polygon_survey", "facade_grid"],
        processing_stages=["dataset_validation", "reconstruction", "defect_projection", "report_generation"],
        report_sections=["summary", "3d_model", "defect_projection", "image_gallery"],
        default_parameters={
            "reconstruction_profile": "standard",
            "execution_mode": "local",
            "use_cache": True,
        },
    ),
    WorkflowTemplate(
        id="construction_progress",
        name="Construction Progress",
        description=(
            "Compare two aligned survey DSMs and deliver mapped surface-rise/fall "
            "regions, measured cut/fill volumes, and a client-readable change report."
        ),
        asset_type="construction_site",
        required_inputs=["earlier_dsm", "later_dsm"],
        optional_inputs=["site_semantic_layer", "approved_design_model"],
        mission_modes=["polygon_survey", "double_grid"],
        processing_stages=["survey_change"],
        report_sections=["summary", "surface_change", "change_map", "limitations"],
        default_parameters={
            "change_threshold_m": 0.05,
            "min_region_area_m2": 1.0,
        },
    ),
    WorkflowTemplate(
        id="facade_inspection",
        name="Facade Inspection",
        description="Systematic facade grid capture and crack / spalling detection.",
        asset_type="building",
        required_inputs=["image_dataset"],
        optional_inputs=["crack_mask_dir"],
        mission_modes=["facade_grid"],
        processing_stages=["dataset_validation", "crack_detection", "structural_detection", "crack_propagation", "report_generation"],
        report_sections=["summary", "crack_analysis", "structural_defects", "propagation_forecast", "image_gallery"],
        default_parameters={"forecast_steps": 6, "propagation_mode": "physics_informed"},
    ),
    WorkflowTemplate(
        id="roof_inspection",
        name="Roof Inspection",
        description="Nadir and oblique roof capture with moisture, crack and ponding detection.",
        asset_type="building",
        required_inputs=["image_dataset"],
        optional_inputs=["thermal_dataset"],
        mission_modes=["polygon_survey"],
        processing_stages=["dataset_validation", "defect_detection", "report_generation"],
        report_sections=["summary", "roof_condition", "defect_table", "image_gallery"],
        default_parameters={},
    ),
    WorkflowTemplate(
        id="bridge_inspection",
        name="Bridge Inspection",
        description="Multi-face bridge inspection with crack propagation analysis and 3D reconstruction.",
        asset_type="bridge",
        required_inputs=["image_dataset"],
        optional_inputs=["capture_bundle", "crack_mask_dir"],
        mission_modes=["facade_grid", "orbit", "waypoint"],
        processing_stages=["dataset_validation", "crack_detection", "structural_detection", "crack_propagation", "reconstruction", "report_generation"],
        report_sections=["summary", "structural_risk", "crack_analysis", "3d_model", "propagation_forecast", "recommendations"],
        default_parameters={
            "forecast_steps": 8,
            "propagation_mode": "physics_informed",
            "structure_type": "bridge",
            "material_type": "concrete",
        },
    ),
    WorkflowTemplate(
        id="solar_inspection",
        name="Solar Inspection",
        description="Solar panel thermal and visual inspection — hotspot detection and IV-curve estimation.",
        asset_type="solar",
        required_inputs=["image_dataset"],
        optional_inputs=["thermal_dataset"],
        mission_modes=["polygon_survey"],
        processing_stages=["dataset_validation", "solar_defect_detection", "report_generation"],
        report_sections=["summary", "hotspot_map", "defect_table", "performance_estimate"],
        default_parameters={"structure_type": "solar"},
    ),
    WorkflowTemplate(
        id="tower_inspection",
        name="Tower Inspection",
        description="Vertical orbit capture of telecom / transmission towers with corrosion and joint analysis.",
        asset_type="tower",
        required_inputs=["image_dataset"],
        optional_inputs=["crack_mask_dir"],
        mission_modes=["orbit", "waypoint"],
        processing_stages=["dataset_validation", "structural_detection", "defect_detection", "report_generation"],
        report_sections=["summary", "structural_risk", "defect_table", "image_gallery"],
        default_parameters={"structure_type": "tower", "material_type": "steel"},
    ),
    WorkflowTemplate(
        id="wind_turbine_inspection",
        name="Wind Turbine Inspection",
        description="Blade surface inspection with edge tracking, erosion and crack detection.",
        asset_type="wind_turbine",
        required_inputs=["image_dataset"],
        optional_inputs=["capture_bundle"],
        mission_modes=["facade_grid", "orbit"],
        processing_stages=["dataset_validation", "crack_detection", "structural_detection", "report_generation"],
        report_sections=["summary", "blade_condition", "defect_table", "image_gallery"],
        default_parameters={"structure_type": "generic"},
    ),
    WorkflowTemplate(
        id="stockpile_measurement",
        name="Stockpile Measurement",
        description=(
            "Reconstruct a stockpile survey, measure an operator-selected pile against "
            "an explicit DTM/base plane, and produce a mapped client deliverable."
        ),
        asset_type="stockpile",
        required_inputs=["image_dataset"],
        optional_inputs=["calibration_profile"],
        mission_modes=["polygon_survey"],
        processing_stages=["dataset_validation", "reconstruction", "volume_estimation", "report_generation"],
        report_sections=["summary", "volume_report", "dsm_view", "image_gallery"],
        default_parameters={
            "reconstruction_profile": "inspection_high_accuracy",
            "volume_polygon_xy": None,
        },
    ),
    WorkflowTemplate(
        id="linear_corridor",
        name="Linear Corridor Inspection",
        description="Pipeline, road or rail corridor inspection with automated path planning.",
        asset_type="pipeline",
        required_inputs=["image_dataset"],
        optional_inputs=["capture_bundle", "calibration_profile"],
        mission_modes=["linear_corridor"],
        processing_stages=["dataset_validation", "defect_detection", "report_generation"],
        report_sections=["summary", "corridor_map", "defect_table", "image_gallery"],
        default_parameters={"structure_type": "pipeline"},
    ),
    WorkflowTemplate(
        id="custom",
        name="Custom Workflow",
        description="Fully configurable workflow — choose your own stages and report sections.",
        asset_type="generic",
        required_inputs=["image_dataset"],
        optional_inputs=["capture_bundle", "calibration_profile", "crack_mask_dir", "thermal_dataset"],
        mission_modes=["waypoint", "polygon_survey", "facade_grid", "orbit", "linear_corridor"],
        processing_stages=["dataset_validation", "crack_detection", "structural_detection", "solar_defect_detection", "crack_propagation", "reconstruction", "report_generation"],
        report_sections=["summary", "crack_analysis", "structural_defects", "propagation_forecast", "3d_model", "image_gallery"],
        default_parameters={},
    ),
    WorkflowTemplate(
        id='stockpile_change',
        name='Selected Stockpile or Pit Change',
        description=(
            'Compare two aligned DSMs only inside an operator-selected stockpile or '
            'pit polygon, with mapped rise/fall and volume change.'
        ),
        asset_type='mining_roi',
        required_inputs=['earlier_dsm', 'later_dsm', 'roi_polygon'],
        optional_inputs=['site_semantic_layer'],
        mission_modes=['polygon_survey', 'double_grid'],
        processing_stages=['selected_roi_change'],
        report_sections=['summary', 'roi_selection', 'surface_change', 'limitations'],
        default_parameters={
            'roi_type': 'stockpile',
            'roi_name': '',
            'change_threshold_m': 0.05,
            'min_region_area_m2': 1.0,
        },
    ),
    WorkflowTemplate(
        id='semantic_mapping',
        name='Shared Semantic Mapping',
        description=(
            'Apply a schema-matched trained semantic model to a georeferenced '
            'orthomosaic and produce class, confidence and polygon GIS layers.'
        ),
        asset_type='survey_map',
        required_inputs=['orthomosaic', 'semantic_model', 'semantic_model_manifest'],
        optional_inputs=[],
        mission_modes=['polygon_survey', 'double_grid', 'linear_corridor'],
        processing_stages=['semantic_segmentation'],
        report_sections=['summary', 'semantic_map', 'class_areas', 'model_provenance'],
        default_parameters={
            'semantic_device': 'cuda',
            'semantic_allow_cpu': False,
            'semantic_max_cpu_pixels': 4_000_000,
        },
    ),
]

_BUILTIN_MAP: dict[str, WorkflowTemplate] = {t.id: t for t in _BUILTIN_TEMPLATES}

# ── Functions ─────────────────────────────────────────────────────────────────

def list_workflow_templates(workspace: Path | None = None) -> list[WorkflowTemplate]:
    """Return built-in + user-defined templates."""
    templates = list(_BUILTIN_TEMPLATES)
    if workspace is not None:
        user_dir = Path(workspace) / "workflow_templates"
        if user_dir.exists():
            for f in sorted(user_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    t = WorkflowTemplate(**{k: v for k, v in data.items() if k in WorkflowTemplate.__dataclass_fields__})
                    t.user_defined = True
                    templates.append(t)
                except Exception:
                    pass
    return templates


def get_workflow_template(template_id: str, workspace: Path | None = None) -> WorkflowTemplate:
    """Load one workflow by id; raise KeyError if not found."""
    if template_id in _BUILTIN_MAP:
        return _BUILTIN_MAP[template_id]
    if workspace is not None:
        user_dir = Path(workspace) / "workflow_templates"
        f = user_dir / f"{template_id}.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                t = WorkflowTemplate(**{k: v for k, v in data.items() if k in WorkflowTemplate.__dataclass_fields__})
                t.user_defined = True
                return t
            except Exception:
                pass
    raise KeyError(f"Workflow template not found: {template_id!r}")


def create_custom_workflow(template: WorkflowTemplate, workspace: Path) -> WorkflowTemplate:
    """Save a custom workflow template to workspace."""
    user_dir = Path(workspace) / "workflow_templates"
    user_dir.mkdir(parents=True, exist_ok=True)
    out = user_dir / f"{template.id}.json"
    out.write_text(json.dumps(template.to_dict(), indent=2), encoding="utf-8")
    template.user_defined = True
    return template


def validate_workflow_readiness(project_meta: dict, template_id: str) -> WorkflowReadiness:
    """Check whether the project has required inputs for the selected workflow."""
    try:
        template = get_workflow_template(template_id)
    except KeyError:
        return WorkflowReadiness(ready=False, missing_required=["workflow_template"], optional_improvements=[], next_action="Select a valid workflow template.")

    root = Path(str(project_meta.get("root_dir", "") or ""))
    missing: list[str] = []
    improvements: list[str] = []

    for req in template.required_inputs:
        if req == "image_dataset":
            datasets_dir = root / "datasets"
            if not datasets_dir.exists() or not any(datasets_dir.iterdir()):
                missing.append("image_dataset — import drone images in Data Library")
        elif req in {"earlier_dsm", "later_dsm"}:
            value = project_meta.get(req) or project_meta.get(f"{req}_path")
            if not value or not Path(str(value)).is_file():
                missing.append(f"{req} - select an existing georeferenced DSM")
        elif req == "thermal_dataset":
            improvements.append("thermal_dataset — import thermal imagery for better analysis")

    for req in template.required_inputs:
        if req in {'orthomosaic', 'semantic_model', 'semantic_model_manifest'}:
            value = project_meta.get(req) or project_meta.get(f'{req}_path')
            if not value or not Path(str(value)).is_file():
                missing.append(f'{req} - select an existing file')

    if 'roi_polygon' in template.required_inputs:
        polygon = project_meta.get('roi_polygon') or project_meta.get('roi_polygon_xy')
        if not polygon or len(polygon) < 3:
            missing.append('roi_polygon - draw a stockpile or pit boundary')

    for opt in template.optional_inputs:
        if opt == "calibration_profile":
            improvements.append("calibration_profile — improves measurement accuracy")
        elif opt == "site_semantic_layer":
            improvements.append(
                "site_semantic_layer - enables cause-specific labels instead of geometric change only"
            )
        elif opt == "approved_design_model":
            improvements.append(
                "approved_design_model - enables design progress percentage"
            )
        elif opt == "capture_bundle":
            improvements.append("capture_bundle — enables multi-sensor processing")

    if missing:
        next_action = f"Import {missing[0].split(' — ')[0].replace('_', ' ')} to proceed."
    elif improvements:
        next_action = "Optional improvements available — workflow is ready to run."
    else:
        next_action = "All inputs ready — run the pipeline."

    return WorkflowReadiness(
        ready=len(missing) == 0,
        missing_required=missing,
        optional_improvements=improvements,
        next_action=next_action,
    )
