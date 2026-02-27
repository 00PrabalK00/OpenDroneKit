"""Core pipeline modules for structural fault analysis."""

from .detection import (
    CrackDetectionResult,
    DefectHit,
    MetalDefectResult,
    SolarDefectResult,
    StructuralDefectResult,
    detect_cracks,
    detect_metal_defects,
    detect_solar_defects,
    detect_structural_defects,
    load_image,
)
from .coverage_validation import CoverageValidationConfig, CoverageValidator
from .multisensor import (
    CalibrationProfile,
    MultiSensorConfig,
    MultiSensorProcessor,
    build_capture_bundle_from_folders,
    create_calibration_profile,
    load_calibration_profile,
    load_capture_bundle,
    write_capture_bundle,
)
from .models import (
    ModelSpec,
    ensure_model_layout,
    get_model_spec,
    load_registry,
    migrate_legacy_models,
    model_status,
    resolve_model_path,
    sync_legacy_checkpoint_archive,
)
from .health_scoring import derive_asset_id, evaluate_asset_health, load_asset_health_history
from .pipeline import PipelineConfig, PipelineResult, StructuralFaultPipeline, run_pipeline
from .propagation import CrackPropagationForecaster, PropagationPhysicsConfig, run_fenicsx_phasefield
from .reconstruction import (
    CustomDroneReconstructor,
    ReconstructionResult,
    available_reconstruction_profiles,
)

__all__ = [
    "CrackDetectionResult",
    "DefectHit",
    "MetalDefectResult",
    "SolarDefectResult",
    "StructuralDefectResult",
    "CoverageValidationConfig",
    "CoverageValidator",
    "CalibrationProfile",
    "MultiSensorConfig",
    "MultiSensorProcessor",
    "build_capture_bundle_from_folders",
    "PipelineConfig",
    "PipelineResult",
    "StructuralFaultPipeline",
    "CrackPropagationForecaster",
    "PropagationPhysicsConfig",
    "CustomDroneReconstructor",
    "ReconstructionResult",
    "available_reconstruction_profiles",
    "detect_cracks",
    "detect_metal_defects",
    "detect_solar_defects",
    "detect_structural_defects",
    "load_image",
    "load_capture_bundle",
    "build_capture_bundle_from_folders",
    "write_capture_bundle",
    "load_calibration_profile",
    "create_calibration_profile",
    "ModelSpec",
    "ensure_model_layout",
    "load_registry",
    "get_model_spec",
    "resolve_model_path",
    "model_status",
    "migrate_legacy_models",
    "sync_legacy_checkpoint_archive",
    "derive_asset_id",
    "evaluate_asset_health",
    "load_asset_health_history",
    "run_pipeline",
    "run_fenicsx_phasefield",
]
