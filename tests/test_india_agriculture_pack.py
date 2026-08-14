"""Registry-visible India agriculture tests using real raster products."""

from training.verification.test_india_agriculture import (
    test_canopy_cover_uses_real_semantic_raster,
    test_indices_use_calibrated_bands_and_mark_missing_band_unavailable,
    test_plant_count_counts_connected_instances_without_inventing_missing_or_health,
    test_stress_zones_require_and_preserve_crop_sensor_scope,
)


__all__ = [name for name in globals() if name.startswith("test_")]
