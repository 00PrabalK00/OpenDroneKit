"""Registry-visible India agriculture tests using real raster products."""

from training.verification.test_india_agriculture import (
    test_canopy_cover_uses_real_semantic_raster,
    test_index_cover_classes_feed_canopy_cover_and_need_a_scope,
    test_indices_use_calibrated_bands_and_mark_missing_band_unavailable,
    test_plant_count_from_canopy_refuses_merged_rows_and_counts_separate_plants,
    test_plant_count_counts_connected_instances_without_inventing_missing_or_health,
    test_stress_zones_require_and_preserve_crop_sensor_scope,
)


__all__ = [name for name in globals() if name.startswith("test_")]
