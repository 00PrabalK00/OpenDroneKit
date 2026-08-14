"""Registry-visible road, power, rail and solar tests using real GIS files."""

from training.verification.test_india_assets import (
    test_power_and_rail_enforce_capture_geometry_and_map_real_vectors,
    test_road_condition_uses_explicit_centerline_for_metric_distance,
    test_solar_inventory_counts_modules_and_only_calls_layout_gaps_missing,
)


__all__ = [name for name in globals() if name.startswith("test_")]
