"""Registry-visible India land tests; implementations use real GeoTIFF/GeoJSON files."""

from training.verification.test_india_land import (
    test_encroachment_uses_imported_boundary_and_aligned_previous_survey,
    test_land_gis_extracts_real_georeferenced_semantic_classes,
    test_metric_land_analysis_refuses_unreferenced_raster,
)


__all__ = [name for name in globals() if name.startswith("test_")]
