"""Registry-visible construction tests using real semantic and design files."""

from training.verification.test_india_construction import (
    test_approved_design_progress_measures_observed_surface_not_contract_completion,
    test_construction_schema_covers_the_registry_contract,
)


__all__ = [name for name in globals() if name.startswith("test_")]
