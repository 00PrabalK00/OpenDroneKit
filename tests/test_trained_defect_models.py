"""Registry rows that claim a trained detector must point at weights that exist.

A capability like "solar defect detection" is only true if a model is installed, its
weights are the ones that were measured, and the numbers behind it are written down
where someone deciding whether to trust it will read them.

These tests check exactly that, and nothing about quality: a detector can pass here and
still be too weak to ship. Quality lives in the description, in the operator's hands.
What is enforced here is that the claim cannot drift away from the artefact -- a row
saying "trained detector with published validation metrics" backed by a missing file, a
digest that no longer matches, or a description with no numbers in it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# Rows in docs/features/registry.py that assert a trained model exists, mapped to the
# registry keys that make them true.
SOLAR_MODELS = ("solar_cell_defect_detector", "solar_thermal_anomaly_classifier")
RAIL_MODELS = ("rail_obstacle_detector", "rail_corridor_segmentation")
CRACK_MODELS = ("crack_segmentation", "crack_presence_classifier")


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]


def installed(registry: dict, key: str) -> dict:
    entry = registry.get(key)
    assert entry is not None, f"{key} is not in the model registry"
    assert entry.get("status") == "installed", f"{key} is registered but not installed"
    return entry


def weights_present(registry: dict, key: str) -> bool:
    """Whether the weights are on this machine.

    They are gitignored on purpose -- ONNX files are large binaries and the registry
    carries their digest instead. So a fresh clone, and therefore CI, has the registry
    and not the weights, and a test that asserted the file exists would fail everywhere
    except the machine that trained it.

    Skipping is right here and lying would not be: the digest check below is the whole
    guarantee, and where the file exists it still runs.
    """
    entry = registry.get(key) or {}
    return bool(entry.get("path")) and (MODELS_DIR / entry["path"]).is_file()


def requires_weights(registry: dict, key: str) -> None:
    if not weights_present(registry, key):
        pytest.skip(f"{key} weights are not on this machine (gitignored; see models/README.md)")


class TestTheWeightsAreReallyThere:
    @pytest.mark.parametrize("key", SOLAR_MODELS + RAIL_MODELS + CRACK_MODELS)
    def test_the_registry_names_a_weights_path(self, registry, key) -> None:
        # Checkable everywhere, including a clone with no weights: the entry must at
        # least say where its file belongs.
        entry = installed(registry, key)
        assert entry.get("path"), f"{key} is installed but names no file"

    @pytest.mark.parametrize("key", SOLAR_MODELS + RAIL_MODELS + CRACK_MODELS)
    def test_the_weights_file_exists(self, registry, key) -> None:
        requires_weights(registry, key)
        weights = MODELS_DIR / registry[key]["path"]
        assert weights.is_file()
        assert weights.stat().st_size > 0

    @pytest.mark.parametrize("key", SOLAR_MODELS + RAIL_MODELS + CRACK_MODELS)
    def test_the_digest_matches_the_file_on_disk(self, registry, key) -> None:
        """Identity, not existence.

        A model swapped after registration keeps every reported number from the run that
        measured a different file. The digest is what makes the metrics belong to these
        weights rather than to their filename.
        """
        entry = installed(registry, key)
        recorded = entry.get("sha256")
        assert recorded, f"{key} has no sha256, so its metrics belong to no particular file"
        requires_weights(registry, key)
        actual = hashlib.sha256((MODELS_DIR / entry["path"]).read_bytes()).hexdigest()
        assert actual == recorded, (
            f"{key} on disk does not match its recorded digest; the published metrics "
            "describe a file that is no longer installed"
        )


class TestTheNumbersArePublished:
    @pytest.mark.parametrize("key", SOLAR_MODELS + RAIL_MODELS + CRACK_MODELS)
    def test_the_description_carries_a_validation_figure(self, registry, key) -> None:
        description = installed(registry, key).get("description", "")
        assert re.search(r"0\.\d{2,}", description), (
            f"{key} publishes no validation number, so 'with published validation "
            "metrics' is not true of it"
        )

    @pytest.mark.parametrize("key", SOLAR_MODELS + RAIL_MODELS + CRACK_MODELS)
    def test_the_description_states_a_limit_not_only_a_score(self, registry, key) -> None:
        """Every model here has a weakness, and the entry has to name it.

        A description that is only a score invites the reader to treat the model as
        finished. These entries carry the weak class, the corpus it was measured on, or
        what the model cannot do -- which is the part that changes how it gets used.
        """
        description = installed(registry, key).get("description", "").lower()
        signals = ("not", "cannot", "scope", "weak", "must", "unmeasured", "read ")
        assert any(word in description for word in signals), (
            f"{key} is described only by its score, with no stated limit"
        )


class TestLabelsMatchTheModel:
    @pytest.mark.parametrize("key", SOLAR_MODELS + RAIL_MODELS + CRACK_MODELS)
    def test_every_model_names_its_classes(self, registry, key) -> None:
        labels = installed(registry, key).get("labels")
        assert labels, f"{key} has no labels; its outputs are anonymous column indices"
        assert len(labels) == len(set(labels)), f"{key} has duplicate labels"

    def test_the_solar_models_cover_different_ground(self, registry) -> None:
        """The two solar models answer different questions and must not be conflated.

        The cell detector works on electroluminescence imagery of individual cells. The
        thermal classifier works on aerial infrared of whole modules. Only the second is
        capturable by a drone, and a caller reaching for "the solar model" needs the
        distinction to survive in the registry.
        """
        cell = installed(registry, "solar_cell_defect_detector")
        thermal = installed(registry, "solar_thermal_anomaly_classifier")
        assert set(cell["labels"]) != set(thermal["labels"])
        assert cell["kind"] != thermal["kind"]
