"""Whether the installed model is the file its recorded metrics belong to.

A registry entry outlives the file it points at. Retrain a model, copy one in by hand,
or restore half a backup, and the entry keeps its labels, its thresholds and its
published accuracy while the bytes underneath change. Nothing raises. Every report the
model feeds goes on quoting an IoU measured on a file that is no longer there.

The four outcomes are deliberately distinct, and the one that matters most is the third:
"nobody recorded a digest" must never be reported as "verified", because that would turn
an absence of checking into a claim of having checked.
"""

from __future__ import annotations

import json

import pytest

from core import models as models_module
from core.models import verify_all_models, verify_model_identity


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A registry and models tree of our own, so nothing here touches the real one."""
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(models_module, "models_root", lambda: root)
    monkeypatch.setattr(models_module, "registry_path", lambda: root / "model_registry.json")
    models_module._DIGEST_CACHE.clear()

    def write(models):
        (root / "model_registry.json").write_text(
            json.dumps({"version": 1, "models": models}), encoding="utf-8")

    def install(name, content=b"a real model graph"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        import hashlib
        return hashlib.sha256(content).hexdigest()

    return type("Fixture", (), {"root": root, "write": staticmethod(write),
                                "install": staticmethod(install)})


class TestVerification:
    def test_a_matching_file_verifies(self, registry):
        digest = registry.install("structural/detector.onnx")
        registry.write({"det": {"kind": "onnx_yolo", "path": "structural/detector.onnx",
                                "labels": ["crack"], "sha256": digest}})

        result = verify_model_identity("det")
        assert result["status"] == "verified"
        assert result["verified"] is True

    def test_a_replaced_file_is_a_mismatch_not_a_pass(self, registry):
        """The exact case: retrained weights dropped in under the old entry."""
        old = registry.install("structural/detector.onnx", b"the measured model")
        registry.write({"det": {"kind": "onnx_yolo", "path": "structural/detector.onnx",
                                "labels": ["crack"], "sha256": old}})
        models_module._DIGEST_CACHE.clear()
        registry.install("structural/detector.onnx", b"a different model entirely")

        result = verify_model_identity("det")
        assert result["status"] == "mismatch"
        assert result["verified"] is False
        assert "describes a different file" in result["detail"]

    def test_no_recorded_digest_is_unrecorded_never_verified(self, registry):
        registry.install("structural/detector.onnx")
        registry.write({"det": {"kind": "onnx_yolo", "path": "structural/detector.onnx",
                                "labels": ["crack"]}})

        result = verify_model_identity("det")
        assert result["status"] == "unrecorded"
        assert result["verified"] is False
        assert "cannot be shown to be" in result["detail"]

    def test_an_absent_file_is_missing_not_mismatched(self, registry):
        registry.write({"det": {"kind": "onnx_yolo", "path": "structural/absent.onnx",
                                "labels": ["crack"], "sha256": "a" * 64}})

        result = verify_model_identity("det")
        assert result["status"] == "missing"
        assert "nothing to run" in result["detail"]

    def test_an_unknown_key_is_refused(self, registry):
        registry.write({})
        assert verify_model_identity("never_registered")["status"] == "unknown_model_key"

    def test_the_recorded_digest_is_compared_case_insensitively(self, registry):
        digest = registry.install("structural/detector.onnx")
        registry.write({"det": {"kind": "onnx_yolo", "path": "structural/detector.onnx",
                                "labels": ["crack"], "sha256": digest.upper()}})

        assert verify_model_identity("det")["status"] == "verified"


class TestWholeRegistryReport:
    def test_bad_news_is_reported_first(self, registry):
        good = registry.install("a.onnx", b"good")
        stale = registry.install("b.onnx", b"stale")
        registry.write({
            "verified_one": {"kind": "onnx_yolo", "path": "a.onnx", "labels": [], "sha256": good},
            "unrecorded_one": {"kind": "onnx_yolo", "path": "b.onnx", "labels": []},
            "missing_one": {"kind": "onnx_yolo", "path": "gone.onnx", "labels": [],
                            "sha256": "b" * 64},
            "mismatched_one": {"kind": "onnx_yolo", "path": "b.onnx", "labels": [],
                               "sha256": "c" * 64},
        })

        report = verify_all_models()
        assert report["models"][0]["status"] == "mismatch"
        assert report["any_mismatch"] is True
        assert report["counts"] == {"mismatch": 1, "missing": 1, "unrecorded": 1, "verified": 1}

    def test_a_clean_registry_reports_no_mismatch(self, registry):
        digest = registry.install("a.onnx", b"good")
        registry.write({"one": {"kind": "onnx_yolo", "path": "a.onnx", "labels": [],
                                "sha256": digest}})

        assert verify_all_models()["any_mismatch"] is False


class TestEntriesWithNoWeights:
    """A registered target and a model that has gone are different facts."""

    def test_a_declared_empty_slot_is_awaiting_weights_not_missing(self, registry):
        registry.write({"planned": {
            "kind": "onnx_yolo", "path": "solar/not_yet.onnx", "labels": ["hotspot"],
            "status": "awaiting_weights",
            "status_note": "Training target; weights not obtained yet.",
        }})

        result = verify_model_identity("planned")
        assert result["status"] == "awaiting_weights"
        assert result["verified"] is False
        assert "not obtained yet" in result["detail"]

    def test_a_model_that_has_gone_is_still_reported_as_missing(self, registry):
        """An installed entry whose file vanished must not hide behind the softer word."""
        registry.write({"installed_once": {
            "kind": "onnx_yolo", "path": "solar/gone.onnx", "labels": [],
            "status": "installed", "sha256": "a" * 64,
        }})

        assert verify_model_identity("installed_once")["status"] == "missing"

    def test_awaiting_weights_keys_are_not_counted_as_available(self, registry):
        digest = registry.install("real.onnx", b"real")
        registry.write({
            "real_one": {"kind": "onnx_yolo", "path": "real.onnx", "labels": [],
                         "sha256": digest},
            "planned_one": {"kind": "onnx_yolo", "path": "planned.onnx", "labels": [],
                            "status": "awaiting_weights"},
        })

        report = verify_all_models()
        assert report["available"] == ["real_one"]
        assert report["awaiting_weights"] == ["planned_one"]


class TestTheRealRegistry:
    def test_both_shipping_models_verify_against_their_recorded_digests(self):
        """Guards the two models that actually answer in the field."""
        report = verify_all_models()
        by_key = {row["model_key"]: row for row in report["models"]}

        for key in ("crack_segmentation", "structural_multiclass_detector"):
            assert by_key[key]["status"] == "verified", by_key[key]["detail"]

    def test_no_installed_model_is_running_under_a_stale_digest(self):
        assert verify_all_models()["any_mismatch"] is False

    def test_no_entry_claims_to_be_installed_without_a_file(self):
        """The registry's whole purpose is to state what is real."""
        report = verify_all_models()
        assert report["counts"].get("missing", 0) == 0, (
            "an entry marked installed has no file behind it: "
            + "; ".join(row["model_key"] for row in report["models"]
                        if row["status"] == "missing")
        )

    def test_the_keys_the_pipeline_routes_to_are_accounted_for(self):
        """solar_defect_detector is a live routing target; it must not read as ready."""
        by_key = {row["model_key"]: row for row in verify_all_models()["models"]}
        assert by_key["solar_defect_detector"]["status"] == "awaiting_weights"
        assert "Kaggle" in by_key["solar_defect_detector"]["detail"]
