"""Every model finding carries the identity of the model that made it.

The moment this record is needed is the moment a model turns out to be wrong. Someone
has to find every finding that model produced and re-examine it, and that is only
possible if each one names the model and the build. A retrained model under the same key
is a different model, so the key alone is not enough -- the digest is what distinguishes
them.

The identity is computed from the installed file rather than read from the registry. A
recorded hash describes what was installed at some point; hashing the file describes
what is about to run. Those differ exactly when it matters: after a model has been
replaced, retrained, or copied in by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.detection import detect_structural_defects
from core.models import model_identity, model_status


class TestModelIdentity:
    def test_an_installed_model_reports_the_digest_of_its_file(self):
        if not model_status("structural_multiclass_detector").get("exists"):
            pytest.skip("No structural model installed in this checkout.")

        identity = model_identity("structural_multiclass_detector")
        assert identity["model_key"] == "structural_multiclass_detector"
        assert len(identity["model_sha256"]) == 64

    def test_the_digest_matches_the_file_on_disk(self):
        """Computed, not recalled: this is what distinguishes it from a registry entry."""
        import hashlib
        from pathlib import Path

        info = model_status("structural_multiclass_detector")
        if not info.get("exists"):
            pytest.skip("No structural model installed in this checkout.")

        digest = hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()
        assert model_identity("structural_multiclass_detector")["model_sha256"] == digest

    def test_a_model_that_is_not_installed_has_no_identity(self):
        identity = model_identity("no_such_model_key")
        assert identity["model_sha256"] == ""

    def test_the_digest_is_cached_but_keyed_on_the_file(self, tmp_path, monkeypatch):
        """A replaced model must produce a new identity, not the cached old one."""
        from core import models as models_module

        target = tmp_path / "fake.onnx"
        target.write_bytes(b"first version")

        monkeypatch.setattr(models_module, "model_status",
                            lambda key: {"exists": True, "path": str(target)})

        first = models_module.model_identity("fake")["model_sha256"]

        # Rewrite with different content and a different size, which is what a
        # retrained model looks like on disk.
        target.write_bytes(b"a second, longer version of the model")
        second = models_module.model_identity("fake")["model_sha256"]

        assert first != second, "replacing the file must change its identity"


class TestDetectionCarriesIdentity:
    def test_a_model_detection_names_the_model_and_its_digest(self):
        import glob

        if not model_status("structural_multiclass_detector").get("exists"):
            pytest.skip("No structural model installed in this checkout.")
        paths = sorted(glob.glob("training/data/prepared/structural_det/test/images/*.jpg"))
        if not paths:
            pytest.skip("Prepared test imagery is not present in this checkout.")

        result = detect_structural_defects(cv2.imread(paths[0]))

        assert result.model_used.startswith("onnx:")
        assert result.model_key == "structural_multiclass_detector"
        assert len(result.model_sha256) == 64

    def test_the_heuristic_path_claims_no_model_identity(self):
        """An identity here would attribute a classical result to a model."""
        image = np.full((320, 320, 3), 180, dtype=np.uint8)
        result = detect_structural_defects(image, use_model=False)

        assert result.model_used == "heuristic"
        assert result.model_key == ""
        assert result.model_sha256 == ""

    def test_identity_and_model_used_cannot_disagree(self):
        """They are derived from one another rather than assigned separately."""
        import inspect

        from core import detection

        source = inspect.getsource(detection._identity_for)
        assert 'startswith("onnx:")' in source

    def test_the_summary_carries_the_identity_through(self):
        image = np.full((320, 320, 3), 180, dtype=np.uint8)
        summary = detect_structural_defects(image, use_model=False).to_summary()

        assert "model_key" in summary
        assert "model_sha256" in summary


class TestApiRefusesUnattributableFindings:
    """The database is where a finding outlives the run that produced it."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("email_validator")
        from fastapi.testclient import TestClient

        monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'prov.db'}")
        monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")

        import services.api.db as db_module

        db_module._engine = None
        db_module._SessionLocal = None
        from services.api.main import app

        with TestClient(app) as test_client:
            yield test_client

        db_module._engine = None
        db_module._SessionLocal = None

    @pytest.fixture
    def project(self, client):
        response = client.post("/auth/register", json={
            "email": "ops@example.com", "password": "longenough1",
            "organization_name": "Acme"})
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        project_id = client.post(f"/organizations/{organization_id}/projects",
                                 headers=headers, json={"name": "Bridge 7"}).json()["id"]
        return headers, project_id

    def _post(self, client, headers, project_id, **overrides):
        payload = {
            "category": "crack", "severity": "high", "source": "model",
            "model_key": "crack_segmentation",
            "model_sha256": "c" * 64, "confidence": 0.87,
        }
        payload.update(overrides)
        return client.post(f"/projects/{project_id}/defects", headers=headers, json=payload)

    def test_a_fully_attributed_finding_is_accepted(self, client, project):
        headers, project_id = project
        assert self._post(client, headers, project_id).status_code in (200, 201)

    def test_a_finding_without_a_digest_is_refused(self, client, project):
        """A retrained model under the same key is a different model."""
        headers, project_id = project
        response = self._post(client, headers, project_id, model_sha256="")

        assert response.status_code == 422
        assert "model_sha256" in response.json()["detail"]

    def test_a_finding_without_confidence_is_refused(self, client, project):
        headers, project_id = project
        response = self._post(client, headers, project_id, confidence=None)

        assert response.status_code == 422
        assert "confidence" in response.json()["detail"]

    def test_the_refusal_names_everything_that_is_missing_at_once(self, client, project):
        headers, project_id = project
        response = self._post(client, headers, project_id, model_key="",
                              model_sha256="", confidence=None)
        detail = response.json()["detail"]

        assert "model_key" in detail and "model_sha256" in detail and "confidence" in detail

    def test_an_impossible_confidence_is_refused(self, client, project):
        headers, project_id = project
        assert self._post(client, headers, project_id, confidence=1.4).status_code == 422

    def test_a_human_finding_needs_no_model_identity(self, client, project):
        """A person is not a model, and demanding a digest from one would be absurd."""
        headers, project_id = project
        response = client.post(f"/projects/{project_id}/defects", headers=headers, json={
            "category": "crack", "severity": "high", "source": "human"})

        assert response.status_code in (200, 201)
