"""Label, build a corpus, train, deploy and run a custom detector -- all through the Api.

The audit found the pieces of custom training each worked alone and none was reachable
from the application: the labelling canvas was loaded by no page, the corpus builders
had no button, and training and registration were command-line tools. A user who drew
boxes in the app could not turn them into a model without leaving it.

These tests drive the same Api calls the AI Inspection toolbar makes (Label -> Build
Corpus -> Train Model -> Deploy), on real photographs carrying real boxes: the CODEBRIM
test and validation splits, with each label file converted to exactly the payload the
canvas posts (normalised top-left x, y, width, height). Nothing is mocked.

The CODEBRIM images are gitignored. They are read from training/data, or from
ODK_TRAINING_DATA_DIR where another checkout holds them; the pretrained checkpoint comes
from models/pretrained or ODK_MODEL_WEIGHTS_DIR. Tests needing either skip without it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "concrete_cracked_public_domain.jpg"
DATA = Path(os.environ.get("ODK_TRAINING_DATA_DIR", "") or ROOT / "training" / "data")
CODEBRIM = DATA / "codebrim_structural"
CLASSES = ["Efflorescence", "CorrosionStain", "Crack", "Spallation", "ExposedBars"]

needs_codebrim = pytest.mark.skipif(
    not (CODEBRIM / "test" / "labels").is_dir(),
    reason=f"CODEBRIM corpus not on this machine ({CODEBRIM})")


def _pretrained_available() -> bool:
    from core.models import extra_weights_root, models_root

    for root in (models_root(), extra_weights_root()):
        if root is not None and any((root / "pretrained").glob("yolo11*.pt")):
            return True
    return False


def _gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 -- no torch means no training here at all
        return False


def canvas_payloads() -> list[dict]:
    """Each CODEBRIM label file as the canvas would post it: normalised top-left boxes."""
    payloads = []
    for split in ("test", "valid"):
        for label_file in sorted((CODEBRIM / split / "labels").glob("*.txt")):
            image = next((CODEBRIM / split / "images").glob(label_file.stem + ".*"), None)
            if image is None:
                continue
            boxes = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) >= len(CLASSES):
                    continue  # "Background" boxes are not a defect class
                cx, cy, w, h = (float(v) for v in parts[1:])
                x, y = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                boxes.append({"label": CLASSES[int(parts[0])], "x": x, "y": y,
                              "width": min(w, 1.0 - x), "height": min(h, 1.0 - y)})
            if boxes:
                payloads.append({"path": str(image), "boxes": boxes, "confirmed_empty": False})
    return payloads


def _wait(api, job_id: str, timeout: float = 3600) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        job = api.job_status(job_id)["job"]
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(2)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def api(tmp_path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    instance = Api(AppSession(ProjectStore(tmp_path / "odk.db")))
    created = instance.create_project("custom model", str(tmp_path / "project"))
    assert created["ok"], created
    return instance


class TestLabelsDrawnInTheAppAreKept:
    """Save Labels in the canvas dialog calls save_label_boxes; Label reopens with them."""

    def test_boxes_are_stored_in_the_project(self, api, tmp_path):
        box = {"label": "crack", "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1}
        saved = api.save_label_boxes({"path": str(FIXTURE), "boxes": [box]})
        assert saved["ok"], saved
        listed = api.list_label_boxes()
        assert listed["images"] == 1 and listed["boxes"] == 1
        assert listed["classes"] == ["crack"]
        assert listed["entries"][0]["boxes"][0] == box
        store = tmp_path / "project" / "training" / "labels" / "boxes.json"
        assert json.loads(store.read_text(encoding="utf-8"))[0]["path"] == str(FIXTURE)

    def test_relabelling_an_image_replaces_its_boxes(self, api):
        first = {"label": "crack", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
        second = {"label": "spalling", "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2}
        api.save_label_boxes({"path": str(FIXTURE), "boxes": [first]})
        api.save_label_boxes({"path": str(FIXTURE), "boxes": [second]})
        listed = api.list_label_boxes()
        assert listed["images"] == 1
        assert listed["entries"][0]["boxes"] == [second]

    def test_malformed_geometry_is_refused_and_not_stored(self, api):
        bad = {"label": "crack", "x": 0.9, "y": 0.1, "width": 0.5, "height": 0.1}
        result = api.save_label_boxes({"path": str(FIXTURE), "boxes": [bad]})
        assert not result["ok"]
        assert "outside the image" in result["error"]
        assert api.list_label_boxes()["images"] == 0

    def test_a_missing_image_is_refused(self, api, tmp_path):
        result = api.save_label_boxes({"path": str(tmp_path / "nope.jpg"), "boxes": []})
        assert not result["ok"]

    def test_too_few_labels_are_refused_by_build_corpus(self, api):
        api.save_label_boxes({"path": str(FIXTURE), "boxes": [
            {"label": "crack", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}]})
        result = api.build_detection_corpus(None, "")
        assert not result["ok"]
        assert "Too few boxes" in result["error"] or "class" in result["error"]


@needs_codebrim
class TestBuildCorpusFromWhatWasDrawn:
    def test_the_stored_labels_become_a_yolo_corpus(self, api, tmp_path):
        for payload in canvas_payloads():
            assert api.save_label_boxes(payload)["ok"]
        built = api.build_detection_corpus(None, "")
        assert built["ok"], built.get("error")
        corpus = Path(built["output_dir"])
        assert corpus.parent == tmp_path / "project" / "training"
        assert (corpus / "data.yaml").is_file()
        assert built["counts"]["val"]["images"] > 0
        assert set(built["classes"]) <= set(CLASSES)
        listed = api.list_training_artifacts()
        assert [c["path"] for c in listed["corpora"]] == [str(corpus)]


@needs_codebrim
class TestTrainingRefusesWhatItCannotRun:
    def test_no_corpus_is_refused(self, api):
        result = api.train_custom_model("mine", "")
        assert not result["ok"] and "Build Corpus" in result["error"]

    def test_an_unnamed_model_is_refused(self, api, tmp_path):
        (tmp_path / "c").mkdir()
        (tmp_path / "c" / "data.yaml").write_text("path: .\n", encoding="utf-8")
        result = api.train_custom_model("  ", str(tmp_path / "c"))
        assert not result["ok"] and "name" in result["error"]

    def test_deploying_a_run_with_no_weights_is_refused(self, api, tmp_path):
        result = api.deploy_custom_model(str(tmp_path))
        assert not result["ok"] and "No trained weights" in result["error"]

    def test_a_corpus_with_an_empty_validation_split_is_refused(self, tmp_path):
        """counts['val'] is a dict, so a truthiness check passed a split of no images."""
        from core.custom_model_training import TrainingRefused, train_detector

        (tmp_path / "data.yaml").write_text("path: .\n", encoding="utf-8")
        (tmp_path / "corpus_report.json").write_text(json.dumps(
            {"counts": {"train": {"images": 12, "crack": 30},
                        "val": {"images": 0, "crack": 0}}}), encoding="utf-8")
        with pytest.raises(TrainingRefused, match="no validation images"):
            train_detector(tmp_path, tmp_path / "runs", name="x", epochs=1)

    def test_a_shipped_key_cannot_be_overwritten(self):
        from core.custom_model_training import TrainingRefused, deploy_detector

        with pytest.raises(TrainingRefused, match="custom_"):
            deploy_detector(Path("."), "structural_multiclass_detector")


@pytest.fixture
def registry_restored():
    """Deploying installs into models/; put the checkout back exactly as it was."""
    from core.models import models_root

    root = models_root()
    kept = {p: p.read_bytes() for p in (
        root / "model_registry.json", root / "manifests" / "model_provenance.json")
        if p.is_file()}
    before = {p for p in root.rglob("custom_*")}
    yield
    for path, content in kept.items():
        path.write_bytes(content)
    for path in set(root.rglob("custom_*")) - before:
        path.unlink(missing_ok=True)


@needs_codebrim
@pytest.mark.skipif(not _pretrained_available(), reason="no pretrained YOLO checkpoint here")
@pytest.mark.skipif(not _gpu(), reason="training a YOLO11x epoch needs a CUDA GPU")
class TestLabelToRunningModel:
    """The whole path, once: labels -> corpus -> train -> metrics -> deploy -> inference."""

    def test_a_user_labelled_detector_trains_deploys_and_runs(self, api, registry_restored):
        for payload in canvas_payloads():
            assert api.save_label_boxes(payload)["ok"]
        corpus = api.build_detection_corpus(None, "")
        assert corpus["ok"], corpus.get("error")

        started = api.train_custom_model("Wall defects", corpus["output_dir"],
                                         epochs=1, image_size=320, batch_size=4)
        assert started["ok"], started
        assert started["key"] == "custom_wall_defects"
        job = _wait(api, started["job_id"])
        assert job["status"] == "done", job.get("error")
        metrics = job["result"]["metrics"]
        # A real validation pass happened: the card holds a number the trainer measured,
        # whatever it is after one epoch -- not a placeholder.
        assert isinstance(metrics.get("map50"), float)

        runs = api.list_training_artifacts()["runs"]
        run = next(r for r in runs if r["key"] == "custom_wall_defects")
        assert not run["deployed"]

        deployed = api.deploy_custom_model(run["run_dir"], run["key"])
        assert deployed["ok"], deployed
        job = _wait(api, deployed["job_id"])
        assert job["status"] == "done", job.get("error")
        assert job["result"]["parity_ok"] is True
        assert set(job["result"]["labels"]) == set(CLASSES)
        assert len(job["result"]["sha256"]) == 64

        status = api.model_status()
        entry = next(m for m in status["models"] if m["key"] == "custom_wall_defects")
        assert entry["exists"]

        ran = api.run_model("custom_wall_defects", str(FIXTURE))
        assert ran["ok"], ran.get("error")
        assert ran["model_key"] == "custom_wall_defects"
        assert ran["model_sha256"] == job["result"]["sha256"]
        assert "detection_count" in ran["output"]

        # Exporting imported ultralytics in-process, which on Windows swaps cv2.imread for
        # a wrapper that reads grayscale as HxWx1; every later mask read in the app then
        # failed. After a deploy, a grayscale read is still two-dimensional.
        import cv2
        import numpy as np

        probe = str(api._session.project_root() / "gray_probe.png")
        assert cv2.imwrite(probe, np.zeros((4, 4), np.uint8))
        assert cv2.imread(probe, cv2.IMREAD_GRAYSCALE).ndim == 2


def test_deploy_restores_what_the_exporter_patched(tmp_path, monkeypatch):
    """The exporter's import of ultralytics rebinds cv2.imread/imwrite/imshow; deploy
    puts them back whatever the export's outcome. The fake stands in for ultralytics
    (the external library) and does exactly what its import does on Windows."""
    import cv2

    import training.export_onnx as export_onnx
    from core.custom_model_training import TrainingRefused, deploy_detector

    originals = (cv2.imread, cv2.imwrite, cv2.imshow)

    def patched(*_a, **_k):
        raise AssertionError("the exporter's cv2 wrapper leaked out of deploy")

    def fake_export(run):
        cv2.imread = cv2.imwrite = cv2.imshow = patched
        return {"parity_ok": False}

    monkeypatch.setattr(export_onnx, "export_yolo", fake_export)
    with pytest.raises(TrainingRefused):
        deploy_detector(tmp_path, "custom_probe")
    assert (cv2.imread, cv2.imwrite, cv2.imshow) == originals
