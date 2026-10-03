"""Release download URLs in the model registry, and the downloader's refusal of bad bytes.

v0.1.0-beta publishes the weights whose licence chain allows public redistribution as
GitHub Release assets. Each such registry row carries a ``download`` entry. These tests
pin three things:

* every download URL points at the v0.1.0-beta release, names the same file the row's
  ``path`` names, and sits on a row with a recorded sha256 the downloader can check;
* only models cleared in WEIGHTS_LICENSES.md carry a URL -- a blocked model gaining one
  would publish weights whose licence does not allow it;
* the first-run downloader (core.model_download, from the installer work) refuses bytes
  that do not hash to the registry's sha256, and installs nothing when it does.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "models" / "model_registry.json"
LICENSES = REPO / "WEIGHTS_LICENSES.md"
RELEASE_BASE = "https://github.com/00PrabalK00/OpenDroneKit/releases/download/v0.1.0-beta/"
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# The licence audit's outcome. Changing either set is a licensing decision, and must be
# made in WEIGHTS_LICENSES.md first.
CLEARED = {"crack_presence_classifier", "solar_thermal_anomaly_classifier"}
BLOCKED = {
    "structural_multiclass_detector",
    "crack_segmentation",
    "solar_cell_defect_detector",
    "rail_corridor_segmentation",
    "rail_obstacle_detector",
    "corrosion_severity_segmentation",
    "shared_semantic_landcover",
}


def _rows() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]


def _urls(spec: dict) -> list[tuple[str, str, str]]:
    """(url, file name, sha256 that file must hash to) for a download entry."""
    if spec.get("files"):
        return [(f["url"], f["name"], f["sha256"]) for f in spec["files"]]
    return [(spec["url"], "", "")]


def test_every_download_url_has_a_matching_sha256():
    rows = _rows()
    published = {key: row for key, row in rows.items() if row.get("download")}
    assert published, "no registry row carries a release download URL"
    for key, row in published.items():
        assert SHA256.match(str(row.get("sha256", ""))), f"{key}: download URL without a sha256"
        assert row.get("status") == "installed", f"{key}: URL on a row that has no weights"
        file_name = Path(row["path"]).name
        for url, name, digest in _urls(row["download"]):
            assert url.startswith(RELEASE_BASE), f"{key}: {url} is not a v0.1.0-beta asset"
            asset = url[len(RELEASE_BASE):]
            assert "/" not in asset, f"{key}: {url} is not a flat release asset"
            if name:
                assert asset == name and SHA256.match(digest), f"{key}: {name} lacks its digest"
            else:
                assert asset == file_name, f"{key}: URL names {asset}, registry path names {file_name}"


def test_only_cleared_models_are_published():
    rows = _rows()
    published = {key for key, row in rows.items() if row.get("download")}
    assert published == CLEARED
    assert not published & BLOCKED
    assert CLEARED | BLOCKED == {k for k, r in rows.items() if r.get("status") == "installed"}


def test_licence_document_names_every_published_asset():
    text = LICENSES.read_text(encoding="utf-8")
    for key, row in _rows().items():
        if row.get("download"):
            assert key in text and row["sha256"] in text, f"{key} missing from WEIGHTS_LICENSES.md"
            assert (REPO / "docs" / "model_cards" / f"{key}.md").is_file(), f"{key} has no model card"


@pytest.fixture()
def downloader(monkeypatch, tmp_path):
    md = pytest.importorskip(
        "core.model_download",
        reason="core.model_download arrives with the installer branch (roadmap/installer)",
    )
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    monkeypatch.setattr(md, "_registry", lambda: registry)
    monkeypatch.setattr(md, "_installed", lambda key: False)
    monkeypatch.setattr(md, "target_root", lambda: tmp_path / "models")
    monkeypatch.delenv(md.SOURCES_ENV, raising=False)
    return md, registry, tmp_path / "models"


@pytest.mark.parametrize("key", sorted(CLEARED))
def test_downloader_refuses_a_release_asset_that_does_not_match(downloader, monkeypatch, key):
    md, registry, root = downloader
    fetched: list[str] = []

    def tampered(url, dest, progress, timeout):
        fetched.append(url)
        Path(dest).write_bytes(b"not the published weights")

    monkeypatch.setattr(md, "_fetch", tampered)
    info = md.availability(key)
    assert info["downloadable"] is True
    assert info["urls"] == [registry["models"][key]["download"]["url"]]

    result = md.download_model(key, force=True)
    assert result["ok"] is False
    assert "refused" in result["error"]
    assert fetched == [registry["models"][key]["download"]["url"]]
    target = root / registry["models"][key]["path"]
    assert not target.exists(), "a mismatching download was installed"
    assert not any(p.is_file() for p in root.rglob("*")), "mismatching bytes were left behind"


def test_downloader_installs_bytes_that_match(downloader, monkeypatch):
    md, registry, root = downloader
    payload = b"stand-in weights for the matching case"
    key = sorted(CLEARED)[0]
    registry["models"][key]["sha256"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(md, "_fetch", lambda url, dest, p, t: Path(dest).write_bytes(payload))

    result = md.download_model(key, force=True)
    assert result["ok"] is True
    assert (root / registry["models"][key]["path"]).read_bytes() == payload
