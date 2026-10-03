"""Release download URLs in the model registry, driven by each model's own licence block.

Models ship under their own licence terms; MIT covers the code only. A model is held
back only when its licence forbids redistribution outright, or when no single set of
terms can satisfy its inputs (recorded as ``redistribution: "held"``). These tests pin:

* every row with a ``download`` carries a ``licence`` block, a licence page at
  ``docs/model_licences/<key>.md`` and a model card at ``docs/model_cards/<key>.md``;
* every download URL is a flat v0.1.0-beta asset with a sha256 the downloader can check;
* a row marked ``redistribution: "forbidden"`` or ``"held"``, or listed in the held-back
  table of WEIGHTS_LICENSES.md, has no ``download``;
* WEIGHTS_LICENSES.md is the document tools/build_weights_licences.py assembles from the
  registry and the licence pages, so it cannot drift from them;
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
USES = {"commercial", "copyleft", "non-commercial"}
NOT_REDISTRIBUTABLE = {"forbidden", "held"}


def _rows() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]


def _published() -> dict:
    return {key: row for key, row in _rows().items() if row.get("download")}


def _urls(spec: dict) -> list[tuple[str, str, str]]:
    """(url, file name, sha256 that file must hash to) for a download entry."""
    if spec.get("files"):
        return [(f.get("url", ""), f.get("name", ""), f.get("sha256", "")) for f in spec["files"]]
    return [(spec["url"], "", "")]


def _held_in_document() -> set[str]:
    """Keys in the held-back table of WEIGHTS_LICENSES.md."""
    text = LICENSES.read_text(encoding="utf-8")
    match = re.search(r"^## Held back.*?(?=^## |\Z)", text, re.M | re.S)
    assert match, "WEIGHTS_LICENSES.md has no 'Held back' section"
    return set(re.findall(r"^\| `([a-z0-9_]+)` \|", match.group(0), re.M))


def test_every_download_url_has_a_matching_sha256():
    published = _published()
    assert published, "no registry row carries a release download URL"
    for key, row in published.items():
        assert SHA256.match(str(row.get("sha256", ""))), f"{key}: download URL without a sha256"
        assert row.get("status") == "installed", f"{key}: URL on a row that has no weights"
        file_name = Path(row["path"]).name
        names = []
        for url, name, digest in _urls(row["download"]):
            assert url.startswith(RELEASE_BASE), f"{key}: {url} is not a v0.1.0-beta asset"
            asset = url[len(RELEASE_BASE):]
            assert asset and "/" not in asset, f"{key}: {url} is not a flat release asset"
            if name:
                assert asset == name, f"{key}: URL names {asset}, entry names {name}"
                assert SHA256.match(str(digest)), f"{key}: {name} lacks its sha256"
                names.append(name)
            else:
                assert asset == file_name, f"{key}: URL names {asset}, registry path names {file_name}"
        if names:
            assert file_name in names, f"{key}: the release files do not include {file_name}"
            if len(names) == 1 and not row.get("companions"):
                # One file, no companions: the per-file digest IS the model digest.
                only = row["download"]["files"][0]["sha256"]
                assert only == row["sha256"], f"{key}: file digest differs from the row's sha256"


def test_asset_names_are_unique_across_the_release():
    seen: dict[str, str] = {}
    for key, row in _published().items():
        for url, _, _ in _urls(row["download"]):
            asset = url[len(RELEASE_BASE):]
            assert asset not in seen, f"{asset} is published by both {seen.get(asset)} and {key}"
            seen[asset] = key


def test_every_published_row_carries_its_licence_block_and_documents():
    for key, row in _published().items():
        licence = row.get("licence")
        assert isinstance(licence, dict) and licence, f"{key}: published without a licence block"
        for field in ("weights", "weights_url", "data", "use", "attribution"):
            assert str(licence.get(field, "")).strip(), f"{key}: licence.{field} is empty"
        assert licence["use"] in USES, f"{key}: licence.use {licence['use']!r} is not one of {USES}"
        assert licence.get("redistribution", "allowed") == "allowed", (
            f"{key}: published although licence.redistribution is {licence.get('redistribution')!r}")
        assert (REPO / "docs" / "model_licences" / f"{key}.md").is_file(), f"{key}: no licence page"
        assert (REPO / "docs" / "model_cards" / f"{key}.md").is_file(), f"{key}: no model card"


def test_rows_that_may_not_be_redistributed_have_no_download():
    rows = _rows()
    held_rows = {key for key, row in rows.items()
                 if (row.get("licence") or {}).get("redistribution") in NOT_REDISTRIBUTABLE}
    held_doc = _held_in_document()
    assert held_rows <= held_doc, f"held in the registry but not in WEIGHTS_LICENSES.md: {held_rows - held_doc}"
    for key in held_rows | held_doc:
        assert key in rows, f"WEIGHTS_LICENSES.md holds back {key}, which is not a registry row"
        assert not rows[key].get("download"), f"{key} is held back but carries a download URL"
        assert str((rows[key].get("licence") or {}).get("held_reason", "")).strip(), (
            f"{key}: held without a held_reason quoting the clause")


def test_licence_document_names_every_published_asset_and_digest():
    text = LICENSES.read_text(encoding="utf-8")
    for key, row in _published().items():
        assert f"`{key}`" in text, f"{key} missing from WEIGHTS_LICENSES.md"
        for _, name, digest in _urls(row["download"]):
            assert (digest or row["sha256"]) in text, f"{key}: sha256 missing from WEIGHTS_LICENSES.md"


def test_licence_document_is_assembled_from_the_registry_and_licence_pages():
    tool = pytest.importorskip("tools.build_weights_licences")
    assert tool.build() == LICENSES.read_text(encoding="utf-8"), (
        "WEIGHTS_LICENSES.md is stale: run python -m tools.build_weights_licences")


def test_the_owner_gate_is_open_and_explicit():
    text = LICENSES.read_text(encoding="utf-8")
    assert "OWNER GATE" in text and "InsPLAD" in text and "AGPL" in text


# -- downloader (core.model_download lives on roadmap/installer) --------------------------


def _download_urls(row: dict) -> list[str]:
    return [url for url, _, _ in _urls(row["download"])]


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


@pytest.mark.parametrize("key", sorted(_published()))
def test_downloader_refuses_a_release_asset_that_does_not_match(downloader, monkeypatch, key):
    md, registry, root = downloader
    fetched: list[str] = []

    def tampered(url, dest, progress, timeout):
        fetched.append(url)
        Path(dest).write_bytes(b"not the published weights")

    monkeypatch.setattr(md, "_fetch", tampered)
    info = md.availability(key)
    assert info["downloadable"] is True
    assert info["urls"] == _download_urls(registry["models"][key])

    result = md.download_model(key, force=True)
    assert result["ok"] is False
    assert "refused" in result["error"]
    assert fetched and fetched[0] == _download_urls(registry["models"][key])[0]
    target = root / registry["models"][key]["path"]
    assert not target.exists(), "a mismatching download was installed"
    assert not any(p.is_file() for p in root.rglob("*")), "mismatching bytes were left behind"


def test_downloader_installs_bytes_that_match(downloader, monkeypatch):
    md, registry, root = downloader
    payload = b"stand-in weights for the matching case"
    key = sorted(k for k, r in _published().items() if r["download"].get("url"))[0]
    registry["models"][key]["sha256"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(md, "_fetch", lambda url, dest, p, t: Path(dest).write_bytes(payload))

    result = md.download_model(key, force=True)
    assert result["ok"] is True
    assert (root / registry["models"][key]["path"]).read_bytes() == payload
