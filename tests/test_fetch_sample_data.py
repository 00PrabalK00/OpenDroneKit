"""tools/fetch_sample_data.py: pinned hashes, upstream URLs and the licence rules.

Downloads are exercised offline against ``file://`` URLs (urllib serves them exactly like
HTTP). With ODK_NETWORK_TESTS=1 one file of every per-file dataset, and one member of
every remote archive the catalog range-reads, is fetched from its
real upstream and checked against the pinned hash.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from tools import fetch_sample_data as fsd

TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "E:/Shit was cluittering/Projects/odk-testdata"))


def test_every_fetchable_dataset_has_a_pinned_manifest_and_a_licence():
    catalog = fsd.load_catalog()
    for name, row in catalog.items():
        assert row["licence"] and row["source"].startswith("https://"), name
        if row["fetch"] == "restricted":
            continue
        entries = fsd.read_manifest(name)
        assert entries, name
        assert all(len(e.sha256) == 64 for e in entries)
        if row.get("base_url"):
            assert "/" + row["commit"] + "/" in row["base_url"], f"{name} is not pinned to a commit"


def test_cc_by_datasets_carry_their_attribution():
    for name, row in fsd.load_catalog().items():
        if row["licence"].startswith("CC-BY"):
            assert row.get("attribution"), f"{name} is CC BY and names no attribution"


def test_no_dataset_is_mirrored_by_opendronekit():
    hosts = {"raw.githubusercontent.com", "media.githubusercontent.com", "zenodo.org",
             "ndownloader.figshare.com"}
    for name, row in fsd.load_catalog().items():
        urls = [row.get("base_url", "")] + [a["url"] for a in row.get("archives", [])]
        urls += list((row.get("rename") or {}).values())
        urls += [rule["url"] for rule in row.get("members") or []]
        for url in filter(None, urls):
            assert url.split("/")[2] in hosts, f"{name}: {url} is not the publisher"


def test_upstream_paths_follow_the_catalog_mapping():
    catalog = fsd.load_catalog()
    sheffield = catalog["odm_sheffield_cross"]
    assert fsd.upstream_url(sheffield, "images/DJI_0054.JPG").endswith(
        "/143f556ea3128835ab46cd694714f8f78319699b/DJI_0054.JPG")
    parser = catalog["thermal_rjpeg_samples_thermal_parser"]
    assert fsd.upstream_url(parser, "H20T/DJI_0001_R.JPG").endswith("/images/H20T/DJI_0001_R.JPG")
    assert fsd.upstream_url(parser, "LICENSE.thermal_parser.txt").endswith(
        "/b513647ef2318ba99dd2e2a543fa0fb071fd9579/LICENSE")


@pytest.mark.parametrize("name", ["dji_thermal_sdk", "university_of_guam_thermal"])
def test_restricted_datasets_are_never_fetched(name, tmp_path):
    with pytest.raises(fsd.FetchRefused, match="not fetched by OpenDroneKit"):
        fsd.fetch(name, tmp_path)
    assert not any(tmp_path.iterdir())


def test_every_pinned_file_has_an_automatic_route():
    """No fetchable dataset leaves files behind: each path is a direct URL, a member of a
    remote archive, or unpacked from a downloaded archive."""
    for name, row in fsd.load_catalog().items():
        if row["fetch"] == "restricted":
            continue
        extracted = [a.get("extract_to", "") for a in row.get("archives") or [] if a.get("extract")]
        saved = {a["save_as"] for a in row.get("archives") or []}
        for entry in fsd.read_manifest(name):
            routed = (entry.path in saved or fsd._rule_for(row, entry.path)
                      or fsd._direct_url(row, entry.path)
                      or any(entry.path.startswith(sub + "/") if sub else True for sub in extracted))
            assert routed, f"{name}: {entry.path} has no automatic route"


def test_a_manual_dataset_is_refused_with_instructions(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"datasets": [{
        "name": "handmade", "title": "t", "licence": "CC0-1.0", "source": "https://example.org",
        "fetch": "manual", "manual": "Assemble it from the archive, then run verify."}]}))
    monkeypatch.setattr(fsd, "CATALOG", catalog)
    with pytest.raises(fsd.FetchRefused, match="run verify"):
        fsd.fetch("handmade", tmp_path / "dest")


def _zip_bytes(members: dict[str, bytes], method=zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=method) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _pin(manifests: Path, name: str, rows: dict[str, str]) -> None:
    manifests.mkdir(exist_ok=True)
    (manifests / f"{name}.sha256").write_text(
        "".join(f"{digest} *./{path}\n" for path, digest in rows.items()), encoding="utf-8")


def test_members_are_read_out_of_a_zip_stored_inside_a_zip(tmp_path, monkeypatch):
    """The RDD2022 layout: a 13 GB archive storing one zip per country. Only the pinned
    members are read, through range reads of the outer file."""
    drone_a = b"\xff\xd8drone-a"
    inner = _zip_bytes({"China_Drone/train/images/a.jpg": drone_a,
                        "China_Drone/train/images/b.jpg": b"\xff\xd8drone-b"})
    outer = tmp_path / "release.zip"
    outer.write_bytes(_zip_bytes({"RDD2022/China_Drone.zip": inner,
                                  "RDD2022/Other.zip": b"x" * 500_000}, zipfile.ZIP_STORED))
    label = b"item { id: 1 name: 'D00' }\n"
    (tmp_path / "label_map.pbtxt").write_bytes(label)
    manifests = tmp_path / "manifests"
    _pin(manifests, "nested", {"China_Drone/train/images/a.jpg": hashlib.sha256(drone_a).hexdigest(),
                               "label_map.pbtxt": hashlib.sha256(label).hexdigest()})
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"datasets": [{
        "name": "nested", "title": "t", "licence": "CC-BY-4.0", "source": "https://example.org",
        "fetch": "files", "rename": {"label_map.pbtxt": (tmp_path / "label_map.pbtxt").as_uri()},
        "members": [{"prefix": "China_Drone/", "url": outer.as_uri(),
                     "nest": ["RDD2022/China_Drone.zip"], "member_prefix": "China_Drone/"}]}]}))
    monkeypatch.setattr(fsd, "MANIFESTS", manifests)
    monkeypatch.setattr(fsd, "CATALOG", catalog)
    dest = tmp_path / "dest"
    result = fsd.fetch("nested", dest, log=lambda *_: None)
    assert sorted(result["downloaded"]) == ["China_Drone/train/images/a.jpg", "label_map.pbtxt"]
    assert not result["failed"] and not result["not_fetched"]
    assert (dest / "China_Drone/train/images/a.jpg").read_bytes() == drone_a
    assert not (dest / "China_Drone/train/images/b.jpg").exists()   # not pinned, not fetched


def test_a_compressed_nested_zip_is_refused(tmp_path):
    outer = tmp_path / "outer.zip"
    outer.write_bytes(_zip_bytes({"inner.zip": _zip_bytes({"a": b"a"})}, zipfile.ZIP_DEFLATED))
    with pytest.raises(fsd.FetchRefused, match="compressed"):
        fsd.open_remote_zip(outer.as_uri(), ["inner.zip"])


def test_an_archive_is_unpacked_into_its_folder_and_checked(tmp_path, monkeypatch):
    good, bad = b"ortho-bytes", b"tampered"
    archive = _zip_bytes({"B1_Aug22.tif": good, "notes.txt": bad, "__MACOSX/._B1": b"junk"})
    (tmp_path / "up").mkdir()
    (tmp_path / "up" / "ortho.zip").write_bytes(archive)
    manifests = tmp_path / "manifests"
    _pin(manifests, "arch", {"ortho.zip": hashlib.sha256(archive).hexdigest(),
                             "orthophoto/B1_Aug22.tif": hashlib.sha256(good).hexdigest(),
                             "orthophoto/notes.txt": "0" * 64})
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"datasets": [{
        "name": "arch", "title": "t", "licence": "CC0-1.0", "source": "https://example.org",
        "fetch": "archive", "archives": [{"url": (tmp_path / "up" / "ortho.zip").as_uri(),
                                          "save_as": "ortho.zip", "extract": True,
                                          "extract_to": "orthophoto"}]}]}))
    monkeypatch.setattr(fsd, "MANIFESTS", manifests)
    monkeypatch.setattr(fsd, "CATALOG", catalog)
    dest = tmp_path / "dest"
    result = fsd.fetch("arch", dest, log=lambda *_: None)
    assert "orthophoto/B1_Aug22.tif" in result["downloaded"]
    assert any(f.startswith("orthophoto/notes.txt") for f in result["failed"])
    assert not (dest / "orthophoto" / "notes.txt").exists()
    assert not (dest / "orthophoto" / "__MACOSX").exists()


def _fake_upstream(tmp_path, files: dict[str, bytes], pinned: dict[str, str] | None = None):
    upstream = tmp_path / "upstream"
    for rel, data in files.items():
        (upstream / rel).parent.mkdir(parents=True, exist_ok=True)
        (upstream / rel).write_bytes(data)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    lines = [f"{(pinned or {}).get(rel) or hashlib.sha256(data).hexdigest()} *./{rel}"
             for rel, data in files.items()]
    (manifests / "toy.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"datasets": [{
        "name": "toy", "title": "toy", "licence": "CC0-1.0", "source": "https://example.org",
        "fetch": "files", "commit": "x", "base_url": upstream.as_uri() + "/"}]}), encoding="utf-8")
    return manifests, catalog


@pytest.fixture
def toy(tmp_path, monkeypatch):
    def make(files, pinned=None):
        manifests, catalog = _fake_upstream(tmp_path, files, pinned)
        monkeypatch.setattr(fsd, "MANIFESTS", manifests)
        monkeypatch.setattr(fsd, "CATALOG", catalog)
        return tmp_path / "dest"
    return make


def test_files_are_fetched_verified_and_not_fetched_twice(toy):
    dest = toy({"license.txt": b"CC0\n", "images/a.jpg": b"\xff\xd8a", "images/b.jpg": b"\xff\xd8b"})
    first = fsd.fetch("toy", dest, log=lambda *_: None)
    assert sorted(first["downloaded"]) == ["images/a.jpg", "images/b.jpg", "license.txt"]
    second = fsd.fetch("toy", dest, log=lambda *_: None)
    assert len(second["already_present"]) == 3 and not second["downloaded"]
    assert fsd.verify("toy", dest) == {"ok": ["license.txt", "images/a.jpg", "images/b.jpg"],
                                       "missing": [], "mismatched": []}


def test_a_file_that_does_not_match_its_pin_is_deleted_and_reported(toy):
    dest = toy({"images/a.jpg": b"\xff\xd8real"}, pinned={"images/a.jpg": "0" * 64})
    result = fsd.fetch("toy", dest, log=lambda *_: None)
    assert result["failed"] and "sha256 mismatch" in result["failed"][0]
    assert not (dest / "images" / "a.jpg").exists()
    assert not list(dest.rglob("*.part"))


def test_a_text_file_is_accepted_only_when_its_crlf_form_matches(toy):
    lf = b"line one\nline two\n"
    crlf_digest = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
    dest = toy({"README.md": lf}, pinned={"README.md": crlf_digest})
    assert fsd.fetch("toy", dest, log=lambda *_: None)["downloaded"] == ["README.md"]
    assert (dest / "README.md").read_bytes() == lf.replace(b"\n", b"\r\n")


def test_limit_keeps_the_licence_and_the_first_data_files(toy):
    dest = toy({"LICENSE": b"MIT\n", "images/1.jpg": b"1", "images/2.jpg": b"2", "images/3.jpg": b"3"})
    result = fsd.fetch("toy", dest, limit=2, log=lambda *_: None)
    assert sorted(result["downloaded"]) == ["LICENSE", "images/1.jpg", "images/2.jpg"]


def test_the_cli_lists_and_refuses(capsys):
    assert fsd.main(["list"]) == 0
    assert "odm_aukerman" in capsys.readouterr().out
    assert fsd.main(["fetch", "dji_thermal_sdk"]) == 3
    assert "never downloads" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["odm_aukerman", "road_damage_rdd2022_subset",
                                  "thermal_rjpeg_samples_thermal_parser"])
def test_the_pinned_hashes_match_the_reference_copies(name):
    folder = TESTDATA / name
    if not folder.is_dir():
        pytest.skip(f"reference copy {folder} not present")
    result = fsd.verify(name, folder)
    assert not result["missing"] and not result["mismatched"], result["mismatched"][:5]


@pytest.mark.skipif(os.environ.get("ODK_NETWORK_TESTS") != "1",
                    reason="set ODK_NETWORK_TESTS=1 to fetch from the real upstreams")
@pytest.mark.parametrize("name", [n for n, r in fsd.load_catalog().items() if r["fetch"] == "files"])
def test_one_real_file_from_every_upstream(name, tmp_path):
    result = fsd.fetch(name, tmp_path / name, limit=1, log=lambda *_: None)
    assert not result["failed"], result["failed"]
    assert not fsd.verify(name, tmp_path / name, limit=1)["mismatched"]


def _network_member_cases():
    cases = []
    for name, row in fsd.load_catalog().items():
        for rule in row.get("members") or []:
            entry = next(e for e in fsd.read_manifest(name) if e.path.startswith(rule["prefix"]))
            cases.append(pytest.param(name, entry.path, entry.sha256, id=f"{name}:{rule['prefix']}"))
    return cases


@pytest.mark.skipif(os.environ.get("ODK_NETWORK_TESTS") != "1",
                    reason="set ODK_NETWORK_TESTS=1 to range-read the real archives")
@pytest.mark.parametrize("name,path,sha256", _network_member_cases())
def test_one_real_member_from_every_remote_archive(name, path, sha256, tmp_path):
    rule = fsd._rule_for(fsd.load_catalog()[name], path)
    archive = fsd.open_remote_zip(rule["url"], rule.get("nest", []))
    assert fsd._accept(archive.read(fsd._member(rule, path)), tmp_path / "f", sha256) == sha256
