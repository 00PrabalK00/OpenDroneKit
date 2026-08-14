"""Provenance that can be checked, not merely asserted.

A sidecar saying "this came from those images, in that CRS" is worth nothing if the
file it describes has since been replaced. So the tests that matter here are the ones
where the record and the file disagree: a modified artifact, a missing one, one with no
record at all. Each must be reported as such rather than passed off as attributed.
"""

from __future__ import annotations

import json

import pytest

from core import provenance


def write_artifact(path, content=b"orthomosaic bytes"):
    path.write_bytes(content)
    return path


class TestRecording:
    def test_a_record_captures_the_lineage_that_matters(self, tmp_path):
        artifact = write_artifact(tmp_path / "orthomosaic.tif")

        entry = provenance.record(
            artifact, engine="colmap", engine_version="4.1.1",
            sources=["DSC00229.JPG", "DSC00230.JPG"], crs_epsg=32617,
            parameters={"profile": "fast", "gsd_m": 0.05},
        )

        assert entry.engine == "colmap"
        assert entry.crs_epsg == 32617
        assert entry.source_count == 2
        assert entry.parameters["profile"] == "fast"
        assert len(entry.sha256) == 64

    def test_the_sidecar_travels_next_to_the_artifact(self, tmp_path):
        """A record in a database the recipient does not have is not provenance."""
        artifact = write_artifact(tmp_path / "dsm.tif")
        provenance.record(artifact, engine="colmap")

        sidecar = tmp_path / "dsm.tif.provenance.json"
        assert sidecar.exists()
        assert json.loads(sidecar.read_text(encoding="utf-8"))["engine"] == "colmap"

    def test_the_software_versions_that_produced_it_are_recorded(self, tmp_path):
        artifact = write_artifact(tmp_path / "mesh.ply")
        entry = provenance.record(artifact, engine="open3d")
        assert "python" in entry.software

    def test_recording_a_file_that_does_not_exist_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            provenance.record(tmp_path / "never_written.tif", engine="colmap")

    def test_a_long_source_list_is_truncated_but_the_count_is_not(self, tmp_path):
        """A truncated list must never read as the complete input set."""
        artifact = write_artifact(tmp_path / "big.tif")
        entry = provenance.record(
            artifact, engine="colmap",
            sources=[f"DSC{i:05d}.JPG" for i in range(1200)],
            max_sources_listed=100,
        )

        assert len(entry.sources) == 100
        assert entry.source_count == 1200
        assert "truncated" in entry.notes


class TestVerification:
    def test_an_untouched_artifact_verifies(self, tmp_path):
        artifact = write_artifact(tmp_path / "orthomosaic.tif")
        provenance.record(artifact, engine="colmap", crs_epsg=32617)

        result = provenance.verify(artifact)
        assert result["ok"] is True
        assert result["status"] == "verified"
        assert result["crs_epsg"] == 32617

    def test_a_modified_artifact_is_reported_as_modified(self, tmp_path):
        """The case the digest exists for: the record now describes a different file."""
        artifact = write_artifact(tmp_path / "dsm.tif")
        provenance.record(artifact, engine="colmap")

        artifact.write_bytes(b"a different raster entirely")

        result = provenance.verify(artifact)
        assert result["ok"] is False
        assert result["status"] == "modified"
        assert result["recorded_sha256"] != result["actual_sha256"]

    def test_even_a_one_byte_change_is_caught(self, tmp_path):
        artifact = write_artifact(tmp_path / "dtm.tif", b"abcdefgh")
        provenance.record(artifact, engine="colmap")
        artifact.write_bytes(b"abcdefgi")

        assert provenance.verify(artifact)["status"] == "modified"

    def test_an_artifact_with_no_record_is_unattributed_not_trusted(self, tmp_path):
        artifact = write_artifact(tmp_path / "mystery.tif")
        result = provenance.verify(artifact)

        assert result["ok"] is False
        assert result["status"] == "no_provenance"
        assert "unattributed" in result["detail"]

    def test_a_recorded_but_deleted_artifact_is_reported_missing(self, tmp_path):
        artifact = write_artifact(tmp_path / "gone.tif")
        provenance.record(artifact, engine="colmap")
        artifact.unlink()

        result = provenance.verify(artifact)
        assert result["status"] == "missing_artifact"

    def test_a_corrupt_sidecar_reads_as_no_provenance_rather_than_crashing(self, tmp_path):
        artifact = write_artifact(tmp_path / "ortho.tif")
        provenance.sidecar_path(artifact).write_text("{ not json", encoding="utf-8")

        assert provenance.read(artifact) is None
        assert provenance.verify(artifact)["status"] == "no_provenance"


class TestReconstructionOutputs:
    def test_every_derived_file_in_a_run_is_recorded(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        for name in ("DSC1.JPG", "DSC2.JPG", "DSC3.JPG"):
            (images / name).write_bytes(b"jpeg")

        out = tmp_path / "out"
        out.mkdir()
        for name in ("orthomosaic.tif", "dsm.tif", "reconstruction.ply"):
            (out / name).write_bytes(b"artifact")

        summary = provenance.record_reconstruction_outputs(
            out, engine="colmap", image_dir=images, crs_epsg=32617,
            parameters={"profile": "fast"},
        )

        assert summary["artifact_count"] == 3
        assert summary["source_count"] == 3
        assert provenance.verify(out / "dsm.tif")["ok"] is True

    def test_sidecars_do_not_get_sidecars_of_their_own(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "dsm.tif").write_bytes(b"raster")
        images = tmp_path / "images"
        images.mkdir()

        provenance.record_reconstruction_outputs(out, engine="colmap", image_dir=images)
        provenance.record_reconstruction_outputs(out, engine="colmap", image_dir=images)

        assert not (out / "dsm.tif.provenance.json.provenance.json").exists()

    def test_a_non_folder_is_refused(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            provenance.record_reconstruction_outputs(
                tmp_path / "nope", engine="colmap", image_dir=tmp_path)


class TestAudit:
    def test_a_clean_folder_audits_clean(self, tmp_path):
        for name in ("a.tif", "b.tif"):
            artifact = write_artifact(tmp_path / name)
            provenance.record(artifact, engine="colmap")

        report = provenance.audit(tmp_path)
        assert report["ok"] is True
        assert len(report["verified"]) == 2

    def test_the_audit_names_the_files_that_fail_and_why(self, tmp_path):
        good = write_artifact(tmp_path / "good.tif")
        provenance.record(good, engine="colmap")

        tampered = write_artifact(tmp_path / "tampered.tif")
        provenance.record(tampered, engine="colmap")
        tampered.write_bytes(b"swapped")

        write_artifact(tmp_path / "orphan.tif")

        report = provenance.audit(tmp_path)
        assert report["ok"] is False
        assert report["verified"] == ["good.tif"]
        assert report["modified"] == ["tampered.tif"]
        assert report["unattributed"] == ["orphan.tif"]


class TestRealReconstruction:
    def test_a_real_run_can_be_recorded_and_verified(self, tmp_path):
        """Provenance must work on the artifacts the pipeline actually writes."""
        pytest.importorskip("pycolmap")
        from pathlib import Path

        from core.reconstruction_colmap import ColmapReconstructor, colmap_available

        imagery = Path("training/data/aukerman_subset")
        if not colmap_available() or not imagery.is_dir():
            pytest.skip("Survey imagery or pycolmap not available.")

        frames = sorted(imagery.glob("*.JPG"))[:8]
        if len(frames) < 8:
            pytest.skip("Not enough frames.")

        images = tmp_path / "images"
        images.mkdir()
        for frame in frames:
            (images / frame.name).write_bytes(frame.read_bytes())

        out = tmp_path / "out"
        ColmapReconstructor(profile="fast", dense=False, max_image_size=1024).reconstruct(
            image_dir=images, output_dir=out)

        summary = provenance.record_reconstruction_outputs(
            out, engine="colmap", image_dir=images, crs_epsg=32617,
            parameters={"profile": "fast", "dense": False},
        )
        assert summary["artifact_count"] > 0
        assert summary["source_count"] == 8

        report = provenance.audit(out)
        assert report["ok"] is True, f"unverified artifacts: {report['unattributed']}"
