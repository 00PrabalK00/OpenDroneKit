"""Reconstructing a real survey with COLMAP, end to end.

Everything else in the geospatial stack is unit-tested against synthetic transforms,
which proves the arithmetic but not that the pipeline holds together on photographs
taken from an actual aircraft. This runs the real engine over real imagery and checks
the properties a survey is judged by: cameras registered, reprojection error, a metric
CRS, and rasters that land in the right place on Earth.

It also pins the honesty behaviours, which are the ones that would otherwise decay
quietly. A reconstruction that registers six of eight images must say so rather than
present six as the whole survey. A sparse cloud that cannot fill a fine grid must
relax the resolution and report it, rather than interpolating 403 points into a
detailed-looking surface that was never measured.

The imagery is gitignored because it is 167 MB, so these skip on a checkout without it.
A skip is not a pass: tools/feature_status.py will decline to mark the feature verified,
which is the correct outcome when the evidence is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pycolmap", reason="COLMAP engine not installed")

from core.reconstruction_colmap import ColmapReconstructor, colmap_available

IMAGERY = Path("training/data/aukerman_subset")

# Enough views to triangulate and register, few enough to run in seconds.
IMAGE_COUNT = 8

# COLMAP's own guidance treats a bundle-adjusted model above this as poorly converged.
MAX_REPROJECTION_ERROR_PX = 1.5


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    """Run the real pipeline once over a handful of real aerial frames."""
    if not colmap_available():
        pytest.skip("pycolmap is not available in this environment.")
    if not IMAGERY.is_dir():
        pytest.skip(f"Survey imagery not present at {IMAGERY} (gitignored, 167 MB).")

    frames = sorted(IMAGERY.glob("*.JPG"))[:IMAGE_COUNT]
    if len(frames) < IMAGE_COUNT:
        pytest.skip(f"Need {IMAGE_COUNT} frames, found {len(frames)}.")

    workspace = tmp_path_factory.mktemp("survey")
    images = workspace / "images"
    images.mkdir()
    for frame in frames:
        (images / frame.name).write_bytes(frame.read_bytes())

    reconstructor = ColmapReconstructor(profile="fast", dense=False, max_image_size=1024)
    result = reconstructor.reconstruct(image_dir=images, output_dir=workspace / "out")

    twin_path = Path(result.digital_twin_path)
    twin = json.loads(twin_path.read_text(encoding="utf-8")) if twin_path.exists() else {}
    return result, twin, workspace / "out"


class TestStructureFromMotion:
    def test_the_colmap_engine_is_the_one_that_ran(self, survey):
        """The pipeline falls back to a custom engine, so the label must be earned."""
        _, twin, _ = survey
        assert twin.get("engine") == "colmap"

    def test_cameras_are_registered_from_the_imagery(self, survey):
        _, twin, _ = survey
        assert twin.get("registered_images", 0) >= 2, "fewer than two cameras is not a model"

    def test_bundle_adjustment_converges_below_the_error_threshold(self, survey):
        """The number that says the geometry is self-consistent."""
        _, twin, _ = survey
        error = twin.get("mean_reprojection_error_px")
        assert error is not None, "a reconstruction that does not report its error proves nothing"
        assert 0 < error < MAX_REPROJECTION_ERROR_PX

    def test_a_sparse_cloud_with_real_points_is_produced(self, survey):
        result, twin, _ = survey
        assert twin.get("point_count", 0) > 0
        assert Path(result.point_cloud_path).exists()

    def test_unregistered_images_are_reported_not_hidden(self, survey):
        """Six of eight is a different survey from eight of eight."""
        _, twin, _ = survey
        registered = twin.get("registered_images", 0)
        frames = twin.get("frame_count", 0)
        if registered < frames:
            assert any("failed to register" in w for w in twin.get("warnings", [])), (
                "images that did not register must be declared"
            )


class TestGeoreferencing:
    def test_the_model_lands_in_a_real_metric_crs(self, survey):
        _, twin, _ = survey
        assert twin.get("georeferenced") is True
        epsg = twin.get("crs_epsg")
        # UTM north/south: the auto-selected zone for the survey's latitude.
        assert epsg and (32600 < epsg < 32661 or 32700 < epsg < 32761)

    def test_the_geo_fit_reports_its_residual(self, survey):
        """A georeference without a residual is an assertion, not a measurement."""
        _, twin, _ = survey
        rmse = twin.get("geo_rmse_m")
        assert rmse is not None
        assert 0 <= rmse < 20.0, f"geo RMSE {rmse} m is too large for a GPS-tagged survey"

    def test_the_anchor_records_how_it_was_solved(self, survey):
        _, _, out = survey
        anchor = json.loads((out / "geo_anchor.json").read_text(encoding="utf-8"))
        assert anchor["method"] == "umeyama_ransac"
        assert anchor["inlier_count"] >= 3, "a similarity needs three non-collinear points"
        assert anchor["inlier_count"] <= anchor["sample_count"]

    def test_rasters_are_written_as_georeferenced_geotiffs(self, survey):
        """A PNG carries no CRS, so a measurement taken from one is in pixels."""
        rasterio = pytest.importorskip("rasterio")
        _, twin, out = survey

        for name in ("dsm.tif", "dtm.tif", "orthomosaic.tif"):
            path = out / name
            assert path.exists(), f"{name} was not produced"
            with rasterio.open(path) as raster:
                assert raster.crs is not None, f"{name} has no CRS"
                assert raster.crs.to_epsg() == twin.get("crs_epsg")

    def test_the_dsm_holds_elevations_in_metres(self, survey):
        rasterio = pytest.importorskip("rasterio")
        numpy = pytest.importorskip("numpy")
        _, _, out = survey

        with rasterio.open(out / "dsm.tif") as raster:
            band = raster.read(1, masked=True)
            finite = band.compressed() if hasattr(band, "compressed") else band[numpy.isfinite(band)]
            if finite.size == 0:
                pytest.skip("DSM has no valid cells at this resolution.")
            spread = float(finite.max() - finite.min())
            # Metres above the ellipsoid over a small site: a real range, not pixel indices.
            assert -500.0 < float(finite.min()) < 9000.0
            assert spread < 2000.0

    def test_the_dtm_holds_ground_elevations_in_metres(self, survey):
        rasterio = pytest.importorskip("rasterio")
        numpy = pytest.importorskip("numpy")
        _, _, out = survey

        with rasterio.open(out / "dtm.tif") as raster:
            band = raster.read(1, masked=True)
            finite = band.compressed() if hasattr(band, "compressed") else band[numpy.isfinite(band)]
            if finite.size == 0:
                pytest.skip("DTM has no valid ground cells at this resolution.")
            assert raster.crs is not None and raster.crs.is_projected
            assert -500.0 < float(finite.min()) < 9000.0
            assert float(finite.max() - finite.min()) < 2000.0


class TestNoFabrication:
    def test_resolution_is_relaxed_openly_rather_than_interpolated_silently(self, survey):
        """A sparse cloud cannot fill a fine grid, and pretending otherwise invents terrain."""
        _, twin, _ = survey
        gsd = twin.get("ground_sample_distance_m")
        assert gsd and gsd > 0
        if gsd > 0.5:
            assert any("resolution relaxed" in w.lower() for w in twin.get("warnings", [])), (
                "a coarsened raster must say it was coarsened"
            )

    def test_a_sparse_run_does_not_claim_dense_reconstruction(self, survey):
        _, twin, _ = survey
        assert twin.get("dense") is False

    def test_the_point_count_is_not_inflated_beyond_what_was_triangulated(self, survey):
        """The removed fake cloned sparse points with jitter and called it MVS."""
        _, twin, out = survey
        reported = twin.get("point_count", 0)

        ply = out / "reconstruction.ply"
        if not ply.exists():
            pytest.skip("No point cloud written.")
        header = ply.read_bytes()[:512].decode("ascii", errors="replace")
        for line in header.splitlines():
            if line.startswith("element vertex"):
                assert int(line.split()[-1]) == reported, (
                    "the reported point count must match the cloud actually written"
                )
                break

    def test_missing_optional_dependencies_are_declared_not_worked_around(self, survey):
        """No mesh is an honest outcome; a fake mesh is not."""
        result, twin, _ = survey
        if not result.mesh_path:
            assert any("mesh" in w.lower() for w in twin.get("warnings", [])), (
                "a missing mesh must be explained"
            )
