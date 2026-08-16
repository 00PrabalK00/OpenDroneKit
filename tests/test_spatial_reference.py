"""What a reconstruction's coordinates mean, and when measuring in it is refused.

Structure-from-motion recovers geometry up to a similarity transform. A GPS-denied
reconstruction with no control still renders, still meshes, still produces a convincing
point cloud -- and every distance in it is wrong by an unknown factor.

That is the failure these tests exist for. Not that the reconstruction fails, but that
it succeeds and someone measures in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.spatial_reference import (
    MIN_GCP_COUNT,
    MIN_GEOTAG_FRACTION,
    MeasurementRefused,
    SpatialReference,
    assess_spatial_reference,
    require_measurable,
)


@pytest.fixture
def images(tmp_path: Path, monkeypatch):
    """A factory making N image paths of which `tagged` report a GPS fix."""
    def build(count: int, tagged: int):
        paths = []
        for i in range(count):
            path = tmp_path / f"img_{i:03d}.jpg"
            path.write_bytes(b"stub")
            paths.append(path)
        tagged_set = {str(p) for p in paths[:tagged]}

        from core import geo

        def fake_read(path):
            return object() if str(path) in tagged_set else None

        monkeypatch.setattr(geo, "read_exif_gps", fake_read)
        return paths
    return build


class TestGeoreferenced:
    def test_fully_geotagged_imagery_is_measurable(self, images) -> None:
        result = assess_spatial_reference(images(20, 20), epsg=32643)
        assert result.mode == "georeferenced"
        assert result.measurements_allowed
        assert result.scale_is_known
        assert result.epsg == 32643

    def test_it_warns_that_gnss_accuracy_is_metres_not_centimetres(self, images) -> None:
        # Georeferenced is not the same as accurate, and a user planning a survey
        # deserves that distinction before flying rather than after.
        result = assess_spatial_reference(images(20, 20))
        assert any("consumer receiver" in w for w in result.warnings)


class TestArbitraryScale:
    def test_no_geotags_and_no_control_refuses_measurement(self, images) -> None:
        result = assess_spatial_reference(images(20, 0))
        assert result.mode == "arbitrary"
        assert not result.measurements_allowed
        assert not result.scale_is_known

    def test_the_note_names_scale_specifically(self, images) -> None:
        # Position and rotation being arbitrary is obvious. Scale being arbitrary is the
        # one that fools people, because the model still looks right.
        result = assess_spatial_reference(images(20, 0))
        assert "SCALE" in result.note
        assert "cannot be measured" in result.note

    def test_no_epsg_is_claimed_for_an_arbitrary_model(self, images) -> None:
        # Carrying a CRS through would let downstream code believe the coordinates mean
        # something on the ground.
        result = assess_spatial_reference(images(20, 0), epsg=32643)
        assert result.epsg is None

    def test_the_guard_refuses(self, images) -> None:
        result = assess_spatial_reference(images(20, 0))
        with pytest.raises(MeasurementRefused, match="arbitrary-scale"):
            require_measurable(result, what="compute a volume")

    def test_the_guard_passes_a_georeferenced_model(self, images) -> None:
        require_measurable(assess_spatial_reference(images(20, 20)))


class TestControlReferenced:
    def test_gcps_alone_restore_measurability(self, images) -> None:
        result = assess_spatial_reference(images(20, 0), gcp_count=MIN_GCP_COUNT)
        assert result.mode == "control_referenced"
        assert result.measurements_allowed
        assert result.scale_is_known

    def test_too_few_gcps_do_not(self, images) -> None:
        # Fewer than three cannot resolve scale and rotation together.
        result = assess_spatial_reference(images(20, 0), gcp_count=MIN_GCP_COUNT - 1)
        assert result.mode == "arbitrary"
        assert not result.measurements_allowed


class TestSparseGeotags:
    def test_partial_geotags_do_not_count_as_georeferenced(self, images) -> None:
        # A handful of tagged frames in a long sequence pins the ends and lets the
        # middle drift, which looks georeferenced and is not.
        count = 20
        tagged = int(count * MIN_GEOTAG_FRACTION) - 1
        result = assess_spatial_reference(images(count, tagged), gcp_count=0)
        assert result.mode == "arbitrary"

    def test_the_sparsity_is_explained_not_just_flagged(self, images) -> None:
        result = assess_spatial_reference(images(20, 5))
        assert any("drift" in w for w in result.warnings)

    def test_enough_geotags_still_qualifies(self, images) -> None:
        count = 20
        tagged = int(count * MIN_GEOTAG_FRACTION) + 1
        result = assess_spatial_reference(images(count, tagged))
        assert result.mode == "georeferenced"


class TestEdges:
    def test_no_images_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to reconstruct"):
            assess_spatial_reference([])

    def test_the_payload_reports_the_fraction(self, images) -> None:
        payload = assess_spatial_reference(images(10, 8)).to_dict()
        assert payload["geotagged_fraction"] == pytest.approx(0.8)
