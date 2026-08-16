"""A 3D modelling mission photographs walls, not just roofs.

A cross-hatch alone sees vertical surfaces at a grazing angle or not at all. A
reconstruction built from nadir imagery still produces buildings -- with sides that are
stretched texture over guessed geometry. The model looks finished, which is what makes
the omission expensive: nobody queries a facade that rendered.

The oblique bands put real observations on those surfaces, and these tests hold three
things in place: that asking for a 3D model adds them, that asking for a plain double
grid does not, and that the extra flight time stays proportionate.

The third one is not housekeeping. The bands first arrived compiling as a second full
nadir grid -- 6,120 poses instead of 48 -- because the primitive dispatcher normalises
its kind through the mission-template alias table, and that table maps anything it does
not recognise to "grid". No error, no warning: a 64-minute survey silently became a
179-minute one that no battery plan supports, and every one of those extra frames was
nadir. The regression test below is the one that would have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mission.planner import (
    MAPPING_3D_ALIASES,
    OBLIQUE_BAND_DEFAULT_TILTS_DEG,
    MissionPlanner,
)

SITE = [
    [-81.7525, 41.3022],
    [-81.7485, 41.3022],
    [-81.7485, 41.3062],
    [-81.7525, 41.3062],
]

LOCAL_SQUARE = [[-200.0, -200.0], [200.0, -200.0], [200.0, 200.0], [-200.0, 200.0], [-200.0, -200.0]]


def plan(mode: str, **kwargs):
    return MissionPlanner().generate(
        polygon_lonlat=SITE, mode=mode, altitude_m=60.0, speed_m_s=8.0, **kwargs
    )


@pytest.fixture(scope="module")
def poses() -> list:
    return MissionPlanner()._compile_oblique_bands_primitive(
        {"polygon_local": LOCAL_SQUARE, "altitude_m": 60.0}
    )


class TestTheBandsAreOblique:
    def test_two_tilts_are_flown_by_default(self, poses) -> None:
        tilts = sorted({pose.gimbal_pitch_deg for pose in poses})
        assert tilts == sorted(OBLIQUE_BAND_DEFAULT_TILTS_DEG)

    def test_no_band_is_effectively_nadir(self, poses) -> None:
        # A "band" at -85 is a nadir pass with extra flying, and would add cost without
        # adding the views the mission exists to collect.
        for pose in poses:
            assert -75.0 <= pose.gimbal_pitch_deg <= -20.0

    def test_the_camera_faces_the_site_from_every_point(self, poses) -> None:
        """A ring that photographs outwards is a perimeter flight, not a survey."""
        for pose in poses:
            bearing_to_centre = np.degrees(np.arctan2(-pose.x_m, -pose.y_m))
            delta = abs((pose.yaw_deg - bearing_to_centre + 180.0) % 360.0 - 180.0)
            assert delta < 1.0, "a capture point is not aimed at the site"

    def test_the_rings_stand_off_outside_the_area(self, poses) -> None:
        radii = [float(np.hypot(pose.x_m, pose.y_m)) for pose in poses]
        # The square's own extent is ~283 m corner to centre.
        assert min(radii) > 283.0, "the ring sits over the site, so the camera looks down into it"

    def test_a_shallower_tilt_stands_further_back(self, poses) -> None:
        """Geometry, not decoration: reach is altitude / tan(tilt)."""
        by_tilt: dict[float, float] = {}
        for pose in poses:
            by_tilt.setdefault(pose.gimbal_pitch_deg, float(np.hypot(pose.x_m, pose.y_m)))
        shallow = by_tilt[-45.0]
        steep = by_tilt[-60.0]
        assert shallow > steep

    def test_each_band_is_its_own_primitive(self, poses) -> None:
        assert len({pose.primitive for pose in poses}) == len(OBLIQUE_BAND_DEFAULT_TILTS_DEG)


class TestOnlyAskedForMissionsPayForIt:
    def test_a_3d_modelling_mission_adds_oblique_capture(self) -> None:
        assert len(plan("mapping_3d").waypoints) > len(plan("double_grid").waypoints)

    def test_a_plain_double_grid_stays_nadir(self) -> None:
        """The extra flight time must be opt-in, not a surprise on an existing workflow."""
        nadir_poses = MissionPlanner()._compile_oblique_bands_primitive(
            {"polygon_local": LOCAL_SQUARE, "altitude_m": 60.0, "oblique_points_per_ring": 8}
        )
        assert nadir_poses, "sanity: the compiler does produce bands when asked"
        assert (
            plan("double_grid").to_dict()["estimated_time_min"]
            < plan("mapping_3d").to_dict()["estimated_time_min"]
        )

    @pytest.mark.parametrize("alias", sorted(MAPPING_3D_ALIASES))
    def test_every_alias_asks_for_the_same_mission(self, alias) -> None:
        assert len(plan(alias).waypoints) == len(plan("mapping_3d").waypoints)


class TestTheCostStaysProportionate:
    """The regression. Silent inflation is the failure mode, not an exception."""

    def test_the_bands_add_tens_of_poses_not_thousands(self) -> None:
        poses = MissionPlanner()._compile_oblique_bands_primitive(
            {"polygon_local": LOCAL_SQUARE, "altitude_m": 60.0}
        )
        assert len(poses) == 2 * 24, (
            f"the oblique bands produced {len(poses)} poses. Two rings of 24 is 48; a "
            "number in the thousands means the primitive compiled as a nadir grid."
        )

    def test_a_3d_mission_does_not_double_the_flight_time(self) -> None:
        nadir = plan("double_grid").to_dict()["estimated_time_min"]
        modelled = plan("mapping_3d").to_dict()["estimated_time_min"]
        assert modelled < nadir * 1.5, (
            f"a 3D mission takes {modelled:.0f} min against {nadir:.0f} min for the "
            "cross-hatch alone. Oblique rings are a perimeter, not another area pass; "
            "this much growth means they are being flown as a grid."
        )

    def test_the_added_capture_is_actually_oblique(self) -> None:
        """Growth alone is not proof -- the extra frames have to be the oblique ones."""
        poses = MissionPlanner()._compile_oblique_bands_primitive(
            {"polygon_local": LOCAL_SQUARE, "altitude_m": 60.0}
        )
        assert all(pose.gimbal_pitch_deg > -80.0 for pose in poses)
        assert all(pose.camera_yaw_locked for pose in poses)


class TestRefusals:
    def test_a_degenerate_polygon_produces_no_bands(self) -> None:
        assert MissionPlanner()._compile_oblique_bands_primitive(
            {"polygon_local": [[0.0, 0.0], [1.0, 1.0]], "altitude_m": 60.0}
        ) == []

    def test_requested_tilts_are_clamped_rather_than_flown_as_given(self) -> None:
        poses = MissionPlanner()._compile_oblique_bands_primitive(
            {"polygon_local": LOCAL_SQUARE, "altitude_m": 60.0, "oblique_tilts_deg": [-5.0, -89.0]}
        )
        tilts = sorted({pose.gimbal_pitch_deg for pose in poses})
        assert tilts == [-75.0, -20.0], "out-of-range tilts were flown rather than clamped"
