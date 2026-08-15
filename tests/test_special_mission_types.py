"""Pylon, thermal and multispectral missions, planned by the real mission engine.

The precedent these tests exist against is a real bug from this repository: facade
inspection was missing from the planner's alias table and quietly compiled to a nadir
grid. Waypoints were produced, the plan looked ordinary, and the aircraft would have
photographed the roof instead of the wall.

So none of these tests assert that a plan came back. They assert that the plan is
different in the way the mission type is supposed to be different -- orbits at the
heights of named fittings, an altitude derived from the thermal sensor rather than the
RGB one, calibration captures at both ends of a multispectral flight -- and that the
refusals fire when the inputs cannot support any of that.
"""

from __future__ import annotations

import pytest

from mission.cameras import require as require_camera
from mission.mission_types import (
    MissionTypeRefused,
    PylonElement,
    plan_multispectral_mission,
    plan_pylon_inspection,
    plan_thermal_mission,
)
from mission.planner import MissionPlanner

FIELD = [
    [77.5940, 12.9700],
    [77.5960, 12.9700],
    [77.5960, 12.9715],
    [77.5940, 12.9715],
]
PYLON = [77.5950, 12.9707]

# A 400 kV lattice tower: body, two crossarm levels, the insulator strings below each,
# and the conductor attachments.
ELEMENTS = [
    {"name": "body", "height_m": 12.0},
    {"name": "crossarm", "height_m": 24.0},
    {"name": "insulator", "height_m": 22.0},
    {"name": "conductor", "height_m": 19.5},
]


@pytest.fixture
def planner():
    return MissionPlanner()


class TestPylonInspection:
    def test_each_named_element_gets_its_own_orbit_at_its_own_height(self, planner):
        result = plan_pylon_inspection(planner, center_lonlat=PYLON, elements=ELEMENTS)

        assert result["element_count"] == 4
        heights = [level["height_m"] for level in result["levels"]]
        assert heights == sorted(heights), "levels should climb the structure in order"
        assert set(result["elements_covered"]) == {"body", "crossarm", "insulator", "conductor"}

    def test_the_gimbal_differs_per_element_rather_than_pointing_down(self, planner):
        """A nadir gimbal photographs the ground beside the tower, not the fitting."""
        result = plan_pylon_inspection(planner, center_lonlat=PYLON, elements=ELEMENTS)
        tilts = {level["element"]: level["gimbal_tilt_deg"] for level in result["levels"]}

        assert tilts["crossarm"] == 0.0
        assert tilts["insulator"] == -20.0
        assert tilts["conductor"] == -30.0
        assert all(tilt > -90.0 for tilt in tilts.values())

    def test_every_level_is_a_real_compiled_plan_with_waypoints(self, planner):
        result = plan_pylon_inspection(planner, center_lonlat=PYLON, elements=ELEMENTS)

        assert len(result["plans"]) == 4
        assert all(len(plan.waypoints) >= 4 for plan in result["plans"])
        assert all(plan.template == "orbit" for plan in result["plans"])
        assert result["waypoint_count"] == sum(len(p.waypoints) for p in result["plans"])

    def test_a_per_element_radius_is_honoured(self, planner):
        close = [{"name": "insulator", "height_m": 22.0, "radius_m": 8.0}]
        result = plan_pylon_inspection(planner, center_lonlat=PYLON, elements=close,
                                       standoff_m=15.0)

        assert result["levels"][0]["radius_m"] == 8.0
        assert result["levels"][0]["capture_range_m"] == pytest.approx(5.0)

    def test_a_structure_with_no_stated_elements_is_refused(self, planner):
        """Guessing crossarm heights near live conductors is a clearance problem."""
        with pytest.raises(MissionTypeRefused, match="what this mission type exists to replace"):
            plan_pylon_inspection(planner, center_lonlat=PYLON, elements=[])

    def test_a_standoff_inside_the_structure_is_refused(self, planner):
        with pytest.raises(MissionTypeRefused, match="through the tower"):
            plan_pylon_inspection(planner, center_lonlat=PYLON, elements=ELEMENTS,
                                  standoff_m=2.0, structure_radius_m=3.0)

    def test_an_element_the_engine_does_not_plan_for_is_named(self, planner):
        with pytest.raises(MissionTypeRefused, match="not a pylon element"):
            PylonElement(name="transformer", height_m=5.0)

    def test_a_height_that_was_not_measured_is_refused(self, planner):
        with pytest.raises(MissionTypeRefused, match="positive measured value"):
            PylonElement(name="crossarm", height_m=0.0)


class TestThermalMission:
    def test_altitude_comes_from_the_thermal_sensor_not_the_rgb_one(self, planner):
        """The whole point: a 640x512 imager needs to fly much lower for the same GSD."""
        thermal = plan_thermal_mission(
            planner, polygon_lonlat=FIELD, thermal_camera="mavic3t_thermal",
            target_gsd_cm=5.0)
        rgb_altitude = require_camera("zenmuse_p1").altitude_for_gsd_m(5.0)

        assert thermal["altitude_m"] < rgb_altitude / 2
        assert thermal["thermal_gsd_cm"] == pytest.approx(5.0, abs=0.01)

    def test_the_plan_is_a_real_grid_over_the_area(self, planner):
        result = plan_thermal_mission(
            planner, polygon_lonlat=FIELD, thermal_camera="flir_vue_pro_r_640",
            target_gsd_cm=8.0)

        assert len(result["plan"].waypoints) > 4
        assert result["plan"].altitude_m == pytest.approx(result["altitude_m"], abs=0.5)

    def test_a_paired_rgb_camera_is_recorded_with_its_own_gsd(self, planner):
        result = plan_thermal_mission(
            planner, polygon_lonlat=FIELD, thermal_camera="mavic3t_thermal",
            target_gsd_cm=5.0, rgb_camera="mavic3t_wide")

        assert result["paired_rgb"] is True
        assert result["capture_contract"]["paired_capture"] is True
        # The RGB sensor at the thermal altitude is much finer, which is the right way round.
        assert result["rgb_gsd_cm"] < result["thermal_gsd_cm"]

    def test_flying_thermal_without_an_rgb_pair_says_what_is_lost(self, planner):
        result = plan_thermal_mission(
            planner, polygon_lonlat=FIELD, thermal_camera="mavic3t_thermal",
            target_gsd_cm=5.0)

        assert result["paired_rgb"] is False
        assert any("without a" in limit and "visual pair" in limit
                   for limit in result["limits"])

    def test_a_non_thermal_camera_is_refused(self, planner):
        """Otherwise the deliverable is ordinary photographs labelled thermal."""
        with pytest.raises(MissionTypeRefused, match="not a radiometric thermal camera"):
            plan_thermal_mission(planner, polygon_lonlat=FIELD,
                                 thermal_camera="zenmuse_p1", target_gsd_cm=5.0)

    def test_an_unknown_camera_is_refused_rather_than_defaulted(self, planner):
        with pytest.raises(MissionTypeRefused, match="not in the database"):
            plan_thermal_mission(planner, polygon_lonlat=FIELD,
                                 thermal_camera="whatever_we_have", target_gsd_cm=5.0)

    def test_the_radiometric_caveat_travels_with_the_plan(self, planner):
        result = plan_thermal_mission(
            planner, polygon_lonlat=FIELD, thermal_camera="mavic3t_thermal",
            target_gsd_cm=5.0)

        assert any("does not correct them" in limit for limit in result["limits"])


class TestMultispectralMission:
    PANEL = [77.5941, 12.9701]

    def test_the_band_set_is_carried_into_the_mission(self, planner):
        result = plan_multispectral_mission(
            planner, polygon_lonlat=FIELD, payload_key="micasense_rededge_mx",
            calibration_panel_lonlat=self.PANEL)

        assert result["band_count"] == 5
        assert 717.0 in result["bands_nm"], "red edge is what makes it a crop sensor"
        assert result["capture_contract"]["synchronised_bands"] is True

    def test_calibration_captures_are_planned_at_both_ends(self, planner):
        """One panel shot is not enough: the light changes during the flight."""
        result = plan_multispectral_mission(
            planner, polygon_lonlat=FIELD, payload_key="micasense_rededge_mx",
            calibration_panel_lonlat=self.PANEL)

        when = [capture["when"] for capture in result["calibration_captures"]]
        assert when == ["before", "after"]
        assert all(capture["command"] == "calibrate"
                   for capture in result["calibration_captures"])

    def test_the_survey_grid_is_a_real_plan(self, planner):
        result = plan_multispectral_mission(
            planner, polygon_lonlat=FIELD, payload_key="mavic3m_multispectral",
            calibration_panel_lonlat=self.PANEL, altitude_m=50.0)

        assert len(result["plan"].waypoints) > 4
        assert result["plan"].altitude_m == pytest.approx(50.0, abs=0.5)

    def test_a_calibrating_payload_without_a_panel_position_is_refused(self, planner):
        with pytest.raises(MissionTypeRefused, match="not be comparable with any other survey"):
            plan_multispectral_mission(planner, polygon_lonlat=FIELD,
                                       payload_key="micasense_rededge_mx")

    def test_a_payload_that_is_not_multispectral_is_refused(self, planner):
        with pytest.raises(MissionTypeRefused, match="would capture no bands"):
            plan_multispectral_mission(planner, polygon_lonlat=FIELD,
                                       payload_key="zenmuse_p1",
                                       calibration_panel_lonlat=self.PANEL)

    def test_an_undescribed_payload_is_refused(self, planner):
        with pytest.raises(MissionTypeRefused, match="not described"):
            plan_multispectral_mission(planner, polygon_lonlat=FIELD,
                                       payload_key="borrowed_sensor",
                                       calibration_panel_lonlat=self.PANEL)

    def test_the_comparability_limit_is_stated(self, planner):
        result = plan_multispectral_mission(
            planner, polygon_lonlat=FIELD, payload_key="micasense_rededge_mx",
            calibration_panel_lonlat=self.PANEL)

        assert any("comparable between surveys" in limit for limit in result["limits"])
