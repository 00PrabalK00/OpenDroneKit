"""Three mission types were implemented, tested, and impossible to invoke.

mission/mission_types.py has planned pylon, thermal and multispectral missions since that
work landed, and tests/test_special_mission_types.py covers the geometry. But none of them
appeared in the planner's template table, in mission_templates(), or on the Api -- so
nothing a user or an operator could do would reach them.

Implemented and unreachable is indistinguishable from missing to everyone outside the
repository, and the registry counted the rows as done. These tests are about the wiring
rather than the geometry: whether the application can actually be asked for one.
"""

from __future__ import annotations

import pytest

from app.api import Api
from app.session import AppSession

AOI = [[77.400, 23.200], [77.404, 23.200], [77.404, 23.204], [77.400, 23.204], [77.400, 23.200]]
PYLON = [77.402, 23.202]
ELEMENTS = [
    {"name": "crossarm", "height_m": 18.0},
    {"name": "insulator", "height_m": 16.0},
    {"name": "conductor", "height_m": 26.0},
]


@pytest.fixture
def api() -> Api:
    return Api(AppSession())


@pytest.fixture
def api_with_aoi(api: Api) -> Api:
    api.set_aoi(AOI)
    return api


class TestPylon:
    def test_the_application_can_be_asked_for_a_pylon_inspection(self, api: Api) -> None:
        result = api.plan_pylon_mission(PYLON, ELEMENTS)
        assert result["ok"], result.get("error")
        assert result["plan"]["template"] == "pylon_inspection"

    def test_it_flies_one_level_per_element(self, api: Api) -> None:
        """The point of the mission type: photographs of identified fittings at the
        height each one sits at, rather than one general circuit of the tower."""
        plan = api.plan_pylon_mission(PYLON, ELEMENTS)["plan"]
        assert len(plan["levels"]) == len(ELEMENTS)
        assert sorted(plan["elements_covered"]) == sorted(e["name"] for e in ELEMENTS)
        assert plan["waypoint_count"] > 0

    def test_an_unknown_element_is_refused_by_name(self, api: Api) -> None:
        """Naming the ones it does plan for is the difference between a refusal a user
        can act on and one they have to go and read the source to understand."""
        result = api.plan_pylon_mission(PYLON, [{"name": "crossarm_lower", "height_m": 18.0}])
        assert not result["ok"]
        assert "crossarm" in result["error"]

    def test_a_standoff_inside_the_structure_is_refused(self, api: Api) -> None:
        result = api.plan_pylon_mission(PYLON, ELEMENTS, standoff_m=2.0, structure_radius_m=3.0)
        assert not result["ok"]
        assert "through the tower" in result["error"]


class TestThermal:
    def test_the_application_can_be_asked_for_a_thermal_survey(self, api_with_aoi: Api) -> None:
        result = api_with_aoi.plan_thermal_survey("flir_vue_pro_r_640", 5.0)
        assert result["ok"], result.get("error")
        assert result["plan"]["template"] == "thermal"

    def test_the_altitude_is_solved_for_the_thermal_sensor(self, api_with_aoi: Api) -> None:
        """A thermal imager has far coarser pixels than the visual camera beside it, so
        a grid planned for the RGB delivers thermal imagery too coarse to read. Asking
        for a finer thermal GSD must bring the aircraft down."""
        coarse = api_with_aoi.plan_thermal_survey("flir_vue_pro_r_640", 10.0)["plan"]
        fine = api_with_aoi.plan_thermal_survey("flir_vue_pro_r_640", 5.0)["plan"]
        assert fine["altitude_m"] < coarse["altitude_m"]

    def test_it_refuses_without_an_area(self, api: Api) -> None:
        result = api.plan_thermal_survey("flir_vue_pro_r_640", 5.0)
        assert not result["ok"]
        assert "area of interest" in result["error"]

    def test_an_unknown_camera_names_the_ones_it_has(self, api_with_aoi: Api) -> None:
        result = api_with_aoi.plan_thermal_survey("not_a_real_camera", 5.0)
        assert not result["ok"]
        assert "not in the database" in result["error"]


class TestMultispectral:
    def test_it_refuses_without_a_reflectance_panel(self, api_with_aoi: Api) -> None:
        """The refusal is the feature. Indices from a flight with no panel capture
        cannot be compared with any other flight, and a plan that quietly omitted it
        would produce numbers that look fine and mean nothing.
        """
        result = api_with_aoi.plan_multispectral_survey("micasense_rededge_mx")
        assert not result["ok"]
        assert "panel" in result["error"].lower()

    def test_with_a_panel_it_plans(self, api_with_aoi: Api) -> None:
        result = api_with_aoi.plan_multispectral_survey(
            "micasense_rededge_mx", calibration_panel_lonlat=[77.3995, 23.1995])
        assert result["ok"], result.get("error")
        plan = result["plan"]
        assert plan["template"] == "multispectral"
        assert plan["calibration_captures"], "the panel captures are not in the plan"
        assert plan["band_count"] >= 5

    def test_it_carries_the_payload_bands(self, api_with_aoi: Api) -> None:
        plan = api_with_aoi.plan_multispectral_survey(
            "micasense_rededge_mx", calibration_panel_lonlat=[77.3995, 23.1995])["plan"]
        assert plan["bands_nm"], "the band wavelengths are not reported"


def test_all_three_are_on_the_api_surface() -> None:
    """The specific failure: present in the module, absent from every route to it."""
    for method in ("plan_pylon_mission", "plan_thermal_survey", "plan_multispectral_survey"):
        assert hasattr(Api, method), f"{method} is not reachable from the application"
