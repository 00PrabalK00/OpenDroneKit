"""A plan's warnings must describe the plan that was made, and offer only real options.

Three defects a re-verification found, each a warning that said the wrong thing:

- a terrain CSV that the planner DID follow was reported as "does not cover the planned
  area", because the coverage check only knew rasters and a CSV has no CRS to read;
- a multispectral survey without a reflectance panel was refused with the advice to
  "accept single-flight relative values explicitly", an option that did not exist;
- a 55 minute facade mapping planned without a word about batteries.

Every test goes through the Api the UI calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

AOI = [[-81.7535, 41.3035], [-81.7521, 41.3035], [-81.7521, 41.3044], [-81.7535, 41.3044]]


@pytest.fixture
def api(tmp_path) -> Api:
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("warnings", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi(AOI)["ok"]
    return api


def terrain_csv(path: Path, west: float, south: float, east: float, north: float) -> Path:
    lines = ["lon,lat,elev"]
    for lat in np.arange(south, north, 0.0002):
        for lon in np.arange(west, east, 0.0002):
            lines.append(f"{lon:.6f},{lat:.6f},{200.0 + 3000.0 * (lon + 81.755):.3f}")
    path.write_text("\n".join(lines) + "\n")
    return path


class TestTerrainCsvCoverage:
    def test_a_csv_that_spans_the_area_is_not_called_uncovered(self, api, tmp_path):
        source = terrain_csv(tmp_path / "hill.csv", -81.755, 41.302, -81.750, 41.306)
        assert api.set_terrain_source(str(source))["ok"]
        planned = api.plan_mission({"template": "grid", "altitude_m": 50})
        assert planned["ok"], planned.get("error")
        text = " ".join(planned["warnings"])
        assert "does not cover" not in text
        assert "coordinate reference system" not in text
        assert "could not be read" not in text
        # The terrain really was followed: altitudes vary with the slope.
        alts = {round(c[2], 1) for f in planned["geojson"]["features"]
                if f["properties"].get("role") == "waypoint"
                for c in [f["geometry"]["coordinates"]]}
        assert len(alts) > 3

    def test_a_csv_that_stops_short_still_says_so(self, api, tmp_path):
        source = terrain_csv(tmp_path / "short.csv", -81.755, 41.302, -81.7528, 41.306)
        assert api.set_terrain_source(str(source))["ok"]
        planned = api.plan_mission({"template": "grid", "altitude_m": 50})
        text = " ".join(planned["warnings"])
        assert "does not cover the planned area" in text and "part of this area" in text


class TestMultispectralWithoutAPanel:
    OPTIONS = {"template": "multispectral", "payload_key": "mavic3m_multispectral"}

    def test_it_is_refused_unless_relative_values_are_accepted(self, api):
        refused = api.plan_mission(dict(self.OPTIONS))
        assert not refused["ok"]
        assert "accept_relative_values" in refused["error"]
        # The cockpit form sends strings; "no" must still refuse.
        assert not api.plan_mission({**self.OPTIONS, "accept_relative_values": "no"})["ok"]

    @pytest.mark.parametrize("accepted", [True, "yes"])
    def test_accepting_relative_values_plans_and_says_what_was_given_up(self, api, tmp_path, accepted):
        planned = api.plan_mission({**self.OPTIONS, "accept_relative_values": accepted})
        assert planned["ok"], planned.get("error")
        assert planned["summary"]["template"] == "multispectral"
        assert any("relative" in w and "must not be compared" in w for w in planned["warnings"])
        contract = planned["plan"]["capture_contract"]
        assert contract["calibrated"] is False and contract["relative_values_only"] is True
        assert planned["plan"]["calibration_captures"] == []
        exported = api.export_mission(["qgc_plan"], str(tmp_path / "ms"))
        assert exported["ok"] and not str(exported["written"]["qgc_plan"]).startswith("failed")


class TestMultispectralWithAPanel:
    """The panel shots make the survey calibrated; the operator has to be told to take them."""

    PANEL = [-81.7536, 41.3034]

    def test_the_panel_steps_are_on_the_plan_and_in_the_geojson(self, api, tmp_path):
        planned = api.plan_mission({"template": "multispectral", "payload_key": "mavic3m_multispectral",
                                    "calibration_panel_lonlat": f"{self.PANEL[0]}, {self.PANEL[1]}"})
        assert planned["ok"], planned.get("error")
        assert planned["plan"]["capture_contract"]["calibrated"] is True
        steps = [w for w in planned["warnings"] if w.startswith("Calibration step")]
        assert len(steps) == 2
        assert "before take-off" in steps[0] and "after landing" in steps[1]
        assert all(f"{self.PANEL[1]:.6f}, {self.PANEL[0]:.6f}" in s for s in steps)

        exported = api.export_mission(["geojson", "qgc_plan"], str(tmp_path / "ms"))
        assert exported["ok"]
        features = json.loads(Path(exported["written"]["geojson"]).read_text())["features"]
        panels = [f for f in features if f["properties"].get("role") == "calibration_panel"]
        assert [f["properties"]["when"] for f in panels] == ["before", "after"]
        assert all(f["geometry"]["coordinates"][:2] == self.PANEL for f in panels)
        # They are ground steps, not survey waypoints: the flight file does not fly to them.
        items = json.loads(Path(exported["written"]["qgc_plan"]).read_text())["mission"]["items"]
        assert not any(abs(i["params"][5] - self.PANEL[0]) < 1e-6 and abs(i["params"][4] - self.PANEL[1]) < 1e-6
                       for i in items if i["command"] in (16, 19))


class TestEnduranceIsWarnedAtPlanTime:
    def test_a_plan_longer_than_one_battery_says_so(self, api):
        planned = api.plan_mission({"template": "facade_mapping", "altitude_m": 50})
        assert planned["ok"], planned.get("error")
        assert planned["summary"]["duration_min"] > 19.0
        text = " ".join(planned["warnings"])
        assert "batteries" in text and "assumed 25 min" in text

    def test_the_operators_endurance_is_used_and_named(self, api):
        planned = api.plan_mission({"template": "facade_mapping", "altitude_m": 50,
                                    "endurance_min": 400})
        assert not any("batteries" in w for w in planned["warnings"])
        short = api.plan_mission({"template": "grid", "altitude_m": 50, "endurance_min": 2})
        assert any("the endurance you gave" in w for w in short["warnings"])

    def test_a_short_plan_is_not_warned(self, api):
        planned = api.plan_mission({"template": "grid", "altitude_m": 50})
        assert planned["summary"]["duration_min"] < 18.0
        assert not any("batteries" in w for w in planned["warnings"])
