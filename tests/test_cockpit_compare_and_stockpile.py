"""Every Compare / Compare Dates / Change / Stockpile button in the real cockpit.

Audit: Compare and Compare Dates called compare_surveys() with no arguments and
Stockpile called build_asset_inventory() with no arguments, so each answered "missing
required positional argument". The handlers were rewired, but four workspaces put a
comparison button on their toolbar (Measurements, Digital Twin twice, Thermal) and only
one of them was ever pressed in a test.

Here each is pressed in workspace.html in headless Chromium against the real Api, on a
project whose DSM holds a 10 x 10 x 3 m block that an earlier survey did not have, so
the change and the pile both have a known answer (300 m^3).
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api
from conftest import BLOCK_CELLS, BLOCK_VOLUME_M3, GRID, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M

pytest.importorskip("rasterio")

import numpy as np  # noqa: E402

from core import geo  # noqa: E402

EPSG = 32617

COMPARE = """
    await ODK.nav("%(workspace)s");
    await ODK.press("%(button)s");
    // Movement needs the error bounds of both surveys and the registration.
    await ODK.answer("0.05, 0.05, 0.05");
    const toast = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("Added")
      || t.includes("rror") || t.includes("missing")), 30000);
    return { toast, all: ODK.toasts() };
"""


@pytest.fixture
def project(tmp_path, surface_pair):
    api = make_api(tmp_path)
    api._session.register_raster_layer(surface_pair["dsm"], name="DSM")
    api._session.register_raster_layer(surface_pair["dtm"], name="DTM")
    # The earlier survey: flat ground on its own grid, before the block was placed.
    earlier = tmp_path / "earlier_dsm.tif"
    geo.write_geotiff(earlier, np.full((60, 60), 100.0, dtype=np.float32), epsg=EPSG,
                      west=ORIGIN_X - 2.0, north=ORIGIN_Y + 2.0, pixel_size=0.7, cog=False)
    return api, str(earlier)


@pytest.mark.parametrize("workspace,button", [
    ("Measurements", "Compare Dates"),
    ("Digital Twin", "Compare Dates"),
    ("Digital Twin", "Change"),
    ("Thermal", "Compare"),
])
def test_each_comparison_button_compares_the_two_surveys(tmp_path, project, workspace, button) -> None:
    api, earlier = project
    with Cockpit(api, tmp_path / "profile") as cockpit:
        cockpit.picks["file"] = earlier
        result = cockpit.run(COMPARE % {"workspace": workspace, "button": button})

    volumes = cockpit.called("compare_survey_volumes")
    moved = cockpit.called("compare_surveys")
    assert volumes and all(c["ok"] for c in volumes), volumes
    assert moved and all(c["ok"] for c in moved), moved
    # The earlier survey that was picked, against this project's DSM, with the bounds typed.
    assert volumes[0]["args"][0] == earlier and volumes[0]["args"][1].endswith("dsm.tif")
    assert moved[0]["args"][2:] == [0.05, 0.05, 0.05], moved[0]["args"]

    # What the toast says is what the Api computed for these two surfaces.
    expected = api.compare_survey_volumes(earlier, volumes[0]["args"][1], None)
    import re

    shown = re.match(r"Added ([\d.]+) m³, removed ([\d.]+) m³", result["toast"])
    assert shown, result
    assert float(shown.group(1)) == pytest.approx(expected["added_volume_m3"]), (result, expected)
    assert float(shown.group(2)) == pytest.approx(expected["removed_volume_m3"]), (result, expected)
    assert abs(expected["added_volume_m3"] - BLOCK_VOLUME_M3) / BLOCK_VOLUME_M3 < 0.15, expected
    assert "region(s) moved" in result["toast"] or "No movement" in result["toast"], result


def test_stockpile_measures_the_drawn_pile_and_writes_its_report(tmp_path, project) -> None:
    api, _ = project
    start = (GRID - BLOCK_CELLS) // 2
    margin = 3
    west = ORIGIN_X + (start - margin) * PIXEL_SIZE_M
    east = ORIGIN_X + (start + BLOCK_CELLS + margin) * PIXEL_SIZE_M
    north = ORIGIN_Y - (start - margin) * PIXEL_SIZE_M
    south = ORIGIN_Y - (start + BLOCK_CELLS + margin) * PIXEL_SIZE_M
    lon, lat = geo.projected_to_wgs84([west, east, east, west], [south, south, north, north], EPSG)
    ring = [[float(a), float(b)] for a, b in zip(lon, lat)]

    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Measurements");
            const map = await ODK.map();
            map.jumpTo({{ center: {ring[0]}, zoom: 19 }});
            await ODK.sleep(500);
            await ODK.press("Stockpile");            // no area yet: it asks for one
            const asked = ODK.lastToast();
            for (const [lon, lat] of {ring}) await ODK.click(map, lon, lat);
            await ODK.key("Enter");
            await ODK.answer("");                     // base: the project's DTM
            const toast = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("Report:")
              || t.includes("rror")), 30000);
            return {{ asked, toast }};
        """)

    calls = cockpit.called("measure_drawn")
    assert calls and calls[-1]["ok"], calls
    assert calls[-1]["args"][0] == "stockpile" and len(calls[-1]["args"][1]) >= 4, calls[-1]
    assert not cockpit.called("build_asset_inventory")
    assert "Draw around the pile's toe" in result["asked"], result
    assert result["toast"] and "Report:" in result["toast"], result
    volume = float(result["toast"].split(" m³")[0].split()[-1].replace(",", ""))
    assert abs(volume - BLOCK_VOLUME_M3) / BLOCK_VOLUME_M3 < 0.05, result


def _strict_json(reply) -> None:
    """What JSON.parse in the UI does with a reply: bare NaN/Infinity are rejected."""
    import json

    def refuse(token):
        raise ValueError(f"not JSON: {token}")

    json.loads(json.dumps(reply, default=str), parse_constant=refuse)


def test_a_compare_leaves_the_project_readable_after_a_restart(tmp_path, project) -> None:
    """Verifier: the difference raster's layer carried nodata NaN, so get_state and
    list_layers sent bare NaN, JSON.parse failed, the Layers panel read "No layers yet",
    and after reloading every project action answered "Open a project first"."""
    api, earlier = project
    for press in (1, 2):
        # A fresh Cockpit is a fresh page: what an app restart reloads from get_state.
        with Cockpit(api, tmp_path / f"profile{press}") as cockpit:
            cockpit.picks["file"] = earlier
            result = cockpit.run(
                COMPARE.replace("return { toast, all: ODK.toasts() };",
                                'await ODK.sleep(1500); '
                                'return { toast, all: ODK.toasts(), panel: ODK.panel("Layers") };')
                % {"workspace": "Measurements", "button": "Compare Dates"})
        assert result["toast"] and result["toast"].startswith("Added"), (press, result)
        assert not any("Open a project first" in t for t in result["all"]), (press, result)
        assert "Elevation change" in result["panel"] and "DSM" in result["panel"], (press, result)
        assert [c["ok"] for c in cockpit.called("list_layers")] and all(
            c["ok"] for c in cockpit.called("list_layers"))

    change = [layer for layer in api.list_layers()["layers"] if layer["name"] == "Elevation change"]
    assert len(change) == 2
    assert "nodata" in change[0]["metadata"] and change[0]["metadata"]["nodata"] is None
    _strict_json(api.get_state())
    _strict_json(api.list_layers())
