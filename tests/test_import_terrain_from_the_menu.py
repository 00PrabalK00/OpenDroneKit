"""File > Import Terrain works end to end, in both windows.

It was the one native menu item of sixty-two that could end in silence:

* the default window passed pywebview a display string, "Terrain (*.tif;*.tiff;*.asc;
  *.csv)", which _file_type_filters lower-cased into "TERRAIN (*.TIF;...) files (*.terrain
  (...))". pywebview's own check refused it, pick_file raised before the chooser opened,
  and the window said nothing. Import Vector, Import Raster and Add Layer pass display
  strings too and failed the same way;
* the cockpit said nothing when the chooser was cancelled, and accepted a DEM without
  saying whether it covered the site, which set_terrain_source already works out;
* Mission Planning's map read "Terrain: flat plane -- no DEM loaded" after one was.

pick_file here is the production method. Only pywebview's window is replaced, by one
whose create_file_dialog applies pywebview's own filter check (webview.util.
parse_file_type) and answers with the file a person would pick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.api import Api, _file_type_filters
from cockpit_harness import Cockpit, make_api

rasterio = pytest.importorskip("rasterio")
webview_util = pytest.importorskip("webview.util")

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


class ChooserWindow:
    """pywebview's window, as far as pick_file uses it: the filter check, then a pick."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.asked: list[tuple[str, ...]] = []

    def create_file_dialog(self, dialog_type, directory="", allow_multiple=False,
                           save_filename="", file_types=()):
        for entry in file_types:          # what webview.Window.create_file_dialog does
            webview_util.parse_file_type(entry)
        self.asked.append(tuple(file_types))
        return (self.answer,) if self.answer else None


def _dem(path: Path, west: float, south: float, east: float, north: float) -> Path:
    """A small metre-height DEM in EPSG:4326 over the given box (a test input)."""
    from rasterio.transform import from_bounds

    rows, cols = 40, 40
    heights = np.linspace(250.0, 262.0, rows * cols, dtype="float32").reshape(rows, cols)
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(west, south, east, north, cols, rows)) as dst:
        dst.write(heights, 1)
    return path


def _real_chooser(api: Api, answer: str) -> ChooserWindow:
    window = ChooserWindow(answer)
    api._window = window  # noqa: SLF001 - the desktop shell sets this at start-up
    api.pick_file = Api.pick_file.__get__(api)  # undo the harness's stub: the real method
    return window


@pytest.mark.parametrize("asked", [
    ["Terrain (*.tif;*.tiff;*.asc;*.csv)"],
    ["GeoJSON (*.geojson;*.json)"],
    ["GeoTIFF (*.tif;*.tiff)"],
    ["Geodata (*.tif;*.tiff;*.geojson;*.json)"],
    ["tif", "tiff", "asc", "csv"],
])
def test_every_filter_the_windows_ask_for_passes_pywebviews_check(asked):
    filters = _file_type_filters(asked)
    for entry in filters:
        webview_util.parse_file_type(entry)
    assert filters[-1] == "All files (*.*)"
    if len(asked) == 1 and "(" in asked[0]:
        assert filters[0] == asked[0], filters


def test_cockpit_import_terrain_covers_the_site(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    dem = _dem(tmp_path / "site_dtm.tif", -81.7530, 41.3020, -81.7470, 41.3070)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        window = _real_chooser(api, str(dem))
        seen = cockpit.run("""
            const out = {};
            await ODK.nav("Home");
            window.odk.onMenu({ action: "data.import_terrain" });
            out.toast = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Terrain source")), 15000);
            out.workspace = ODK.$(".nav-item.active").textContent;
            out.chip = await ODK.waitFor(() => {
              const bl = ODK.$(".canvas-overlay.bl");
              return bl && /site_dtm\\.tif/.test(bl.textContent) ? bl.textContent : null;
            }, 15000);
            await ODK.press("Plan");
            out.planned = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Planned")), 30000);
            return out;
        """)
    assert window.asked, "the chooser never opened"
    assert api.get_state()["terrain_source"] == str(dem)
    assert "contains the area" in seen["toast"], seen
    assert seen["workspace"] == "Mission Planning", seen
    assert "flat plane" not in (seen["chip"] or ""), seen
    # The plan follows it: no flat-earth warning, terrain following on.
    assert "No terrain model loaded" not in seen["planned"], seen
    assert api._session.mission_plan_dict["terrain_follow_enabled"] is True  # noqa: SLF001


def test_cockpit_import_terrain_says_when_it_misses_or_is_cancelled(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    elsewhere = _dem(tmp_path / "other_valley.tif", 10.0, 45.0, 10.01, 45.01)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        window = _real_chooser(api, str(elsewhere))
        seen = cockpit.run("""
            const out = {};
            window.odk.onMenu({ action: "data.import_terrain" });
            out.miss = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Terrain source")), 15000);
            return out;
        """)
        assert ", but other_valley.tif covers ground nowhere near this site" in seen["miss"], seen

        window.answer = ""          # the operator presses Cancel
        cancelled = cockpit.run("""
            window.odk.onMenu({ action: "data.import_terrain" });
            return { toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Import Terrain")), 15000) };
        """)
    assert cancelled["toast"] == "Import Terrain: no file chosen; the terrain source is unchanged.", cancelled
    assert api.get_state()["terrain_source"] == str(elsewhere)


def test_classic_import_terrain_opens_the_chooser_and_sets_the_source(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    dem = _dem(tmp_path / "site_dtm.tif", -81.7530, 41.3020, -81.7470, 41.3070)
    with Cockpit(api, tmp_path / "profile", page="index.html") as classic:
        window = _real_chooser(api, str(dem))
        seen = classic.run("""
            await ODK.sleep(1000);
            ODK.$("#status-message").textContent = "";
            window.odk.onMenu({ action: "data.import_terrain" });
            const status = await ODK.waitFor(() => {
              const t = ODK.$("#status-message").textContent;
              return /Terrain source set|pick_file/.test(t) ? t : null;
            }, 15000);
            return { status, field: ODK.$("#mission-terrain").value };
        """)
    assert window.asked == [("Terrain (*.tif;*.tiff;*.asc;*.csv)", "All files (*.*)")], window.asked
    assert "Terrain source set" in seen["status"] and "contains the area" in seen["status"], seen
    assert seen["field"] == str(dem), seen
    assert api.get_state()["terrain_source"] == str(dem)
