"""The classic window's map with no network: what it shows, and that it says so.

Audit (classic UI, the default launch): with no signal every basemap is an internet
tile server, so the map went dark. The fallback to the offline cache existed, but its
explanation went to the one-line status bar and "OpenDroneKit ready." replaced it a
moment later, so an operator saw a black map and no reason. A cache holding z12-z17 was
also asked for z18-z20 tiles it never had, so zooming in on the site went dark again.

These drive index.html in headless Chromium against the real Api, every host but
loopback unresolvable, with the offline tiles answered by core.tile_cache.tile_response
-- the code the desktop shell serves them with.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cockpit_harness import Cockpit, make_api

REPO_ROOT = Path(__file__).resolve().parents[1]
ORTHO = Path(os.environ.get(
    "ODK_DEMO_ORTHO", REPO_ROOT / "demo_project" / "aukerman_full" / "reconstruction" / "orthomosaic.tif"))

# Waits for the map, then for the offline note, and reads what is on screen.
SCENARIO = """
    const note = await ODK.waitFor(() => {
      const n = ODK.$("#basemap-note");
      return n && !n.hidden && /offline tile cache|Basemap: offline cache/.test(n.textContent) ? n : null;
    }, 40000);
    if (!note) return { note: null, select: ODK.$("#basemap-select").value };
    %s
    await ODK.sleep(2500);
    const box = note.getBoundingClientRect();
    return {
      note: note.textContent,
      visible: box.width > 0 && box.height > 0 && getComputedStyle(note).display !== "none",
      status: ODK.$("#status-message").textContent,
      select: ODK.$("#basemap-select").selectedOptions[0].textContent,
    };
"""


@pytest.fixture
def tile_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    from core.tile_cache import tile_root

    return tile_root()


def _cache(root: Path, zoom_levels) -> int:
    """Tiles as "Cache view" leaves them on disk: <provider>/<z>/<x>/<y>.png."""
    from PIL import Image

    count = 0
    for z in zoom_levels:
        for x in range(2 ** z):
            for y in range(2 ** z):
                folder = root / "satellite" / str(z) / str(x)
                folder.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (256, 256), (40, 90 + 40 * z, 60)).save(folder / f"{y}.png")
                count += 1
    return count


def test_a_fresh_install_offline_says_there_is_no_basemap_and_keeps_saying_it(
        tmp_path, tile_cache) -> None:
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile", page="index.html", window_size="1600,900") as cockpit:
        result = cockpit.run(SCENARIO % "")
    assert result["note"], result
    assert result["note"].startswith("No basemap: the offline tile cache is empty"), result
    assert result["visible"], result
    # Still there after the start-up messages that used to replace it.
    assert result["select"] == "Offline cache", result
    # The offline source asked the local cache, which answered "not cached".
    assert cockpit.tiles and {code for _, code in cockpit.tiles} == {404}, cockpit.tiles


def test_cached_tiles_are_the_basemap_and_deeper_zooms_enlarge_the_deepest(
        tmp_path, tile_cache) -> None:
    cached = _cache(tile_cache, [0, 1])
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile", page="index.html", window_size="1600,900") as cockpit:
        result = cockpit.run(SCENARIO % "")
    assert result["note"] == (f"Basemap: offline cache, {cached} satellite tile(s) at zoom 0, 1. "
                              "Areas never cached stay empty. Project layers still draw from disk."), result
    assert result["visible"], result
    # At the start view the map wants z3, which was never cached (404). Once the source
    # is capped at the deepest cached level it asks for z1 and enlarges those tiles.
    first_hit = next(i for i, (_, code) in enumerate(cockpit.tiles) if code == 200)
    after = cockpit.tiles[first_hit:]
    assert all(code == 200 and path.startswith(("/tiles/0/", "/tiles/1/"))
               for path, code in after), cockpit.tiles
    assert any(path.startswith("/tiles/3/") and code == 404
               for path, code in cockpit.tiles[:first_hit]), cockpit.tiles


@pytest.mark.skipif(not ORTHO.is_file(), reason="The demo orthomosaic is not present (gitignored).")
def test_the_projects_orthomosaic_is_shown_from_disk_with_no_network(tmp_path, tile_cache) -> None:
    import shutil

    api = make_api(tmp_path)
    ortho = tmp_path / "orthomosaic.tif"
    shutil.copy(ORTHO, ortho)
    added = api.add_layer_from_file(str(ortho))
    assert added["ok"], added
    with Cockpit(api, tmp_path / "profile", page="index.html", window_size="1600,900") as cockpit:
        result = cockpit.run(SCENARIO % """
            await ODK.waitFor(() => /orthomosaic is drawn from disk/.test(note.textContent), 30000);
        """)
    assert result["note"].startswith("No basemap: the offline tile cache is empty"), result
    assert result["note"].endswith("The project's orthomosaic is drawn from disk."), result
    previews = cockpit.called("raster_preview")
    assert previews and all(call["ok"] for call in previews), previews
