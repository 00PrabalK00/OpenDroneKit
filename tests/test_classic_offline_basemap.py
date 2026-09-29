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


# Keeps the WebGL canvas readable so the test can look at what the map actually drew.
_READABLE_MAP = ('<script>(function(){const M=maplibregl.Map;window.__maps=[];'
                 'maplibregl.Map=class extends M{constructor(o){o=Object.assign({},o,'
                 '{canvasContextAttributes:Object.assign({},o.canvasContextAttributes||{},'
                 '{preserveDrawingBuffer:true})});super(o);window.__maps.push(this);}};})();</script>')


class _ReadableCockpit(Cockpit):
    def _page(self) -> str:
        html = super()._page()
        tag = '<script src="vendor/maplibre/maplibre-gl-draw.js"></script>'
        assert tag in html, "maplibre-gl-draw is no longer loaded where this hook expects"
        return html.replace(tag, _READABLE_MAP + tag, 1)


# z1 tiles are painted this colour by _cache.
_Z1 = (40, 130, 60)

ENLARGED = """
    const note = await ODK.waitFor(() => {
      const n = ODK.$("#basemap-note");
      return n && !n.hidden && /Basemap: offline cache/.test(n.textContent) ? n : null;
    }, 60000);
    if (!note) return { note: null };
    // Wait for what the map draws, not for a fixed time: on a loaded machine the
    // cached tiles can take many seconds to arrive and paint. /__tick waits in real
    // time; the page's own clock is virtual and would run out before a slow decode.
    const sample = () => {
      const m = window.__maps[0];
      m.triggerRepaint();
      const canvas = m.getCanvas();
      const c2 = document.createElement("canvas");
      c2.width = canvas.width; c2.height = canvas.height;
      const ctx = c2.getContext("2d"); ctx.drawImage(canvas, 0, 0);
      let hit = 0, n = 0; const seen = {};
      for (let i = 1; i < 12; i++) for (let j = 1; j < 12; j++) {
        const d = ctx.getImageData(Math.floor(c2.width * i / 12), Math.floor(c2.height * j / 12), 1, 1).data;
        n++; const key = `${d[0]},${d[1]},${d[2]}`; seen[key] = (seen[key] || 0) + 1;
        if (Math.abs(d[0] - %(r)d) < 12 && Math.abs(d[1] - %(g)d) < 12 && Math.abs(d[2] - %(b)d) < 12) hit++;
      }
      return { hit, n, seen };
    };
    let last = { hit: 0, n: 0 };
    for (let i = 0; i < 300 && last.hit / Math.max(last.n, 1) < 0.9; i++) {
      await fetch("/__tick?ms=300");
      await ODK.sleep(100);
      last = sample();
    }
    const m = window.__maps[0];
    return {
      note: note.textContent,
      visible: note.getBoundingClientRect().width > 0 && getComputedStyle(note).display !== "none",
      zoom: m.getZoom(),
      maxzoom: m.getStyle().sources.basemap.maxzoom,
      ...last,
    };
""" % {"r": _Z1[0], "g": _Z1[1], "b": _Z1[2]}


def test_cached_tiles_are_the_basemap_and_deeper_zooms_enlarge_the_deepest(
        tmp_path, tile_cache) -> None:
    cached = _cache(tile_cache, [0, 1])
    api = make_api(tmp_path)
    with _ReadableCockpit(api, tmp_path / "profile", page="index.html",
                          window_size="1600,900") as cockpit:
        result = cockpit.run(ENLARGED, virtual_time_ms=150000, timeout_s=480)
    assert result["note"] == (f"Basemap: offline cache, {cached} satellite tile(s) at zoom 0, 1. "
                              "Areas never cached stay empty. Project layers still draw from disk."), result
    assert result["visible"], result
    # The source is capped at the deepest cached level, and the map, deeper than that,
    # shows those z1 tiles enlarged across the view rather than an empty background.
    assert result["maxzoom"] == 1, result
    assert result["zoom"] > 1, result
    assert result["hit"] / result["n"] >= 0.9, (result, cockpit.tiles)
    # Every tile that was served came from the cache on disk, at the cached levels. A
    # deeper request may or may not happen before the cap is set; either way it is 404.
    served = [path for path, code in cockpit.tiles if code == 200]
    assert served and all(path.startswith(("/tiles/0/", "/tiles/1/")) for path in served), cockpit.tiles
    assert all(code == 404 for path, code in cockpit.tiles
               if not path.startswith(("/tiles/0/", "/tiles/1/"))), cockpit.tiles


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


def test_the_desktop_server_answers_a_burst_of_tile_requests(tile_cache, monkeypatch) -> None:
    """The map asks for many tiles at once. With the default listen backlog of 5, a
    burst that arrived while the server was slow to accept (a loaded machine) was
    refused, and MapLibre never re-asks for a tile that failed, so the offline map kept
    holes: 1 run in 3 showed only the z0 tile. Every request in a burst is answered."""
    import socket
    import socketserver
    import threading
    from urllib.parse import urlsplit

    from app.shell import Shell

    # Hold the accept loop, as a busy machine does, until every connection is queued.
    gate = threading.Event()
    serve = socketserver.BaseServer.serve_forever
    monkeypatch.setattr(socketserver.BaseServer, "serve_forever",
                        lambda self, *a, **k: (gate.wait(30), serve(self, *a, **k)))
    _cache(tile_cache, [0, 1, 2, 3])
    url = Shell._serve(REPO_ROOT / "app" / "web", "index.html", {})
    port = urlsplit(url).port
    sockets = [socket.create_connection(("127.0.0.1", port), timeout=30) for _ in range(60)]
    gate.set()
    answers: list[bytes] = [b""] * len(sockets)

    def ask(i: int, sock: socket.socket) -> None:
        z, rest = 3, i % 64
        sock.sendall(f"GET /tiles/{z}/{rest // 8}/{rest % 8}.png?provider=satellite HTTP/1.0\r\n\r\n".encode())
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        answers[i] = b"".join(chunks)
        sock.close()

    threads = [threading.Thread(target=ask, args=(i, s)) for i, s in enumerate(sockets)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert all(answer.startswith(b"HTTP/1.0 200") for answer in answers), [a[:40] for a in answers]
