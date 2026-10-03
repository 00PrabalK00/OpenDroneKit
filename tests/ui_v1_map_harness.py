"""Serve the v1 map module (js/odk/map) against a real Api, and drive it in Chromium.

The same arrangement as tests/ui_live_bridge.py -- the production UI code and the
production Api, with only pywebview's transport replaced by HTTP on loopback -- plus
the two tile routes the desktop shell serves from its own origin:

  * ``/raster-tiles/<token>/{z}/{x}/{y}.png``  project rasters cut from the GeoTIFF
    (core.raster_tiles, the deep-zoom route), and
  * ``/tiles/{z}/{x}/{y}.png``                 the offline basemap cache (core.tile_cache).

Every tile request is recorded with its status so a test can say which levels the map
really drew from. The page is a bare map stage (no shell): the furniture, the theme and
the image viewer are the units under test.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ui_live_bridge import BRIDGE_JS, Browser, LoopbackServer, _CONTENT_TYPES, chromium_path, free_port

WEB_ROOT = Path(__file__).resolve().parents[1] / "app" / "web"

STAGE = """<!doctype html>
<html lang="en" data-mode="__MODE__">
<head>
<meta charset="utf-8">
<script src="/__odk_bridge.js"></script>
<link rel="stylesheet" href="css/odk/fonts.css">
<link rel="stylesheet" href="css/odk/tokens.css">
<link rel="stylesheet" href="css/odk/base.css">
<link rel="stylesheet" href="css/odk/components.css">
<link rel="stylesheet" href="css/odk/field.css">
<link rel="stylesheet" href="vendor/maplibre/maplibre-gl.css">
<link rel="stylesheet" href="css/odk/map.css">
<script src="vendor/maplibre/maplibre-gl.js"></script>
<script src="js/basemap-depth.js"></script>
<style>
  #stage { position: fixed; inset: 0; display: flex; }
  #map, #photo { flex: 1 1 auto; position: relative; }
</style>
</head>
<body>
<div id="stage">__BODY__</div>
<script type="module">
__SCRIPT__
</script>
</body>
</html>
"""


def stage_page(script: str, *, mode: str = "desk", body: str = '<div id="map"></div>') -> bytes:
    return (STAGE.replace("__MODE__", mode).replace("__BODY__", body)
            .replace("__SCRIPT__", script)).encode("utf-8")


class MapServer:
    """Loopback server: app/web, the Api bridge, both tile routes and a stage page."""

    def __init__(self, api: Any, page: bytes, *, pick: str | None = None):
        self.api = api
        self.page = page
        self.pick = pick
        self.raster_tiles: list[tuple[str, int]] = []
        self.basemap_tiles: list[tuple[str, int]] = []
        self.calls: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                return

            def _send(self, status: int, body: bytes, kind: str, headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for key, value in (headers or {}).items():
                    if key.lower() != "cache-control":
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                path, _, query = self.path.partition("?")
                if path == "/__stage.html":
                    return self._send(200, server.page, _CONTENT_TYPES[".html"])
                if path == "/__odk_bridge.js":
                    return self._send(200, BRIDGE_JS.encode("utf-8"), _CONTENT_TYPES[".js"])
                from core.raster_tiles import raster_tile_response
                from core.tile_cache import tile_response

                raster = raster_tile_response(path)
                if raster is not None:
                    code, kind, body, headers = raster
                    server.raster_tiles.append((path, code))
                    return self._send(code, body, kind, headers)
                if path.startswith("/tiles/"):
                    answer = tile_response(path, query)
                    code, kind, body = (answer[0], answer[1], answer[2]) if answer else (404, "text/plain", b"")
                    server.basemap_tiles.append((path, code))
                    return self._send(code, body, kind)
                target = (WEB_ROOT / path.lstrip("/")).resolve()
                if WEB_ROOT not in target.parents or not target.is_file():
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, target.read_bytes(),
                                  _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                if not self.path.startswith("/__odk_api/"):
                    return self._send(404, b"not found", "text/plain")
                name = self.path[len("/__odk_api/"):]
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                server.calls.append(name)
                method = getattr(server.api, name, None) if not name.startswith("_") else None
                if name == "pick_file" and server.pick:
                    result: Any = {"ok": True, "path": server.pick}  # the native dialog, answered
                elif not callable(method):
                    result = {"ok": False, "error": f"unwired: {name}"}
                else:
                    try:
                        result = method(*payload.get("args", []))
                    except Exception as exc:  # noqa: BLE001 - reported to the page like guard() does
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                return self._send(200, json.dumps(result, default=str).encode("utf-8"), "application/json")

        self.httpd = LoopbackServer(("127.0.0.1", free_port()), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


class MapBrowser(Browser):
    """The DevTools browser with a viewport, screenshots and real mouse input."""

    def __init__(self, executable: str):
        super().__init__(executable)
        # Software WebGL on a busy machine can take more than a minute to settle a
        # deep-zoom view; the page is still working, not hung.
        self.socket.settimeout(300)

    def viewport(self, width: int, height: int) -> None:
        self._send("Emulation.setDeviceMetricsOverride", width=width, height=height,
                   deviceScaleFactor=1, mobile=False)

    def screenshot(self, path: Path | None = None, clip: dict[str, float] | None = None) -> bytes:
        params: dict[str, Any] = {"format": "png"}
        if clip:
            params["clip"] = {**clip, "scale": 1}
        data = base64.b64decode(self._send("Page.captureScreenshot", **params)["data"])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return data

    def click(self, x: float, y: float) -> None:
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            self._send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)


def decode_png(data: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


__all__ = ["MapBrowser", "MapServer", "chromium_path", "decode_png", "stage_page"]
