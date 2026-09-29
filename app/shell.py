"""Native application shell.

A pywebview window hosting the web UI, with a real operating-system menu bar. On
Windows the renderer is the Edge WebView2 runtime that ships with the OS, so there is
no bundled browser and startup is immediate.

Menu actions do not implement behaviour themselves; each one dispatches into the web
layer, which then calls the same `Api` methods the in-page controls use. That keeps a
single code path per action regardless of how the user triggered it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import webview
from webview.menu import Menu, MenuAction, MenuSeparator

from .api import Api
from .session import AppSession

WEB_ROOT = Path(__file__).resolve().parent / "web"
# Window/taskbar icon. Without it Windows shows the python.exe icon.
WINDOW_ICON = WEB_ROOT / "assets" / "brand" / "derived" / "favicon.ico"
WINDOW_TITLE = "OpenDroneKit — Inspection & Geospatial Toolkit"


class Shell:
    """Owns the window, the menu, and the bridge object."""

    def __init__(self, session: AppSession | None = None):
        self.api = Api(session)
        self.window: Any = None
        # Served to the pages as odk-runtime.json (see _serve).
        self.runtime: dict[str, Any] = {"hub_status": "not started", "hub_api": None, "hub_error": ""}
        # Reselect the dataset this project last imported. The field lives in memory, so
        # without this the application opened with nothing selected even when the project
        # obviously had a dataset, and every dataset-dependent action refused until the
        # user re-imported a folder it already knew about.
        try:
            self.api._session.restore_active_dataset()  # noqa: SLF001 - startup wiring
            self.api._session.restore_layers()  # noqa: SLF001 - startup wiring
        except Exception:  # noqa: BLE001 - a first run has no project yet
            pass

    # -- menu dispatch ---------------------------------------------------

    def _dispatch(self, action: str, payload: dict[str, Any] | None = None) -> Callable[[], None]:
        """Build a menu callback that forwards an action to the web layer."""

        def handler() -> None:
            if self.window is None:
                return
            message = json.dumps({"action": action, "payload": payload or {}})
            self.window.evaluate_js(f"window.odk && window.odk.onMenu({message})")

        return handler

    def build_menu(self) -> list[Menu]:
        """The top menu bar: the primary navigation surface for the application."""
        act = self._dispatch

        file_menu = Menu(
            "File",
            [
                MenuAction("New Project", act("project.new")),
                MenuAction("Open Project...", act("project.open")),
                MenuSeparator(),
                MenuAction("Import Imagery Folder...", act("data.import_imagery")),
                MenuAction("Import Terrain (GeoTIFF/ASC/CSV)...", act("data.import_terrain")),
                MenuAction("Import Vector (GeoJSON)...", act("data.import_vector")),
                MenuAction("Import Raster (GeoTIFF)...", act("data.import_raster")),
                MenuSeparator(),
                MenuAction("Open Project Folder", act("project.reveal")),
                MenuSeparator(),
                MenuAction("Exit", self._quit),
            ],
        )

        mission_menu = Menu(
            "Mission",
            [
                MenuAction("Plan Mission", act("mission.plan")),
                MenuAction("Mission Settings...", act("mission.settings")),
                MenuSeparator(),
                MenuAction("Clear Area of Interest", act("mission.clear_aoi")),
                MenuAction("Clear No-Fly Zones", act("mission.clear_nofly")),
                MenuSeparator(),
                MenuAction("Save Mission Version", act("mission.save")),
                MenuAction("Mission History...", act("mission.history")),
                MenuSeparator(),
                MenuAction("Export All Formats", act("mission.export_all")),
                MenuAction("Export QGroundControl .plan", act("mission.export", {"format": "qgc_plan"})),
                MenuAction("Export QGC WPL .waypoints", act("mission.export", {"format": "qgc_wpl"})),
                MenuAction("Export DJI WPML .kmz", act("mission.export", {"format": "dji_wpml"})),
                MenuAction("Export Litchi CSV", act("mission.export", {"format": "litchi"})),
                MenuAction("Export KML", act("mission.export", {"format": "kml"})),
                MenuAction("Export GeoJSON", act("mission.export", {"format": "geojson"})),
            ],
        )

        fly_menu = Menu(
            "Fly",
            [
                MenuAction("Connect Vehicle...", act("fly.connect")),
                MenuAction("Disconnect", act("fly.disconnect")),
                MenuSeparator(),
                MenuAction("Upload Mission to Vehicle", act("fly.upload")),
                MenuSeparator(),
                MenuAction("Arm", act("fly.command", {"command": "arm"})),
                MenuAction("Start Mission", act("fly.command", {"command": "start"})),
                MenuAction("Pause", act("fly.command", {"command": "pause"})),
                MenuAction("Resume", act("fly.command", {"command": "resume"})),
                MenuSeparator(),
                MenuAction("Return to Home", act("fly.command", {"command": "rth"})),
                MenuAction("Abort Mission", act("fly.command", {"command": "abort"})),
            ],
        )

        analysis_menu = Menu(
            "Analysis",
            [
                MenuAction("Run Full Pipeline", act("analysis.pipeline")),
                MenuSeparator(),
                MenuAction("Model Manager", act("analysis.models")),
            ],
        )

        reconstruct_menu = Menu(
            "Reconstruct",
            [
                MenuAction("Run Reconstruction (COLMAP)", act("recon.run", {"engine": "colmap"})),
                MenuAction("Run Reconstruction (fallback engine)", act("recon.run", {"engine": "custom"})),
                MenuSeparator(),
                MenuAction("Reconstruction Settings...", act("recon.settings")),
                MenuAction("Open Output Folder", act("recon.reveal")),
            ],
        )

        reports_menu = Menu(
            "Reports",
            [
                MenuAction("Generate Report...", act("report.generate")),
                MenuAction("Export Findings as PDF", act("report.export", {"format": "pdf"})),
                MenuAction("Export Findings as Word", act("report.export", {"format": "docx"})),
                MenuAction("Generated Reports...", act("report.list")),
                MenuSeparator(),
                MenuAction("Record Finding...", act("report.finding")),
                MenuAction("Findings and Review...", act("report.review")),
                MenuSeparator(),
                MenuAction("Verify Provenance", act("report.provenance")),
                MenuAction("Export Audit Log (CSV)", act("report.audit_export")),
                MenuAction("Notifications...", act("report.notifications")),
            ],
        )

        view_menu = Menu(
            "View",
            [
                MenuAction("Layers Panel", act("view.panel", {"panel": "layers"})),
                MenuAction("Mission Panel", act("view.panel", {"panel": "mission"})),
                MenuAction("Telemetry Panel", act("view.panel", {"panel": "telemetry"})),
                MenuAction("Jobs Panel", act("view.panel", {"panel": "jobs"})),
                MenuSeparator(),
                MenuAction("Basemap: Satellite", act("view.basemap", {"basemap": "satellite"})),
                MenuAction("Basemap: Street", act("view.basemap", {"basemap": "street"})),
                MenuAction("Basemap: Topographic", act("view.basemap", {"basemap": "topo"})),
                MenuAction("Basemap: Offline Cache", act("view.basemap", {"basemap": "offline"})),
                MenuAction("Cache Visible Map for Offline Use", act("view.cache_tiles")),
                MenuSeparator(),
                MenuAction("Zoom to Area of Interest", act("view.zoom_aoi")),
                MenuAction("Zoom to Mission", act("view.zoom_mission")),
            ],
        )

        tools_menu = Menu(
            "Tools",
            [
                MenuAction("Measure Distance", act("tools.measure", {"mode": "distance"})),
                MenuAction("Measure Area", act("tools.measure", {"mode": "area"})),
                MenuSeparator(),
                MenuAction("Environment Capabilities", act("tools.capabilities")),
                MenuAction("Audit Log", act("tools.audit")),
            ],
        )

        help_menu = Menu(
            "Help",
            [
                MenuAction("About OpenDroneKit", act("help.about")),
                MenuAction("Keyboard Shortcuts", act("help.shortcuts")),
            ],
        )

        return [file_menu, mission_menu, fly_menu, analysis_menu, reconstruct_menu, reports_menu,
                view_menu, tools_menu, help_menu]

    def _quit(self) -> None:
        if self.window is not None:
            self.window.destroy()

    # -- lifecycle -------------------------------------------------------

    def run(self, *, debug: bool = False, width: int = 1600, height: int = 980,
            ui: str | None = None) -> None:
        page = UI_PAGES[resolve_ui(ui)]
        index = WEB_ROOT / page
        if not index.exists():
            raise FileNotFoundError(f"UI assets are missing: {index}")

        # Served over loopback rather than opened as a file. The cockpit is built from
        # ES modules, and a module fetched from file:// has origin "null", which the
        # webview refuses -- the window opens, the menu works, and the page is silently
        # blank with nothing in any log. index.html survives on file:// only because it
        # uses a classic script tag, which is why one UI worked and the other did not.
        #
        # pywebview's own http_server=True does not apply when the window is given an
        # absolute path, so the server is started here where the behaviour is explicit.
        # It binds 127.0.0.1 on an ephemeral port: nothing leaves the machine, which
        # this application guarantees elsewhere and must not quietly break here.
        url = self._serve(WEB_ROOT, page, runtime=self.runtime)

        # The Hub page (linked from the toolbar) talks to the REST service. Nothing in
        # the launch path used to start it, so the link opened a page that could only
        # say "API unavailable". It runs in-process on loopback, in the background so
        # the window does not wait for it.
        if os.environ.get("ODK_HUB", "1") != "0":
            start_hub_service(self.runtime, ui_origin=url.rsplit("/", 1)[0])
        else:
            self.runtime.update(hub_status="disabled", hub_error="Disabled with ODK_HUB=0.")

        self.window = webview.create_window(
            WINDOW_TITLE,
            url,
            js_api=self.api,
            width=width,
            height=height,
            min_size=(1100, 700),
            background_color="#12161c",
            text_select=True,
        )
        self.api.bind_window(self.window)
        webview.start(menu=self.build_menu(), debug=debug,
                      icon=str(WINDOW_ICON) if WINDOW_ICON.is_file() else None)

    @staticmethod
    def _serve(root: Path, page: str, runtime: dict[str, Any] | None = None) -> str:
        """Serve the UI directory on loopback and return the URL for `page`.

        A daemon thread on an ephemeral port. Bound to 127.0.0.1 explicitly rather than
        0.0.0.0: an offline-first inspection tool must not open a port to the network
        because its UI happens to need an origin.

        Besides the static files it answers two things the pages need at runtime:

        - `odk-runtime.json`: where the in-process Hub service is listening, if it is,
          and why not when it is not.
        - `tiles/{z}/{x}/{y}.png?provider=`: the offline basemap cache, read straight
          from disk. The "Offline cache" basemap used to load from the Hub on port
          8000, which the launcher never started, so with no network the map was blank.
        """
        import functools
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        state = runtime if runtime is not None else {}

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                """Silent: a request line per asset would bury the application's own log."""

            def do_GET(self) -> None:  # noqa: N802 - http.server naming
                parts = urlsplit(self.path)
                if parts.path == "/odk-runtime.json":
                    body = json.dumps(state).encode("utf-8")
                    self._send(200, "application/json", body, {"Cache-Control": "no-store"})
                    return
                match = _TILE_ROUTE.match(parts.path)
                if match:
                    from core.tile_cache import TRANSPARENT_PNG, read_tile, tile_media_type

                    provider = (parse_qs(parts.query).get("provider") or ["satellite"])[0]
                    z, x, y = (int(value) for value in match.groups())
                    tile = read_tile(provider, z, x, y)
                    if tile is None:
                        # A gap in coverage looks like a gap, not like terrain.
                        self._send(404, "image/png", TRANSPARENT_PNG, {"X-ODK-Tile": "not-cached"})
                    else:
                        self._send(200, tile_media_type(tile), tile, {"X-ODK-Tile": "cached"})
                    return
                super().do_GET()

            def _send(self, code: int, kind: str, body: bytes, headers: dict[str, str]) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

        handler = functools.partial(Handler, directory=str(root))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/{page}"


_TILE_ROUTE = re.compile(r"^/tiles/(\d+)/(\d+)/(\d+)\.png$")

HUB_DEFAULT_PORT = 8000


def _free_port(preferred: int) -> int:
    """`preferred` when nothing is listening there, otherwise any free loopback port."""
    import socket

    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise OSError("No free loopback port.")


def start_hub_service(runtime: dict[str, Any], *, ui_origin: str,
                      port: int | None = None) -> threading.Thread:
    """Run the Hub REST API in this process, on loopback, in a daemon thread.

    `runtime` is the dict served as odk-runtime.json: it records where the service is
    listening, or exactly why it is not (a missing package is named, not hidden). The
    UI origin is added to the allowed CORS origins because the pages are served from
    the shell's own loopback port. Port 8000 -- the Hub page's documented default -- is
    used when free, any free port otherwise.
    """
    runtime.update(hub_status="starting", hub_api=None, hub_error="")

    def serve() -> None:
        try:
            import uvicorn

            from starlette.middleware.cors import CORSMiddleware

            from services.api.main import app as hub_app

            # The page's origin is allowed by wrapping the app, not by editing
            # ODK_CORS_ORIGINS: services.api.main reads that once, at import, so if
            # anything had imported it first the page's requests were refused.
            served = CORSMiddleware(hub_app, allow_origins=[ui_origin], allow_credentials=True,
                                    allow_methods=["*"], allow_headers=["*"])

            chosen = port if port is not None else _free_port(HUB_DEFAULT_PORT)
            server = uvicorn.Server(uvicorn.Config(
                served, host="127.0.0.1", port=chosen, log_level="warning",
                lifespan="on", access_log=False,
            ))
            # Not the main thread: uvicorn must not install signal handlers here.
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

            def mark_running() -> None:
                import time

                while runtime.get("hub_status") == "starting":
                    if server.started:
                        # Address first: a reader that sees "running" must see where.
                        runtime["hub_api"] = f"http://127.0.0.1:{chosen}"
                        runtime["hub_status"] = "running"
                        return
                    time.sleep(0.05)

            threading.Thread(target=mark_running, daemon=True).start()
            server.run()
            if runtime.get("hub_status") != "running":
                runtime.update(hub_status="unavailable",
                               hub_error=f"The Hub service could not listen on 127.0.0.1:{chosen}.")
        except BaseException as exc:  # noqa: BLE001 - reported to the page, never raised
            runtime.update(
                hub_status="unavailable",
                hub_error=(
                    f"{type(exc).__name__}: {exc}. Install requirements.txt into this "
                    "environment (python main.py --check-deps lists what is missing)."
                ),
            )

    thread = threading.Thread(target=serve, name="odk-hub-service", daemon=True)
    thread.start()
    return thread


UI_PAGES = {"classic": "index.html", "cockpit": "workspace.html"}
DEFAULT_UI = "classic"


def resolve_ui(choice: str | None = None) -> str:
    """Which interface to open: --ui, else ODK_UI, else the classic shell.

    The classic single-screen UI is the default: every control and all of the native
    menu bar is wired and exercised there. The cockpit (fourteen workspaces) is bridged
    to the same Api and opens with `--ui cockpit` or ODK_UI=cockpit. ODK_UI=classic was
    documented and did nothing special, and an unknown value fell back silently; both
    are explicit now.
    """
    raw = (choice or os.environ.get("ODK_UI") or DEFAULT_UI).strip().lower()
    if raw not in UI_PAGES:
        raise ValueError(f"Unknown UI {raw!r}: choose one of {', '.join(UI_PAGES)}.")
    return raw


class UsageError(Exception):
    """A command line or ODK_UI value the shell cannot honour.

    argparse reports these by writing to stderr and exiting 2. Under `pythonw` -- the
    way the launcher starts the app -- stderr is None, so a mistyped ODK_UI closed the
    app with no window, no message and no log. Raising lets main.py route it through
    report_startup_failure like every other startup failure.
    """

    def __init__(self, message: str, usage: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.usage = usage

    def __str__(self) -> str:
        return f"{self.usage}error: {self.message}" if self.usage else self.message


def _parser():
    import argparse

    class _Parser(argparse.ArgumentParser):
        def error(self, message: str):  # noqa: D401 - argparse hook
            raise UsageError(message, self.format_usage())

    return _Parser(description="Launch the OpenDroneKit desktop application.")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    parser.add_argument("--debug", action="store_true", help="Open developer tools.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--ui", choices=sorted(UI_PAGES), default=None,
                        help="Interface to open (default: ODK_UI, else classic).")
    args = parser.parse_args(argv)

    try:
        ui = resolve_ui(args.ui)
    except ValueError as exc:
        parser.error(str(exc))
    Shell().run(debug=args.debug, width=args.width, height=args.height, ui=ui)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UsageError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2)
