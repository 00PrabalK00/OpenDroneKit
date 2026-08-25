"""Native application shell.

A pywebview window hosting the web UI, with a real operating-system menu bar. On
Windows the renderer is the Edge WebView2 runtime that ships with the OS, so there is
no bundled browser and startup is immediate.

Menu actions do not implement behaviour themselves; each one dispatches into the web
layer, which then calls the same `Api` methods the in-page controls use. That keeps a
single code path per action regardless of how the user triggered it.
"""

from __future__ import annotations

import os

import json
from pathlib import Path
from typing import Any, Callable

import webview
from webview.menu import Menu, MenuAction, MenuSeparator

from .api import Api
from .session import AppSession

WEB_ROOT = Path(__file__).resolve().parent / "web"
WINDOW_TITLE = "OpenDroneKit — Inspection & Geospatial Toolkit"


class Shell:
    """Owns the window, the menu, and the bridge object."""

    def __init__(self, session: AppSession | None = None):
        self.api = Api(session)
        self.window: Any = None

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
                MenuAction("Basemap: Offline Grid", act("view.basemap", {"basemap": "offline"})),
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

        return [file_menu, mission_menu, fly_menu, analysis_menu, reconstruct_menu, view_menu, tools_menu, help_menu]

    def _quit(self) -> None:
        if self.window is not None:
            self.window.destroy()

    # -- lifecycle -------------------------------------------------------

    def run(self, *, debug: bool = False, width: int = 1600, height: int = 980) -> None:
        # The cockpit is the operations interface the documentation has described all
        # along -- fourteen workspaces around one canvas. It was never loaded: this line
        # opened index.html, so the UI in docs/UI_GUIDE.md and the README was one no user
        # could reach, and the one they did reach was documented nowhere.
        #
        # ODK_UI=classic still opens the older single-screen shell, which remains the
        # more completely wired of the two while the cockpit's workspaces are connected
        # to the Api one at a time.
        page = "index.html" if os.environ.get("ODK_UI") == "classic" else "workspace.html"
        index = WEB_ROOT / page
        if not index.exists():
            raise FileNotFoundError(f"UI assets are missing: {index}")

        self.window = webview.create_window(
            WINDOW_TITLE,
            str(index),
            js_api=self.api,
            width=width,
            height=height,
            min_size=(1100, 700),
            background_color="#12161c",
            text_select=True,
        )
        self.api.bind_window(self.window)
        webview.start(menu=self.build_menu(), debug=debug)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the OpenDroneKit desktop application.")
    parser.add_argument("--debug", action="store_true", help="Open developer tools.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=980)
    args = parser.parse_args(argv)

    Shell().run(debug=args.debug, width=args.width, height=args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
