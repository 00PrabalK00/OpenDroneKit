"""The API surface exposed to the web UI.

Every method here is callable from JavaScript as `pywebview.api.<name>(...)`. The
methods are thin adapters: they validate input, call into `core/` or `mission/`, and
return JSON-serializable results. Long operations return a job id immediately and
report progress through `job_status`.

Errors are returned as `{"ok": false, "error": ...}` rather than raised, so the UI can
always show the real reason a step did not work.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

from .jobs import JobManager
from .session import AppSession, MapLayer


def ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def fail(error: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "error": str(error), **payload}


def guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Convert exceptions into structured failures the UI can display."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return fail(f"{type(exc).__name__}: {exc}")

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class Api:
    """Bridge object bound to the webview window."""

    def __init__(self, session: AppSession | None = None):
        self._session = session or AppSession()
        self._jobs = JobManager()
        self._window: Any = None

    def bind_window(self, window: Any) -> None:
        self._window = window

    # -- environment -----------------------------------------------------

    @guard
    def capabilities(self) -> dict[str, Any]:
        """What this installation can actually do. Drives honest UI state."""
        from core import geo
        from core.reconstruction_colmap import engine_capabilities

        caps = engine_capabilities()
        try:
            import torch

            caps["torch"] = torch.__version__
            caps["cuda"] = bool(torch.cuda.is_available())
            caps["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception:  # noqa: BLE001
            caps["torch"] = ""
            caps["cuda"] = False
            caps["gpu"] = ""
        try:
            from pymavlink import mavutil  # noqa: F401

            caps["pymavlink"] = True
        except Exception:  # noqa: BLE001
            caps["pymavlink"] = False
        caps["geo"] = geo.geo_capabilities()
        caps["platform"] = f"{platform.system()} {platform.release()}"
        return ok(capabilities=caps)

    @guard
    def model_status(self) -> dict[str, Any]:
        """Which detection models are present on disk, so the UI never implies AI it lacks."""
        from core.models import model_status as status_for

        registry_path = Path("models/model_registry.json")
        keys: list[str] = []
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                keys = list(registry.get("models", registry).keys())
            except json.JSONDecodeError:
                keys = []
        rows = []
        for key in keys:
            try:
                info = status_for(key)
                rows.append({"key": key, **(info if isinstance(info, dict) else {"status": str(info)})})
            except Exception as exc:  # noqa: BLE001
                rows.append({"key": key, "exists": False, "error": str(exc)})
        available = sum(1 for r in rows if r.get("exists"))
        return ok(models=rows, available=available, total=len(rows))

    @guard
    def verify_models(self) -> dict[str, Any]:
        """Whether each installed model is the file its recorded metrics were measured on.

        A model replaced on disk keeps its registry entry, its labels and its published
        accuracy, so every report it feeds continues to quote figures belonging to a
        different file. This is the check that says so.
        """
        from core.models import verify_all_models

        report = verify_all_models()
        if report["any_mismatch"]:
            self._session.audit("model_identity_mismatch", {
                "keys": [row["model_key"] for row in report["models"]
                         if row["status"] == "mismatch"],
            })
        return ok(**report)

    # -- projects --------------------------------------------------------

    @guard
    def get_project(self) -> dict[str, Any]:
        return ok(project=self._session.ensure_active_project())

    @guard
    def list_projects(self) -> dict[str, Any]:
        return ok(projects=self._session.store.list_projects())

    @guard
    def create_project(self, name: str, root_dir: str = "", description: str = "") -> dict[str, Any]:
        if not str(name).strip():
            return fail("Project name cannot be empty.")
        return ok(project=self._session.create_project(name.strip(), root_dir, description))

    @guard
    def set_active_project(self, project_id: int) -> dict[str, Any]:
        return ok(project=self._session.set_active_project(int(project_id)))

    @guard
    def audit_log(self, limit: int = 200) -> dict[str, Any]:
        return ok(events=self._session.store.list_audit_events(self._session.project_id(), int(limit)))

    # -- map geometry ----------------------------------------------------

    @guard
    def set_aoi(self, polygon_lonlat: list[list[float]]) -> dict[str, Any]:
        """Store the area of interest drawn on the map.

        This replaces the hardcoded survey polygon the previous UI used for every
        mission: planning now uses whatever the operator actually drew.
        """
        ring = [[float(p[0]), float(p[1])] for p in polygon_lonlat if len(p) >= 2]
        if len(ring) < 3:
            return fail("An area of interest needs at least 3 vertices.")
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        self._session.aoi_polygon = ring
        from core import geo

        return ok(
            vertices=len(ring),
            area_m2=round(geo.polygon_area_m2(ring), 2),
            suggested_epsg=geo.auto_utm_epsg(ring[0][1], ring[0][0]),
        )

    @guard
    def set_no_fly_zones(self, polygons: list[list[list[float]]]) -> dict[str, Any]:
        cleaned: list[list[list[float]]] = []
        for polygon in polygons or []:
            ring = [[float(p[0]), float(p[1])] for p in polygon if len(p) >= 2]
            if len(ring) >= 3:
                cleaned.append(ring[:-1] if ring[0] == ring[-1] else ring)
        self._session.no_fly_polygons = cleaned
        return ok(count=len(cleaned))

    @guard
    def set_terrain_source(self, path: str) -> dict[str, Any]:
        source = Path(path)
        if path and not source.exists():
            return fail(f"Terrain file not found: {path}")
        self._session.terrain_source_path = str(source) if path else ""

        # Selecting a terrain file says nothing about whether it covers the site, and a
        # DEM for the wrong valley plans as quietly as the right one. Answer that here,
        # while the operator is still choosing, rather than only at plan time.
        coverage: dict[str, Any] | None = None
        if self._session.terrain_source_path and len(self._session.aoi_polygon) >= 3:
            from core.terrain_cache import source_covers_area

            coverage = source_covers_area(
                self._session.terrain_source_path, self._session.aoi_polygon)
        return ok(path=self._session.terrain_source_path, coverage=coverage)

    @guard
    def mark_boundary_corner(self, note: str = "") -> dict[str, Any]:
        """Record the aircraft's current position as a corner of the survey boundary.

        For the field edge under canopy, or the stockpile toe that moved since the
        basemap was flown: the boundary comes out of the flight rather than out of
        imagery that may be years old.
        """
        from mission.fly_to_draw import BoundaryRefused, mark_from_telemetry

        try:
            mark = mark_from_telemetry(self._session.telemetry(), note=note)
        except BoundaryRefused as exc:
            # Written for an operator standing next to the aircraft, who can act on it.
            return fail(str(exc))

        marks = list(getattr(self._session, "boundary_marks", []))
        marks.append(mark)
        self._session.boundary_marks = marks
        self._session.audit("boundary_corner_marked",
                            {"corner": len(marks), "fix_type": mark.fix_type})
        return ok(corner=len(marks), mark=mark.to_dict())

    @guard
    def clear_boundary_marks(self) -> dict[str, Any]:
        self._session.boundary_marks = []
        return ok(corner_count=0)

    @guard
    def boundary_from_marks(self, apply_as_aoi: bool = True) -> dict[str, Any]:
        """Close the flown corners into an area of interest."""
        from mission.fly_to_draw import BoundaryRefused, boundary_from_marks as build

        marks = list(getattr(self._session, "boundary_marks", []))
        try:
            boundary = build(marks)
        except BoundaryRefused as exc:
            return fail(str(exc))

        if apply_as_aoi:
            self._session.aoi_polygon = [list(point) for point in boundary["polygon"]]
            self._session.audit("boundary_flown", {
                "corners": boundary["corner_count"],
                "area_hectares": boundary["area_hectares"],
            })
        return ok(**boundary)

    @guard
    def search_places(self, query: str, provider: str = "") -> dict[str, Any]:
        """Find a site by address, place name, or typed coordinates.

        Geocoding is the one part of this toolkit that reaches outward, so it happens
        only when the operator searches, and the response carries a note stating where
        the query went. A typed coordinate is resolved locally either way.
        """
        from core.geocoding import search_places as run_search

        selected = provider or getattr(self._session, "geocoding_provider", "") or "nominatim"
        result = run_search(
            query,
            provider=selected,
            gazetteer_path=self._session.project_root() / "places.json",
        )
        if result.get("results"):
            self._session.audit("place_searched", {"query": query, "provider": result.get("provider")})
        return ok(**result)

    @guard
    def set_geocoding_provider(self, provider: str) -> dict[str, Any]:
        """Choose the search backend, including a self-hosted or offline one."""
        from core.geocoding import PROVIDERS, build_provider

        name = str(provider or "nominatim").strip().lower()
        if name not in PROVIDERS:
            return fail(f"Unknown provider {provider!r}. Available: {', '.join(sorted(PROVIDERS))}.")
        self._session.geocoding_provider = name
        return ok(provider=name, note=build_provider(name).describe())

    @guard
    def get_state(self) -> dict[str, Any]:
        """Everything the UI needs for a full refresh in one call."""
        return ok(
            project=self._session.ensure_active_project(),
            aoi=self._session.aoi_polygon,
            no_fly=self._session.no_fly_polygons,
            terrain_source=self._session.terrain_source_path,
            layers=self._session.layer_list(),
            vehicle=self._session.vehicle.to_dict(),
            has_mission=bool(self._session.mission_plan_dict),
            active_dataset=self._session.active_dataset_dir,
            jobs=self._jobs.active(),
        )

    # -- layers ----------------------------------------------------------

    @guard
    def list_layers(self) -> dict[str, Any]:
        return ok(layers=self._session.layer_list())

    @guard
    def set_layer_visible(self, layer_id: str, visible: bool) -> dict[str, Any]:
        return ok(changed=self._session.set_layer_visibility(layer_id, bool(visible)))

    @guard
    def set_layer_opacity(self, layer_id: str, opacity: float) -> dict[str, Any]:
        return ok(changed=self._session.set_layer_opacity(layer_id, float(opacity)))

    @guard
    def remove_layer(self, layer_id: str) -> dict[str, Any]:
        return ok(removed=self._session.remove_layer(layer_id))

    @guard
    def add_layer_from_file(self, path: str) -> dict[str, Any]:
        source = Path(path)
        if not source.exists():
            return fail(f"File not found: {path}")
        suffix = source.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            return ok(layer=self._session.register_raster_layer(source))
        if suffix in {".geojson", ".json"}:
            return ok(layer=self._session.register_vector_layer(source))
        return fail(f"Unsupported layer format: {suffix}. Use GeoTIFF or GeoJSON.")

    @guard
    def read_vector_layer(self, layer_id: str) -> dict[str, Any]:
        """Return a vector layer's GeoJSON so the map can render it."""
        layer = self._session.layers.get(layer_id)
        if layer is None:
            return fail(f"Unknown layer: {layer_id}")
        if layer.kind != "vector" or not layer.path:
            return fail("Layer is not a vector layer.")
        return ok(geojson=json.loads(Path(layer.path).read_text(encoding="utf-8")))

    @guard
    def raster_preview(self, layer_id: str, max_size: int = 1024) -> dict[str, Any]:
        """Render a raster layer to a PNG data URI with its geographic bounds.

        MapLibre cannot read GeoTIFF directly, so a georeferenced image overlay is the
        route to showing real reconstruction output on the map.
        """
        import base64

        import cv2
        import numpy as np

        from core import geo

        layer = self._session.layers.get(layer_id)
        if layer is None or not layer.path:
            return fail(f"Unknown layer: {layer_id}")
        if not layer.crs_epsg:
            return fail("Raster has no CRS, so it cannot be placed on the map.")

        data, meta = geo.read_geotiff(layer.path)
        nodata = meta.get("nodata")
        if data.shape[0] >= 3:
            image = np.moveaxis(data[:3], 0, 2).astype(np.float32)
            alpha = np.any(data[:3] != (nodata if nodata is not None else 0), axis=0).astype(np.uint8) * 255
        else:
            band = data[0].astype(np.float32)
            valid = np.isfinite(band)
            if nodata is not None:
                valid &= ~np.isclose(band, float(nodata))
            scaled = np.zeros_like(band, dtype=np.uint8)
            if valid.any():
                low, high = float(band[valid].min()), float(band[valid].max())
                span = max(high - low, 1e-9)
                scaled[valid] = np.clip((band[valid] - low) / span * 255.0, 0, 255).astype(np.uint8)
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[:, :, ::-1].astype(np.float32)
            alpha = valid.astype(np.uint8) * 255

        height, width = image.shape[:2]
        scale = min(1.0, float(max_size) / max(height, width))
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_NEAREST)

        rgba = np.dstack([np.clip(image, 0, 255).astype(np.uint8)[:, :, ::-1], alpha])
        success, buffer = cv2.imencode(".png", rgba)
        if not success:
            return fail("Could not encode raster preview.")

        bounds = meta.get("bounds") or []
        lon, lat = geo.projected_to_wgs84([bounds[0], bounds[2]], [bounds[1], bounds[3]], int(layer.crs_epsg))
        return ok(
            data_uri="data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii"),
            # MapLibre image sources take corners clockwise from the top left.
            coordinates=[
                [float(lon[0]), float(lat[1])],
                [float(lon[1]), float(lat[1])],
                [float(lon[1]), float(lat[0])],
                [float(lon[0]), float(lat[0])],
            ],
            crs_epsg=layer.crs_epsg,
        )

    # -- missions --------------------------------------------------------

    @guard
    def mission_templates(self) -> dict[str, Any]:
        """Every mission type the planner can actually generate.

        This was a hand-written list of fifteen while the planner accepted twenty-two, so
        wind turbine, dome, box, closed loop, multi-facade and smart adaptive were
        plannable and invisible. A mission type nobody can see is a mission type nobody
        uses, which is indistinguishable from one that was never built.
        """
        from mission.planner import MissionPlanner, available_templates

        return ok(templates=available_templates(), planner=MissionPlanner.__name__)

    @guard
    def cache_terrain(self, path: str, name: str = "") -> dict[str, Any]:
        """Copy a terrain raster into the project for use with no connectivity."""
        from core.terrain_cache import TerrainCacheError, cache_terrain as store

        try:
            tile = store(path, self._session.project_root(), name=name)
        except TerrainCacheError as exc:
            return fail(str(exc))

        self._session.audit("terrain_cached", {"tile": tile.name, "source": str(path)})
        return ok(tile=tile.to_dict())

    @guard
    def terrain_coverage(self) -> dict[str, Any]:
        """Whether the drawn area can be flown terrain-following offline.

        A partially covered area is reported as uncovered on purpose: terrain following
        that works over part of a site and silently flies level over the rest is worse
        than not following at all, because the transition is invisible in the plan.
        """
        from core.terrain_cache import coverage_report

        polygon = self._session.aoi_polygon
        if len(polygon) < 3:
            return fail("Draw an area of interest before checking terrain coverage.")
        return ok(**coverage_report(polygon, self._session.project_root()))

    @guard
    def describe_terrain_cache(self) -> dict[str, Any]:
        """What terrain is cached for this project, and whether the files are present."""
        from core.terrain_cache import describe_cache

        return ok(**describe_cache(self._session.project_root()))

    @guard
    def repeat_mission(self, version_num: int | None = None, mode: str = "exact",
                       camera: str = "", terrain_source: str = "",
                       name: str = "") -> dict[str, Any]:
        """Plan a repeat of an earlier survey so the two can be compared.

        A camera change moves the altitude to hold ground resolution rather than
        holding altitude, because preserving the flight while losing the comparison is
        the wrong thing to preserve.
        """
        from mission.repeat import repeat_mission as plan_repeat

        source: dict[str, Any] | None = None
        if version_num is None:
            source = self._session.mission_plan_dict or None
        else:
            versions = self._session.store.list_mission_versions(
                self._session.project_id(), name or None)
            entry = next((v for v in versions
                          if int(v.get("version_num", 0)) == int(version_num)), None)
            if entry is None:
                available = ", ".join(str(v.get("version_num")) for v in versions) or "none"
                return fail(f"No version {version_num}. Saved versions: {available}.")
            source = entry

        if not source:
            return fail("No mission to repeat. Plan one, or name a saved version.")

        try:
            repeat = plan_repeat(source, mode=mode, camera=camera,
                                 terrain_source=terrain_source,
                                 boundary=self._session.aoi_polygon or None)
        except ValueError as exc:
            return fail(str(exc))

        self._session.audit("mission_repeat_planned",
                            {"mode": mode, "camera": camera or "unchanged",
                             "comparability": repeat.comparability})
        return ok(repeat=repeat.to_dict())

    @guard
    def compare_survey_specifications(self, first_version: int,
                                      second_version: int,
                                      name: str = "") -> dict[str, Any]:
        """Whether two surveys were flown to specifications that can be differenced."""
        from mission.repeat import compare_specifications

        versions = self._session.store.list_mission_versions(
            self._session.project_id(), name or None)
        by_num = {int(v.get("version_num", 0)): v for v in versions}

        missing = [n for n in (first_version, second_version) if n not in by_num]
        if missing:
            available = ", ".join(str(n) for n in sorted(by_num)) or "none"
            return fail(f"No such version(s): {missing}. Saved: {available}.")

        return ok(comparison=compare_specifications(by_num[first_version],
                                                    by_num[second_version]))

    @guard
    def import_gcps(self, path: str, epsg: int | None = None) -> dict[str, Any]:
        """Load surveyed ground control points for this project."""
        from core.gcp import GcpError, read_gcp_file

        try:
            points = read_gcp_file(path, default_epsg=epsg)
        except GcpError as exc:
            return fail(str(exc))

        self._session.ground_control = points
        self._session.audit("gcps_imported", {"path": str(path), "count": len(points)})
        return ok(points=[p.to_dict() for p in points], count=len(points))

    @guard
    def mark_gcp(self, name: str, image: str, pixel_x: float, pixel_y: float) -> dict[str, Any]:
        """Record where a control point appears in one image."""
        from core.gcp import add_mark

        points = getattr(self._session, "ground_control", None) or []
        point = next((p for p in points if p.name == name), None)
        if point is None:
            known = ", ".join(p.name for p in points) or "none imported"
            return fail(f"No control point named {name!r}. Imported: {known}.")

        add_mark(point, image, float(pixel_x), float(pixel_y))
        return ok(point=point.to_dict())

    @guard
    def gcp_accuracy_report(self, computed: dict[str, list[float]] | None = None) -> dict[str, Any]:
        """How far each control point landed from where it was surveyed."""
        from core.gcp import accuracy_report, residuals_from_positions, write_report

        points = getattr(self._session, "ground_control", None) or []
        if not points:
            return fail("No ground control points have been imported.")

        placed = {k: tuple(v) for k, v in (computed or {}).items() if len(v) >= 3}
        report = accuracy_report(points, residuals_from_positions(points, placed))

        target = self._session.project_root() / "gcp_report.json"
        write_report(report, target)
        return ok(report=report, path=str(target))

    @guard
    def camera_capabilities(self) -> dict[str, Any]:
        """What the connected payload has said it can do."""
        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")
        camera = getattr(client, "camera", None)
        if camera is None:
            return fail(f"The {self._session.vehicle.driver!r} driver has no camera control.")
        return ok(camera=camera().describe())

    @guard
    def camera_command(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """Photo, video, mode, zoom and focus on the connected payload.

        Exposure settings are refused with the reason rather than silently dropped: an
        operator who believes they set an ISO that never arrived will not find out
        until the imagery comes back.
        """
        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")
        builder = getattr(client, "camera", None)
        if builder is None:
            return fail(f"The {self._session.vehicle.driver!r} driver has no camera control.")

        camera = builder()
        handlers = {
            "take_photo": lambda: camera.take_photo(
                count=int(kwargs.get("count", 1)),
                interval_s=float(kwargs.get("interval_s", 0.0))),
            "stop_photo_sequence": camera.stop_photo_sequence,
            "start_video": camera.start_video,
            "stop_video": camera.stop_video,
            "set_mode": lambda: camera.set_mode(str(kwargs.get("mode", "photo"))),
            "set_zoom": lambda: camera.set_zoom(float(kwargs.get("level_pct", 0.0))),
            "set_focus": lambda: camera.set_focus(float(kwargs.get("level_pct", 0.0))),
            "set_exposure_setting": lambda: camera.set_exposure_setting(
                str(kwargs.get("setting", "")), kwargs.get("value")),
        }

        handler = handlers.get(action)
        if handler is None:
            return fail(f"Unknown camera action {action!r}. "
                        f"Use one of: {', '.join(sorted(handlers))}.")

        result = handler()
        self._session.audit("camera_command", {"action": action, "ok": result.ok})
        payload = result.to_dict()
        return ok(**payload) if result.ok else fail(result.message, **payload)

    @guard
    def check_interrupted_flight(self) -> dict[str, Any]:
        """Whether a flight was in progress when this software last stopped.

        Nothing is resumed. The aircraft may still be airborne and no file on disk can
        say otherwise, so the operator is given what was recorded and left to decide.
        """
        from core.flight_state import recover

        return ok(**recover(self._session.project_root()))

    @guard
    def discard_interrupted_flight(self) -> dict[str, Any]:
        """Forget a recorded flight once the operator has resolved it."""
        from core.flight_state import clear_state

        clear_state(self._session.project_root())
        self._session.audit("flight_state_discarded", {})
        return ok(cleared=True)

    @guard
    def control_state(self) -> dict[str, Any]:
        """Who is flying the aircraft right now."""
        from core.flight_control import control_state as describe_control

        client = self._session._drone_client
        if client is None:
            return ok(state={
                "control": "unknown", "mode": "", "armed": False,
                "pilot_has_control": False,
                "description": "No vehicle is connected.",
            })
        return ok(state=describe_control(client.get_telemetry()).to_dict())

    @guard
    def take_manual_control(self, preferred_mode: str = "") -> dict[str, Any]:
        """Interrupt autonomy and hand the aircraft back to the pilot.

        Confirms against the vehicle's reported mode rather than against having sent the
        command, because telling a pilot they have control while the aircraft is still
        flying its mission is the worst outcome available here.
        """
        from core.flight_control import take_manual_control as hand_back

        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")

        result = hand_back(client, preferred=preferred_mode)
        self._session.audit("manual_control_requested", {
            "ok": result.get("ok"),
            "mode": result.get("mode", ""),
        })
        return ok(**result) if result.get("ok") else fail(result["error"], **result)

    @guard
    def verify_site(self, image_folder: str = "", quality_sample_limit: int = 200) -> dict[str, Any]:
        """Decide whether this survey can be left as flown.

        Every problem this catches is cheap on site and expensive afterwards, so it is
        meant to be run before the aircraft goes back in its case.
        """
        from core.site_verification import verify_site as run_verification

        folder = image_folder or self._session.active_dataset_dir
        if not folder:
            return fail("Select a dataset, or pass the folder the images were copied to.")
        if not Path(folder).is_dir():
            return fail(f"Not a folder of images: {folder}")

        try:
            verdict = run_verification(
                folder,
                self._session.mission_plan_dict or None,
                quality_sample_limit=int(quality_sample_limit),
            )
        except NotADirectoryError as exc:
            return fail(str(exc))

        payload = verdict.to_dict()
        self._session.audit("site_verified", {
            "folder": str(folder),
            "ok": payload["ok"],
            "blocking": len(payload["blocking"]),
        })
        return ok(**payload)

    @guard
    def export_flight_log(self, output_dir: str = "", output_format: str = "all") -> dict[str, Any]:
        """Write the recorded flight to CSV, JSON, GPX and KML.

        A telemetry stream is transient; the moment it matters is afterwards, when
        someone asks where the aircraft was or what altitude something was flown at.
        """
        from core.flight_log import export, export_all

        log = getattr(self._session, "flight_log", None)
        if log is None or not getattr(log, "samples", None):
            return fail(
                "No flight has been recorded in this session. Connect to a vehicle and "
                "fly, or load a log, before exporting one."
            )

        target = Path(output_dir) if output_dir else (self._session.project_root() / "flights")
        stem = (log.mission_name or "flight").replace(" ", "_").lower()

        try:
            if output_format == "all":
                written = export_all(log, target, stem=stem)
            else:
                path = export(log, target / f"{stem}.{output_format}", output_format)
                written = {output_format: str(path)}
        except ValueError as exc:
            return fail(str(exc))

        self._session.audit("flight_log_exported",
                            {"formats": sorted(written), "samples": len(log.samples)})
        return ok(files=written, summary=log.summary())

    @guard
    def plan_battery_segments(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Split the current mission into sorties that each fit inside one battery."""
        from mission.estimates import AircraftProfile
        from mission.resume import plan_battery_segments

        plan = getattr(self._session, "mission_plan", None)
        if plan is None:
            return fail("Plan a mission before splitting it across batteries.")

        opts = dict(options or {})
        aircraft = AircraftProfile(
            name=str(opts.get("aircraft_name", "generic multirotor")),
            endurance_min=float(opts.get("endurance_min", 25.0)),
            reserve_pct=float(opts.get("reserve_pct", 25.0)),
            batteries_owned=int(opts.get("batteries_owned", 0)),
        )
        try:
            return ok(**plan_battery_segments(plan, aircraft))
        except ValueError as exc:
            return fail(str(exc))

    @guard
    def resume_from_images(self, image_folder: str) -> dict[str, Any]:
        """Work out what is left to fly from the images already on the card.

        The imagery is the record of what was captured, which is a better source than a
        progress counter: a counter can be confident about a photograph that was never
        written to disk.
        """
        from mission.resume import resume_plan, state_from_images

        plan = getattr(self._session, "mission_plan_dict", None)
        if not plan:
            return fail("Plan a mission before resuming one.")
        if not Path(image_folder).is_dir():
            return fail(f"Not a folder of images: {image_folder}")

        try:
            state = state_from_images(plan, image_folder)
        except (ValueError, NotADirectoryError) as exc:
            return fail(str(exc))

        resumed = resume_plan(plan, state)
        self._session.audit("mission_resumed", {
            "remaining": resumed.get("capture_count", 0),
            "completed": state.to_dict()["completed"],
        })
        return ok(resume=resumed, progress=state.to_dict())

    @guard
    def linked_mission_progress(self, image_folder: str,
                                resume: bool = False) -> dict[str, Any]:
        """Which survey inside a linked sortie is finished, and which is part flown.

        An overall percentage cannot distinguish "every survey is nearly done" from
        "three are finished and the fourth was never started", and those call for
        opposite actions on site.
        """
        from mission.linking import NotLinked, linked_progress, resume_linked_mission

        plan = getattr(self._session, "mission_plan_dict", None)
        if not plan:
            return fail("Plan a mission before asking how much of it has been flown.")
        if not Path(image_folder).is_dir():
            return fail(f"Not a folder of images: {image_folder}")

        try:
            handler = resume_linked_mission if resume else linked_progress
            report = handler(plan, image_folder)
        except NotLinked as exc:
            # Written for an operator; passing it through beats a generic message.
            return fail(str(exc))
        except (ValueError, NotADirectoryError) as exc:
            return fail(str(exc))

        self._session.audit("linked_progress_checked", {
            "complete": report["complete_segments"],
            "partial": report["partial_segments"],
        })
        return ok(**report)

    @guard
    def list_cameras(self) -> dict[str, Any]:
        """Every camera profile available for planning, built-in and user-defined."""
        from mission.cameras import all_profiles

        profiles = all_profiles()
        return ok(cameras=[p.to_dict() for p in sorted(profiles.values(),
                                                       key=lambda c: c.name)])

    @guard
    def describe_camera(self, name: str, altitude_m: float = 60.0) -> dict[str, Any]:
        """A camera's geometry and what it yields at a working altitude."""
        from mission.cameras import describe

        return ok(camera=describe(name, altitude_m=float(altitude_m)))

    @guard
    def add_camera(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Store an operator's own camera so it can be planned with."""
        from mission.cameras import CameraProfile, save_user_profile

        required = ("key", "name", "sensor_w_mm", "sensor_h_mm", "focal_mm",
                    "image_w_px", "image_h_px")
        missing = [field for field in required if field not in profile]
        if missing:
            return fail(f"A camera profile needs: {', '.join(missing)}.")

        try:
            entry = CameraProfile(
                key=str(profile["key"]).strip().lower(),
                name=str(profile["name"]),
                sensor_w_mm=float(profile["sensor_w_mm"]),
                sensor_h_mm=float(profile["sensor_h_mm"]),
                focal_mm=float(profile["focal_mm"]),
                image_w_px=int(profile["image_w_px"]),
                image_h_px=int(profile["image_h_px"]),
                thermal=bool(profile.get("thermal", False)),
                source="user",
                notes=str(profile.get("notes", "")),
            )
            path = save_user_profile(entry)
        except (ValueError, TypeError) as exc:
            # The validation messages explain why the geometry cannot be real.
            return fail(str(exc))

        return ok(camera=entry.to_dict(), stored_at=str(path))

    @guard
    def altitude_for_gsd(self, camera: str, gsd_cm: float) -> dict[str, Any]:
        """The altitude that achieves a required GSD, which is how surveys are specified."""
        from mission.cameras import UnknownCamera, require

        try:
            profile = require(camera)
            altitude = profile.altitude_for_gsd_m(float(gsd_cm))
        except UnknownCamera as exc:
            return fail(str(exc))
        except ValueError as exc:
            return fail(str(exc))

        return ok(camera=profile.key, gsd_cm=float(gsd_cm),
                  altitude_m=round(altitude, 1))

    @guard
    def measure_on_raster(self, raster_path: str, pixels: list[list[float]],
                          kind: str = "distance") -> dict[str, Any]:
        """Measure distance, area or perimeter on a georeferenced raster.

        The raster supplies its own scale, so nothing here depends on the operator
        typing one in. A raster without a CRS is refused rather than measured in
        pixels and labelled metres.
        """
        from core.raster_measurement import (
            NotGeoreferenced,
            NotProjected,
            measure_area,
            measure_distance,
            measure_perimeter,
        )

        handlers = {
            "distance": measure_distance,
            "area": measure_area,
            "perimeter": measure_perimeter,
        }
        handler = handlers.get(kind)
        if handler is None:
            return fail(f"Unknown measurement kind {kind!r}. Use distance, area or perimeter.")

        points = [(float(p[0]), float(p[1])) for p in pixels if len(p) >= 2]
        try:
            measurement = handler(raster_path, points)
        except (NotGeoreferenced, NotProjected) as exc:
            # Written for an operator; passing them through beats a generic message.
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))

        return ok(measurement=measurement.to_dict())

    @guard
    @guard
    def tag_annotations(self, annotation_ids: list[str], tags: list[str]) -> dict[str, Any]:
        """Apply tags to many findings at once.

        The bulk case is the normal case: an inspector reviews forty roof photographs and
        wants them all marked "north elevation". One at a time is why people stop tagging,
        and an untagged set cannot be filtered, reported on or handed over.

        A stale id in the selection is reported rather than fatal -- something deleted
        since the selection was made should not discard the other thirty-nine.
        """
        from core.annotations import add_tags

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        if not annotation_ids:
            return fail("Select the findings to tag first.")
        try:
            result = add_tags(Path(root), [str(i) for i in annotation_ids],
                              [str(t) for t in tags])
        except ValueError as exc:
            return fail(str(exc))
        self._session.audit("annotations_tagged",
                            {"count": len(result["updated"]), "tags": result["tags"]})
        return ok(**result)

    @guard
    def untag_annotations(self, annotation_ids: list[str], tags: list[str]) -> dict[str, Any]:
        """Take tags off many findings at once."""
        from core.annotations import remove_tags

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        if not annotation_ids:
            return fail("Select the findings to untag first.")
        try:
            result = remove_tags(Path(root), [str(i) for i in annotation_ids],
                                 [str(t) for t in tags])
        except ValueError as exc:
            return fail(str(exc))
        return ok(**result)

    @guard
    def list_annotation_tags(self, project_id: str = "") -> dict[str, Any]:
        """Every tag in use, with how many findings carry it.

        The count is what makes the list usable: a tag on one finding out of four hundred
        is usually a typo, and it shows up beside the one it should have been.
        """
        from core.annotations import all_tags

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        return ok(tags=all_tags(Path(root), project_id or None))

    @guard
    def find_annotations(self, tag: str = "", source_type: str = "",
                         source_id: str = "") -> dict[str, Any]:
        """Findings filtered by tag, surface or image."""
        from core.annotations import list_annotations

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        found = list_annotations(
            Path(root),
            source_id=source_id or None,
            source_type=source_type or None,
            tag=tag or None,
        )
        return ok(annotations=[a.to_dict() for a in found], count=len(found))

    @guard
    def list_views(self) -> dict[str, Any]:
        """The named ways this project's model can be opened."""
        from core.saved_views import ViewStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        views = ViewStore(Path(root) / "reconstruction").load()
        return ok(views=[v.to_dict() for v in views])

    @guard
    def save_view(self, name: str, position: list[float], target: list[float],
                  fov_deg: float = 50.0, visible_clips: list[str] | None = None,
                  facade_mode: bool = False, show_annotations: bool = True,
                  is_default: bool = False) -> dict[str, Any]:
        """Save how the model should open: camera, clips, facade mode, annotations.

        A view is a set of numbers pointing at the live model, not a render. The
        recipient still orbits away from it, measures, and opens the source photographs
        -- a picture would be smaller and would answer a different question.
        """
        from core.saved_views import SavedView, ViewRefused, ViewStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            views = ViewStore(Path(root) / "reconstruction").add(SavedView(
                name=str(name),
                position=tuple(float(v) for v in position),
                target=tuple(float(v) for v in target),
                fov_deg=float(fov_deg),
                visible_clips=[str(c) for c in (visible_clips or [])],
                facade_mode=bool(facade_mode),
                show_annotations=bool(show_annotations),
                is_default=bool(is_default),
            ))
        except (ViewRefused, TypeError, ValueError) as exc:
            return fail(str(exc))
        self._session.audit("view_saved", {"name": name})
        return ok(views=[v.to_dict() for v in views])

    @guard
    def open_view(self, name: str) -> dict[str, Any]:
        """Read a saved view back, reporting any clip it can no longer apply.

        A clip can be deleted after a view referring to it was saved. Failing to open the
        view is unhelpful; opening it silently showing more of the model than intended is
        worse. So it opens, and says what it could not apply.
        """
        from core.model_clipping import ClipStore
        from core.saved_views import ViewStore, resolve_clips

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        recon = Path(root) / "reconstruction"
        view = next((v for v in ViewStore(recon).load() if v.name == str(name)), None)
        if view is None:
            return fail(f"No view named {name!r}.")

        applied, missing = resolve_clips(view, [c.name for c in ClipStore(recon).load()])
        result = ok(view=view.to_dict(), applied_clips=applied, missing_clips=missing)
        if missing:
            result["warning"] = (
                f"This view refers to {len(missing)} clip(s) that no longer exist: "
                + ", ".join(missing)
                + ". It has opened without them, so more of the model is showing than "
                "when it was saved."
            )
        return result

    @guard
    def set_default_view(self, name: str) -> dict[str, Any]:
        """Choose the view a share link opens at."""
        from core.saved_views import ViewRefused, ViewStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            views = ViewStore(Path(root) / "reconstruction").set_default(str(name))
        except ViewRefused as exc:
            return fail(str(exc))
        self._session.audit("default_view_set", {"name": name})
        return ok(views=[v.to_dict() for v in views])

    @guard
    def remove_view(self, name: str) -> dict[str, Any]:
        from core.saved_views import ViewRefused, ViewStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            views = ViewStore(Path(root) / "reconstruction").remove(str(name))
        except ViewRefused as exc:
            return fail(str(exc))
        return ok(views=[v.to_dict() for v in views])

    @guard
    def list_clips(self) -> dict[str, Any]:
        """The named cuts saved against this project's model."""
        from core.model_clipping import ClipStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        clips = ClipStore(Path(root) / "reconstruction").load()
        return ok(clips=[c.to_dict() for c in clips])

    @guard
    def add_polygon_clip(self, name: str, polygon_xy: list[list[float]]) -> dict[str, Any]:
        """Keep what is inside a footprint; remove the neighbours and the car park."""
        from core.model_clipping import Clip, ClipRefused, ClipStore, points_inside_polygon

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            # Validate the ring now rather than at render time, so a bad boundary is
            # refused while the operator is still looking at the drawing they made.
            import numpy as np

            points_inside_polygon(np.zeros((1, 3)), polygon_xy)
            clips = ClipStore(Path(root) / "reconstruction").add(
                Clip(name=str(name), kind="polygon",
                     polygon_xy=[list(map(float, p)) for p in polygon_xy]))
        except ClipRefused as exc:
            return fail(str(exc))
        self._session.audit("clip_added", {"name": name, "kind": "polygon"})
        return ok(clips=[c.to_dict() for c in clips])

    @guard
    def add_plane_clip(self, name: str, point_xyz: list[float],
                       heading_deg: float = 0.0, pitch_deg: float = 0.0,
                       roll_deg: float = 0.0) -> dict[str, Any]:
        """Cut on an oriented plane, to isolate a facade or expose a section."""
        from core.model_clipping import Clip, ClipPlane, ClipRefused, ClipStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            plane = ClipPlane.from_orientation(
                point_xyz, float(heading_deg), float(pitch_deg), float(roll_deg))
            clips = ClipStore(Path(root) / "reconstruction").add(
                Clip(name=str(name), kind="plane", plane=plane))
        except ClipRefused as exc:
            return fail(str(exc))
        self._session.audit("clip_added", {"name": name, "kind": "plane"})
        return ok(clips=[c.to_dict() for c in clips])

    @guard
    def set_clip_visible(self, name: str, visible: bool) -> dict[str, Any]:
        """Show or hide a clip without deleting it.

        Hiding is what makes a clip a view rather than an edit: the geometry it removed
        comes back, and no survey data was ever destroyed to produce the deliverable.
        """
        from core.model_clipping import ClipRefused, ClipStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            clips = ClipStore(Path(root) / "reconstruction").set_visible(str(name), bool(visible))
        except ClipRefused as exc:
            return fail(str(exc))
        return ok(clips=[c.to_dict() for c in clips])

    @guard
    def remove_clip(self, name: str) -> dict[str, Any]:
        from core.model_clipping import ClipRefused, ClipStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            clips = ClipStore(Path(root) / "reconstruction").remove(str(name))
        except ClipRefused as exc:
            return fail(str(exc))
        self._session.audit("clip_removed", {"name": name})
        return ok(clips=[c.to_dict() for c in clips])

    @guard
    def export_clipped_model(self, output_path: str = "") -> dict[str, Any]:
        """Write the model as the visible clips leave it.

        The export is a new file. Cutting the only copy of a reconstruction would turn a
        presentation decision into data loss, so the source mesh is never touched.
        """
        from core.model_clipping import ClipStore, clip_mesh
        from core.model_measurement import read_mesh

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        recon = Path(root) / "reconstruction"
        source = recon / "mesh.obj"
        if not source.is_file():
            return fail("No mesh in this project yet. Run a reconstruction first.")

        clips = [c for c in ClipStore(recon).load() if c.visible]
        if not clips:
            return fail("No visible clips, so there is nothing to cut to.")

        vertices, faces = read_mesh(source)
        kept_v, kept_f = clip_mesh(vertices, faces, clips)
        if len(kept_f) == 0:
            return fail(
                "Those clips remove the entire model. Check the boundary is drawn around "
                "the asset rather than beside it."
            )

        target = Path(output_path) if output_path else recon / "mesh_clipped.obj"
        lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in kept_v]
        lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in kept_f]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self._session.audit("clipped_model_exported", {"path": str(target)})
        return ok(path=str(target), vertices=int(len(kept_v)), faces=int(len(kept_f)),
                  source_vertices=int(len(vertices)), source_faces=int(len(faces)))

    @guard
    def plan_pylon_mission(self, center_lonlat: list[float],
                           elements: list[dict[str, Any]],
                           standoff_m: float = 12.0,
                           structure_radius_m: float = 3.0,
                           camera: str = "mavic2pro") -> dict[str, Any]:
        """A stacked orbit per named pylon element, at the height that element sits at.

        mission/mission_types.py has planned this since the pylon work landed, and nine
        tests cover it, but nothing could reach it: it was absent from the planner's
        template table, from mission_templates() and from this class. Implemented and
        unreachable is indistinguishable from missing to anyone using the application.
        """
        from mission.mission_types import MissionTypeRefused, plan_pylon_inspection
        from mission.planner import MissionPlanner

        try:
            plan = plan_pylon_inspection(
                MissionPlanner(),
                center_lonlat=list(center_lonlat),
                elements=elements,
                standoff_m=float(standoff_m),
                structure_radius_m=float(structure_radius_m),
                camera=str(camera),
            )
        except MissionTypeRefused as exc:
            return fail(str(exc))
        self._session.audit("pylon_planned", {"elements": len(elements)})
        return ok(plan=plan)

    @guard
    def plan_thermal_survey(self, thermal_camera: str, target_gsd_cm: float,
                            rgb_camera: str = "",
                            front_overlap_pct: float = 80.0,
                            side_overlap_pct: float = 70.0) -> dict[str, Any]:
        """A grid flown at the altitude the THERMAL sensor needs, with paired RGB.

        Thermal imagers have far coarser pixels than the visual camera beside them, so a
        grid planned for the RGB sensor delivers thermal imagery too coarse to read. The
        altitude here is solved for the thermal GSD and the RGB is simply along for the
        ride, which is the opposite of the usual assumption.
        """
        from mission.mission_types import MissionTypeRefused, plan_thermal_mission
        from mission.planner import MissionPlanner

        polygon = self._session.aoi_polygon
        if len(polygon) < 3:
            return fail("Draw an area of interest on the map before planning a survey.")
        try:
            plan = plan_thermal_mission(
                MissionPlanner(),
                polygon_lonlat=[list(p) for p in polygon],
                thermal_camera=str(thermal_camera),
                target_gsd_cm=float(target_gsd_cm),
                rgb_camera=str(rgb_camera),
                front_overlap_pct=float(front_overlap_pct),
                side_overlap_pct=float(side_overlap_pct),
            )
        except MissionTypeRefused as exc:
            return fail(str(exc))
        self._session.audit("thermal_planned", {"camera": thermal_camera})
        return ok(plan=plan)

    @guard
    def plan_multispectral_survey(self, payload_key: str,
                                  camera: str = "mavic3e",
                                  altitude_m: float = 60.0,
                                  calibration_panel_lonlat: list[float] | None = None,
                                  front_overlap_pct: float = 80.0,
                                  side_overlap_pct: float = 75.0) -> dict[str, Any]:
        """A grid for a multispectral array, with the calibration panel in the plan.

        The payload is the subject here, not the camera: a multispectral array has its
        own band set and its own reflectance panel, and a survey flown without a panel
        shot gives indices that cannot be compared with any other flight.
        """
        from mission.mission_types import MissionTypeRefused, plan_multispectral_mission
        from mission.planner import MissionPlanner

        polygon = self._session.aoi_polygon
        if len(polygon) < 3:
            return fail("Draw an area of interest on the map before planning a survey.")
        try:
            plan = plan_multispectral_mission(
                MissionPlanner(),
                polygon_lonlat=[list(p) for p in polygon],
                payload_key=str(payload_key),
                camera=str(camera),
                altitude_m=float(altitude_m),
                calibration_panel_lonlat=(
                    list(calibration_panel_lonlat) if calibration_panel_lonlat else None
                ),
                front_overlap_pct=float(front_overlap_pct),
                side_overlap_pct=float(side_overlap_pct),
            )
        except MissionTypeRefused as exc:
            return fail(str(exc))
        self._session.audit("multispectral_planned", {"payload": payload_key})
        return ok(plan=plan)

    def plan_complex_facade(self, polygon_xy: list[list[float]],
                            courtyards: list[list[list[float]]] | None = None,
                            standoff_m: float = 10.0,
                            overhang_depth_m: float = 0.0,
                            camera_fov_deg: float = 78.0) -> dict[str, Any]:
        """Facade passes for a building with courtyards, recesses and overhangs.

        A courtyard inverts the standoff: outside the building the aircraft stands off
        outward, inside a courtyard it must offset inward, and treating the two alike
        plans a mission that flies into masonry.

        Overhangs are reported rather than planned around. A balcony or deep reveal
        hides wall a single sweep never photographs, and the reconstruction renders the
        unseen part as smooth surface rather than a hole -- so the omission is invisible
        in the deliverable unless something says it is there.
        """
        from mission.footprints import (
            FootprintRefused, analyse_footprint, assess_occlusion, courtyard_segments,
        )

        try:
            analysis = analyse_footprint(polygon_xy)
            segments = courtyard_segments(
                polygon_xy, courtyards or [], standoff_m=float(standoff_m)
            )
            occlusion = assess_occlusion(
                float(overhang_depth_m), standoff_m=float(standoff_m),
                camera_fov_deg=float(camera_fov_deg),
            )
        except FootprintRefused as exc:
            return fail(str(exc))
        return ok(
            footprint=analysis.to_dict(),
            segments=[s.to_dict() for s in segments],
            segment_count=len(segments),
            courtyard_count=len(courtyards or []),
            occlusion=occlusion.to_dict(),
        )

    @guard
    def size_reconstruction_job(self, image_count: int,
                                work_dir: str = ".") -> dict[str, Any]:
        """Whether this machine can finish a reconstruction of this size.

        Asked up front because the failure it prevents is expensive: a job that runs for
        hours, exhausts memory during bundle adjustment, and dies having produced
        nothing. Feature matching is the binding constraint and grows with the SQUARE of
        the image count, so past a few hundred images chunking is a requirement rather
        than a tuning choice.

        Estimates are rough and say so. A wrong estimate costs a conversation; no
        estimate costs an afternoon.
        """
        from core.job_sizing import size_job

        try:
            estimate = size_job(int(image_count), work_dir=work_dir)
        except ValueError as exc:
            return fail(str(exc))
        return ok(**estimate.to_dict())

    @guard
    def plan_job_chunks(self, image_paths: list[str], chunk_size: int,
                        overlap: int = 10) -> dict[str, Any]:
        """Split a large capture into overlapping chunks that can be merged.

        Overlap is not optional: chunks reconstructed independently share no geometry,
        so without images appearing in both the result is several disconnected models
        rather than one survey.
        """
        from core.job_sizing import chunk_images

        try:
            chunks = chunk_images([str(p) for p in image_paths], int(chunk_size),
                                  overlap=int(overlap))
        except ValueError as exc:
            return fail(str(exc))
        return ok(
            chunk_count=len(chunks),
            chunks=chunks,
            overlap=int(overlap),
            note=("Consecutive chunks share images on purpose; those shared views are "
                  "what lets the sub-models be tied into one reconstruction."),
        )

    @guard
    def check_spatial_reference(self, image_paths: list[str],
                                gcp_count: int = 0,
                                epsg: int | None = None) -> dict[str, Any]:
        """What a reconstruction of these images will mean, before running it.

        GPS-denied work -- indoor, handheld, under-bridge, ground robot -- is a
        legitimate use and reconstructs fine. The catch is that structure-from-motion
        recovers geometry only up to a similarity transform, so without geotags or
        control the model has arbitrary position, rotation and SCALE. It still renders
        and still meshes; every distance in it is simply wrong by an unknown factor.

        Reported up front so a user knows whether the deliverable they want is a survey
        or a shape.
        """
        from core.spatial_reference import assess_spatial_reference

        try:
            reference = assess_spatial_reference(
                [str(p) for p in image_paths], gcp_count=int(gcp_count), epsg=epsg
            )
        except ValueError as exc:
            return fail(str(exc))
        return ok(**reference.to_dict())

    @guard
    def build_asset_inventory(self, instances: list[dict[str, Any]],
                              model_key: str = "",
                              model_sha256: str = "",
                              crs: str = "",
                              min_confidence: float = 0.0) -> dict[str, Any]:
        """Inventory detected assets across packs, in one vocabulary.

        Every pack grew its own class set -- power knows "pole", rail knows "signal",
        solar knows "module" -- so a survey covering a substation and the rail line
        beside it produced two inventories with no shared terms.

        The refusals are the substance. An instance with no location, no confidence or
        no model digest is rejected rather than counted, because a count is a claim
        somebody acts on -- a crew sent to seventeen poles, a client invoiced for a
        module count -- and a claim whose origin cannot be checked a month later is
        indistinguishable from a guess.
        """
        from core.asset_taxonomy import AssetRefused, build_asset_inventory, filter_by_confidence

        try:
            inventory = build_asset_inventory(
                instances,
                model={"key": model_key, "sha256": model_sha256},
                crs=crs,
            )
            if float(min_confidence) > 0.0:
                inventory = filter_by_confidence(inventory, float(min_confidence))
        except AssetRefused as exc:
            return fail(str(exc))
        return ok(**inventory)

    @guard
    def processing_queue_report(self) -> dict[str, Any]:
        """What the processing queue is doing, including how long work has waited.

        The wait time is reported because priority here is strict: a steady stream of
        high-priority jobs will hold a low-priority one indefinitely, and the only
        symptom is a job that never starts. A number makes that visible instead of
        leaving it to be inferred.
        """
        queue = getattr(self._session, "processing_queue", None)
        if queue is None:
            return fail(
                "No processing queue is attached to this session. The desktop shell runs "
                "jobs directly; the queue is for batch and service deployments."
            )
        return ok(**queue.report())

    @guard
    def build_training_corpus(self, labelled: list[dict[str, Any]],
                              salt: str = "custom-corpus-v1") -> dict[str, Any]:
        """Turn a user's labelled images into splits, or refuse and say what is missing.

        Someone labelling their own defects is doing the most valuable thing here and
        the most fragile. The failure modes are silent: a corpus builds, trains and
        reports without anyone noticing that a class had four examples or that the same
        photograph sat in both training and validation. None of those look like errors
        -- they look like a model that did unusually well.
        """
        from core.custom_training import CorpusRefused, LabelledImage, build_custom_corpus

        try:
            samples = [
                LabelledImage(path=Path(str(entry.get("path", ""))),
                              label=str(entry.get("label", "")))
                for entry in labelled
            ]
            splits, report = build_custom_corpus(samples, salt=salt)
        except CorpusRefused as exc:
            return fail(str(exc))
        return ok(
            splits={name: [str(s.path) for s in entries] for name, entries in splits.items()},
            **report.to_dict(),
        )

    @guard
    def build_detection_corpus(self, labelled: list[dict[str, Any]],
                               output_dir: str,
                               salt: str = "custom-boxes-v1",
                               allow_empty_images: bool = False) -> dict[str, Any]:
        """Turn boxes drawn in the labelling canvas into a trainable corpus, or refuse.

        The other half of build_training_corpus. That one takes a class per image, which
        answers "is there a crack in this tile"; this one takes the regions someone drew,
        which is what a user actually needs when they want to know WHERE.

        Refusals are the point, and they are stricter here because a box can be wrong in
        ways a class label cannot: a click stored as a zero-area target, coordinates off
        the edge of the image, or an image opened and never labelled, which teaches the
        model the defect is absent rather than being skipped.
        """
        from core.custom_training import CorpusRefused
        from core.label_sets import build_detection_corpus, regions_from_payload

        try:
            regions = regions_from_payload(labelled)
            report = build_detection_corpus(
                regions, output_dir, salt=salt, allow_empty_images=allow_empty_images
            )
        except CorpusRefused as exc:
            return fail(str(exc))
        except (OSError, ValueError) as exc:
            return fail(str(exc))
        return ok(**report)

    # ---------------------------------------------------------------- fleet
    #
    # Fleet, sharing, webhooks, reports and review were reachable only through the web
    # service, so twenty-three buttons in the cockpit reported themselves unavailable.
    # They are not missing capabilities -- every one carries a verified registry row.
    # app/desktop_ops.py opens the same database the service uses and calls the same
    # code, so a local-first user gets the real thing rather than a stub.

    @guard
    def list_dataset_images(self, limit: int = 500) -> dict[str, Any]:
        """The images in the active dataset, so a panel can list something real."""
        from app.session import SUPPORTED_IMAGE_EXTENSIONS

        folder = self._session.active_dataset_dir
        if not folder:
            return fail("Select a dataset first.")
        root = Path(folder)
        if not root.is_dir():
            return fail(f"Dataset folder is missing: {folder}")
        # The same extension set the importer accepted, so the panel cannot list an
        # image the application would then refuse to open.
        names = sorted(
            entry.name for entry in root.iterdir()
            if entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )[: max(1, int(limit))]
        return ok(folder=str(root), images=names, count=len(names))

    @guard
    def image_preview(self, name: str, max_size: int = 1400) -> dict[str, Any]:
        """One image from the active dataset, as a data URI the canvas can draw.

        The desktop UI is served over loopback from app/web, so it cannot read a file
        anywhere else on disk -- a survey folder is outside the document root and an
        <img src="D:/..."> is blocked. Handing back a downscaled data URI is what lets
        the canvas show the actual photograph rather than a placeholder saying one
        exists.

        Downscaled because these are 24-megapixel frames: a full-size data URI is tens
        of megabytes of base64 across the bridge for a preview nobody zooms into.
        """
        import base64

        import cv2

        folder = self._session.active_dataset_dir
        if not folder:
            return fail("Select a dataset first.")
        # Resolved against the dataset and checked, so a crafted name cannot walk out of
        # it -- the same containment rule the upload path enforces.
        root = Path(folder).resolve()
        target = (root / str(name)).resolve()
        if root not in target.parents and target != root:
            return fail("That image is outside the active dataset.")
        if not target.is_file():
            return fail(f"No such image in the dataset: {name}")

        image = cv2.imread(str(target))
        if image is None:
            return fail(f"Could not read {name}. It may not be an image this build understands.")
        height, width = image.shape[:2]
        limit = max(64, int(max_size))
        if max(height, width) > limit:
            scale = limit / float(max(height, width))
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        okay, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not okay:
            return fail(f"Could not encode a preview for {name}.")
        return ok(
            name=str(name),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            source_width=int(width),
            source_height=int(height),
            data_uri="data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii"),
        )

    @guard
    def add_aircraft(self, organization_id: int, name: str, model: str = "",
                     serial: str = "") -> dict[str, Any]:
        """Register an aircraft."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.add_aircraft(organization_id, name, model, serial))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def add_battery(self, organization_id: int, serial: str, capacity_mah: int = 0,
                    cycle_limit: int = 0) -> dict[str, Any]:
        """Register a battery so its cycles can be tracked."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.add_battery(organization_id, serial, capacity_mah, cycle_limit))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def add_pilot(self, organization_id: int, display_name: str, licence_number: str = "",
                  licence_expires_on: str = "") -> dict[str, Any]:
        """Add a pilot, refusing an unparseable licence expiry rather than defaulting it."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.add_pilot(organization_id, display_name,
                                              licence_number, licence_expires_on))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def log_maintenance(self, aircraft_id: int, kind: str, description: str = "",
                        performed_by: str = "") -> dict[str, Any]:
        """Record maintenance and reset the aircraft's service clock."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.log_maintenance(aircraft_id, kind, description, performed_by))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def assign_mission_to_aircraft(self, aircraft_id: int, mission_name: str) -> dict[str, Any]:
        """Note which aircraft is flying which mission."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.assign_mission(aircraft_id, mission_name))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def fleet_status(self, organization_id: int = 1) -> dict[str, Any]:
        """Aircraft, batteries, pilots and what is due for service."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.fleet_status(organization_id))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    # -------------------------------------------------------------- sharing

    @guard
    def create_share_link(self, project_id: int, note: str = "",
                          allow_download: bool = False) -> dict[str, Any]:
        """Issue a share link. The token is returned ONCE and only its hash is stored."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.create_share_link(project_id, note, allow_download))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def list_share_links(self, project_id: int) -> dict[str, Any]:
        """Every link issued for this project, by prefix rather than by token."""
        from app import desktop_ops

        try:
            return ok(links=desktop_ops.list_share_links(project_id))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def revoke_share_link(self, share_id: int) -> dict[str, Any]:
        """Revoke a link that has been shared too widely."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.revoke_share_link(share_id))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    # ------------------------------------------------------------- webhooks

    @guard
    def add_webhook(self, organization_id: int, url: str, events: list[str] | None = None,
                    description: str = "") -> dict[str, Any]:
        """Register a webhook. The signing secret is returned once."""
        from app import desktop_ops

        try:
            return ok(**desktop_ops.add_webhook(organization_id, url, events, description))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def list_webhooks(self, organization_id: int = 1) -> dict[str, Any]:
        """Registered webhooks and their delivery counts."""
        from app import desktop_ops

        try:
            return ok(webhooks=desktop_ops.list_webhooks(organization_id))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    # -------------------------------------------------------------- reports

    def _report_project_id(self) -> str:
        """The active project, as core.report_engine knows it.

        There are two project stores. app/store.py holds what the desktop opened;
        core/project.py holds what the report engine reads, with its own ids. Assuming
        they were the same made every report refuse with "Open a project" while a project
        was plainly open.

        So the desktop project is mirrored into the engine's store on demand, matched by
        name, and the engine's id is returned. Mirrored rather than merged: merging two
        schemas that have diverged this far is a migration, and a report should not be
        the thing that performs one.
        """
        from core.project import get_manager

        project = self._session.ensure_active_project()
        name = str(project.get("name") or "").strip()
        if not name:
            raise ValueError("Open a project before generating a report.")

        manager = get_manager()
        for candidate in manager.list_projects(include_archived=True):
            if str(candidate.get("name", "")).strip() == name:
                return str(candidate["id"])

        # No root_dir: the manager refuses a folder that already has files in it, which
        # is right for its own workspace and wrong to argue with here. The desktop
        # project keeps its folder; the engine keeps its own and writes the report there.
        created = manager.create_project(
            name=name,
            description=str(project.get("description") or ""),
        )
        return str(created["id"] if isinstance(created, dict) else created)

    @guard
    def report_readiness(self, project_id: str = "", report_type: str = "standard") -> dict[str, Any]:
        """What a report still needs, before anyone presses Generate."""
        from app import desktop_ops

        try:
            target = str(project_id) if project_id else self._report_project_id()
        except Exception as exc:  # noqa: BLE001 - no project open yet
            return fail(str(exc))
        try:
            return ok(**desktop_ops.report_readiness(target, report_type=report_type))
        except Exception as exc:  # noqa: BLE001 - the engine raises its own error type
            return fail(str(exc))

    @guard
    def generate_report(self, project_id: str = "", title: str = "Inspection report",
                        report_type: str = "standard", author: str = "") -> dict[str, Any]:
        """Build the report, or return the engine refusal naming what is missing."""
        from app import desktop_ops

        try:
            target = str(project_id) if project_id else self._report_project_id()
        except Exception as exc:  # noqa: BLE001 - no project open yet
            return fail(str(exc))
        try:
            return ok(**desktop_ops.generate_report(
                target, title=title, report_type=report_type, author=author))
        except Exception as exc:  # noqa: BLE001 - AppError carries the readiness detail
            return fail(str(exc))

    # --------------------------------------------------------------- review

    @guard
    def review_finding(self, annotation_id: str, decision: str,
                       reviewer: str = "operator") -> dict[str, Any]:
        """Accept, reject or flag a finding without overwriting what the model claimed."""
        from app import desktop_ops

        root = self._session.project_root()
        if not root:
            return fail("Open a project before reviewing findings.")
        try:
            return ok(**desktop_ops.review_finding(root, annotation_id, decision, reviewer))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def list_plugins(self) -> dict[str, Any]:
        """What the plugin registry has actually loaded."""
        from app import desktop_ops

        try:
            return ok(plugins=desktop_ops.list_plugins())
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def asset_taxonomy(self, domain: str = "") -> dict[str, Any]:
        """The shared asset vocabulary, so a caller can see it before detecting anything."""
        from core.asset_taxonomy import ASSET_TYPES, AssetRefused, DOMAINS, assets_for_domain

        try:
            types = assets_for_domain(domain) if str(domain).strip() else ASSET_TYPES
        except AssetRefused as exc:
            return fail(str(exc))
        return ok(
            domains=list(DOMAINS),
            asset_types=[
                {
                    "name": asset.name,
                    "domain": asset.domain,
                    "geometry": asset.geometry,
                    "countable": asset.countable,
                    "description": asset.description,
                    "aliases": list(asset.aliases),
                }
                for asset in types
            ],
        )

    @guard
    def reconstruction_capabilities(self) -> dict[str, Any]:
        """What this machine can actually reconstruct, before anyone starts a job.

        Dense point clouds need patch-match multi-view stereo, which needs a CUDA
        COLMAP -- either the native binary or CUDA-enabled pycolmap bindings. Without
        one, only the sparse cloud is available and no amount of post-processing
        changes that: densifying a sparse cloud by cloning points adds no observations,
        it only inflates the point count. That fake existed here once and was removed.

        Reported rather than discovered mid-job, because a user deserves to know before
        a long reconstruction whether the deliverable they want is possible at all.
        """
        from core.reconstruction_colmap import engine_capabilities

        caps = engine_capabilities()
        dense = bool(caps.get("dense_stereo"))
        return ok(
            capabilities=caps,
            dense_available=dense,
            note=(
                "Dense patch-match stereo is available; a dense cloud can be produced."
                if dense else
                "Dense point clouds are NOT available in this environment: no CUDA "
                "COLMAP binary and no CUDA-enabled pycolmap. Sparse reconstruction, "
                "mesh and orthomosaic still work. The sparse cloud is never inflated "
                "to imitate a dense one."
            ),
        )

    @guard
    def find_ponding(self, surface_path: str,
                     vertical_accuracy_m: float | None = None) -> dict[str, Any]:
        """Where water can collect on a surface, from the DSM.

        ``vertical_accuracy_m`` has no default on purpose. Without it there is no way to
        separate a real basin from reconstruction noise, and a list of depressions with
        no error bound is exactly the confident-but-uncheckable output this refuses to
        produce. The caller must state what the survey was capable of.

        Reports where water CAN collect, not whether water is there now. A dry
        depression and a flooded one are identical to this.
        """
        from core.dsm_analysis import NotGeoreferenced, load_surface
        from core.ponding import PondingRefused, find_ponding as run_ponding
        from core.slope import NotProjected

        try:
            surface = load_surface(surface_path)
            report = run_ponding(surface, vertical_accuracy_m=vertical_accuracy_m)
        except (NotGeoreferenced, NotProjected, PondingRefused) as exc:
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        return ok(**report.to_dict())

    @guard
    def compare_surveys(self, earlier_path: str, later_path: str,
                        earlier_accuracy_m: float | None = None,
                        later_accuracy_m: float | None = None,
                        registration_residual_m: float | None = None) -> dict[str, Any]:
        """Vertical movement between two surveys of the same ground.

        All three uncertainties are required. Two surveys of unchanged ground never
        difference to zero, so without them this cannot tell movement from measurement
        error -- and a subsidence figure nobody can check is worse than none.
        """
        from core.deformation import DeformationRefused, compare_surfaces
        from core.dsm_analysis import NotGeoreferenced, load_surface
        from core.slope import NotProjected

        try:
            report = compare_surfaces(
                load_surface(earlier_path), load_surface(later_path),
                earlier_accuracy_m=earlier_accuracy_m,
                later_accuracy_m=later_accuracy_m,
                registration_residual_m=registration_residual_m,
            )
        except (NotGeoreferenced, NotProjected, DeformationRefused) as exc:
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        return ok(**report.to_dict())

    @guard
    def plan_irregular_facade(self, polygon_xy: list[list[float]],
                              standoff_m: float = 10.0) -> dict[str, Any]:
        """Facade passes around an L-shaped or otherwise concave footprint.

        Reports reflex corners rather than smoothing them, and drops any pass that would
        fall inside the building -- which is what a naive offset ring does at a concave
        corner, silently, until the aircraft reaches the wall.
        """
        from mission.footprints import (
            FootprintRefused, analyse_footprint, facade_segments,
        )

        try:
            analysis = analyse_footprint(polygon_xy)
            segments = facade_segments(polygon_xy, standoff_m=float(standoff_m))
        except FootprintRefused as exc:
            return fail(str(exc))
        return ok(
            footprint=analysis.to_dict(),
            segments=[s.to_dict() for s in segments],
            segment_count=len(segments),
        )

    @guard
    def demo_workflow(self, name: str = "Demo inspection") -> dict[str, Any]:
        """A complete workflow with no hardware, marked synthetic throughout.

        Every artefact carries ``synthetic: True`` recursively, so a single finding
        lifted out of this still declares itself. Demo output exists to show what the
        software does, never to be reported as a survey result.
        """
        from core.demo_mode import demo_project

        return ok(**demo_project(name=name))

    @guard
    def measure_slope(self, surface_path: str,
                      polygon_xy: list[list[float]] | None = None) -> dict[str, Any]:
        """Gradient over a DSM or DTM: roof pitch, pavement fall, ramp steepness.

        The polygon is in the raster's own projected coordinates, because that is what
        the elevation surface is in and converting here would hide which CRS the numbers
        belong to.
        """
        from core.dsm_analysis import NotGeoreferenced
        from core.slope import NotProjected, measure_slope as run_slope

        region = [[float(p[0]), float(p[1])] for p in (polygon_xy or []) if len(p) >= 2]
        try:
            result = run_slope(surface_path, polygon_xy=region or None)
        except (NotGeoreferenced, NotProjected) as exc:
            # Written for an operator; passing them through beats a generic message.
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))

        if not result.get("ok"):
            return fail(result["reason"], **{k: v for k, v in result.items() if k != "reason"})
        return ok(**{k: v for k, v in result.items() if k != "ok"})

    @guard
    def check_ppk_inputs(self, events_path: str, rinex_path: str,
                         leap_seconds: int = 18, margin_s: float = 0.0) -> dict[str, Any]:
        """Whether the camera events and base observations can support a PPK survey.

        Checked before processing rather than after, because a run over partial base
        coverage still succeeds -- it just produces a deliverable that is centimetre
        accurate in places and metre accurate in others.
        """
        from core.rtk import RtkError, positioning_report

        try:
            report = positioning_report(events_path, rinex_path,
                                        leap_seconds=int(leap_seconds),
                                        margin_s=float(margin_s))
        except RtkError as exc:
            # Written for an operator; passing them through beats a generic message.
            return fail(str(exc))

        self._session.audit("ppk_inputs_checked", {
            "events": str(events_path), "rinex": str(rinex_path),
            "usable": report["usable_for_ppk"],
        })
        return ok(**{k: v for k, v in report.items() if k != "ok"})

    @guard
    def measure_in_model(self, model_path: str, kind: str = "length",
                         points_xyz: list[list[float]] | None = None) -> dict[str, Any]:
        """Length, height, area or volume inside a reconstructed 3D model.

        Refused unless the model's provenance records the CRS it was aligned to, since
        an ungeoreferenced reconstruction is in structure-from-motion units that read
        exactly like metres.
        """
        from core.model_measurement import (
            NotClosed,
            NotMetric,
            measure_area,
            measure_height,
            measure_length,
            measure_volume,
        )

        points = [[float(v) for v in p[:3]] for p in (points_xyz or []) if len(p) >= 3]
        try:
            if kind == "volume":
                measurement = measure_volume(model_path)
            elif kind == "area":
                measurement = measure_area(model_path, points)
            elif kind in {"length", "height"}:
                if len(points) < 2:
                    return fail(f"A {kind} needs two picked points.")
                handler = measure_length if kind == "length" else measure_height
                measurement = handler(model_path, points[0], points[1])
            else:
                return fail(
                    f"Unknown measurement kind {kind!r}. Use length, height, area or volume."
                )
        except (NotMetric, NotClosed) as exc:
            # Written for an operator; passing them through beats a generic message.
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))

        return ok(measurement=measurement)

    @guard
    def import_boundary(self, path: str) -> dict[str, Any]:
        """Load an area of interest from a KML, KMZ, GeoJSON, GPX or CSV file.

        The imported boundary becomes the session AOI, so the operator can plan against
        the client's own file rather than tracing it by hand on the map.
        """
        from mission.boundary_import import BoundaryImportError, describe_boundary, read_boundary

        try:
            polygon = read_boundary(path)
        except BoundaryImportError as exc:
            # These messages are written for an operator, so pass them through rather
            # than replacing them with something generic.
            return fail(str(exc))

        self._session.aoi_polygon = [list(point) for point in polygon]
        summary = describe_boundary(polygon)
        self._session.audit("boundary_imported",
                            {"path": str(path), "points": summary["point_count"],
                             "area_hectares": summary["area_hectares"]})
        return ok(polygon=self._session.aoi_polygon, summary=summary)

    @guard
    def mission_estimates(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Batteries, storage and duration for the current plan."""
        from mission.estimates import AircraftProfile, estimate_mission

        plan = getattr(self._session, "mission_plan", None)
        if plan is None:
            return fail("Plan a mission before asking what it will cost to fly.")

        opts = dict(options or {})
        aircraft = AircraftProfile(
            name=str(opts.get("aircraft_name", "generic multirotor")),
            endurance_min=float(opts.get("endurance_min", 25.0)),
            cruise_speed_m_s=float(opts.get("cruise_speed_m_s", 8.0)),
            reserve_pct=float(opts.get("reserve_pct", 25.0)),
            batteries_owned=int(opts.get("batteries_owned", 0)),
        )
        return ok(estimates=estimate_mission(
            plan, aircraft=aircraft,
            image_format=str(opts.get("image_format", "jpeg")),
        ))

    @guard
    def plan_mission(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plan a mission over the drawn AOI with the operator's real constraints."""
        from mission.planner import MissionConstraints, MissionPlanner

        opts = dict(options or {})
        polygon = self._session.aoi_polygon
        if len(polygon) < 3:
            return fail("Draw an area of interest on the map before planning a mission.")

        constraints = MissionConstraints(
            geofence=[list(p) for p in polygon],
            min_altitude_m=float(opts.get("min_altitude_m", 20.0)),
            max_altitude_m=float(opts.get("max_altitude_m", 120.0)),
            standoff_m=float(opts.get("standoff_m", 0.0)),
            rth_altitude_m=float(opts.get("rth_altitude_m", 80.0)),
            no_fly_polygons=[list(map(list, ring)) for ring in self._session.no_fly_polygons],
            rth_action=str(opts.get("rth_action", "return_home")),
            obstacle_avoidance_profile=str(opts.get("obstacle_profile", "standard")),
        )

        kwargs: dict[str, Any] = {
            "polygon_lonlat": polygon,
            "altitude_m": float(opts.get("altitude_m", 55.0)),
            "front_overlap_pct": float(opts.get("front_overlap_pct", 75.0)),
            "side_overlap_pct": float(opts.get("side_overlap_pct", 65.0)),
            "speed_m_s": float(opts.get("speed_m_s", 8.0)),
            "mode": str(opts.get("template", "grid")),
            "gimbal_tilt_deg": float(opts.get("gimbal_tilt_deg", -90.0)),
            "inspection_dwell_s": float(opts.get("dwell_s", 0.0)),
            "wind_speed_m_s": float(opts.get("wind_speed_m_s", 0.0)),
            "wind_direction_deg": float(opts.get("wind_direction_deg", 0.0)),
            "wind_gust_m_s": float(opts.get("wind_gust_m_s", 0.0)),
            "constraints": constraints,
        }
        if opts.get("camera"):
            kwargs["camera"] = str(opts["camera"])
        if self._session.terrain_source_path:
            kwargs["terrain_follow_enabled"] = True
            kwargs["terrain_source_path"] = self._session.terrain_source_path
            kwargs["terrain_follow_mode"] = str(opts.get("terrain_follow_mode", "agl"))
            kwargs["terrain_normal_camera_enabled"] = bool(opts.get("terrain_normal_camera", False))
        if opts.get("orbit_radius_m"):
            kwargs["orbit_radius_m"] = float(opts["orbit_radius_m"])
            kwargs["orbit_center_lonlat"] = opts.get("orbit_center_lonlat") or _centroid(polygon)

        plan = MissionPlanner().generate(**kwargs)
        self._session.mission_plan = plan
        self._session.mission_plan_dict = _plan_to_dict(plan)
        self._session.audit("mission_planned", {"template": kwargs["mode"], "waypoints": len(plan.waypoints)})

        # The recipe carries the resolved terrain model under `metadata`; reading it
        # from the recipe root silently yielded None, so no terrain warning ever fired.
        recipe = self._session.mission_plan_dict.get("flight_recipe") or {}
        terrain = (recipe.get("metadata") or {}).get("terrain_model") or recipe.get("terrain_model") or {}
        warnings: list[str] = []
        if self._session.terrain_source_path and terrain.get("source") == "missing_terrain_source":
            warnings.append(
                "Terrain following was requested but the source could not be read; "
                "the plan assumes flat ground."
            )
        elif not self._session.terrain_source_path and terrain.get("type") == "flat":
            # Flat-earth planning is only safe over genuinely flat ground. Saying so
            # every time is the difference between an assumption and a silent one.
            warnings.append(
                "No terrain model loaded: altitudes are relative to a flat plane at the "
                "launch elevation. Over sloping ground the true height above surface will "
                "differ. Load a terrain source via File > Import Terrain to plan AGL."
            )
        if terrain.get("type") == "plane" and terrain.get("source") == "fitted_plane":
            warnings.append(
                "Terrain was approximated by a single fitted plane, so local relief "
                "inside the area is not represented."
            )

        # What is actually fitted decides what the aircraft is told at a capture point.
        # A LiDAR sent a shutter trigger flies the whole mission and lands with nothing.
        payload_block: dict[str, Any] | None = None
        if opts.get("payload"):
            from mission.payloads import UnknownPayload, get_payload, payload_plan_notes

            try:
                fitted = get_payload(str(opts["payload"]))
            except UnknownPayload as exc:
                return fail(str(exc))
            payload_block = {
                **fitted.to_dict(),
                "plan_notes": payload_plan_notes(fitted),
            }
            self._session.mission_plan_dict["payload"] = payload_block
            warnings.extend(payload_block["plan_notes"])

        # A terrain source that loads perfectly well but stops short of the area is the
        # case none of the checks above can see: the model is real, the plan is clean,
        # and the aircraft flies level over whatever the DEM does not reach.
        if self._session.terrain_source_path and terrain.get("source") != "missing_terrain_source":
            from core.terrain_cache import source_covers_area

            extent = source_covers_area(self._session.terrain_source_path, polygon)
            if not extent["covered"]:
                warnings.append(
                    f"Terrain source does not cover the planned area: {extent['detail']}"
                )

        return ok(
            summary={
                "template": kwargs["mode"],
                "waypoints": len(plan.waypoints),
                "distance_m": round(float(plan.path_distance_m), 1),
                "duration_min": round(float(plan.estimated_time_min), 1),
                "gsd_cm": round(float(plan.estimated_gsd_cm), 2),
                "altitude_m": float(plan.altitude_m),
                "adjustments": plan.safety_adjustments,
            },
            geojson=plan.geojson,
            warnings=warnings,
            payload=payload_block,
        )

    @guard
    def list_payloads(self) -> dict[str, Any]:
        """Every payload the engine can plan for, built-in and operator-defined."""
        from mission.payloads import list_payloads as read_payloads

        return ok(payloads=read_payloads())

    @guard
    def describe_payload(self, key: str) -> dict[str, Any]:
        """One payload, with what fitting it changes about the mission."""
        from mission.payloads import UnknownPayload, get_payload, payload_plan_notes

        try:
            profile = get_payload(key)
        except UnknownPayload as exc:
            return fail(str(exc))
        return ok(payload=profile.to_dict(), plan_notes=payload_plan_notes(profile))

    @guard
    def add_payload(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Describe a payload this operator actually owns."""
        from mission.payloads import PayloadProfile, save_user_profile

        fields = dict(spec or {})
        try:
            profile = PayloadProfile(
                key=str(fields.get("key", "")).strip().lower(),
                name=str(fields.get("name", "")),
                kind=str(fields.get("kind", "custom")),
                commands=tuple(fields.get("commands") or ()),
                mass_g=None if fields.get("mass_g") in (None, "") else float(fields["mass_g"]),
                power_w=None if fields.get("power_w") in (None, "") else float(fields["power_w"]),
                mount=str(fields.get("mount", "")),
                bands_nm=tuple(float(b) for b in (fields.get("bands_nm") or ())),
                continuous=bool(fields.get("continuous", False)),
                requires_calibration=bool(fields.get("requires_calibration", False)),
                source="user", notes=str(fields.get("notes", "")),
            )
        except (TypeError, ValueError) as exc:
            return fail(str(exc))

        try:
            store = save_user_profile(profile)
        except ValueError as exc:
            return fail(str(exc))
        self._session.audit("payload_added", {"key": profile.key, "kind": profile.kind})
        return ok(payload=profile.to_dict(), stored_at=str(store))

    @guard
    def mission_geojson(self) -> dict[str, Any]:
        if not self._session.mission_plan_dict:
            return fail("No mission has been planned yet.")
        return ok(geojson=self._session.mission_plan_dict.get("geojson", {}))

    @guard
    def export_mission(self, formats: list[str] | None = None, directory: str = "") -> dict[str, Any]:
        from mission import exporters

        if not self._session.mission_plan_dict:
            return fail("No mission has been planned yet.")
        target = Path(directory) if directory else self._session.project_root() / "missions"
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"mission_{stamp}"

        selected = formats or list(exporters.EXPORTERS.keys())
        written: dict[str, str] = {}
        for name in selected:
            entry = exporters.EXPORTERS.get(name)
            if entry is None:
                written[name] = f"failed: unknown format {name}"
                continue
            writer, suffix = entry
            try:
                written[name] = writer(target / f"{stem}{suffix}", self._session.mission_plan_dict)
            except Exception as exc:  # noqa: BLE001
                written[name] = f"failed: {exc}"
        self._session.audit("mission_exported", {"formats": selected, "directory": str(target)})
        return ok(written=written, directory=str(target))

    @guard
    def save_mission(self, name: str = "", note: str = "") -> dict[str, Any]:
        if not self._session.mission_plan_dict:
            return fail("No mission has been planned yet.")
        plan = self._session.mission_plan_dict
        entry = self._session.store.save_mission_version(
            project_id=self._session.project_id(),
            mission_name=name or "mission",
            template=str(plan.get("template", "grid")),
            flight_recipe=plan.get("flight_recipe") or {},
            plan_summary={
                "waypoints": len(plan.get("waypoints") or []),
                "altitude_m": plan.get("altitude_m"),
                "path_distance_m": plan.get("path_distance_m"),
                "estimated_time_min": plan.get("estimated_time_min"),
                "estimated_gsd_cm": plan.get("estimated_gsd_cm"),
                "safety_constraints": plan.get("safety_constraints") or {},
                "safety_adjustments": plan.get("safety_adjustments") or {},
                "geojson": plan.get("geojson") or {},
            },
            note=note,
        )
        return ok(version=entry)

    @guard
    def list_mission_versions(self, name: str = "") -> dict[str, Any]:
        return ok(
            versions=self._session.store.list_mission_versions(self._session.project_id(), name or None)
        )

    @guard
    def mission_version_history(self, name: str = "") -> dict[str, Any]:
        """Every saved version with what changed from the one before it."""
        from mission.versioning import version_history

        versions = self._session.store.list_mission_versions(
            self._session.project_id(), name or None)
        return ok(history=version_history(versions))

    @guard
    def diff_mission_versions(self, from_version: int, to_version: int,
                              name: str = "") -> dict[str, Any]:
        """Compare two saved versions of a mission."""
        from mission.versioning import diff_versions

        versions = self._session.store.list_mission_versions(
            self._session.project_id(), name or None)
        by_num = {int(v.get("version_num", 0)): v for v in versions}

        older, newer = by_num.get(int(from_version)), by_num.get(int(to_version))
        missing = [n for n, v in ((from_version, older), (to_version, newer)) if v is None]
        if missing:
            available = ", ".join(str(n) for n in sorted(by_num)) or "none"
            return fail(f"No such version(s): {missing}. Saved versions: {available}.")

        return ok(diff=diff_versions(older, newer).to_dict())

    @guard
    def restore_mission_version(self, version_num: int, name: str = "",
                                note: str = "") -> dict[str, Any]:
        """Reinstate an earlier version by saving it again as the newest.

        Nothing is deleted: the versions in between remain on the record, because an
        audit trail that can be rewritten is not one.
        """
        from mission.versioning import restore_version

        versions = self._session.store.list_mission_versions(
            self._session.project_id(), name or None)
        source = next((v for v in versions
                       if int(v.get("version_num", 0)) == int(version_num)), None)
        if source is None:
            available = ", ".join(str(v.get("version_num")) for v in versions) or "none"
            return fail(f"No version {version_num}. Saved versions: {available}.")

        entry = restore_version(self._session.store, self._session.project_id(),
                                source, note=note)
        self._session.audit("mission_version_restored",
                            {"from_version": version_num,
                             "new_version": entry.get("version_num")})
        return ok(version=entry)

    # -- vehicle ---------------------------------------------------------

    @guard
    def connect_vehicle(self, uri: str, driver: str = "mavlink") -> dict[str, Any]:
        return ok(vehicle=self._session.connect_vehicle(uri, driver))

    @guard
    def disconnect_vehicle(self) -> dict[str, Any]:
        self._session.disconnect_vehicle()
        return ok(vehicle=self._session.vehicle.to_dict())

    @guard
    def telemetry(self) -> dict[str, Any]:
        return ok(telemetry=self._session.telemetry())

    @guard
    def upload_mission(self) -> dict[str, Any]:
        """Upload the planned mission, geofence, and rally points to the vehicle."""
        from mission import exporters

        if not self._session.mission_plan_dict:
            return fail("No mission has been planned yet.")
        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")
        uploader = getattr(client, "upload_mission", None)
        if uploader is None:
            return fail(f"The {self._session.vehicle.driver!r} driver does not support mission upload.")

        items = [item.to_dict() for item in exporters.build_mission_items(self._session.mission_plan_dict)]
        result = uploader(items)
        self._session.audit("mission_uploaded", {"items": len(items), "driver": self._session.vehicle.driver})

        # Recorded so that, if this software dies before the mission ends, the next
        # session knows a mission was on the aircraft rather than guessing.
        from core.flight_state import PHASE_UPLOADED, record_transition

        record_transition(
            self._session.project_root(), PHASE_UPLOADED,
            mission_name=str(self._session.mission_plan_dict.get("template", "")),
            vehicle_driver=self._session.vehicle.driver,
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else {"result": str(result)}
        return ok(items=len(items), result=payload)

    @guard
    def vehicle_command(self, command: str, **kwargs: Any) -> dict[str, Any]:
        """Arm / start / pause / resume / RTL / abort against the connected vehicle."""
        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")
        # The command set a ground station is expected to offer. Anything the
        # connected driver does not implement is reported as unsupported rather than
        # failing silently, because a pilot pressing a button needs to know whether
        # the aircraft heard it.
        mapping = {
            "start": "start_mission",
            "pause": "pause_mission",
            "resume": "resume_mission",
            "rth": "return_to_home",
            "abort": "abort_mission",
            "arm": "arm",
            "disarm": "disarm",
            "takeoff": "takeoff",
            "land": "land",
            "loiter": "loiter",
            "emergency_stop": "emergency_stop",
            "set_home": "set_home",
        }
        method_name = mapping.get(str(command).lower())
        if method_name is None:
            return fail(
                f"Unknown command {command!r}. Available: {', '.join(sorted(mapping))}."
            )
        method = getattr(client, method_name, None)
        if method is None:
            return fail(f"The {self._session.vehicle.driver!r} driver does not support {command!r}.")

        # Takeoff needs a target altitude; the rest take none.
        if method_name == "takeoff":
            altitude = float(kwargs.get("altitude_m", 0.0) or 0.0)
            if altitude <= 0:
                return fail("Takeoff requires a positive altitude_m.")
            result = method(altitude)
        else:
            result = method()
        self._session.audit("vehicle_command", {"command": command, "driver": self._session.vehicle.driver})
        return ok(result=result.to_dict() if hasattr(result, "to_dict") else str(result))

    # -- datasets and processing ----------------------------------------

    @guard
    def import_dataset(self, folder: str, name: str = "") -> dict[str, Any]:
        return ok(dataset=self._session.import_dataset(folder, name))

    @guard
    def list_datasets(self) -> dict[str, Any]:
        return ok(datasets=self._session.list_datasets())

    @guard
    def run_reconstruction(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start a COLMAP reconstruction as a background job."""
        opts = dict(options or {})
        images = opts.get("image_dir") or self._session.active_dataset_dir
        if not images:
            return fail("Select a dataset before running reconstruction.")
        if not Path(images).exists():
            return fail(f"Image folder not found: {images}")

        output = Path(opts.get("output_dir") or (self._session.project_root() / "reconstruction"))
        session = self._session

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from core.reconstruction_colmap import build_reconstructor

            reconstructor = build_reconstructor(
                opts.get("engine", "auto"),
                profile=opts.get("profile", "standard"),
                dense=opts.get("dense"),
                target_epsg=opts.get("epsg"),
            )
            result = reconstructor.reconstruct(
                image_dir=images, output_dir=str(output), progress_callback=progress
            )
            payload = result.to_dict()
            # Reconstruction products become map layers immediately, which is the whole
            # point of doing this inside a GIS tool.
            for key, kind, label in (
                ("orthomosaic_cog_path", "raster", "Orthomosaic"),
                ("dsm_cog_path", "raster", "DSM"),
                ("dtm_cog_path", "raster", "DTM"),
                ("hillshade_path", "raster", "DSM hillshade"),
            ):
                path = payload.get(key)
                if path and Path(path).exists():
                    session.register_raster_layer(path, name=label, kind=kind)
            track = payload.get("camera_track_geojson_path")
            if track and Path(track).exists():
                session.register_vector_layer(track, name="Camera positions")
            session.audit("reconstruction_complete", {"engine": payload.get("engine"), "output": str(output)})
            return payload

        job_id = self._jobs.submit("reconstruction", work)
        return ok(job_id=job_id)

    @guard
    def run_pipeline(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start the full inspection pipeline as a background job."""
        opts = dict(options or {})
        images = opts.get("image_dir") or self._session.active_dataset_dir
        if not images:
            return fail("Select a dataset before running the pipeline.")
        output = str(opts.get("output_dir") or (self._session.project_root() / "analysis"))

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from core.pipeline import PipelineConfig, StructuralFaultPipeline

            config = PipelineConfig()
            config.reconstruction_engine = opts.get("engine", "auto")
            config.reconstruction_profile = opts.get("profile", "standard")
            pipeline = StructuralFaultPipeline(config)
            result = pipeline.run(image_dir=images, output_root=output, progress_callback=progress)
            return result.to_dict() if hasattr(result, "to_dict") else dict(result)

        return ok(job_id=self._jobs.submit("pipeline", work))

    @guard
    def set_active_dataset(self, folder: str) -> dict[str, Any]:
        if folder and not Path(folder).exists():
            return fail(f"Folder not found: {folder}")
        self._session.active_dataset_dir = folder
        return ok(active_dataset=folder)

    # -- jobs ------------------------------------------------------------

    @guard
    def job_status(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        return ok(job=job) if job else fail(f"Unknown job: {job_id}")

    @guard
    def list_jobs(self) -> dict[str, Any]:
        return ok(jobs=self._jobs.list())

    @guard
    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return ok(cancelled=self._jobs.cancel(job_id))

    # -- files -----------------------------------------------------------

    @guard
    def pick_folder(self) -> dict[str, Any]:
        import webview

        if self._window is None:
            return fail("Window is not ready.")
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        return ok(path=result[0] if result else "")

    @guard
    def pick_file(self, extensions: list[str] | None = None) -> dict[str, Any]:
        """Open the native file chooser, filtered to the extensions the caller wants.

        pywebview's `file_types` are display strings -- "Rasters (*.tif;*.tiff)" -- not
        extensions. Passing the bare list the UI sends made it raise
        `ValueError: tif is not a valid file filter`, and the exception travelled all the
        way to the toolbar as the result of pressing the button. Every layer-import
        control was dead in exactly that way.

        So the conversion happens here, at the boundary between what the UI naturally has
        and what the toolkit demands, rather than being every caller's problem.
        """
        import webview

        if self._window is None:
            return fail("Window is not ready.")

        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG, file_types=_file_type_filters(extensions)
        )
        return ok(path=result[0] if result else "")

    @guard
    def open_path(self, path: str) -> dict[str, Any]:
        """Reveal a file or folder in the operating system's file manager."""
        target = Path(path)
        if not target.exists():
            return fail(f"Path not found: {path}")
        if platform.system() == "Windows":
            os.startfile(str(target))  # noqa: S606 - intentional shell-less open
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(target)], check=False)
        else:
            subprocess.run(["xdg-open", str(target)], check=False)
        return ok(opened=str(target))


#: Extensions the UI asks for, grouped the way a person picking a file thinks about them.
_FILE_TYPE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Rasters", ("tif", "tiff", "img", "vrt")),
    ("Vectors", ("geojson", "json", "shp", "gpkg", "kml")),
    ("Imagery", ("jpg", "jpeg", "png")),
    ("Point clouds", ("las", "laz", "ply")),
)


def _file_type_filters(extensions: list[str] | None) -> tuple[str, ...]:
    """Turn bare extensions into the display strings pywebview requires.

    pywebview validates `file_types` against "Description (*.ext;*.ext)" and raises
    ValueError on anything else -- which is how "tif" became a toolbar error message.

    Requested extensions are grouped under the labels above so the chooser reads like a
    file chooser rather than a list of suffixes, and anything unrecognised still gets an
    entry of its own rather than being dropped: silently narrowing what a user is allowed
    to open is worse than an ugly label.
    """
    wanted = [str(e).strip().lstrip("*.").lower() for e in (extensions or []) if str(e).strip()]
    if not wanted:
        return ("All files (*.*)",)

    filters: list[str] = []
    claimed: set[str] = set()
    for label, known in _FILE_TYPE_GROUPS:
        present = [e for e in wanted if e in known and e not in claimed]
        if present:
            filters.append(f"{label} ({';'.join('*.' + e for e in present)})")
            claimed.update(present)

    for extension in wanted:
        if extension not in claimed:
            filters.append(f"{extension.upper()} files (*.{extension})")
            claimed.add(extension)

    # Every supported type in one entry, so a user with a mixed folder is not forced to
    # cycle the dropdown to find their file.
    if len(filters) > 1:
        combined = ";".join(f"*.{e}" for e in wanted)
        filters.insert(0, f"Supported files ({combined})")
    filters.append("All files (*.*)")
    return tuple(filters)


def _centroid(ring: list[list[float]]) -> list[float]:
    return [sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)]


def _plan_to_dict(plan: Any) -> dict[str, Any]:
    """Serialize a MissionPlan dataclass without depending on a to_dict method."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(plan):
        return asdict(plan)
    if isinstance(plan, dict):
        return plan
    return {key: getattr(plan, key) for key in dir(plan) if not key.startswith("_") and not callable(getattr(plan, key))}
