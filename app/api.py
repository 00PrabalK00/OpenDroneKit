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
import time
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable

from .jobs import JobManager
from .session import AppSession, MapLayer

#: The largest area of interest one mission is planned over. Beyond this the planner runs
#: for minutes to hours; see set_aoi.
MAX_AOI_AREA_M2 = 50_000_000.0


def ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def fail(error: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "error": str(error), **payload}


def _optional_float(value: Any) -> float | None:
    """A number from the UI, where an empty field means "not supplied"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Convert exceptions into structured failures the UI can display."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ModuleNotFoundError as exc:
            # Features import their packages when used, so this is where a package the
            # launcher's environment lacks shows up. Say which one and how to add it.
            missing = (exc.name or "").split(".")[0]
            if not missing or (Path(__file__).resolve().parents[1] / missing).exists():
                return fail(f"{type(exc).__name__}: {exc}")
            return fail(
                f"{missing} is not installed in the Python running OpenDroneKit "
                f"({sys.executable}), so this feature cannot run. Install it with: "
                f"\"{sys.executable}\" -m pip install -r requirements.txt  (or run "
                "tools/check_environment.py --install)."
            )
        except Exception as exc:  # noqa: BLE001
            return fail(f"{type(exc).__name__}: {exc}")

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _data_paths() -> dict[str, str]:
    """Where this install keeps its state (core/paths.py), for the UI to show."""
    from core.paths import describe

    return describe()


def _register_reconstruction_layers(session: Any, payload: dict[str, Any]) -> list[str]:
    """Add a reconstruction's georeferenced products to the layer tree.

    Only files that exist are registered, so a run that produced no DSM adds no DSM
    layer. Returns the names that were added.
    """
    added: list[str] = []
    for key, kind, label in (
        ("orthomosaic_cog_path", "raster", "Orthomosaic"),
        ("dsm_cog_path", "raster", "DSM"),
        ("dtm_cog_path", "raster", "DTM"),
        ("hillshade_path", "raster", "DSM hillshade"),
    ):
        path = payload.get(key)
        if path and Path(path).exists():
            session.register_raster_layer(path, name=label, kind=kind)
            added.append(label)
    track = payload.get("camera_track_geojson_path")
    if track and Path(track).exists():
        session.register_vector_layer(track, name="Camera positions")
        added.append("Camera positions")
    return added


class Api:
    """Bridge object bound to the webview window."""

    def __init__(self, session: AppSession | None = None):
        self._session = session or AppSession()
        self._jobs = JobManager()
        # A reconstruction runs for minutes and the operator has switched workspace long
        # before it lands. Without this the result appears in a panel nobody is looking
        # at, and they cannot tell whether it finished two seconds or forty minutes ago.
        self._jobs.on_finished(self._session.announce_finished_job)
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
    def rest_endpoints(self) -> dict[str, Any]:
        """The REST service's routes, read from its source rather than written down.

        The Developers screen listed nine endpoints by hand -- GET /projects, GET
        /findings, GET /processing/jobs -- and most of them did not exist; the real routes
        are /organizations/{id}/projects, /projects/{id}/defects and /projects/{id}/jobs.
        Reading the decorators means the list is the service's own, and cannot drift.
        The service is a separate process (uvicorn services.api.main:app); nothing here
        claims it is running.
        """
        import ast

        root = Path(__file__).resolve().parents[1] / "services" / "api"
        main = root / "main.py"
        if not main.exists():
            return fail("The REST service (services/api) is not part of this installation.")

        def routes_in(path: Path, owner: str) -> tuple[str, list[dict[str, Any]]]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            prefix = ""
            found: list[dict[str, Any]] = []
            for node in ast.walk(tree):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and getattr(node.value.func, "id", "") == "APIRouter"):
                    for kw in node.value.keywords:
                        if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                            prefix = str(kw.value.value)
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for deco in node.decorator_list:
                    if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
                        continue
                    verb = deco.func.attr.upper()
                    target = getattr(deco.func.value, "id", "")
                    if target != owner or verb not in {"GET", "POST", "PUT", "PATCH", "DELETE", "WEBSOCKET"}:
                        continue
                    if not deco.args or not isinstance(deco.args[0], ast.Constant):
                        continue
                    # Pages such as the API docs, not endpoints of the REST API.
                    if any(kw.arg == "include_in_schema" and isinstance(kw.value, ast.Constant)
                           and kw.value.value is False for kw in deco.keywords):
                        continue
                    doc =(ast.get_docstring(node) or "").strip().splitlines()
                    found.append({"method": verb, "path": str(deco.args[0].value),
                                  "summary": doc[0] if doc else "", "handler": node.name})
            return prefix, found

        endpoints: list[dict[str, Any]] = []
        _, app_routes = routes_in(main, "app")
        for route in app_routes:
            endpoints.append({**route, "router": "main"})
        for path in sorted((root / "routers").glob("*.py")):
            prefix, found = routes_in(path, "router")
            for route in found:
                endpoints.append({**route, "path": prefix + route["path"], "router": path.stem})
        return ok(endpoints=endpoints, count=len(endpoints),
                  started_with="uvicorn services.api.main:app",
                  note="Routes of the separate REST service. This desktop app does not start it.")

    @guard
    def rest_request(self, path: str, base_url: str = "http://127.0.0.1:8000") -> dict[str, Any]:
        """Send one GET to the REST service on this machine and report what came back.

        Loopback only, and GET only: this is the Developers screen's "does the service
        answer, and with what", not a general HTTP client -- nothing leaves the machine.
        A refused connection is reported as the service not running, which is the usual
        and truthful answer on a desktop install.
        """
        from urllib import error, parse, request

        base = parse.urlparse(str(base_url))
        if base.scheme != "http" or base.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return fail("Send Request only talks to a service on this machine (127.0.0.1).")
        if not str(path).startswith("/"):
            return fail(f"Not a route path: {path!r}")
        url = f"{base.scheme}://{base.netloc}{path}"
        try:
            with request.urlopen(request.Request(url, method="GET"), timeout=5) as response:
                body = response.read(4000).decode("utf-8", "replace")
                return ok(url=url, status=response.status, body=body)
        except error.HTTPError as exc:
            body = exc.read(4000).decode("utf-8", "replace") if exc.fp else ""
            return ok(url=url, status=exc.code, body=body)
        except (error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            return fail(f"The REST service is not answering at {base.netloc} ({reason}). "
                        "Start it with: uvicorn services.api.main:app")

    @guard
    def model_status(self) -> dict[str, Any]:
        """Which detection models are present on disk, so the UI never implies AI it lacks.

        The registry is found through core.models rather than a path relative to the
        working directory: launched from a shortcut, the relative path pointed nowhere
        and the app reported "0 of 0" models on a machine with nine installed.

        `detectors` answers the question a user actually has -- for each detector the
        pipeline runs, is it a trained model or a heuristic -- because one global count
        hid that solar and metal detection were heuristic on every machine.
        """
        from core.model_runner import detector_backends
        from core.models import load_registry, model_status as status_for

        keys = list((load_registry().get("models") or {}).keys())
        rows = []
        for key in keys:
            try:
                info = status_for(key)
                rows.append({"key": key, **(info if isinstance(info, dict) else {"status": str(info)})})
            except Exception as exc:  # noqa: BLE001
                rows.append({"key": key, "exists": False, "error": str(exc)})
        available = sum(1 for r in rows if r.get("exists"))
        detectors = detector_backends()
        heuristic = [d["detector"] for d in detectors if d["backend"] == "heuristic"]
        return ok(models=rows, available=available, total=len(rows),
                  detectors=detectors, heuristic_detectors=heuristic)

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
        # Names are unique in the store. Checked here so the operator reads which name is
        # taken, rather than "IntegrityError: UNIQUE constraint failed: projects.name".
        wanted = name.strip()
        if any(str(p.get("name", "")).casefold() == wanted.casefold()
               for p in self._session.store.list_projects()):
            return fail(f"A project named {wanted!r} already exists. Open it, or choose another name.")
        return ok(project=self._session.create_project(wanted, root_dir, description))

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
        from core import geo

        # Clearing is a real request (Mission > Clear AOI sends null), and iterating None
        # raised a TypeError at the operator instead of clearing anything.
        if not polygon_lonlat:
            self._session.aoi_polygon = []
            return ok(vertices=0, area_m2=0.0, cleared=True)
        ring = [[float(p[0]), float(p[1])] for p in polygon_lonlat if len(p) >= 2]
        if len(ring) < 3:
            return fail("An area of interest needs at least 3 vertices.")
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        # An area drawn with the map zoomed out to the world was 2.7 billion hectares,
        # and plan_mission then ran until the process had taken 6.4 GB and never came
        # back. Planning time grows faster than the area (33 km2 takes ~14 s here), so the
        # refusal comes now, while the operator is still drawing, not after a hang.
        area_m2 = geo.polygon_area_m2(ring)
        if area_m2 > MAX_AOI_AREA_M2:
            return fail(
                f"That area is {area_m2 / 1e6:,.1f} km2, larger than one mission can cover "
                f"(limit {MAX_AOI_AREA_M2 / 1e6:,.0f} km2). Zoom in and draw the site, or "
                "split it into several areas.")
        self._session.aoi_polygon = ring

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
    def measure_geometry(self, coordinates_lonlat: list[list[float]],
                         kind: str = "distance") -> dict[str, Any]:
        """Length or area of a shape drawn on the map, in metres on the ground.

        The cockpit's Distance, Area and Measure tools armed themselves and then nothing
        read the clicks, so no measurement was ever taken. The drawn vertices come here
        rather than being summed in the browser so the map shares one geodesy with the
        rest of the toolkit (core.geo).

        These are horizontal distances between the clicked positions. A drawn shape has no
        heights, so a slope length or a surface area needs the DSM, and `method` says
        which this is.
        """
        from core import geo

        points = [[float(p[0]), float(p[1])] for p in coordinates_lonlat or [] if len(p) >= 2]
        for lon, lat in points:
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                return fail(f"Not a longitude/latitude pair: {lon}, {lat}.")

        def path_length(path: list[list[float]]) -> float:
            return sum(geo.haversine_m(a[1], a[0], b[1], b[0]) for a, b in zip(path, path[1:]))

        if kind == "distance":
            if len(points) < 2:
                return fail("A distance needs at least two points.")
            segments = [round(geo.haversine_m(a[1], a[0], b[1], b[0]), 3)
                        for a, b in zip(points, points[1:])]
            return ok(kind="distance", length_m=round(path_length(points), 3),
                      segments_m=segments, vertices=len(points),
                      method="horizontal great-circle distance between the drawn positions")
        if kind == "area":
            ring = points[:-1] if len(points) > 3 and points[0] == points[-1] else points
            if len(ring) < 3:
                return fail("An area needs at least three points.")
            return ok(kind="area", area_m2=round(geo.polygon_area_m2(ring), 3),
                      perimeter_m=round(path_length(ring + [ring[0]]), 3),
                      vertices=len(ring),
                      method="planimetric area of the drawn polygon")
        return fail(f"Unknown measurement kind {kind!r}. Use distance or area.")

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
        from core.geocoding import Place, search_places as run_search

        # What the open project already puts on the map is searchable with no network:
        # a layer by its name, the area of interest as "Area of interest".
        local: list[Place] = []
        for layer in self._session.layer_list():
            b = layer.get("bounds_lonlat") or []
            if len(b) == 4:
                local.append(Place(name=str(layer.get("name") or layer.get("id")),
                                   longitude=(b[0] + b[2]) / 2, latitude=(b[1] + b[3]) / 2,
                                   kind=f"{layer.get('kind', 'layer')} layer",
                                   bounds=(b[0], b[1], b[2], b[3]), source="project"))
        ring = self._session.aoi_polygon or []
        if len(ring) >= 3:
            lons, lats = [p[0] for p in ring], [p[1] for p in ring]
            local.append(Place(name="Area of interest", longitude=sum(lons) / len(lons),
                               latitude=sum(lats) / len(lats), kind="area of interest",
                               bounds=(min(lons), min(lats), max(lons), max(lats)),
                               source="project"))

        selected = provider or getattr(self._session, "geocoding_provider", "") or "nominatim"
        result = run_search(
            query,
            provider=selected,
            gazetteer_path=self._session.project_root() / "places.json",
            local_places=local,
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

    # -- offline basemap ----------------------------------------------------

    @guard
    def basemap_cache_status(self) -> dict[str, Any]:
        """What the offline basemap can actually show: tiles on disk per provider.

        The desktop shell serves these itself (tiles/{z}/{x}/{y}.png on its loopback
        origin), so the "Offline cache" basemap works with no Hub and no network.
        """
        from core.tile_cache import MAX_TILES_PER_REQUEST, PROVIDERS, summary

        return ok(
            **summary(),
            available_providers={k: {"max_zoom": v["max_zoom"], "attribution": v["attribution"]}
                                 for k, v in PROVIDERS.items()},
            max_tiles_per_request=MAX_TILES_PER_REQUEST,
        )

    @guard
    def cache_basemap_area(self, west: float, south: float, east: float, north: float,
                           min_zoom: int = 12, max_zoom: int = 17,
                           provider: str = "satellite") -> dict[str, Any]:
        """Download basemap tiles for an area so the map works on a site with no signal.

        Explicit only: panning never downloads anything. The request is validated (zoom
        ceiling, tile budget) before anything is fetched, and runs as a job reporting
        how many tiles arrived, were already cached, or failed -- a partial cache is
        reported as partial.
        """
        from core.tile_cache import Area, TileCacheRefused, fetch_area, validate

        area = Area(provider=str(provider or "satellite"), west=float(west), south=float(south),
                    east=float(east), north=float(north), min_zoom=int(min_zoom),
                    max_zoom=int(max_zoom))
        try:
            total = validate(area)
        except TileCacheRefused as exc:
            return fail(str(exc))

        session = self._session

        def work(progress: Callable[..., None], should_cancel: Callable[[], bool]) -> dict[str, Any]:
            seen = {"n": 0}

            def on_tile(_success: bool) -> None:
                seen["n"] += 1
                if seen["n"] % 10 == 0 or seen["n"] == total:
                    progress(int(100 * seen["n"] / max(1, total)), f"{seen['n']}/{total} tiles")

            counts = fetch_area(area, on_tile=on_tile, should_cancel=should_cancel)
            session.audit("basemap_cached", {"provider": area.provider, "tiles": total, **counts})
            return {"provider": area.provider, "requested": total, **counts}

        return ok(job_id=self._jobs.submit("basemap_cache", work), tiles=total, provider=area.provider)

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
            display_epsg=self._session.display_epsg,
            data_paths=_data_paths(),
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
        from mission.mission_types import SPECIAL_MISSION_TYPES
        from mission.planner import (
            MissionPlanner, TEMPLATE_LABELS, available_templates, default_gimbal_for_template,
        )

        templates = available_templates()
        return ok(
            templates=templates,
            # Planned by mission/mission_types.py rather than one planner template, and
            # reached through plan_mission with the same template field.
            special_templates=list(SPECIAL_MISSION_TYPES),
            labels={t: TEMPLATE_LABELS.get(t, t.replace("_", " ").title())
                    for t in [*templates, *SPECIAL_MISSION_TYPES]},
            default_gimbal_deg={t: default_gimbal_for_template(t) for t in templates},
            planner=MissionPlanner.__name__,
        )

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

    def _control_points(self) -> list:
        """This project's control points, read from its folder (the only copy that lasts)."""
        from core.gcp import load_control

        points = load_control(self._session.project_root())
        self._session.ground_control = points
        return points

    @guard
    def import_gcps(self, path: str, epsg: int | None = None) -> dict[str, Any]:
        """Load surveyed ground control points for this project.

        Written into the project folder with every mark made on them. They used to live
        only in memory, so an afternoon of marking targets was lost on restart.
        """
        from core.gcp import GcpError, read_gcp_file, save_control

        try:
            points = read_gcp_file(path, default_epsg=epsg)
        except GcpError as exc:
            return fail(str(exc))

        # Re-importing the same file keeps the marks already made on points it still has.
        previous = {p.name: p for p in self._control_points()}
        for point in points:
            if point.name in previous:
                point.marks = previous[point.name].marks
        self._session.ground_control = points
        saved = save_control(self._session.project_root(), points)
        self._session.audit("gcps_imported", {"path": str(path), "count": len(points)})
        return ok(points=[p.to_dict() for p in points], count=len(points), path=str(saved))

    @guard
    def list_gcps(self) -> dict[str, Any]:
        """The project's control points with their marks."""
        points = self._control_points()
        return ok(points=[{**p.to_dict(), "marks": [m.to_dict() for m in p.marks]}
                          for p in points], count=len(points))

    @guard
    def mark_gcp(self, name: str, image: str, pixel_x: float, pixel_y: float,
                 image_width: int = 0, image_height: int = 0) -> dict[str, Any]:
        """Record where a control point appears in one image.

        Pixels are full-resolution unless ``image_width`` x ``image_height`` says which
        picture they were read on (a downscaled preview, say); they are rescaled then.
        """
        from core.gcp import add_mark, save_control

        points = self._control_points()
        point = next((p for p in points if p.name == name), None)
        if point is None:
            known = ", ".join(p.name for p in points) or "none imported"
            return fail(f"No control point named {name!r}. Imported: {known}.")
        if image_width and image_height and not (
                0 <= float(pixel_x) <= float(image_width)
                and 0 <= float(pixel_y) <= float(image_height)):
            return fail(f"({float(pixel_x):.0f}, {float(pixel_y):.0f}) is outside the "
                        f"{int(image_width)}x{int(image_height)} image.")

        add_mark(point, Path(str(image)).name, float(pixel_x), float(pixel_y),
                 int(image_width or 0), int(image_height or 0))
        save_control(self._session.project_root(), points)
        self._session.audit("gcp_marked", {"name": name, "image": Path(str(image)).name})
        return ok(point=point.to_dict())

    @guard
    def gcp_accuracy_report(self, computed: dict[str, list[float]] | None = None,
                            reconstruction_dir: str = "") -> dict[str, Any]:
        """How far each control point landed from where it was surveyed.

        Without ``computed`` the positions come from the model: each point is located by
        intersecting the rays through its image marks with the reconstruction's own
        camera poses, and placed with the georeferencing that run actually used. When
        that georeferencing was fitted to these points the residuals are fit residuals
        (with the leave-one-out check the run wrote); when it came from camera geotags
        they are an independent check.
        """
        import json as _json

        from core import geo
        from core.gcp import (GcpError, accuracy_report, check_against_anchor,
                              poses_from_records, residuals_from_positions, write_report)

        points = self._control_points()
        if not points:
            return fail("No ground control points have been imported.")

        root = self._session.project_root()
        if computed:
            placed = {k: tuple(v) for k, v in computed.items() if len(v) >= 3}
            report = accuracy_report(points, residuals_from_positions(points, placed))
        else:
            run = Path(reconstruction_dir) if reconstruction_dir else root / "reconstruction"
            poses_path = run / "camera_poses.json"
            anchor_path = run / "geo_anchor.json"
            if not poses_path.is_file():
                return fail(f"No reconstruction at {run}: run Process first, then the "
                            "control points can be located in it.")
            if not anchor_path.is_file():
                return fail("That reconstruction is not georeferenced, so there is no "
                            "position to compare the control with.")
            anchor = geo.GeoAnchor(**_json.loads(anchor_path.read_text(encoding="utf-8")))
            poses = poses_from_records(
                _json.loads(poses_path.read_text(encoding="utf-8")).get("cameras", []))
            fitted = anchor.method == "ground_control"
            try:
                report = check_against_anchor(points, poses, anchor,
                                              role="fit" if fitted else "check")
            except GcpError as exc:
                return fail(str(exc))
            report["georeferenced_by"] = anchor.method
            report["reconstruction"] = str(run)
            usable = sum(1 for p in points if p.is_usable)
            if not fitted and usable >= 3:
                report["warnings"].append(
                    f"{usable} control points are now marked well enough to georeference "
                    "the model; run Process again to fit it to them."
                )
            previous = run / "gcp_report.json"
            if fitted and previous.is_file():
                fit_report = _json.loads(previous.read_text(encoding="utf-8"))
                if "leave_one_out" in fit_report:
                    report["leave_one_out"] = fit_report["leave_one_out"]

        target = root / "gcp_report.json"
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

    # -- capture verification ---------------------------------------------

    @guard
    def match_captures(self, match_radius_m: float = 15.0, image_folder: str = "") -> dict[str, Any]:
        """Match the flown images to the planned capture points.

        The Verification screen's Match Captures button counted saved mission versions;
        core.capture_matching, which does the actual comparison, had no route from the
        interface. This runs it on the planned mission and the selected dataset and keeps
        the report, so the screen and Request Reflight can read the same answer.
        """
        from core.capture_matching import verify_capture

        plan = self._session.mission_plan_dict
        if not plan:
            return fail("No mission is planned in this session. Plan it, or restore the "
                        "version that was flown from History, then match again.")
        folder = image_folder or self._session.active_dataset_dir
        if not folder:
            return fail("Select the flight's images first: Load Flight, or Import.")
        if not Path(folder).is_dir():
            return fail(f"Image folder not found: {folder}")
        target = self._session.project_root() / "verification" / "capture_report.json"
        report = verify_capture(plan, folder, match_radius_m=float(match_radius_m),
                                output_path=target)
        if not report.get("ok"):
            return fail(report.get("error", "Capture matching failed."))
        report["image_folder"] = str(folder)
        self._session.capture_report = report
        self._session.audit("captures_matched", {
            "planned": report["planned_total"], "matched": report["matched"],
            "missed": len(report["missed"]), "folder": str(folder)})
        return ok(**{k: v for k, v in report.items() if k != "ok"})

    @guard
    def capture_report(self) -> dict[str, Any]:
        """The last capture match in this session, or a refusal saying none was run."""
        report = getattr(self._session, "capture_report", None)
        if not report:
            return fail("No capture match has been run. Use Match Captures.")
        return ok(**{k: v for k, v in report.items() if k != "ok"})

    @guard
    def request_reflight(self, reason: str = "") -> dict[str, Any]:
        """Turn the missed capture points into a reflight layer on the map.

        This used to log a maintenance record against aircraft 1, which is not a
        reflight and fails on any install without that aircraft. A reflight is the set of
        planned points that produced no image: they are written as GeoJSON in the project,
        added as a layer, and recorded in the audit log with the reason.
        """
        from core import geo

        report = getattr(self._session, "capture_report", None)
        if not report:
            return fail("Run Match Captures first: a reflight is the planned points that "
                        "produced no image, and none have been found yet.")
        missed = report.get("missed") or []
        if not missed:
            return fail("Every planned capture point was matched to an image. "
                        "Nothing needs reflying.")
        features = [
            geo.point_feature(float(m["lon"]), float(m["lat"]),
                              {"planned_index": m["index"], "reason": reason or ""},
                              alt=m.get("alt_m"))
            for m in missed
        ]
        target = self._session.project_root() / "verification" / "reflight_points.geojson"
        geo.write_geojson(target, features)
        layer = self._session.register_vector_layer(target, name="Reflight points")
        self._session.audit("reflight_requested", {"points": len(missed), "reason": reason})
        return ok(points=len(missed), path=str(target), layer=layer)

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
        if result.get("ok"):
            return ok(**result)
        # The refusal is the case this feature exists for, and it used to raise
        # "fail() got multiple values for argument 'error'" -- the pilot saw a Python
        # exception instead of "the aircraft is still in AUTO".
        details = {key: value for key, value in result.items() if key not in ("ok", "error")}
        return fail(result.get("error") or "The vehicle did not hand back control.", **details)

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
        # A simulated log is exported -- it is useful for rehearsal -- but it is marked
        # in its file names, its summary and every sample, so it cannot be passed off
        # as the record of a flight.
        stem = log.default_stem() if hasattr(log, "default_stem") else "flight"

        try:
            if output_format == "all":
                written = export_all(log, target, stem=stem)
            else:
                path = export(log, target / f"{stem}.{output_format}", output_format)
                written = {output_format: str(path)}
        except ValueError as exc:
            return fail(str(exc))

        simulated = bool(getattr(log, "is_simulated", False))
        self._session.audit("flight_log_exported",
                            {"formats": sorted(written), "samples": len(log.samples),
                             "simulated": simulated})
        return ok(files=written, summary=log.summary(), simulated=simulated,
                  warning=log.summary().get("simulated_notice", "") if simulated else "")

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
    def import_cad_overlay(self, path: str, source_epsg: int,
                           name: str = "") -> dict[str, Any]:
        """Lay a DXF over the survey, in the project's coordinate system.

        source_epsg is required rather than guessed. A DXF carries no CRS of its own, and
        assuming it matches the project puts the drawing somewhere plausible and wrong --
        which on top of an orthomosaic reads as a construction error rather than a
        placement error, and sends somebody to measure a wall that is exactly where it
        should be.

        The alignment against the project's own rasters is checked in their CRS; the
        overlay is written in WGS84 (as GeoJSON requires) and added to the layer tree,
        so both the map and the layer list draw it.
        """
        from core.cad_overlay import OverlayRefused, alignment_report, read_dxf, reproject

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")

        def georeferenced() -> list:
            return [l for l in self._session.layers.values()
                    if l.kind == "raster" and getattr(l, "crs_epsg", None)]

        layers = georeferenced()
        if not layers:
            # The layer tree is in memory; the products are on disk.
            self._session.restore_layers()
            layers = georeferenced()
        if not layers:
            return fail(
                "This project has no georeferenced products yet, so there is nothing to "
                "lay the drawing over. Run a reconstruction first."
            )
        target_epsg = int(layers[0].crs_epsg)

        try:
            drawing = reproject(read_dxf(path), int(source_epsg), target_epsg)
            lonlat = reproject(read_dxf(path), int(source_epsg), 4326)
        except OverlayRefused as exc:
            return fail(str(exc))

        # Bounds of the products, in their own CRS, to check the drawing landed on them.
        boxes = [l.metadata.get("bounds") for l in layers
                 if (l.metadata or {}).get("bounds") and int(l.crs_epsg) == target_epsg]
        report = {}
        if boxes:
            target = [
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            ]
            report = alignment_report(drawing, target)

        overlay_dir = Path(root) / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        label = name or Path(path).stem
        target_path = overlay_dir / f"{label}.geojson"
        import json as _json

        collection = lonlat.to_geojson()
        collection["properties"] = {"source": str(path), "source_epsg": int(source_epsg),
                                    "kind": "cad_overlay"}
        target_path.write_text(_json.dumps(collection), encoding="utf-8")
        layer = self._session.register_vector_layer(target_path, name=f"CAD: {label}")

        self._session.audit("cad_overlay_imported",
                            {"name": label, "source_epsg": int(source_epsg)})
        result = ok(
            name=label,
            path=str(target_path),
            epsg=target_epsg,
            layer_id=layer["id"],
            entities=drawing.entity_counts,
            skipped=drawing.skipped_entities,
            alignment=report,
        )
        if drawing.skipped_entities:
            named = ", ".join(f"{k} x{v}" for k, v in sorted(drawing.skipped_entities.items()))
            result["warning"] = (
                f"Not everything in the drawing was flattened: {named}. Explode blocks "
                "and hatches in the CAD package and export again if geometry is missing."
            )
        if report.get("warning"):
            result["warning"] = (result.get("warning", "") + " " + report["warning"]).strip()
        return result

    @guard
    def place_image_overlay(self, path: str, west: float, south: float,
                            east: float, north: float, epsg: int = 4326,
                            name: str = "") -> dict[str, Any]:
        """Georeference a plain scan or screenshot by its bounding box.

        The picture is written out as a GeoTIFF in the stated CRS, so it becomes an
        ordinary raster layer that every map and the layer tree can already show. A
        GeoTIFF that is already georeferenced needs none of this: add it as a layer.
        """
        from core.cad_overlay import OverlayRefused, place_raster

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            placed = place_raster(path, float(west), float(south),
                                  float(east), float(north), int(epsg))
        except OverlayRefused as exc:
            return fail(str(exc))

        import cv2
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        picture = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if picture is None:
            return fail(f"{Path(path).name} is not an image this can read.")
        if picture.ndim == 2:
            picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2RGB)
        elif picture.shape[2] == 4:
            picture = cv2.cvtColor(picture, cv2.COLOR_BGRA2RGB)
        else:
            picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        height, width = picture.shape[:2]
        label = name or Path(path).stem
        overlay_dir = Path(root) / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        target = overlay_dir / f"{label}.tif"
        with rasterio.open(
            target, "w", driver="GTiff", width=width, height=height, count=3,
            dtype="uint8", crs=f"EPSG:{int(epsg)}",
            transform=from_bounds(placed.west, placed.south, placed.east, placed.north,
                                  width, height),
        ) as dst:
            dst.write(np.transpose(picture, (2, 0, 1)).astype(np.uint8))
        layer = self._session.register_raster_layer(target, name=f"Overlay: {label}")
        self._session.audit("image_overlay_placed", {"name": label})
        return ok(overlay=placed.to_dict(), name=label, path=str(target),
                  layer_id=layer["id"])

    @guard
    def list_cad_overlays(self) -> dict[str, Any]:
        """Every overlay in this project, with its geometry for drawing."""
        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        overlay_dir = Path(root) / "overlays"
        if not overlay_dir.is_dir():
            return ok(overlays=[])
        import json as _json

        overlays = []
        for p in sorted(overlay_dir.glob("*.geojson")):
            try:
                geojson = _json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            overlays.append({"name": p.stem, "path": str(p), "kind": "vector",
                             "geojson": geojson})
        for p in sorted(overlay_dir.glob("*.tif")):
            overlays.append({"name": p.stem, "path": str(p), "kind": "image"})
        return ok(overlays=overlays)

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
    def model_scene(self, kind: str = "mesh") -> dict[str, Any]:
        """The project's mesh or point cloud as buffers the cockpit's 3D viewer draws.

        Textured Mesh and Point Cloud used to fall back to a 2D raster, and Save View,
        Clip and Measure in 3D all said "the 3D viewer is not built" -- with the model,
        the view store and the clip store already on disk.
        """
        from core.model_scene import SceneRefused, build_scene

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            scene = build_scene(Path(root) / "reconstruction", str(kind))
        except SceneRefused as exc:
            return fail(str(exc))
        views = self.list_views()
        default = next((v for v in (views.get("views") or []) if v.get("is_default")), None)
        return ok(scene=scene, default_view=default)

    @guard
    def measure_on_model(self, kind: str, points_xyz: list[list[float]]) -> dict[str, Any]:
        """Length, height or area between points picked in the 3D viewer, in metres.

        Points arrive in the model file's frame and are carried into the CRS through the
        reconstruction's recorded georeference; an ungeoreferenced model is refused.
        """
        from core.model_scene import SceneRefused, measure_scene

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            measurement = measure_scene(Path(root) / "reconstruction", str(kind), points_xyz or [])
        except SceneRefused as exc:
            return fail(str(exc))
        self._session.audit("model_measured", {"kind": kind, "points": len(points_xyz or [])})
        # Kept in the project's measurement history, as every other measurement is, so a
        # length taken on the model reaches the report.
        value = {"length": measurement.get("slope_distance_m"),
                 "height": abs(float(measurement.get("vertical_distance_m") or 0.0)),
                 "area": measurement.get("surface_area_m2")}.get(str(kind))
        measurement_id = None
        if value is not None:
            from core import geo

            crs = [p for p in measurement.get("points_crs") or [] if len(p) >= 2]
            lonlat: list[list[float]] = []
            if crs:
                lon, lat = geo.projected_to_wgs84([p[0] for p in crs], [p[1] for p in crs],
                                                  int(measurement["crs_epsg"]))
                lonlat = [[float(a), float(b)] for a, b in zip(lon, lat)]
            measurement_id = self._record_measurement(
                f"model_{kind}", float(value), "m2" if kind == "area" else "m", lonlat,
                str(Path(root) / "reconstruction"), dict(measurement))
        return ok(measurement=measurement, measurement_id=measurement_id)

    @guard
    def annotate_model(self, point_xyz: list[float], label: str, severity: str = "medium",
                       note: str = "", source: str = "mesh") -> dict[str, Any]:
        """Pin a finding to a point on the 3D model; it joins the project's annotations."""
        from core.annotations import create_annotation

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        point = [float(v) for v in (point_xyz or [])[:3]]
        if len(point) != 3:
            return fail("A 3D annotation needs a point on the model as [x, y, z].")
        try:
            annotation = create_annotation(
                Path(root), str(self._session.project_id()), "3d", str(source), "point",
                {"type": "Point", "coordinates": point}, str(label).strip(),
                str(severity or "medium").lower(), "open", note=str(note or "") or None,
            )
        except ValueError as exc:
            return fail(str(exc))
        self._session.audit("model_annotated", {"id": annotation.id, "label": annotation.label})
        return ok(annotation=annotation.to_dict())

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
        from mission.mission_types import combine_plans

        combined = combine_plans(plan["plans"], "pylon_inspection")
        adopted = self._adopt_plan(combined, "pylon_inspection", list(plan.get("limits") or []),
                                   extra={"pylon_levels": plan["levels"]})
        self._session.audit("pylon_planned", {"elements": len(elements)})
        summary_plan = {k: v for k, v in plan.items() if k != "plans"}
        return ok(plan=summary_plan, **adopted)

    @guard
    def plan_thermal_survey(self, thermal_camera: str, target_gsd_cm: float,
                            rgb_camera: str = "",
                            front_overlap_pct: float = 80.0,
                            side_overlap_pct: float = 70.0,
                            options: dict[str, Any] | None = None) -> dict[str, Any]:
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
                constraints=self._mission_constraints(options or {}),
                **self._terrain_kwargs(options or {}),
            )
        except MissionTypeRefused as exc:
            return fail(str(exc))
        extra = {k: v for k, v in plan.items() if k != "plan"}
        adopted = self._adopt_plan(plan["plan"], "thermal", list(plan.get("limits") or []),
                                   extra={"thermal": extra}, options=options)
        self._session.audit("thermal_planned", {"camera": thermal_camera})
        return ok(plan=extra, **adopted)

    @guard
    def plan_multispectral_survey(self, payload_key: str,
                                  camera: str = "mavic3e",
                                  altitude_m: float = 60.0,
                                  calibration_panel_lonlat: list[float] | None = None,
                                  front_overlap_pct: float = 80.0,
                                  side_overlap_pct: float = 75.0,
                                  options: dict[str, Any] | None = None,
                                  accept_relative_values: bool = False) -> dict[str, Any]:
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
                accept_relative_values=bool(accept_relative_values),
                constraints=self._mission_constraints(options or {}),
                **self._terrain_kwargs(options or {}),
            )
        except MissionTypeRefused as exc:
            return fail(str(exc))
        extra = {k: v for k, v in plan.items() if k != "plan"}
        adopted = self._adopt_plan(plan["plan"], "multispectral", list(plan.get("limits") or []),
                                   extra={"multispectral": extra, "payload": plan.get("payload")},
                                   options=options)
        self._session.audit("multispectral_planned", {"payload": payload_key})
        return ok(plan=extra, **adopted)

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
    def save_label_boxes(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Keep the boxes drawn on one image, replacing any earlier set for that image.

        The labelling canvas produces one payload per image; stored in the project so a
        labelling session survives closing the window, and so Build Corpus reads what
        was actually drawn rather than whatever the page still held in memory.
        """
        from core.custom_training import CorpusRefused
        from core.label_sets import regions_from_payload

        try:
            regions_from_payload([entry])
        except (CorpusRefused, ValueError, TypeError) as exc:
            return fail(str(exc))
        if not Path(str(entry.get("path", ""))).is_file():
            return fail(f"No such image: {entry.get('path')}")
        store = self._label_store()
        entries = [e for e in self._read_label_store(store)
                   if e.get("path") != entry.get("path")]
        entries.append({
            "path": str(entry["path"]),
            "boxes": list(entry.get("boxes") or []),
            "confirmed_empty": bool(entry.get("confirmed_empty", False)),
        })
        store.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return ok(saved=str(entry["path"]), images=len(entries),
                  boxes=sum(len(e["boxes"]) for e in entries))

    @guard
    def list_label_boxes(self) -> dict[str, Any]:
        """Every image labelled in this project, with its boxes."""
        entries = self._read_label_store(self._label_store())
        classes = sorted({b.get("label", "") for e in entries for b in e.get("boxes", [])})
        return ok(entries=entries, images=len(entries),
                  boxes=sum(len(e.get("boxes", [])) for e in entries), classes=classes)

    def _label_store(self) -> Path:
        folder = self._session.project_root() / "training" / "labels"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "boxes.json"

    @staticmethod
    def _read_label_store(store: Path) -> list[dict[str, Any]]:
        if not store.exists():
            return []
        try:
            data = json.loads(store.read_text(encoding="utf-8"))
        except ValueError:
            return []
        return data if isinstance(data, list) else []

    @guard
    def build_detection_corpus(self, labelled: list[dict[str, Any]] | None = None,
                               output_dir: str = "",
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

        # With nothing passed, the corpus is built from what was drawn in this project.
        if labelled is None:
            labelled = self._read_label_store(self._label_store())
        if not output_dir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = str(self._session.project_root() / "training" / f"corpus_{stamp}")
        try:
            regions = regions_from_payload(labelled)
            report = build_detection_corpus(
                regions, output_dir, salt=salt, allow_empty_images=allow_empty_images
            )
        except CorpusRefused as exc:
            return fail(str(exc))
        except (OSError, ValueError) as exc:
            return fail(str(exc))
        return ok(output_dir=str(output_dir), **report)

    @guard
    def list_training_artifacts(self) -> dict[str, Any]:
        """The corpora built and the training runs finished in this project."""
        root = self._session.project_root() / "training"
        corpora = []
        for folder in sorted(root.glob("corpus_*")):
            report = folder / "corpus_report.json"
            if (folder / "data.yaml").is_file():
                info = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
                corpora.append({"path": str(folder), "name": folder.name,
                                "classes": info.get("classes", []),
                                "total_images": info.get("total_images"),
                                "total_boxes": info.get("total_boxes")})
        runs = []
        for summary in sorted((root / "runs").glob("*/summary.json")):
            info = json.loads(summary.read_text(encoding="utf-8"))
            runs.append({"run_dir": str(summary.parent), "key": summary.parent.name,
                         "metrics": info.get("metrics", {}),
                         "deployed": (summary.parent / "export_report.json").is_file()})
        return ok(corpora=corpora, runs=runs)

    @guard
    def train_custom_model(self, name: str, corpus_dir: str, epochs: int = 50,
                           image_size: int = 640, batch_size: int = 8) -> dict[str, Any]:
        """Train a detector on a corpus built from this project's labels, as a job.

        The result is a metrics card, not an installed model: Deploy Model is a separate
        step taken after reading the numbers.
        """
        from core.custom_model_training import TrainingRefused, custom_key

        if not corpus_dir or not (Path(corpus_dir) / "data.yaml").is_file():
            return fail("Choose a corpus built with Build Corpus first.")
        try:
            key = custom_key(name)
        except TrainingRefused as exc:
            return fail(str(exc))
        runs = self._session.project_root() / "training" / "runs"
        session = self._session

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from core.custom_model_training import train_detector

            progress(2, f"Training {key}")
            result = train_detector(corpus_dir, runs, name=key, epochs=int(epochs),
                                    image_size=int(image_size), batch_size=int(batch_size),
                                    progress=progress, should_cancel=should_cancel)
            session.audit("custom_model_trained", {"key": key, "metrics": result["metrics"]})
            return result

        return ok(job_id=self._jobs.submit("training", work), key=key)

    @guard
    def deploy_custom_model(self, run_dir: str, key: str = "") -> dict[str, Any]:
        """Export a finished run, gate it on ONNX parity, and install it as a custom_* key."""
        from core.custom_model_training import custom_key

        run = Path(run_dir)
        if not (run / "weights" / "best.pt").is_file():
            return fail(f"No trained weights in {run_dir}.")
        target = custom_key(key or run.name)
        session = self._session

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from core.custom_model_training import deploy_detector

            progress(10, "Exporting to ONNX and checking parity")
            result = deploy_detector(run, target)
            session.audit("custom_model_deployed", {"key": target})
            return {"key": target, "parity_ok": result["export"].get("parity_ok"),
                    "labels": result["export"].get("labels"),
                    "sha256": (result.get("registered") or {}).get("sha256", "")}

        return ok(job_id=self._jobs.submit("deploy_model", work), key=target)

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
    def image_preview(self, name: str, max_size: int = 1400,
                      crop: list[float] | None = None) -> dict[str, Any]:
        """One image from the active dataset, as a data URI the canvas can draw.

        The desktop UI is served over loopback from app/web, so it cannot read a file
        anywhere else on disk -- a survey folder is outside the document root and an
        <img src="D:/..."> is blocked. Handing back a downscaled data URI is what lets
        the canvas show the actual photograph rather than a placeholder saying one
        exists.

        Downscaled because these are 24-megapixel frames: a full-size data URI is tens
        of megabytes of base64 across the bridge for a preview nobody zooms into.

        ``crop`` ([x, y, width, height] in the frame's pixels) returns a close-up of that
        window instead, clamped to the frame; the answer carries the window actually
        used. Marking a ground control target needs this: on a whole-frame preview one
        screen pixel is several pixels of the photograph.
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
        window = None
        if crop:
            x0, y0, cw, ch = (float(v) for v in list(crop)[:4])
            cw, ch = max(8.0, min(cw, width)), max(8.0, min(ch, height))
            x0 = int(round(min(max(0.0, x0), width - cw)))
            y0 = int(round(min(max(0.0, y0), height - ch)))
            window = [x0, y0, int(cw), int(ch)]
            image = image[y0:y0 + int(ch), x0:x0 + int(cw)]
        limit = max(64, int(max_size))
        if max(image.shape[:2]) > limit:
            height_now, width_now = image.shape[:2]
            scale = limit / float(max(height_now, width_now))
            image = cv2.resize(image, (int(width_now * scale), int(height_now * scale)),
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
            crop=window,
            data_uri="data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii"),
        )

    @guard
    def add_aircraft(self, organization_id: int, name: str, model: str = "",
                     serial: str = "") -> dict[str, Any]:
        """Register an aircraft.

        The fleet rows reference the service database's organisations, which a fresh
        install has none of, so the cockpit's default organisation 1 failed every Add
        Aircraft, Add Battery and Add Pilot with a foreign-key error. Each fleet call
        resolves the organisation the way sharing does (app/local_tenant.py).
        """
        from app import desktop_ops

        try:
            _, org = self._service_ids(organization_id=organization_id)
            return ok(**desktop_ops.add_aircraft(org, name, model, serial))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def add_battery(self, organization_id: int, serial: str, capacity_mah: int = 0,
                    cycle_limit: int = 0) -> dict[str, Any]:
        """Register a battery so its cycles can be tracked."""
        from app import desktop_ops

        try:
            _, org = self._service_ids(organization_id=organization_id)
            return ok(**desktop_ops.add_battery(org, serial, capacity_mah, cycle_limit))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def add_pilot(self, organization_id: int, display_name: str, licence_number: str = "",
                  licence_expires_on: str = "") -> dict[str, Any]:
        """Add a pilot, refusing an unparseable licence expiry rather than defaulting it."""
        from app import desktop_ops

        try:
            _, org = self._service_ids(organization_id=organization_id)
            return ok(**desktop_ops.add_pilot(org, display_name,
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
            _, org = self._service_ids(organization_id=organization_id)
            return ok(**desktop_ops.fleet_status(org))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def list_fleet(self, organization_id: int = 1) -> dict[str, Any]:
        """The fleet row by row: which aircraft, which battery, which pilot.

        fleet_status() returns counts, which is what a dashboard tile needs and not what
        the Fleet screen needs. Without this the screen had nothing to draw and drew
        four invented aircraft instead.
        """
        from app import desktop_ops

        try:
            _, org = self._service_ids(organization_id=organization_id)
            return ok(**desktop_ops.list_fleet(org))
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    # -------------------------------------------------------------- sharing

    def _service_ids(self, desktop_project_id: int | None = None,
                     organization_id: int | None = None) -> tuple[int | None, int]:
        """The service-database project and organisation for a desktop id.

        The cockpit sends DESKTOP project ids, and share links reference the service
        database's projects -- a different table in a different file -- so every Share
        failed with a foreign-key error. See app/local_tenant.py.
        """
        from app import desktop_ops, local_tenant

        with desktop_ops._session() as db:
            org = local_tenant.organization_id(db, organization_id)
            project = None
            if desktop_project_id is not None:
                wanted = int(desktop_project_id)
                desktop = next((p for p in self._session.store.list_projects()
                                if int(p["id"]) == wanted), None)
                if desktop is None:
                    raise ValueError(f"No project with id {desktop_project_id} on this machine.")
                project = local_tenant.service_project_id(db, desktop)
            db.commit()
        return project, org

    @guard
    def create_share_link(self, project_id: int, note: str = "",
                          allow_download: bool = False) -> dict[str, Any]:
        """Issue a share link. The token is returned ONCE and only its hash is stored."""
        from app import desktop_ops

        try:
            service_project, _ = self._service_ids(project_id)
            link = desktop_ops.create_share_link(service_project, note, allow_download)
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))
        self._session.audit("share_link_issued", {"prefix": link["prefix"], "note": link["note"]})
        return ok(**link, service_project_id=service_project,
                  reaches=("The REST service's record of this project, served at "
                           "/public/shares/{token} while the service runs. Desktop files are not "
                           "uploaded by issuing a link."))

    @guard
    def list_share_links(self, project_id: int) -> dict[str, Any]:
        """Every link issued for this project, by prefix rather than by token."""
        from app import desktop_ops

        try:
            service_project, _ = self._service_ids(project_id)
            return ok(links=desktop_ops.list_share_links(service_project))
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
            _, org = self._service_ids(organization_id=organization_id)
            return ok(**desktop_ops.add_webhook(org, url, events, description), organization_id=org)
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def list_webhooks(self, organization_id: int = 1) -> dict[str, Any]:
        """Registered webhooks and their delivery counts."""
        from app import desktop_ops

        try:
            _, org = self._service_ids(organization_id=organization_id)
            return ok(webhooks=desktop_ops.list_webhooks(org), organization_id=org)
        except (ValueError, RuntimeError) as exc:
            return fail(str(exc))

    @guard
    def issue_api_token(self, email: str, password: str, name: str = "api token") -> dict[str, Any]:
        """Issue a REST API token for a service account, exactly as POST /auth/tokens does.

        The Developers screen's New Key used to issue a SHARE link and call it an API key.
        A key belongs to a user of the REST service, so this asks for that account's
        credentials and writes the same ApiToken row the service would; the secret is
        returned once and only its hash is stored.
        """
        from sqlalchemy import select

        from app import desktop_ops
        from services.api.models import ApiToken, User
        from services.api.security import hash_token, verify_password

        with desktop_ops._session() as db:
            user = db.scalar(select(User).where(User.email == str(email).strip().lower()))
            # One message for both branches, as the service does: the difference would
            # tell whoever is at the keyboard which accounts exist.
            if user is None or not verify_password(str(password), user.password_hash):
                return fail("Incorrect email or password for the REST service. Accounts are "
                            "created with POST /auth/register on the service.")
            if not user.is_active:
                return fail("This service account is disabled.")
            secret, prefix = ApiToken.generate()
            token = ApiToken(user_id=user.id, name=str(name or "api token"),
                             token_hash=hash_token(secret), prefix=prefix)
            db.add(token)
            db.commit()
            db.refresh(token)
            self._session.audit("api_token_issued", {"prefix": prefix, "user_id": user.id})
            return ok(id=token.id, name=token.name, prefix=prefix, secret=secret,
                      user=user.email)

    # -------------------------------------------------------------- reports
    #
    # Reports are built from the project the desktop has open: its store (datasets,
    # mission versions, audit) and its folder (findings, measurements, reconstruction,
    # pipeline runs). They used to go through a mirror project created by name in
    # core.project's separate workspace, with no root folder -- so every report described
    # an empty project and was written outside the one the user chose.

    def _report_config(self, title: str = "Inspection report", report_type: str = "standard",
                       author: str = "", organization: str = "", client: str = "",
                       logo_path: str = "", formats: list[str] | None = None,
                       summary: str = "", recommendations: list[str] | str | None = None,
                       sections: list[str] | None = None):
        from core.report_engine import REPORT_TYPES, ReportConfig

        if str(report_type) not in REPORT_TYPES:
            raise ValueError(f"Unknown report type {report_type!r}. "
                             f"Use one of: {', '.join(REPORT_TYPES)}.")
        if isinstance(recommendations, str):
            recommendations = [line.strip() for line in recommendations.splitlines() if line.strip()]
        chosen = [str(f).lower() for f in (formats or ["html", "pdf"])]
        unknown = [f for f in chosen if f not in ("html", "pdf", "docx")]
        if unknown:
            raise ValueError(f"Unknown report format(s) {unknown}. Use html, pdf or docx.")
        if logo_path and not Path(logo_path).is_file():
            raise ValueError(f"Logo file not found: {logo_path}")
        return ReportConfig(
            project_id=str(self._session.project_id()),
            title=str(title or "Inspection report"),
            report_type=str(report_type),
            author=str(author or ""),
            organization=str(organization or ""),
            client=str(client or ""),
            logo_path=str(logo_path or ""),
            summary=str(summary or ""),
            recommendations=list(recommendations or []),
            formats=chosen,
            sections=list(sections or []),
        )

    @guard
    def report_types(self) -> dict[str, Any]:
        """The report types the engine builds, with the sections each one contains."""
        from core.report_engine import REPORT_TYPES, SECTION_TITLES

        return ok(types=[
            {"key": key, "sections": [{"key": s, "title": SECTION_TITLES.get(s, s)} for s in sections]}
            for key, sections in REPORT_TYPES.items()
        ])

    @guard
    def report_readiness(self, project_id: str = "", report_type: str = "standard",
                         formats: list[str] | None = None) -> dict[str, Any]:
        """What a report still needs, judged on the open project, before Generate."""
        from app import report_builder

        try:
            config = self._report_config(report_type=report_type, formats=formats)
            readiness = report_builder.readiness(self._session, config)
        except ValueError as exc:
            return fail(str(exc))
        # The call itself succeeded; whether the report can be written is `ready`.
        # readiness.to_dict() carries its own "ok", which must not override the call's,
        # or a refused report type reaches both UIs as an error with no message and the
        # list of what is missing is never shown.
        checklist = {k: v for k, v in readiness.to_dict().items() if k != "ok"}
        return ok(ready=readiness.ok, **checklist)

    @guard
    def generate_report(self, project_id: str = "", title: str = "Inspection report",
                        report_type: str = "standard", author: str = "",
                        organization: str = "", client: str = "", logo_path: str = "",
                        formats: list[str] | None = None, summary: str = "",
                        recommendations: list[str] | str | None = None) -> dict[str, Any]:
        """Write the report into <project>/reports: HTML, plus PDF and/or DOCX if asked.

        Refuses, naming what is missing, when the report type has nothing to be about
        or a requested format cannot be written here.
        """
        from app import report_builder

        try:
            config = self._report_config(title, report_type, author, organization, client,
                                         logo_path, formats, summary, recommendations)
            result = report_builder.generate(self._session, config)
        except ValueError as exc:
            return fail(str(exc))
        # store.save_report (in report_builder.generate) records report_generated in
        # the audit trail with the report's path.
        return ok(**result.to_dict())

    @guard
    def list_reports(self) -> dict[str, Any]:
        """Reports written into the open project's folder, newest first."""
        from app import report_builder

        return ok(reports=report_builder.list_reports(self._session))

    @guard
    def open_report(self, report_id: str, kind: str = "html") -> dict[str, Any]:
        """Open a generated report (html, pdf, docx) or its folder in the OS."""
        from app import report_builder

        for report in report_builder.list_reports(self._session):
            if report["id"] == str(report_id) or report["id"].startswith(str(report_id)):
                key = {"html": "html_path", "pdf": "pdf_path", "docx": "docx_path",
                       "folder": "output_dir"}.get(str(kind))
                if key is None:
                    return fail(f"Unknown kind {kind!r}. Use html, pdf, docx or folder.")
                target = report.get(key)
                if not target:
                    return fail(f"This report has no {kind.upper()} file.")
                return self.open_path(target)
        return fail(f"No report {report_id!r} in this project.")

    @guard
    def save_report_template(self, name: str, settings: dict[str, Any]) -> dict[str, Any]:
        """Keep branding and type choices so the next report starts from them."""
        name = str(name or "").strip()
        if not name:
            return fail("A report template needs a name.")
        allowed = {"title", "report_type", "organization", "client", "logo_path", "author",
                   "formats"}
        clean = {k: v for k, v in dict(settings or {}).items() if k in allowed}
        templates = dict(self._session.store.get_setting("report_templates", {}) or {})
        templates[name] = clean
        self._session.store.set_setting("report_templates", templates)
        self._session.audit("report_template_saved", {"name": name})
        return ok(name=name, template=clean)

    @guard
    def report_templates(self) -> dict[str, Any]:
        templates = self._session.store.get_setting("report_templates", {}) or {}
        return ok(templates=[{"name": k, **v} for k, v in sorted(templates.items())])

    @guard
    def record_finding(self, image: str, label: str, severity: str = "medium",
                       note: str = "", bbox: list[float] | None = None,
                       tags: list[str] | None = None) -> dict[str, Any]:
        """Record an inspector's finding on one of the project's images.

        This is the step that had no caller anywhere in the application: the cockpit's
        Annotate tool only armed a cursor, so no report could ever contain a finding
        a person had made. The finding is stored as an annotation, created_by "user",
        which is how reports tell an inspector's finding from a model's proposal.

        `image` is a path, or a file name in one of the project's datasets. `bbox` is
        [x0, y0, x1, y1] in the image's pixels; without it the finding covers the frame.
        """
        from core.annotations import add_tags, create_annotation
        from core.report_engine import resolve_image

        root = self._session.project_root()
        image_dirs = [Path(str(d.get("path", ""))) for d in self._session.list_datasets()]
        if self._session.active_dataset_dir:
            image_dirs.insert(0, Path(self._session.active_dataset_dir))
        source = resolve_image(str(image or ""), [d for d in image_dirs if d.is_dir()])
        if source is None:
            return fail(f"Image not found in this project's datasets: {image!r}. "
                        "Import the imagery folder first, or give the full path.")
        from PIL import Image

        with Image.open(source) as opened:
            width, height = opened.size
        if bbox:
            x0, y0, x1, y1 = (float(v) for v in list(bbox)[:4])
            if x1 <= x0 or y1 <= y0:
                return fail("The box must have x1 > x0 and y1 > y0 (image pixels).")
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                return fail(f"The box lies outside the image ({width} x {height} pixels).")
        else:
            # No box drawn: the finding is about the photo as a whole, and its geometry
            # says exactly that rather than pretending to a location.
            x0, y0, x1, y1 = 0.0, 0.0, float(width), float(height)
        geometry = {"type": "Polygon",
                    "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
        kind = "rectangle"
        try:
            annotation = create_annotation(
                root, str(self._session.project_id()), "image", str(source), kind, geometry,
                str(label or "").strip(), str(severity).lower(), "open",
                note=str(note or "") or None, created_by="user",
            )
        except ValueError as exc:
            return fail(str(exc))
        if tags:
            add_tags(root, [annotation.id], [str(t) for t in tags])
        self._session.audit("finding_recorded", {
            "id": annotation.id, "label": annotation.label, "severity": annotation.severity,
            "image": source.name,
        })
        return ok(finding=annotation.to_dict())

    @guard
    def verify_provenance(self, folder: str = "") -> dict[str, Any]:
        """Check every derived file in a folder against its recorded provenance.

        Defaults to the project's reconstruction output. Files with no record are
        listed as unattributed rather than passed as fine.
        """
        from core.provenance import audit

        target = Path(folder) if folder else self._session.project_root() / "reconstruction"
        if not target.is_dir():
            return fail(f"No folder to verify at {target}. Run a reconstruction first.")
        report = audit(target)
        # audit() carries its own "ok" (every file verified); it must not be mistaken
        # for the call failing, so it travels as all_verified.
        report["all_verified"] = report.pop("ok")
        return ok(**report)

    @guard
    def record_provenance(self, folder: str = "") -> dict[str, Any]:
        """Record provenance for a reconstruction folder that has none yet.

        Outputs produced before provenance was recorded automatically are unattributed.
        This records them now, against the active dataset, and says so in the record:
        what it certifies is that the files have not changed SINCE this moment.
        """
        from core.provenance import read, record_reconstruction_outputs

        target = Path(folder) if folder else self._session.project_root() / "reconstruction"
        if not target.is_dir():
            return fail(f"No folder at {target}.")
        images = self._session.active_dataset_dir
        if not images:
            return fail("Select the dataset these outputs were built from first.")
        twin = target / "digital_twin.json"
        meta: dict[str, Any] = {}
        if twin.is_file():
            try:
                meta = json.loads(twin.read_text(encoding="utf-8"))
            except ValueError:
                meta = {}
        already = [p.name for p in target.iterdir() if p.is_file() and read(p) is not None]
        result = record_reconstruction_outputs(
            target, engine=str(meta.get("engine") or "unknown"), image_dir=images,
            crs_epsg=meta.get("crs_epsg"),
            parameters={"profile": meta.get("profile"), "recorded_after_the_fact": True},
        )
        self._session.audit("provenance_recorded", {"folder": target.name,
                                                    "artifacts": result["artifact_count"]})
        return ok(**result, previously_recorded=already)

    @guard
    def export_audit_log(self, output_format: str = "csv", output_path: str = "") -> dict[str, Any]:
        """Write the open project's audit trail to CSV or JSONL."""
        import csv

        fmt = str(output_format).lower()
        if fmt not in ("csv", "jsonl"):
            return fail("Use csv or jsonl.")
        events = list(reversed(self._session.store.list_audit_events(
            self._session.project_id(), 100000)))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = Path(output_path) if output_path else (
            self._session.project_root() / "reports" / f"audit_{stamp}.{fmt}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["created_at", "event_type", "payload"])
                for e in events:
                    writer.writerow([e.get("created_at"), e.get("event_type"), e.get("payload_json")])
        else:
            with target.open("w", encoding="utf-8") as handle:
                for e in events:
                    handle.write(json.dumps({
                        "created_at": e.get("created_at"), "event_type": e.get("event_type"),
                        "payload": json.loads(e.get("payload_json") or "{}"),
                    }) + "\n")
        return ok(path=str(target), events=len(events), format=fmt)

    @guard
    def thermal_palettes(self) -> dict[str, Any]:
        """Palettes this build can draw a thermal frame in."""
        from core.thermal_scaling import PALETTES

        return ok(palettes=sorted(PALETTES))

    @guard
    def scale_thermal(self, path: str, mode: str = "auto",
                      min_c: float | None = None, max_c: float | None = None,
                      low_percentile: float = 1.0, high_percentile: float = 99.0,
                      sigma: float = 2.0, palette: str = "ironbow") -> dict[str, Any]:
        """Choose the temperature range a radiometric frame is drawn in.

        This decision changes what an inspector sees more than any other step in the
        thermal workflow. One 80 C flue in the frame stretches a full-range scale until
        the 3 C delta across a wet roof renders as flat grey; a range tight enough to show
        that delta clips the flue, so a dangerous hotspot is drawn the same colour as a
        merely warm one.

        The second is the one that gets missed, because the image looks fine. So the
        result always carries what the range excluded, and a warning when the hottest
        pixels in the frame are saturated.
        """
        from core.thermal import NotRadiometric, load_radiometric_file
        from core.thermal_scaling import (
            ScalingRefused, anomaly_scale, auto_scale, legend, manual_scale,
        )

        try:
            frame = load_radiometric_file(path)
        except (NotRadiometric, FileNotFoundError, OSError) as exc:
            return fail(str(exc))

        chosen = str(mode).strip().lower()
        try:
            if chosen == "manual":
                if min_c is None or max_c is None:
                    return fail("A manual range needs both min_c and max_c.")
                scale = manual_scale(frame.celsius, float(min_c), float(max_c), palette)
            elif chosen == "anomaly":
                scale = anomaly_scale(frame.celsius, float(sigma), palette)
            elif chosen == "auto":
                scale = auto_scale(frame.celsius, float(low_percentile),
                                   float(high_percentile), palette)
            else:
                return fail(f"Unknown mode {mode!r}. Use auto, manual or anomaly.")
        except ScalingRefused as exc:
            return fail(str(exc))

        result = ok(
            scale=scale.to_dict(),
            legend=legend(scale),
            stats=frame.stats,
        )
        warning = scale.warning()
        if warning:
            result["warning"] = warning
        return result

    @guard
    def marker_kinds(self) -> dict[str, Any]:
        """What a site marker can be, and what each kind means."""
        from core.site_markers import MARKER_KINDS

        return ok(kinds=[{"kind": k, "describes": v} for k, v in sorted(MARKER_KINDS.items())])

    @guard
    def list_site_markers(self) -> dict[str, Any]:
        from core.site_markers import MarkerStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        store = MarkerStore(Path(root))
        return ok(markers=[m.to_dict() for m in store.load()],
                  geojson=store.to_geojson())

    @guard
    def add_site_marker(self, name: str, kind: str, points: list[list[float]],
                        note: str = "", radius_m: float = 0.0) -> dict[str, Any]:
        """Record something on the ground that is not part of the flight.

        Where the pilot stands, where the van parks, which corner has the overhead line.
        Those live in somebody's head on the day and in nobody's head six months later
        when a different crew reflies the site.
        """
        from core.site_markers import MarkerRefused, MarkerStore, SiteMarker

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            markers = MarkerStore(Path(root)).add(SiteMarker(
                name=str(name), kind=str(kind),
                points=[[float(c) for c in p] for p in points],
                note=str(note), radius_m=float(radius_m),
            ))
        except (MarkerRefused, TypeError, ValueError) as exc:
            return fail(str(exc))
        self._session.audit("site_marker_added", {"name": name, "kind": kind})
        result = ok(markers=[m.to_dict() for m in markers])
        far = self._distance_from_site_m(points)
        if far is not None and far > 2000.0:
            # Recorded -- the operator may know better -- but a marker kilometres from
            # the site is far more often a typo (a swapped digit, lat/lon reversed).
            result["warning"] = (
                f"This marker is {far / 1000.0:.1f} km from the area of interest. Check "
                "the coordinates; a swapped digit or reversed lat/lon lands here."
            )
        return result

    def _distance_from_site_m(self, points: list[list[float]]) -> float | None:
        """Metres from the nearest marker point to the AOI (or mission), None if neither."""
        from core import geo

        area = list(self._session.aoi_polygon or [])
        if not area and self._session.mission_plan_dict:
            area = [[w.get("lon"), w.get("lat")]
                    for w in self._session.mission_plan_dict.get("waypoints", [])
                    if w.get("lon") is not None and w.get("lat") is not None]
        if not area or not points:
            return None
        lons = [float(p[0]) for p in area]
        lats = [float(p[1]) for p in area]
        best = None
        for point in points:
            lon = min(max(float(point[0]), min(lons)), max(lons))
            lat = min(max(float(point[1]), min(lats)), max(lats))
            distance = geo.haversine_m(float(point[1]), float(point[0]), lat, lon)
            best = distance if best is None else min(best, distance)
        return best

    @guard
    def remove_site_marker(self, marker_id: str) -> dict[str, Any]:
        from core.site_markers import MarkerRefused, MarkerStore

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        try:
            markers = MarkerStore(Path(root)).remove(str(marker_id))
        except MarkerRefused as exc:
            return fail(str(exc))
        return ok(markers=[m.to_dict() for m in markers])

    @guard
    def check_hazards(self, clearance_m: float = 30.0) -> dict[str, Any]:
        """Which hazard markers the planned mission passes close to.

        Reports; does not reroute. Markers are information the crew wrote down, not
        constraints the planner enforces -- a planner that silently flew around a note
        somebody typed would be worse than one that says nothing, because the crew would
        stop trusting that the plan is the plan.
        """
        from core.site_markers import MarkerStore, hazards_near

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")
        plan = self._session.mission_plan_dict
        if not plan:
            return fail("No mission has been planned yet, so there is no route to check.")

        # The planner emits [lon, lat, alt] lists. Older recipes and some exporters carry
        # dicts instead, so both are read -- a hazard check that silently found no
        # positions would report "no hazards" and be believed.
        waypoints: list[list[float]] = []
        for entry in plan.get("waypoints") or []:
            if isinstance(entry, dict):
                lon, lat = entry.get("lon"), entry.get("lat")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                lon, lat = entry[0], entry[1]
            else:
                continue
            if lon is not None and lat is not None:
                waypoints.append([float(lon), float(lat)])
        if not waypoints:
            return fail("The planned mission carries no positions to check.")

        found = hazards_near(MarkerStore(Path(root)).load(), waypoints, float(clearance_m))
        result = ok(hazards=found, checked_waypoints=len(waypoints),
                    clearance_m=float(clearance_m))
        if found:
            closest = found[0]
            # A negative clearance means the route passes INSIDE the hazard's own radius,
            # which is the most serious case and the one that reads as a typo if it is
            # printed as "at -21 m".
            if closest["clearance_m"] < 0:
                how = (f"the route passes through {closest['name']}, "
                       f"{abs(closest['clearance_m']):.0f} m inside its {'' if not closest.get('note') else ''}"
                       f"marked radius")
            else:
                how = f"closest is {closest['name']} at {closest['clearance_m']:.0f} m"
            result["warning"] = (
                f"{len(found)} hazard marker(s) within {clearance_m:g} m of the route: {how}. "
                "Markers are a briefing note, not a constraint -- the plan has not been changed."
            )
        return result

    def _notifications(self):
        """The notification centre for the open project, or None."""
        from core.notifications import NotificationCentre

        root = self._session.project_root()
        return NotificationCentre(Path(root)) if root else None

    @guard
    def list_notifications(self, unread_only: bool = False) -> dict[str, Any]:
        """What has happened while the operator was looking somewhere else."""
        centre = self._notifications()
        if centre is None:
            return fail("Open a project first.")
        items = centre.unread() if unread_only else centre.load()
        return ok(notifications=[n.to_dict() for n in items],
                  unread=centre.unread_count())

    @guard
    def mark_notification_read(self, notification_id: str) -> dict[str, Any]:
        centre = self._notifications()
        if centre is None:
            return fail("Open a project first.")
        if not centre.mark_read(str(notification_id)):
            return fail(f"No notification {notification_id!r}.")
        return ok(unread=centre.unread_count())

    @guard
    def mark_all_notifications_read(self) -> dict[str, Any]:
        centre = self._notifications()
        if centre is None:
            return fail("Open a project first.")
        return ok(cleared=centre.mark_all_read(), unread=centre.unread_count())

    @guard
    def clear_notifications(self) -> dict[str, Any]:
        centre = self._notifications()
        if centre is None:
            return fail("Open a project first.")
        return ok(removed=centre.clear())

    @guard
    def export_report(self, output_format: str = "pdf", title: str = "Inspection report",
                      organization: str = "", client: str = "", author: str = "",
                      output_path: str = "", logo_path: str = "") -> dict[str, Any]:
        """Write the findings out as a PDF or Word deliverable.

        Findings come from this project's annotations, so the deliverable describes what
        was actually recorded rather than a fixed example. Each finding carries its
        source image, the photo position, a crop of the marked area and the reviewer's
        decision; the summary, methodology and follow-up items are derived from the same
        project data as the full report. Every export gets its own timestamped file --
        the fixed reports/report.<fmt> name overwrote the previous deliverable.
        """
        from app import report_builder
        from core.report_engine import build_blocks, deliverable_payload, prepare_assets
        from core.report_formats import ReportTemplate, write_docx, write_pdf

        root = self._session.project_root()
        if root is None:
            return fail("Open a project first.")

        fmt = str(output_format).strip().lower()
        if fmt not in ("pdf", "docx"):
            return fail(f"Unknown report format {output_format!r}. Use pdf or docx.")

        try:
            config = self._report_config(title, "standard", author, organization, client,
                                         logo_path, [fmt])
        except ValueError as exc:
            return fail(str(exc))
        context = report_builder.desktop_context(self._session, config)
        findings = context.get("findings") or []
        if not findings:
            return fail(
                "No findings recorded yet, so there is nothing to report. Record a finding "
                "(Reports > Record Finding) or run an AI inspection first."
            )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = (Path(output_path) if output_path
                  else Path(root) / "reports" / f"findings_{stamp}.{fmt}")
        target.parent.mkdir(parents=True, exist_ok=True)
        assets = prepare_assets(context, target.parent / f"{target.stem}_assets")
        blocks = build_blocks(context, assets)
        payload = deliverable_payload(context, blocks, assets)
        template = ReportTemplate(
            title=str(title),
            organization=str(organization),
            client=str(client),
            author=str(author),
            logo_path=str(logo_path or ""),
        )

        writer = write_pdf if fmt == "pdf" else write_docx
        try:
            written = writer(target, payload, template)
        except Exception as exc:  # noqa: BLE001 - reportlab/docx raise their own types
            return fail(f"Could not write the {fmt.upper()}: {exc}")
        try:
            from core.provenance import record

            record(written, engine="opendronekit.report_formats",
                   sources=[f.get("image_source_path") for f in findings
                            if f.get("image_source_path")],
                   parameters={"format": fmt, "findings": len(findings)})
        except Exception:  # noqa: BLE001 - provenance must never lose a written file
            pass

        self._session.audit("report_exported",
                            {"format": fmt, "findings": len(findings), "file": Path(written).name})
        return ok(path=str(written), format=fmt, findings=len(findings))

    @guard
    def report_formats(self) -> dict[str, Any]:
        """Which deliverable formats this build can actually write."""
        available = []
        try:
            import reportlab  # noqa: F401
            available.append("pdf")
        except ImportError:
            pass
        try:
            import docx  # noqa: F401
            available.append("docx")
        except ImportError:
            pass
        return ok(formats=available)

    # --------------------------------------------------------------- review

    @guard
    def review_finding(self, annotation_id: str, decision: str,
                       reviewer: str = "operator", comment: str = "") -> dict[str, Any]:
        """Accept, reject or flag a finding without overwriting what the model claimed.

        Decisions append to the finding's review history; a rejection takes it out of
        reports.
        """
        from app import desktop_ops

        root = self._session.project_root()
        if not root:
            return fail("Open a project before reviewing findings.")
        try:
            return ok(**desktop_ops.review_finding(root, annotation_id, decision, reviewer,
                                                   comment))
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
                "Dense point clouds are NOT available in this environment: "
                f"{caps.get('dense_stereo_reason') or 'no working CUDA COLMAP.'} "
                "Sparse reconstruction, mesh and orthomosaic still work. The sparse "
                "cloud is never inflated to imitate a dense one."
            ),
        )

    @guard
    def find_ponding(self, surface_path: str = "",
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

        surface_path = surface_path or self._product_path("dsm")
        if not surface_path:
            return fail("This project has no DSM to look for ponding on. Run Process first.")
        try:
            surface = load_surface(surface_path)
            report = run_ponding(surface,
                                 vertical_accuracy_m=_optional_float(vertical_accuracy_m))
        except (NotGeoreferenced, NotProjected, PondingRefused) as exc:
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        return ok(**report.to_dict())

    @guard
    def compare_surveys(self, earlier_path: str, later_path: str,
                        earlier_accuracy_m: float | None = None,
                        later_accuracy_m: float | None = None,
                        registration_residual_m: float | None = None,
                        align: bool = True) -> dict[str, Any]:
        """Vertical movement between two surveys of the same ground.

        All three uncertainties are required. Two surveys of unchanged ground never
        difference to zero, so without them this cannot tell movement from measurement
        error -- and a subsidence figure nobody can check is worse than none.
        """
        from core.deformation import DeformationRefused, compare_surfaces
        from core.dsm_analysis import (
            NotGeoreferenced, align_to_grid, load_surface, same_grid,
        )
        from core.slope import NotProjected

        try:
            earlier = load_surface(earlier_path)
            later = load_surface(later_path)
            # Two real surveys are never on one grid. The later one is resampled onto
            # the earlier one and the result says so; the uncertainty inputs still
            # decide what counts as movement.
            resampled = False
            if align and not same_grid(earlier, later):
                later = align_to_grid(later, earlier)
                resampled = True
            report = compare_surfaces(
                earlier, later,
                earlier_accuracy_m=_optional_float(earlier_accuracy_m),
                later_accuracy_m=_optional_float(later_accuracy_m),
                registration_residual_m=_optional_float(registration_residual_m),
            )
        except (NotGeoreferenced, NotProjected, DeformationRefused) as exc:
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        payload = report.to_dict()
        payload["resampled"] = resampled
        if resampled:
            payload["note"] = (
                "The later survey was resampled (bilinear) onto the earlier survey's grid "
                "before differencing. " + payload.get("note", "")
            ).strip()
        return ok(**payload)

    # -- measurement on the project's own surfaces -------------------------
    #
    # Every analysis below existed and was correct, and none could be reached: the
    # cockpit buttons passed null where a raster path and a drawn geometry belonged. The
    # map gives WGS84 lon/lat, the rasters are in a projected CRS, and nothing converted
    # between them. These methods take what the map gives, find the project's own
    # DSM/DTM/orthomosaic, and do the conversion in one place.

    _PRODUCT_FILES = {"dsm": "dsm.tif", "dtm": "dtm.tif", "orthomosaic": "orthomosaic.tif"}

    def _product_path(self, product: str) -> str:
        """This project's DSM, DTM or orthomosaic on disk, or '' when it has none."""
        wanted = product.lower()
        filename = self._PRODUCT_FILES.get(wanted, "")

        def search() -> str:
            for layer in self._session.layers.values():
                if layer.kind != "raster" or not layer.path:
                    continue
                if layer.name.lower() == wanted or Path(layer.path).name.lower() == filename:
                    if Path(layer.path).is_file():
                        return str(layer.path)
            return ""

        found = search()
        if not found:
            # Layers are in memory and cleared on project switch; the files are not.
            self._session.restore_layers()
            found = search()
        return found

    @staticmethod
    def _raster_epsg(path: str) -> int | None:
        import rasterio

        with rasterio.open(path) as raster:
            return raster.crs.to_epsg() if raster.crs else None

    def _lonlat_to_raster_xy(self, path: str,
                             lonlat: list[list[float]] | None) -> list[list[float]]:
        """Project map clicks into the raster's CRS, refusing anything non-finite."""
        import numpy as np

        from core import geo
        from core.dsm_analysis import require_projected_epsg

        points = [[float(p[0]), float(p[1])] for p in (lonlat or []) if len(p) >= 2]
        if not points:
            return []
        if not Path(path).is_file():
            raise FileNotFoundError(f"Raster not found: {path}")
        epsg = self._raster_epsg(path)
        require_projected_epsg(epsg, path)
        east, north = geo.wgs84_to_projected(
            [p[0] for p in points], [p[1] for p in points], int(epsg))
        if not (np.isfinite(east).all() and np.isfinite(north).all()):
            raise ValueError(
                f"The drawn geometry cannot be expressed in EPSG:{epsg}; it is not a "
                "location that raster covers."
            )
        return [[float(x), float(y)] for x, y in zip(east, north)]

    def _record_measurement(self, kind: str, value: float, unit: str,
                            coordinates: list[list[float]] | None, source: str,
                            details: dict[str, Any], label: str = "") -> str:
        """Keep a measurement in the project's history, which the report reads."""
        from core.measurements import create_measurement

        points = [[float(p[0]), float(p[1])] for p in (coordinates or []) if len(p) >= 2]
        geometry: dict[str, Any] = {"crs": "EPSG:4326"}
        polygonal = {"area", "perimeter", "volume", "cut_fill", "stockpile", "slope",
                     "surface_change"}
        if kind in polygonal and len(points) >= 3:
            geometry.update(type="Polygon", coordinates=[points + [points[0]]])
        elif points:
            geometry.update(type="LineString", coordinates=points)
        else:
            geometry.update(type="Extent", coordinates=[])
        measurement = create_measurement(
            project_root=self._session.project_root(),
            project_id=str(self._session.project_id()),
            source_type="map",
            source_id=str(source),
            measurement_type=kind,
            geometry=geometry,
            value=float(value),
            unit=unit,
            label=label or f"{kind.replace('_', ' ').title()} {value:,.3f} {unit}",
            details=details,
        )
        return measurement.id

    def _crs_setting_key(self) -> str:
        return f"project:{self._session.project_id()}:target_epsg"

    @guard
    def get_project_crs(self) -> dict[str, Any]:
        """The projected CRS this project's reconstructions are written in, if chosen."""
        epsg = self._session.store.get_setting(self._crs_setting_key())
        if not epsg:
            return ok(epsg=None, name="automatic UTM zone from the imagery")
        from pyproj import CRS

        return ok(epsg=int(epsg), name=CRS.from_epsg(int(epsg)).name)

    @guard
    def set_project_crs(self, epsg: int | None = None) -> dict[str, Any]:
        """Choose the projected CRS reconstructions are written in; None = automatic.

        A geographic CRS is refused: every product here is measured in metres, and a DSM
        written in degrees would make every later volume and distance meaningless.
        """
        if epsg in (None, "", 0):
            self._session.store.set_setting(self._crs_setting_key(), None)
            self._session.audit("project_crs_set", {"epsg": None})
            return ok(epsg=None, name="automatic UTM zone from the imagery")
        from pyproj import CRS
        from pyproj.exceptions import CRSError

        try:
            crs = CRS.from_epsg(int(epsg))
        except (CRSError, ValueError, TypeError) as exc:
            return fail(f"EPSG:{epsg} is not a coordinate reference system PROJ knows ({exc}).")
        if not crs.is_projected:
            return fail(f"EPSG:{epsg} ({crs.name}) is not projected: its units are degrees, "
                        "and every product is measured in metres. Choose a projected CRS "
                        "such as a UTM zone.")
        unit = crs.axis_info[0].unit_name if crs.axis_info else "unknown"
        factor = crs.axis_info[0].unit_conversion_factor if crs.axis_info else None
        if factor != 1.0:
            return fail(f"EPSG:{epsg} ({crs.name}) measures in {unit}, and every product "
                        "is written with elevations in metres: its DSM would pair "
                        "horizontal feet with vertical metres, and the volume, slope and "
                        "ponding tools refuse such a surface. Choose a projected CRS in "
                        "metres, such as the survey's UTM zone.")
        self._session.store.set_setting(self._crs_setting_key(), int(epsg))
        self._session.audit("project_crs_set", {"epsg": int(epsg)})
        return ok(epsg=int(epsg), name=crs.name)

    @guard
    def project_surfaces(self) -> dict[str, Any]:
        """The DSM, DTM and orthomosaic this project's measurements will run on."""
        found = {name: self._product_path(name) for name in self._PRODUCT_FILES}
        return ok(**found, missing=[name for name, path in found.items() if not path])

    @guard
    def measure_volume(self, polygon_lonlat: list[list[float]] | None = None,
                       base_elevation_m: float | None = None,
                       dsm_path: str = "", dtm_path: str = "",
                       label: str = "") -> dict[str, Any]:
        """Volume (or cut and fill) inside a drawn polygon, against a stated reference.

        With no base elevation the reference is the project's DTM when it has one; with
        one it is that horizontal plane -- the design level a cut-and-fill is asked
        against. Every reference used is reported, never just the convenient one.
        """
        from core.dsm_analysis import (
            IncompatibleReferenceSurface, NotGeoreferenced, estimate_volume,
        )

        dsm = dsm_path or self._product_path("dsm")
        if not dsm:
            return fail("This project has no DSM. Run Process (reconstruction) first, "
                        "or choose a DSM GeoTIFF.")
        base = None if base_elevation_m in (None, "") else float(base_elevation_m)
        # A stated design level IS the reference for cut and fill; passing the DTM too
        # would make it the preferred figure and answer a different question.
        dtm = None if base is not None else (dtm_path or self._product_path("dtm") or None)
        try:
            polygon = self._lonlat_to_raster_xy(dsm, polygon_lonlat)
            if polygon_lonlat and len(polygon) < 3:
                return fail("A volume needs a polygon of at least three points.")
            result = estimate_volume(dsm, dtm_path=dtm, polygon_xy=polygon or None,
                                     base_elevation_m=base)
        except (NotGeoreferenced, IncompatibleReferenceSurface, FileNotFoundError,
                ValueError) as exc:
            return fail(str(exc))
        if not result.get("ok"):
            return fail(result.get("reason") or "No volume could be measured.")

        preferred = result["preferred"] or {}
        kind = "cut_fill" if base is not None else "volume"
        if not polygon:
            result["warning"] = (
                "No polygon was drawn, so this is the volume over the WHOLE DSM. Draw an "
                "area on the map to measure one pile or pit."
            )
        result["measurement_id"] = self._record_measurement(
            kind, float(preferred.get("net_volume_m3", 0.0)), "m3", polygon_lonlat, dsm,
            {k: result[k] for k in ("references", "preferred", "crs_epsg",
                                    "region_area_m2", "note") if k in result},
            label=label,
        )
        self._session.audit("volume_measured",
                            {"dsm": dsm, "reference": preferred.get("reference")})
        return ok(**{k: v for k, v in result.items() if k != "ok"}, kind=kind)

    @guard
    def measure_stockpile(self, polygon_lonlat: list[list[float]],
                          base_elevation_m: float | None = None,
                          label: str = "") -> dict[str, Any]:
        """One selected stockpile, measured and written out as a report package.

        Needs either the project's DTM or an operator-stated base level: the lowest point
        of the selection is an over-estimate, not a base, and is not offered.
        """
        from datetime import datetime, timezone

        from core.dsm_analysis import IncompatibleReferenceSurface, NotGeoreferenced
        from core.survey_intelligence import create_stockpile_package

        dsm = self._product_path("dsm")
        if not dsm:
            return fail("This project has no DSM. Run Process (reconstruction) first.")
        base = None if base_elevation_m in (None, "") else float(base_elevation_m)
        dtm = None if base is not None else (self._product_path("dtm") or None)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = self._session.project_root() / "analysis" / "stockpiles" / stamp
        try:
            polygon = self._lonlat_to_raster_xy(dsm, polygon_lonlat)
            package = create_stockpile_package(dsm, out, polygon_xy=polygon,
                                               dtm_path=dtm, base_elevation_m=base)
        except (NotGeoreferenced, IncompatibleReferenceSurface, FileNotFoundError,
                ValueError) as exc:
            return fail(str(exc))
        measured = package.measurement
        preferred = measured.get("preferred") or {}
        measurement_id = self._record_measurement(
            "stockpile", float(preferred.get("cut_volume_m3", 0.0)), "m3",
            polygon_lonlat, dsm,
            {"references": measured.get("references"), "preferred": preferred,
             "report": package.report_path}, label=label,
        )
        return ok(measurement=measured, preferred=preferred,
                  report_path=package.report_path, summary_path=package.summary_path,
                  selection_path=package.selection_geojson_path,
                  measurement_id=measurement_id)

    @guard
    def elevation_profile(self, line_lonlat: list[list[float]], dsm_path: str = "",
                          spacing_m: float | None = None) -> dict[str, Any]:
        """Elevation along a drawn line, sampled from the project's DSM."""
        from core.dsm_analysis import NotGeoreferenced, elevation_profile

        dsm = dsm_path or self._product_path("dsm")
        if not dsm:
            return fail("This project has no DSM to take a profile from. Run Process first.")
        try:
            line = self._lonlat_to_raster_xy(dsm, line_lonlat)
            result = elevation_profile(dsm, line, spacing_m=spacing_m)
        except (NotGeoreferenced, FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        if not result.get("ok"):
            return fail(result.get("reason", "No profile could be taken."),
                        sample_count=result.get("sample_count"))
        samples = result["samples"]
        stride = max(1, -(-len(samples) // 500))
        details = {k: v for k, v in result.items() if k != "samples"}
        # Every stride-th sample, for drawing the chart; the figures above use them all.
        details["chart_samples"] = [[s["distance_m"], s["z"]] for s in samples[::stride]]
        details["chart_stride"] = stride
        result["measurement_id"] = self._record_measurement(
            # The walked length only when every sample had an elevation; a partial sum
            # is shorter than the line and is not the profile's length.
            "profile", float(result["surface_length_m"] if result.get("surface_length_complete")
                             else result["planar_length_m"]), "m", line_lonlat, dsm, details,
        )
        return ok(**{k: v for k, v in result.items() if k != "ok"})

    @guard
    def measure_drawn(self, kind: str, coordinates_lonlat: list[list[float]],
                      raster_path: str = "",
                      options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Measure what was drawn on the map, on the project's own rasters.

        ``kind`` is distance, area or perimeter (planar, in the raster's CRS, through
        measure_on_raster), or slope, profile, volume or cut_fill (on the DSM). The map
        gives lon/lat; converting to the raster's pixels or projected coordinates happens
        here, so the browser never computes a number it then presents as measured.
        """
        opts = dict(options or {})
        points = [[float(p[0]), float(p[1])] for p in (coordinates_lonlat or []) if len(p) >= 2]
        kind = str(kind or "").lower()
        if kind in {"volume", "cut_fill"}:
            base = opts.get("base_elevation_m") if kind == "cut_fill" else None
            if kind == "cut_fill" and base in (None, ""):
                return fail("Cut and fill needs the design (base) elevation to cut and "
                            "fill against.")
            return self.measure_volume(points, base, raster_path)
        if kind == "stockpile":
            return self.measure_stockpile(points, opts.get("base_elevation_m"))
        if kind == "profile":
            return self.elevation_profile(points, raster_path, opts.get("spacing_m"))
        if kind == "slope":
            dsm = raster_path or self._product_path("dsm")
            if not dsm:
                return fail("This project has no DSM to measure slope on. Run Process first.")
            try:
                polygon = self._lonlat_to_raster_xy(dsm, points)
            except (FileNotFoundError, ValueError) as exc:
                return fail(str(exc))
            result = self.measure_slope(dsm, polygon or None)
            if result.get("ok"):
                result["measurement_id"] = self._record_measurement(
                    "slope", float((result.get("slope") or {}).get("mean_deg") or 0.0), "deg",
                    points, dsm,
                    {k: v for k, v in result.items() if k != "ok"})
            return result
        if kind not in {"distance", "area", "perimeter"}:
            return fail(f"Unknown measurement kind {kind!r}. Use distance, area, perimeter, "
                        "profile, slope, volume, cut_fill or stockpile.")

        from core.dsm_analysis import elevation_profile as sample_profile
        from core.raster_measurement import NotGeoreferenced, NotProjected, lonlat_to_pixels

        raster = raster_path or self._product_path("orthomosaic") or self._product_path("dsm")
        if not raster:
            return fail("This project has no georeferenced raster to measure on. Run "
                        "Process first, or add a GeoTIFF layer.")
        try:
            pixels, caveats = lonlat_to_pixels(raster, points)
        except (NotGeoreferenced, NotProjected, FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        result = self.measure_on_raster(raster, [list(p) for p in pixels], kind)
        if not result.get("ok"):
            return result
        measurement = result["measurement"]
        measurement["caveats"] = list(measurement.get("caveats") or []) + caveats
        value_m = self._planar_value_in_metres(raster, kind, measurement)
        if value_m is None:
            return fail(f"{Path(raster).name} is in {measurement.get('unit')}, a unit with "
                        "no known length in metres, so the measurement cannot be given "
                        "in metres. Reproject the raster to a CRS in metres first.")
        payload: dict[str, Any] = {"measurement": measurement, "raster_path": raster,
                                   "value_m": value_m}
        dsm = self._product_path("dsm")
        if kind == "distance" and dsm:
            # The planar figure is the map distance; over sloping ground the walked
            # distance is longer, and the DSM knows by how much.
            try:
                along = sample_profile(dsm, self._lonlat_to_raster_xy(dsm, points))
            except (FileNotFoundError, ValueError):
                along = {}
            if along.get("ok") and along.get("surface_length_complete"):
                payload["surface_length_m"] = along["surface_length_m"]
            elif along.get("ok"):
                # A walked length summed over only the spans with elevation came out
                # shorter than the map distance on a real survey (96 m for a 100 m line).
                # Partial, so not offered as the over-the-ground length.
                payload["surface_note"] = (
                    f"Over-the-ground length not given: {along['samples_without_data']} of "
                    f"{along['sample_count']} samples along the line have no elevation "
                    f"(DSM holes or outside it). Over the {along['surface_covers_planar_m']} m "
                    f"that has data it is {along['surface_length_m']} m.")
        unit = "m2" if kind == "area" else "m"
        payload["measurement_id"] = self._record_measurement(
            kind, value_m, unit, points, raster,
            {"measurement": measurement,
             **{k: v for k, v in payload.items() if k.startswith("surface")}},
        )
        return ok(**payload)

    def _planar_value_in_metres(self, raster: str, kind: str,
                                measurement: dict[str, Any]) -> float | None:
        """The planar measurement in metres (m2 for an area), whatever the raster's unit.

        A raster in US survey feet measured a 100 m line as 328.18 and the history
        stored it as "328.18 m". A horizontal length converts exactly by the CRS's own
        unit factor; the native value and unit stay in the measurement for reference.
        """
        import rasterio

        from core.dsm_analysis import linear_unit_factor

        with rasterio.open(raster) as src:
            name, factor = linear_unit_factor(src.crs)
        if not (factor > 0):
            return None
        value = float(measurement["value"])
        if factor != 1.0:
            measurement["native_unit"] = name
            measurement["native_value"] = value
            measurement.setdefault("caveats", []).append(
                f"The raster is in {name}; converted to metres with its CRS factor "
                f"({factor} m per unit).")
        return value * (factor ** 2 if kind == "area" else factor)

    @guard
    def list_measurements(self) -> dict[str, Any]:
        """Every measurement taken in this project, newest first."""
        from core.measurements import list_measurements

        items = list_measurements(self._session.project_root(),
                                  str(self._session.project_id()))
        rows = sorted((m.to_dict() for m in items), key=lambda m: m["created_at"],
                      reverse=True)
        return ok(measurements=rows, count=len(rows))

    @guard
    def delete_measurement(self, measurement_id: str) -> dict[str, Any]:
        from core.measurements import delete_measurement

        removed = delete_measurement(self._session.project_root(), str(measurement_id))
        return ok(removed=True) if removed else fail(f"No measurement {measurement_id}.")

    @guard
    def compare_survey_volumes(self, earlier_path: str, later_path: str,
                               polygon_lonlat: list[list[float]] | None = None,
                               change_threshold_m: float = 0.05) -> dict[str, Any]:
        """Material added and removed between two surveys' DSMs.

        Two independent reconstructions are never on the same grid, so the later one is
        resampled onto the earlier one -- and the result says so, along with any
        systematic vertical offset between them, which a volume difference silently
        includes unless the surveys were co-registered.
        """
        from datetime import datetime, timezone

        from core.change_detection import IncomparableSurveys, compare_surfaces
        from core.dsm_analysis import NotGeoreferenced

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (self._session.project_root() / "analysis" / "change"
                  / f"difference_{stamp}.tif")
        try:
            roi = self._lonlat_to_raster_xy(earlier_path, polygon_lonlat) or None
            change = compare_surfaces(
                earlier_path, later_path, output_path=output,
                change_threshold_m=float(change_threshold_m), roi_polygon_xy=roi,
                resample=True,
            )
        except (NotGeoreferenced, IncomparableSurveys, FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        payload = change.to_dict()
        if payload.get("difference_path"):
            try:
                layer = self._session.register_raster_layer(
                    payload["difference_path"], name="Elevation change")
                # Only the handle: the layer's metadata carries a NaN nodata value,
                # which is not JSON.
                payload["layer"] = {"id": layer["id"], "name": layer["name"]}
            except Exception:  # noqa: BLE001 - the numbers stand without the layer
                pass
        payload["measurement_id"] = self._record_measurement(
            "surface_change", float(payload["net_volume_m3"]), "m3", polygon_lonlat,
            later_path, {k: v for k, v in payload.items() if k != "layer"},
        )
        return ok(**payload)

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

    def _project_mesh(self, model_path: str = "") -> Path:
        """The mesh to show or measure: the one named, or this project's reconstruction."""
        if model_path:
            return Path(model_path)
        recon = self._session.project_root() / "reconstruction"
        for name in ("mesh.obj", "mesh.ply"):
            if (recon / name).is_file():
                return recon / name
        raise FileNotFoundError(
            "This project has no reconstructed mesh yet. Run Process with a dataset; "
            "the mesh needs Open3D (`pip install open3d`)."
        )

    def _metric_mesh(self, model_path: str = "") -> Path:
        """The mesh in metres: the reconstruction's own, georeferenced when it needs it."""
        from core.model_measurement import NotMetric, georeference_model, model_scale

        mesh = self._project_mesh(model_path)
        try:
            model_scale(mesh)
            return mesh
        except NotMetric:
            if not (mesh.parent / "geo_anchor.json").is_file():
                raise
        return georeference_model(mesh)

    def _mesh_arrays(self, path: Path):
        """Vertices and faces, cached while the file is unchanged (a pick per click)."""
        from core.model_measurement import read_mesh

        key = (str(path), path.stat().st_mtime_ns)
        cache = getattr(self, "_mesh_cache", None)
        if cache is None or cache[0] != key:
            self._mesh_cache = (key, read_mesh(path))
        return self._mesh_cache[1]

    @guard
    def model_mesh_scene(self, model_path: str = "", max_faces: int = 120000) -> dict[str, Any]:
        """The project's mesh, georeferenced into its CRS, as display buffers.

        The cockpit's Digital Twin draws model_scene (the model file's own frame, mesh or
        cloud); this is the same mesh already carried into the CRS, for a client that
        works in survey coordinates and picks with pick_model_point.

        Coordinates are sent relative to ``origin`` so a WebGL float32 buffer keeps
        centimetres (UTM eastings do not fit in one). Large meshes are thinned by taking
        every n-th triangle for display only; picking and measuring use the full mesh.
        """
        import numpy as np

        from core.model_measurement import NotMetric, model_scale

        try:
            mesh = self._metric_mesh(model_path)
        except (NotMetric, FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        vertices, faces = self._mesh_arrays(mesh)
        total = int(len(faces))
        step = max(1, int(np.ceil(total / max(1, int(max_faces)))))
        shown = faces[::step]
        used, remapped = np.unique(shown.reshape(-1), return_inverse=True)
        local = vertices[used]
        origin = local.mean(axis=0) if len(local) else np.zeros(3)
        scale = model_scale(mesh)
        return ok(
            model_path=str(mesh),
            epsg=scale.epsg,
            origin=[float(v) for v in origin],
            positions=[round(float(v), 4) for v in (local - origin).reshape(-1)],
            indices=[int(v) for v in remapped],
            face_count=total,
            shown_face_count=int(len(shown)),
            bounds=[[float(v) for v in vertices.min(axis=0)],
                    [float(v) for v in vertices.max(axis=0)]],
        )

    @guard
    def pick_model_point(self, model_path: str, origin_xyz: list[float],
                         direction_xyz: list[float]) -> dict[str, Any]:
        """Where a ray from the 3D view first meets the full-resolution mesh."""
        from core.model_measurement import NotMetric, pick_on_mesh

        try:
            mesh = self._metric_mesh(model_path)
            vertices, faces = self._mesh_arrays(mesh)
            hit = pick_on_mesh(vertices, faces, origin_xyz, direction_xyz)
        except (NotMetric, FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        if hit is None:
            return fail("That click is not on the model.")
        return ok(point=[float(v) for v in hit], model_path=str(mesh))

    @guard
    def measure_in_model(self, model_path: str = "", kind: str = "length",
                         points_xyz: list[list[float]] | None = None) -> dict[str, Any]:
        """Length, height, area or volume inside a reconstructed 3D model.

        Refused unless the model's vertices are in a recorded CRS. The reconstruction's
        own mesh is in the structure-from-motion frame with the transform beside it, so
        it is measured on its georeferenced copy; a model with no georeferencing is
        refused, since SfM units read exactly like metres. The result is kept in the
        project's measurement history.
        """
        from core import geo
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
            mesh = self._metric_mesh(model_path)
            if kind == "volume":
                measurement = measure_volume(mesh)
                value, unit = measurement["volume_m3"], "m3"
            elif kind == "area":
                measurement = measure_area(mesh, points)
                value, unit = measurement.get("surface_area_m2"), "m2"
            elif kind in {"length", "height"}:
                if len(points) < 2:
                    return fail(f"A {kind} needs two picked points.")
                handler = measure_length if kind == "length" else measure_height
                measurement = handler(mesh, points[0], points[1])
                value = (measurement.get("slope_distance_m") if kind == "length"
                         else measurement.get("height_m"))
                unit = "m"
            else:
                return fail(
                    f"Unknown measurement kind {kind!r}. Use length, height, area or volume."
                )
        except (NotMetric, NotClosed) as exc:
            # Written for an operator; passing them through beats a generic message.
            return fail(str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))

        lonlat = []
        if points:
            lon, lat = geo.projected_to_wgs84([p[0] for p in points], [p[1] for p in points],
                                              int(measurement["crs_epsg"]))
            lonlat = [[float(a), float(b)] for a, b in zip(lon, lat)]
        measurement_id = None
        if value is not None:
            measurement_id = self._record_measurement(
                f"model_{kind}", float(value), unit, lonlat, str(mesh),
                {**measurement, "points_xyz": points})
        return ok(measurement=measurement, model_path=str(mesh), measurement_id=measurement_id)

    # ------------------------------------------------------------ survey packs
    #
    # The India-first packs (land GIS, design progress, agriculture, road/power/rail/
    # solar assets, thermal module temperatures, anomaly candidates) were library code
    # that nothing in the application could reach. Each entry below runs one pack on
    # files the operator names -- or, for the semantic ones, on this project's latest
    # semantic map -- writes into the project folder, and adds what it produced to the
    # layer tree. Every refusal the packs raise is passed through as written.

    def _pack_dir(self, pack: str) -> Path:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target = self._session.project_root() / "survey_packs" / pack / stamp
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _latest_semantic(self) -> tuple[str, str]:
        """This project's most recent semantic class map and the manifest beside it."""
        root = self._session.project_root() / "semantic"
        runs = sorted((p for p in root.glob("*/semantic_manifest.json")), key=lambda p: p.stat().st_mtime) \
            if root.is_dir() else []
        if not runs:
            return "", ""
        manifest = runs[-1]
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        class_map = (payload.get("artifacts") or {}).get("class_map") or ""
        return str(class_map), str(manifest)

    def _semantic_inputs(self, semantic_raster: str, schema: str) -> tuple[str, str]:
        if semantic_raster:
            if not schema:
                beside = Path(semantic_raster).parent / "semantic_manifest.json"
                if beside.is_file():
                    schema = str(beside)
            if not schema:
                raise ValueError(
                    "Name the class schema (a semantic manifest or schema JSON) for this "
                    "raster: class ids mean nothing without it.")
            return semantic_raster, schema
        raster, manifest = self._latest_semantic()
        if not raster:
            raise ValueError(
                "This project has no semantic map yet. Run Semantic Map on the "
                "orthomosaic, or name a semantic class raster.")
        return raster, schema or manifest

    def _finish_pack(self, pack: str, result: Any) -> dict[str, Any]:
        """Read the pack's summary and put its GeoJSON and GeoTIFF outputs on the map."""
        summary_path = getattr(result, "summary_path", "")
        artifacts = (result.artifact_paths() if callable(getattr(result, "artifact_paths", None))
                     else list(getattr(result, "artifact_paths", ())))
        summary = {}
        if summary_path and Path(summary_path).is_file():
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        layers = []
        for path in artifacts:
            target = Path(path)
            if not target.is_file():
                continue
            label = f"{pack}: {target.stem}"
            try:
                if target.suffix.lower() in (".geojson", ".json") and target != Path(summary_path):
                    payload = json.loads(target.read_text(encoding="utf-8"))
                    if payload.get("type") == "FeatureCollection":
                        layers.append(self._session.register_vector_layer(target, name=label)["id"])
                elif target.suffix.lower() in (".tif", ".tiff"):
                    layers.append(self._session.register_raster_layer(target, name=label)["id"])
            except Exception:  # noqa: BLE001 - a layer that will not register is still on disk
                continue
        self._session.audit("survey_pack_run", {"pack": pack, "summary": str(summary_path)})
        return ok(pack=pack, summary=summary, summary_path=str(summary_path),
                  artifacts=[str(p) for p in artifacts], layer_ids=layers)

    def _pack(self, pack: str, run: Callable[[Path], Any]) -> dict[str, Any]:
        from core.india_geospatial import IndiaPackRefused

        try:
            result = run(self._pack_dir(pack))
        except (IndiaPackRefused, ValueError, FileNotFoundError) as exc:
            return fail(str(exc))
        return self._finish_pack(pack, result)

    @guard
    def semantic_map(self, orthomosaic_path: str = "", model_path: str = "",
                     manifest_path: str = "", device: str = "auto") -> dict[str, Any]:
        """Run the installed semantic model over an orthomosaic, as a background job.

        The model's own registry entry says which classes hold up (vegetation and
        building for shared_semantic v4) and which do not; that is returned with the
        map, because a road or water polygon from it looks exactly as confident as a
        building one.
        """
        from core.models import resolve_model_path

        model = Path(model_path) if model_path else resolve_model_path("shared_semantic_landcover")
        if model is None or not Path(model).is_file():
            return fail("No semantic model is installed (shared_semantic_landcover). Install "
                        "its ONNX package into models/, or name one.")
        manifest = Path(manifest_path) if manifest_path else Path(
            str(model).replace(".onnx", ".manifest.json"))
        if not manifest.is_file():
            return fail(f"The semantic model's manifest is missing: {manifest}")
        source = orthomosaic_path
        if not source:
            ortho = next((l for l in self._session.layers.values()
                          if l.kind == "raster" and "ortho" in (l.name or "").lower()), None)
            source = ortho.path if ortho else ""
        if not source or not Path(source).is_file():
            return fail("Name a georeferenced orthomosaic, or run Process to produce one.")
        from datetime import datetime, timezone

        from core.models import load_registry

        entry = (load_registry().get("models") or {}).get("shared_semantic_landcover") or {}
        usable = list(entry.get("usable_labels") or [])
        output = (self._session.project_root() / "semantic"
                  / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
        session = self._session

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            import onnxruntime as ort

            from core.semantic_engine import (ONNXSemanticPredictor, SemanticInferenceConfig,
                                              load_semantic_manifest, run_semantic_inference)

            schema, metadata, defaults = load_semantic_manifest(manifest)
            chosen = device
            if chosen == "auto":
                chosen = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
            predictor = ONNXSemanticPredictor(model, device=chosen,
                                              mean=defaults.get("mean", (0.485, 0.456, 0.406)),
                                              std=defaults.get("std", (0.229, 0.224, 0.225)))
            # ONNX Runtime can list CUDA and then load on the CPU (cuBLAS/cuDNN missing);
            # the predictor reports the device that actually loaded. "auto" means run on
            # whatever works, so a CPU fallback is allowed -- still under the CPU pixel
            # limit -- while an explicit "cuda" request that could not get a GPU refuses.
            actual = str(getattr(predictor, "device", chosen))
            progress(5, f"Semantic inference on {actual}"
                        + (" (CUDA did not load)" if chosen == "cuda" and actual == "cpu" else ""))
            package = run_semantic_inference(
                source, output, schema=schema, model=metadata, predictor=predictor,
                config=SemanticInferenceConfig(
                    tile_size=int(defaults.get("tile_size", 518)),
                    overlap=int(defaults.get("overlap", 126)),
                    device=actual,
                    allow_cpu=device != "cuda",
                    max_cpu_pixels=int(defaults.get("max_cpu_pixels", 4_000_000)),
                    min_polygon_area_m2=float(defaults.get("min_polygon_area_m2", 1.0)),
                ))
            session.register_raster_layer(package.class_map_path, name="Semantic classes")
            session.register_vector_layer(package.polygons_path, name="Semantic polygons")
            return {"class_map": package.class_map_path, "polygons": package.polygons_path,
                    "manifest": package.manifest_path, "usable_labels": usable,
                    "caveat": ("Only " + ", ".join(usable) + " are measured to hold up for this "
                               "model; treat every other class as unvalidated.") if usable else
                              "This model has no validated classes recorded."}

        return ok(job_id=self._jobs.submit("semantic_map", work), usable_labels=usable)

    @guard
    def india_land_gis(self, semantic_raster: str = "", schema: str = "",
                       cadastral_geojson: str = "", previous_semantic_raster: str = "",
                       boundary_tolerance_m: float = 0.0) -> dict[str, Any]:
        """Land-cover vectors from a semantic map; encroachment when a boundary is given."""
        from core.india_land import detect_cadastral_encroachment, extract_land_gis

        try:
            raster, classes = self._semantic_inputs(semantic_raster, schema)
        except ValueError as exc:
            return fail(str(exc))
        if cadastral_geojson:
            return self._pack("land_encroachment", lambda out: detect_cadastral_encroachment(
                raster, classes, cadastral_geojson, out,
                previous_semantic_class_raster=previous_semantic_raster or None,
                boundary_tolerance_m=float(boundary_tolerance_m)))
        return self._pack("land_gis", lambda out: extract_land_gis(raster, classes, out))

    @guard
    def india_design_progress(self, design_geojson: str, semantic_raster: str = "",
                              schema: str = "", complete_at_percent: float = 95.0) -> dict[str, Any]:
        """Observed coverage inside each approved design element (not contract completion)."""
        from core.india_construction import measure_approved_design_progress

        try:
            raster, classes = self._semantic_inputs(semantic_raster, schema)
        except ValueError as exc:
            return fail(str(exc))
        return self._pack("design_progress", lambda out: measure_approved_design_progress(
            raster, classes, design_geojson, out, complete_at_percent=float(complete_at_percent)))

    @guard
    def india_canopy_cover(self, semantic_raster: str = "", schema: str = "") -> dict[str, Any]:
        """Canopy and bare/missing regions from explicit semantic classes."""
        from core.india_agriculture import analyse_canopy_cover

        try:
            raster, classes = self._semantic_inputs(semantic_raster, schema)
        except ValueError as exc:
            return fail(str(exc))
        return self._pack("canopy", lambda out: analyse_canopy_cover(raster, classes, out))

    @guard
    def india_vegetation_indices(self, multispectral_raster: str,
                                 calibrations: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """NDVI/NDRE/GNDVI from explicitly calibrated reflectance bands only.

        ``calibrations`` maps a band role (red, green, red_edge, nir) to {band, scale,
        offset, source}; a role with no calibration leaves its indices unavailable.
        """
        from core.india_agriculture import ReflectanceBandCalibration, compute_vegetation_indices

        try:
            parsed = {str(role): ReflectanceBandCalibration(
                band=int(c["band"]), scale=float(c["scale"]), offset=float(c.get("offset", 0.0)),
                source=str(c.get("source", ""))) for role, c in (calibrations or {}).items()}
        except (KeyError, TypeError, ValueError) as exc:
            return fail(f"Calibration not usable: {exc}")
        return self._pack("vegetation_indices", lambda out: compute_vegetation_indices(
            multispectral_raster, parsed, out))

    @guard
    def india_stress_zones(self, index_raster: str, severe_below: float, moderate_below: float,
                           validation_scope: str, canopy_mask_raster: str = "") -> dict[str, Any]:
        """Stress zones from a calibrated index, with thresholds the operator states."""
        from core.india_agriculture import create_stress_zones

        return self._pack("stress_zones", lambda out: create_stress_zones(
            index_raster, out, severe_below=float(severe_below),
            moderate_below=float(moderate_below), validation_scope=str(validation_scope),
            canopy_mask_raster=canopy_mask_raster or None))

    @guard
    def india_plant_count(self, instance_raster: str, validation_scope: str,
                          instance_encoding: str = "binary") -> dict[str, Any]:
        """Geolocated plant/tree instances from a validated instance mask."""
        from core.india_agriculture import count_plant_instances

        return self._pack("plant_count", lambda out: count_plant_instances(
            instance_raster, out, validation_scope=str(validation_scope),
            instance_encoding=str(instance_encoding)))

    @guard
    def india_road_condition(self, road_centerline_geojson: str, defect_findings_geojson: str,
                             semantic_raster: str = "", schema: str = "") -> dict[str, Any]:
        """Road surface, surveyed distance and validated findings along a centreline."""
        from core.india_assets import create_road_condition_package

        try:
            raster, classes = self._semantic_inputs(semantic_raster, schema)
        except ValueError as exc:
            return fail(str(exc))
        return self._pack("road_condition", lambda out: create_road_condition_package(
            raster, classes, road_centerline_geojson, defect_findings_geojson, out))

    @guard
    def india_corridor_inspection(self, kind: str, detections_geojson: str,
                                  corridor_centerline_geojson: str, capture_geometry: str,
                                  validation_scope: str) -> dict[str, Any]:
        """Power-line or railway corridor findings mapped along the corridor."""
        from core.india_assets import create_power_inspection_package, create_rail_inspection_package

        builder = {"power": create_power_inspection_package,
                   "rail": create_rail_inspection_package}.get(str(kind))
        if builder is None:
            return fail("Corridor kind must be 'power' or 'rail'.")
        return self._pack(f"{kind}_corridor", lambda out: builder(
            detections_geojson, corridor_centerline_geojson, out,
            capture_geometry=str(capture_geometry), validation_scope=str(validation_scope)))

    @guard
    def india_solar_inventory(self, module_instance_raster: str, validation_scope: str,
                              approved_layout_geojson: str = "",
                              findings_geojson: str = "") -> dict[str, Any]:
        """Module inventory from instances, compared with an approved layout if given."""
        from core.india_assets import create_solar_module_inventory

        return self._pack("solar_inventory", lambda out: create_solar_module_inventory(
            module_instance_raster, out, validation_scope=str(validation_scope),
            approved_layout_geojson=approved_layout_geojson or None,
            findings_geojson=findings_geojson or None))

    @guard
    def india_module_temperatures(self, thermal_path: str, rgb_path: str,
                                  control_points_path: str, modules_path: str,
                                  validated_by: str, warning_delta_c: float,
                                  critical_delta_c: float, hot_cell_delta_c: float,
                                  threshold_basis: str) -> dict[str, Any]:
        """Measured module temperatures, after the RGB/thermal tie points are checked.

        Positional on purpose: the bridge calls with positional arguments only, and
        keyword-only parameters made the cockpit's call a TypeError every time.
        """
        from core.india_thermal import associate_module_temperatures, score_rgb_thermal_registration

        def run(out: Path):
            registration = score_rgb_thermal_registration(
                rgb_path, thermal_path, control_points_path, validated_by=str(validated_by))
            return associate_module_temperatures(
                thermal_path, registration, modules_path, out,
                warning_delta_c=float(warning_delta_c), critical_delta_c=float(critical_delta_c),
                hot_cell_delta_c=float(hot_cell_delta_c), threshold_basis=str(threshold_basis))

        return self._pack("module_temperatures", run)

    @guard
    def india_anomaly_baseline(self, raster_paths: list[str], validated_by: str,
                               validation_scope: str) -> dict[str, Any]:
        """Fit a normal baseline from rasters someone has confirmed to be normal."""
        from core.india_anomaly import fit_validated_baseline
        from core.india_geospatial import IndiaPackRefused

        target = self._pack_dir("anomaly_baseline") / "baseline.json"
        try:
            baseline = fit_validated_baseline(list(raster_paths), target,
                                              validated_by=str(validated_by),
                                              validation_scope=str(validation_scope))
        except (IndiaPackRefused, ValueError, FileNotFoundError) as exc:
            return fail(str(exc))
        return ok(pack="anomaly_baseline", baseline_path=str(target), baseline=baseline.to_dict())

    @guard
    def india_find_anomalies(self, feature_raster: str, baseline_path: str,
                             threshold_sigma: float = 3.5) -> dict[str, Any]:
        """Deviation candidates against a validated baseline (never named as defects)."""
        from core.india_anomaly import detect_anomaly_candidates

        return self._pack("anomalies", lambda out: detect_anomaly_candidates(
            feature_raster, baseline_path, out, threshold_sigma=float(threshold_sigma)))

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

    def _mission_constraints(self, opts: dict[str, Any]) -> Any:
        """The operator's constraints for a plan over the drawn AOI."""
        from mission.planner import MissionConstraints

        polygon = self._session.aoi_polygon
        return MissionConstraints(
            geofence=[list(p) for p in polygon],
            min_altitude_m=float(opts.get("min_altitude_m", 20.0)),
            max_altitude_m=float(opts.get("max_altitude_m", 120.0)),
            # 0 means "not given": structure templates then use their own stand-off.
            standoff_m=float(opts.get("standoff_m", 0.0) or 0.0),
            rth_altitude_m=float(opts.get("rth_altitude_m", 80.0)),
            no_fly_polygons=[list(map(list, ring)) for ring in self._session.no_fly_polygons],
            rth_action=str(opts.get("rth_action", "return_home")),
            obstacle_avoidance_profile=str(opts.get("obstacle_profile", "standard")),
        )

    def _terrain_kwargs(self, opts: dict[str, Any]) -> dict[str, Any]:
        if not self._session.terrain_source_path:
            return {}
        return {
            "terrain_follow_enabled": True,
            "terrain_source_path": self._session.terrain_source_path,
            "terrain_follow_mode": str(opts.get("terrain_follow_mode", "agl")),
            "terrain_normal_camera_enabled": bool(opts.get("terrain_normal_camera", False)),
        }

    @staticmethod
    def _given(opts: dict[str, Any], key: str) -> bool:
        """Whether the operator actually supplied a value, rather than leaving it blank."""
        value = opts.get(key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, float) and value != value:  # NaN from an empty number box
            return False
        return True

    def _adopt_plan(self, plan: Any, template: str, warnings: list[str],
                    extra: dict[str, Any] | None = None,
                    options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make `plan` the session's mission and describe it the way plan_mission does.

        Every planner route ends here, so a plan from the pylon, thermal or
        multispectral planners can be exported, uploaded and saved exactly like one from
        plan_mission -- before this, those three returned a plan nothing could use.
        """
        self._session.mission_plan = plan
        self._session.mission_plan_dict = _plan_to_dict(plan)
        if extra:
            self._session.mission_plan_dict.update(extra)
        # Whatever was uploaded before is not this plan, and nothing about it has been
        # confirmed by the operator yet.
        self._session.mission_uploaded = False
        self._session.preflight_acknowledged = set()
        self._session.audit("mission_planned", {"template": template, "waypoints": len(plan.waypoints)})
        # Extended in place: plan_mission keeps appending to this same list afterwards.
        warnings.extend(self._endurance_warnings(plan, options or {}))
        return {
            "summary": {
                "template": template,
                "waypoints": len(plan.waypoints),
                "distance_m": round(float(plan.path_distance_m), 1),
                "duration_min": round(float(plan.estimated_time_min), 1),
                "gsd_cm": round(float(plan.estimated_gsd_cm), 2),
                "altitude_m": float(plan.altitude_m),
                "speed_m_s": round(float(getattr(plan, "cruise_speed_m_s", 0.0) or 0.0), 2),
                "altitude_frame": str(getattr(plan, "altitude_frame", "relative_to_home")),
                "adjustments": plan.safety_adjustments,
            },
            "geojson": plan.geojson,
            "warnings": warnings,
        }

    @staticmethod
    def _endurance_warnings(plan: Any, opts: dict[str, Any]) -> list[str]:
        """Say so when the plan is longer than one battery, on every planner route.

        A 55 minute facade mapping used to plan with no word about batteries; the
        estimate existed only behind a separate call. The endurance is the operator's
        (endurance_min / reserve_pct) or a stated generic default, and the warning
        names which, so an assumption is not presented as the aircraft's figure.
        """
        from mission.estimates import AircraftProfile, estimate_batteries

        minutes = float(getattr(plan, "estimated_time_min", 0.0) or 0.0)
        given = opts.get("endurance_min") not in (None, "")
        try:
            aircraft = AircraftProfile(
                endurance_min=float(opts.get("endurance_min") or 25.0),
                reserve_pct=float(opts.get("reserve_pct") or 25.0),
            )
            estimate = estimate_batteries(minutes, aircraft)
        except (TypeError, ValueError):
            return []
        if estimate.fits_in_one_flight:
            return []
        basis = ("the endurance you gave" if given else
                 "an assumed 25 min endurance (set endurance_min for your aircraft)")
        return [
            f"This plan needs about {minutes:.0f} min of flying; with {basis} and a "
            f"{aircraft.reserve_pct:.0f}% reserve one battery gives "
            f"{estimate.usable_endurance_min:.0f} min, so it takes "
            f"{estimate.batteries_required} batteries. Split it into sorties "
            "(plan_battery_segments) before flying."
        ]

    def _plan_special(self, template: str, opts: dict[str, Any]) -> dict[str, Any]:
        """pylon_inspection / thermal / multispectral, reached with the same form."""
        if template == "pylon_inspection":
            center = opts.get("pylon_center_lonlat") or opts.get("center_lonlat")
            if not center:
                if len(self._session.aoi_polygon) < 3:
                    return fail("Draw an area around the pylon, or give its centre, before "
                                "planning a pylon inspection.")
                center = _centroid(self._session.aoi_polygon)
            elements = opts.get("pylon_elements") or []
            if isinstance(elements, str):
                elements = _parse_pylon_elements(elements)
            return self.plan_pylon_mission(
                list(center), list(elements),
                standoff_m=float(opts.get("standoff_m") or 12.0),
                structure_radius_m=float(opts.get("structure_radius_m") or 3.0),
                camera=str(opts.get("camera") or "mavic2pro"),
            )
        if template == "thermal":
            if not self._given(opts, "thermal_camera"):
                return fail("A thermal survey needs the thermal camera it will be flown with.")
            if not self._given(opts, "target_gsd_cm"):
                return fail("A thermal survey needs the thermal ground sample distance to fly "
                            "for (target_gsd_cm); the altitude is solved from it.")
            return self.plan_thermal_survey(
                str(opts.get("thermal_camera")), float(opts.get("target_gsd_cm")),
                rgb_camera=str(opts.get("rgb_camera") or ""),
                front_overlap_pct=float(opts.get("front_overlap_pct", 80.0)),
                side_overlap_pct=float(opts.get("side_overlap_pct", 70.0)),
                options=opts,
            )
        if not self._given(opts, "payload_key") and not self._given(opts, "payload"):
            return fail("A multispectral survey needs the multispectral payload it will carry.")
        panel = opts.get("calibration_panel_lonlat")
        if isinstance(panel, str):
            panel = [float(v) for v in panel.replace(";", ",").split(",") if v.strip()] or None
        return self.plan_multispectral_survey(
            str(opts.get("payload_key") or opts.get("payload")),
            camera=str(opts.get("camera") or "mavic3e"),
            altitude_m=float(opts.get("altitude_m", 60.0)),
            calibration_panel_lonlat=panel,
            front_overlap_pct=float(opts.get("front_overlap_pct", 80.0)),
            side_overlap_pct=float(opts.get("side_overlap_pct", 75.0)),
            options=opts,
            # The cockpit form sends strings; "no" must not read as True.
            accept_relative_values=str(opts.get("accept_relative_values") or "").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    @guard
    def plan_linked_mission(self, version_ids: list[int], speed_m_s: float = 5.0,
                            optimize_order: bool = True) -> dict[str, Any]:
        """Join saved missions into one sortie, with transitions planned between them.

        A linked mission is made of missions, so its input is two or more saved mission
        versions (by id). The transition dry run checks each connecting leg against the
        no-fly zones and altitude band; a failed dry run is returned as a warning on the
        plan, not buried in a field nobody reads.
        """
        from mission.planner import MissionPlanner

        wanted = [int(v) for v in version_ids or []]
        if len(wanted) < 2:
            return fail("A linked mission joins two or more saved missions. Save each mission "
                        "(Mission > Save Mission Version), then link them by version.")
        versions = {int(v.get("id", 0)): v for v in
                    self._session.store.list_mission_versions(self._session.project_id())}
        missing = [v for v in wanted if v not in versions]
        if missing:
            return fail(f"No saved mission version with id {missing}.")
        recipes = [json.loads(versions[v].get("flight_recipe_json") or "{}") for v in wanted]
        constraints = None
        if self._session.no_fly_polygons:
            # The zones drawn now, on top of the saved missions' own altitude band. A
            # partial override resets the band to defaults and clamps every pose.
            bands = [r.get("constraints") or {} for r in recipes]
            constraints = {
                "no_fly_polygons": [list(map(list, r)) for r in self._session.no_fly_polygons],
                "min_altitude_m": min(float(b.get("min_altitude_m", 20.0)) for b in bands),
                "max_altitude_m": max(float(b.get("max_altitude_m", 120.0)) for b in bands),
                "rth_altitude_m": max(float(b.get("rth_altitude_m", 80.0)) for b in bands),
            }
        try:
            plan = MissionPlanner().generate_linked_mission(
                recipes,
                speed_m_s=float(speed_m_s),
                optimize_order=bool(optimize_order),
                constraints=constraints,
            )
        except ValueError as exc:
            return fail(str(exc))
        metadata = (plan.flight_recipe or {}).get("metadata") or {}
        warnings: list[str] = []
        if not plan.linked_dry_run_ok:
            warnings.append(
                "The transition dry run failed: "
                f"{int(metadata.get('linked_dry_run_obstacle_hits', 0))} connecting leg(s) cross a "
                f"no-fly zone and {int(metadata.get('linked_dry_run_altitude_violations', 0))} "
                "leave the altitude band. Review the transitions before flying this sortie."
            )
        adopted = self._adopt_plan(plan, "linked_mission", warnings)
        return ok(linked_segment_count=int(plan.linked_segment_count),
                  linked_dry_run_ok=bool(plan.linked_dry_run_ok), **adopted)

    @guard
    def plan_mission(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plan a mission over the drawn AOI with the operator's real constraints."""
        from mission.mission_types import SPECIAL_MISSION_TYPES
        from mission.planner import MissionPlanner, _normalize_template

        opts = dict(options or {})
        requested = str(opts.get("template", "grid")).strip().lower()
        if requested in SPECIAL_MISSION_TYPES:
            return self._plan_special(requested, opts)
        template = _normalize_template(requested)
        if template == "linked_mission":
            ids = opts.get("linked_version_ids")
            if isinstance(ids, str):
                ids = [v for v in ids.replace(";", ",").replace(" ", ",").split(",") if v.strip()]
            elif isinstance(ids, (int, float)):
                ids = [ids]
            if ids:
                return self.plan_linked_mission([int(float(v)) for v in ids],
                                                speed_m_s=float(opts.get("speed_m_s", 5.0)))
            # Planning a "linked mission" from one drawn area used to compile a mix of
            # facade, orbit and grid passes and report success with zero linked segments.
            return fail("A linked mission joins two or more saved missions, so it cannot be "
                        "planned from an area alone. Save each mission (Mission > Save "
                        "Mission Version), then link them by version.")

        polygon = self._session.aoi_polygon
        if len(polygon) < 3:
            return fail("Draw an area of interest on the map before planning a mission.")

        constraints = self._mission_constraints(opts)
        kwargs: dict[str, Any] = {
            "polygon_lonlat": polygon,
            "altitude_m": float(opts.get("altitude_m", 55.0)),
            "front_overlap_pct": float(opts.get("front_overlap_pct", 75.0)),
            "side_overlap_pct": float(opts.get("side_overlap_pct", 65.0)),
            "speed_m_s": float(opts.get("speed_m_s", 8.0)),
            "mode": str(opts.get("template", "grid")),
            "wind_speed_m_s": float(opts.get("wind_speed_m_s", 0.0)),
            "wind_direction_deg": float(opts.get("wind_direction_deg", 0.0)),
            "wind_gust_m_s": float(opts.get("wind_gust_m_s", 0.0)),
            "constraints": constraints,
        }
        # Left blank, the gimbal and dwell come from the template: an orbit looks at its
        # subject and a roof inspection stops at each point. Sending -90 and 0 for every
        # template made every structure mission a set of nadir photographs.
        if self._given(opts, "gimbal_tilt_deg"):
            kwargs["gimbal_tilt_deg"] = float(opts["gimbal_tilt_deg"])
        if self._given(opts, "dwell_s"):
            kwargs["inspection_dwell_s"] = float(opts["dwell_s"])
        if opts.get("camera"):
            kwargs["camera"] = str(opts["camera"])
        kwargs.update(self._terrain_kwargs(opts))
        if opts.get("orbit_radius_m"):
            kwargs["orbit_radius_m"] = float(opts["orbit_radius_m"])
            kwargs["orbit_center_lonlat"] = opts.get("orbit_center_lonlat") or _centroid(polygon)
        if self._given(opts, "dome_radius_m"):
            kwargs["dome_radius_m"] = float(opts.get("dome_radius_m"))
        if self._given(opts, "dome_base_altitude_m"):
            kwargs["dome_base_altitude_m"] = float(opts.get("dome_base_altitude_m"))
        if opts.get("box_altitude_levels_m"):
            levels = opts["box_altitude_levels_m"]
            if isinstance(levels, str):
                levels = [v for v in levels.replace(";", ",").split(",") if v.strip()]
            kwargs["box_altitude_levels_m"] = [float(v) for v in levels]

        warnings: list[str] = []
        # A line the operator drew: the route for linear / lateral / waypoint missions,
        # the row for solar, the wall line for a curved facade.
        path = [list(map(float, p[:2])) for p in (opts.get("path_lonlat") or []) if len(p) >= 2]
        if len(path) >= 2:
            if template == "linear_inspection":
                kwargs["linear_path_lonlat"] = path
            elif template == "lateral_capture":
                kwargs["lateral_target_path_lonlat"] = path
            elif template == "waypoints":
                kwargs["waypoint_path_lonlat"] = path
            elif template == "solar_inspection":
                kwargs["solar_rows_lonlat"] = [path]
            elif template in {"facade", "facade_mapping"}:
                kwargs["facade_curve_path_lonlat"] = path
                kwargs["facade_curvature_alignment"] = True
        elif template in {"linear_inspection", "lateral_capture", "waypoints"}:
            warnings.append(
                "No route line was drawn, so this plan follows the area's outline or long "
                "axis. Draw the route with the Route tool to fly the actual line."
            )
        regions = [r for r in (opts.get("interest_regions") or []) if len(r) >= 3]
        defects = [p for p in (opts.get("prior_defects_lonlat") or []) if len(p) >= 2]
        if template == "smart_adaptive":
            if regions:
                kwargs["adaptive_interest_regions_lonlat"] = regions
            if defects:
                kwargs["adaptive_prior_defects_lonlat"] = defects
            if not regions and not defects:
                warnings.append(
                    "No interest regions or prior defects were given, so there is nothing "
                    "to adapt to: this plan is a uniform grid and is labelled as one. Draw "
                    "interest regions with the Interest tool to add the detail passes."
                )

        plan = MissionPlanner().generate(**kwargs)
        adopted = self._adopt_plan(plan, kwargs["mode"], warnings, options=opts)
        adopted["summary"]["template"] = kwargs["mode"]

        # The recipe carries the resolved terrain model under `metadata`; reading it
        # from the recipe root silently yielded None, so no terrain warning ever fired.
        recipe = self._session.mission_plan_dict.get("flight_recipe") or {}
        terrain = (recipe.get("metadata") or {}).get("terrain_model") or recipe.get("terrain_model") or {}
        if self._session.terrain_source_path and terrain.get("source") == "missing_terrain_source":
            detail = ""
            if self._session.terrain_source_path.lower().endswith((".csv", ".tsv")):
                from mission.planner import TERRAIN_CSV_ELEVATION_COLUMNS, terrain_csv_columns

                columns = terrain_csv_columns(self._session.terrain_source_path)
                detail = (f" Its columns are {columns}; a terrain CSV needs lon/lat (or x/y) and "
                          f"an elevation column named one of {TERRAIN_CSV_ELEVATION_COLUMNS}.")
            warnings.append(
                "Terrain following was requested but the source could not be read; "
                "the plan assumes flat ground." + detail
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
        if str(opts.get("terrain_follow_mode", "agl")).lower() == "amsl" and plan.altitude_frame != "amsl":
            warnings.append(
                "AMSL altitudes need a terrain model to know the ground elevation at home; "
                "without one the mission is exported relative to home instead."
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
                    extent["detail"] if extent.get("reason") == "local_coordinates" else
                    f"Terrain source does not cover the planned area: {extent['detail']}"
                )

        return ok(payload=payload_block, **adopted)

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
                # A format with a waypoint limit (Litchi) writes several parts; each is
                # listed so none of them is silently left behind in the folder.
                exporters.record_written(
                    written, name, writer(target / f"{stem}{suffix}", self._session.mission_plan_dict))
            except Exception as exc:  # noqa: BLE001
                written[name] = f"failed: {type(exc).__name__}: {exc}"
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
    def connect_vehicle(self, uri: str, driver: str = "pymavlink") -> dict[str, Any]:
        vehicle = self._session.connect_vehicle(uri, driver)
        log = self._session.flight_log
        return ok(vehicle=vehicle, flight_log={
            "driver": getattr(log, "driver", ""),
            "simulated": bool(getattr(log, "is_simulated", False)),
            "archived": (self._session.archived_flight_logs[-1]
                         if self._session.archived_flight_logs else None),
        })

    @guard
    def disconnect_vehicle(self) -> dict[str, Any]:
        self._session.disconnect_vehicle()
        return ok(vehicle=self._session.vehicle.to_dict())

    @guard
    def telemetry(self) -> dict[str, Any]:
        return ok(telemetry=self._session.telemetry())

    @guard
    def upload_mission(self) -> dict[str, Any]:
        """Upload the planned mission, geofence, and rally points to the vehicle.

        Reports what the vehicle answered. This returned ok whether the vehicle accepted
        the mission, rejected it, or never answered, so both UIs said "Mission uploaded"
        over a MISSION_ACK that said the opposite -- and the crash-recovery record then
        claimed a mission was on the aircraft.
        """
        from mission import exporters

        if not self._session.mission_plan_dict:
            return fail("No mission has been planned yet.")
        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")
        uploader = getattr(client, "upload_mission", None)
        if uploader is None:
            return fail(f"The {self._session.vehicle.driver!r} driver does not support mission upload.")

        plan = self._session.mission_plan_dict
        items = [item.to_dict() for item in exporters.build_mission_items(plan)]
        fence = [item.to_dict() for item in exporters.build_fence_items(plan)]
        rally = [item.to_dict() for item in exporters.build_rally_items(plan)]

        manager = self._session.flight_manager
        logged = manager.active_session() is not None

        def run(name: str, action: Callable[[], Any]) -> Any:
            return manager.execute(name, action, reason="operator_upload") if logged else action()

        mission_result = run("upload_mission", lambda: uploader(items))
        report: dict[str, Any] = {"mission": _command_payload(mission_result, len(items))}
        self._session.audit("mission_uploaded", {
            "items": len(items), "driver": self._session.vehicle.driver,
            "success": bool(getattr(mission_result, "success", False)),
        })
        if not getattr(mission_result, "success", False):
            self._session.mission_uploaded = False
            return fail(
                getattr(mission_result, "message", "") or "The vehicle did not accept the mission.",
                items=len(items), result=report["mission"], upload=report,
            )

        # The mission is on the aircraft now, whatever happens to fence and rally.
        self._session.mission_uploaded = True
        from core.flight_state import PHASE_UPLOADED, record_transition

        # Recorded so that, if this software dies before the mission ends, the next
        # session knows a mission was on the aircraft rather than guessing. Only after
        # the vehicle accepted it: a rejected upload leaves nothing on the aircraft.
        record_transition(
            self._session.project_root(), PHASE_UPLOADED,
            mission_name=str(plan.get("template", "")),
            vehicle_driver=self._session.vehicle.driver,
        )

        # Fence and rally, when the plan has them and the driver can carry them. The
        # docstring always promised these; the call never made them.
        problems: list[str] = []
        for key, entries, method in (("fence", fence, "upload_geofence"),
                                     ("rally", rally, "upload_rally_points")):
            if not entries:
                report[key] = None
                continue
            sender = getattr(client, method, None)
            if sender is None:
                report[key] = {"success": False, "count": len(entries),
                               "message": f"The {self._session.vehicle.driver!r} driver cannot upload {key}."}
                problems.append(report[key]["message"])
                continue
            result = run(method, lambda sender=sender, entries=entries: sender(entries))
            report[key] = _command_payload(result, len(entries))
            if not report[key]["success"]:
                problems.append(f"{key}: {report[key]['message']}")

        if problems:
            # The mission is on the aircraft but its fence or rally points are not, and
            # a pilot relying on the fence needs to hear that as a failure.
            return fail(
                "The mission was uploaded, but not all of it: " + "; ".join(problems),
                items=len(items), result=report["mission"], upload=report, mission_uploaded=True,
            )
        return ok(items=len(items), result=report["mission"], upload=report)

    @guard
    def vehicle_command(self, command: str, **kwargs: Any) -> dict[str, Any]:
        """Arm / start / pause / resume / RTL / abort against the connected vehicle.

        Reports what the vehicle answered: ok only when it acknowledged the command.
        Every command used to return ok with "Command sent." inside, so a denied RTL
        looked exactly like an accepted one.
        """
        client = self._session._drone_client
        if client is None:
            return fail("No vehicle connected.")
        # The command set a ground station is expected to offer. Anything the
        # connected driver does not implement is reported as unsupported rather than
        # failing silently, because a pilot pressing a button needs to know whether
        # the aircraft heard it. "rtl" is accepted beside "rth": it is what every
        # ArduPilot and PX4 user calls it, and the cockpit's RTL button sent it.
        mapping = {
            "start": "start_mission",
            "pause": "pause_mission",
            "resume": "resume_mission",
            "rth": "return_to_home",
            "rtl": "return_to_home",
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
            action = lambda: method(altitude)  # noqa: E731
        else:
            action = method

        # Through the flight session, so every command lands in commands.jsonl and the
        # session's state follows the named ones (starting, paused, rth, aborted).
        manager = self._session.flight_manager
        if manager.active_session() is not None:
            dispatch = {
                "start_mission": manager.start_mission,
                "pause_mission": manager.pause_flight,
                "resume_mission": manager.resume_flight,
                "return_to_home": manager.trigger_rth,
                "abort_mission": manager.abort_flight,
            }
            named = dispatch.get(method_name)
            result = named(reason="operator") if named else manager.execute(method_name, action)
        else:
            result = action()

        payload = result.to_dict() if hasattr(result, "to_dict") else {"message": str(result)}
        self._session.audit("vehicle_command", {
            "command": command, "driver": self._session.vehicle.driver,
            "success": bool(getattr(result, "success", False)),
            "ack": getattr(result, "ack", ""),
        })
        if not getattr(result, "success", False):
            return fail(payload.get("message") or f"The vehicle did not accept {command!r}.",
                        result=payload)
        return ok(result=payload)

    @guard
    def flight_session(self) -> dict[str, Any]:
        """The live flight session: state, warnings, and the commands sent so far."""
        status = self._session.flight_manager.status()
        if status is None:
            return ok(session=None, reason="No vehicle connected, so there is no flight session.")
        return ok(session=status)

    @guard
    def run_preflight(self, acknowledged: list[str] | None = None,
                      free_storage_mb: float | None = None) -> dict[str, Any]:
        """Preflight checks against the connected aircraft and the planned mission.

        Everything is read from something: GPS, home, battery, compass and IMU from the
        vehicle's own telemetry; the camera from the payload's CAMERA_INFORMATION; the
        geofence and no-fly checks from the plan's path against its own fence and the
        zones drawn here. What nothing reports is a manual check, which the operator
        confirms by id in `acknowledged` -- a confirmation is recorded as theirs, not
        presented as a measurement.
        """
        from core.drone import NoDroneClient
        from core.mission_manager import MissionManager
        from core.preflight import confirm_manual_check, plan_airspace, run_preflight
        from core.settings import load_drone_profiles

        client = self._session._drone_client or NoDroneClient()
        plan_dict = self._session.mission_plan_dict or None
        plan = self._session.mission_plan

        profile = load_drone_profiles()[0]
        mission_summary: dict[str, Any] | None = None
        geofence: dict[str, Any] | None = None
        no_fly: list[dict[str, Any]] | None = None
        if plan_dict:
            # The plan's own limits are what will fly, so they are what is checked --
            # not the defaults of a generic profile.
            constraints = plan_dict.get("safety_constraints") or {}
            for key in ("rth_altitude_m", "min_altitude_m", "max_altitude_m"):
                if constraints.get(key) is not None:
                    setattr(profile, key, float(constraints[key]))
            mission_summary = (MissionManager.summarize_plan(plan) if plan is not None
                               else {"waypoints": len(plan_dict.get("waypoints") or [])})
            mission_summary["issues"] = ([] if mission_summary.get("waypoints")
                                         else ["error: the plan has no waypoints"])
            geofence, no_fly = plan_airspace(plan_dict, self._session.no_fly_polygons)

        report = run_preflight(
            project_id=str(self._session.project_id()),
            mission_id=str((plan_dict or {}).get("template", "")),
            drone_id=f"{self._session.vehicle.driver}:{self._session.vehicle.uri}",
            drone=client,
            drone_profile=profile,
            mission_summary=mission_summary,
            mission_uploaded=bool(self._session.mission_uploaded),
            geofence=geofence,
            no_fly_zones=no_fly,
            free_storage_mb=free_storage_mb,
        )
        # Confirmations persist for this connection and plan, so confirming the weather
        # once is not undone by the next refresh of the checklist.
        confirmed = self._session.preflight_acknowledged
        confirmed.update(str(check_id) for check_id in (acknowledged or []))
        # Only checks that need a human can be confirmed by one. A blocking check the
        # vehicle or the plan failed -- no GPS fix, a path outside the fence -- is not
        # something an acknowledgement makes true.
        known = {check.id for check in report.checks if check.requires_manual}
        for check_id in sorted(confirmed & known):
            confirm_manual_check(report, check_id, operator_note="confirmed by the operator")

        payload = report.to_dict()
        payload["simulated"] = self._session.vehicle.to_dict().get("is_simulated", False)
        payload["pending_confirmation"] = [
            c["id"] for c in payload["checks"] if c.get("requires_manual") and not c.get("confirmed")
        ]
        self._session.audit("preflight_run", {
            "can_start": payload["can_start"], "blocking": payload["blocking_count"],
        })
        return ok(**payload)

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
        busy = self._job_writing_to(output)
        if busy:
            return fail(busy)
        session = self._session
        # Set CRS stores the operator's choice; it used to be read by nothing.
        chosen_epsg = self._session.store.get_setting(self._crs_setting_key()) or None
        # Marked control points georeference the model (three or more marked in three
        # images each) or are reported as a check on the geotag fit.
        control = self._control_points()

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from core.reconstruction_colmap import build_reconstructor

            reconstructor = build_reconstructor(
                opts.get("engine", "auto"),
                profile=opts.get("profile", "standard"),
                dense=opts.get("dense"),
                use_gpu=bool(opts.get("use_gpu", True)),
                max_image_size=opts.get("max_image_size"),
                target_epsg=opts.get("epsg") or chosen_epsg,
                ground_control=control,
            )
            result = reconstructor.reconstruct(
                image_dir=images, output_dir=str(output), progress_callback=progress,
                should_cancel=should_cancel,
            )
            payload = result.to_dict()
            if control and payload.get("engine") != "colmap":
                payload.setdefault("warnings", []).append(
                    "Ground control points were not used: only the COLMAP engine fits the "
                    "model to them.")
            payload["ground_control"] = getattr(reconstructor, "control_report", None)
            # Reconstruction products become map layers immediately, which is the whole
            # point of doing this inside a GIS tool.
            _register_reconstruction_layers(session, payload)
            # Every derived file gets a provenance sidecar (inputs, engine, CRS, sha256)
            # the moment it exists. record_reconstruction_outputs had no caller, so every
            # real output was "unattributed" and no report could say what built it.
            try:
                from core.provenance import record_reconstruction_outputs

                payload["provenance"] = record_reconstruction_outputs(
                    output,
                    engine=str(payload.get("engine") or opts.get("engine", "auto")),
                    image_dir=images,
                    crs_epsg=payload.get("crs_epsg") or payload.get("epsg") or opts.get("epsg"),
                    parameters={"profile": opts.get("profile", "standard"),
                                "dense": opts.get("dense")},
                )
            except Exception as exc:  # noqa: BLE001 - never lose a finished reconstruction
                payload["provenance_error"] = f"{type(exc).__name__}: {exc}"
            session.audit("reconstruction_complete", {"engine": payload.get("engine"), "output": str(output)})
            return payload

        job_id = self._jobs.submit("reconstruction", work)
        self._writers()[str(output.resolve())] = job_id
        return ok(job_id=job_id)

    def _writers(self) -> dict[str, str]:
        """Output folder -> the job writing into it."""
        if not hasattr(self, "_output_writers"):
            self._output_writers: dict[str, str] = {}
        return self._output_writers

    def _job_writing_to(self, output: Path) -> str:
        """Why a second run into `output` must wait, or "" when nothing is writing there.

        Two reconstructions into one folder overwrite each other's files, and on Windows
        the second one died with "PermissionError: [WinError 32]" on a file the first
        held open -- after the operator had been told it started.
        """
        job_id = self._writers().get(str(output.resolve()))
        job = self._jobs.get(job_id) if job_id else None
        if not job or job.get("status") not in {"pending", "running"}:
            return ""
        return (f"A {job.get('name', 'job')} is already writing to {output} "
                f"(job {job_id}, {job.get('percent', 0)}%). Wait for it to finish, or "
                "Cancel it, before starting another into the same folder.")

    @guard
    def run_pipeline(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start the full inspection pipeline as a background job.

        When it finishes, every detection a trained model made becomes a finding in
        review, with the model key, file digest and confidence on the record, so the
        Findings panel shows what the models found. Heuristic regions stay in the run's
        summary and are counted, not recorded as findings.
        """
        opts = dict(options or {})
        images = opts.get("image_dir") or self._session.active_dataset_dir
        if not images:
            return fail("Select a dataset before running the pipeline.")
        if not Path(images).exists():
            return fail(f"Image folder not found: {images}")
        root = self._session.project_root()
        output = str(opts.get("output_dir") or (root / "analysis"))
        project_id = str(self._session.project_id())
        session = self._session
        chosen_epsg = self._session.store.get_setting(self._crs_setting_key()) or None
        terrain = self._session.terrain_source_path
        aoi = list(self._session.aoi_polygon or [])
        # Start / Run AI used to reconstruct without the project's marked control points,
        # silently: only Process used them.
        control = self._control_points()

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from app import desktop_ops
            from core.pipeline import PipelineConfig, StructuralFaultPipeline

            config = PipelineConfig()
            config.reconstruction_engine = opts.get("engine", "auto")
            config.reconstruction_profile = opts.get("profile", "standard")
            config.reconstruction_target_epsg = int(opts["epsg"]) if opts.get("epsg") else chosen_epsg
            # Coverage QA sizes each image's footprint from its height above ground.
            config.coverage_flight_height_agl_m = float(opts.get("flight_height_agl_m") or 0.0)
            if opts.get("ground_elevation_m") not in (None, ""):
                config.coverage_ground_elevation_m = float(opts["ground_elevation_m"])
            config.coverage_terrain_path = str(opts.get("terrain_path") or terrain or "")
            config.coverage_aoi_lonlat = aoi or None
            if opts.get("dense") is not None:
                config.reconstruction_dense = bool(opts["dense"])
            for key in ("structure_type", "material_type", "asset_id", "solar_imagery"):
                if opts.get(key):
                    setattr(config, key, str(opts[key]))
            for key in ("run_solar_detection", "run_metal_detection",
                        "crack_presence_gate", "grade_corrosion"):
                if opts.get(key) is not None:
                    setattr(config, key, bool(opts[key]))
            pipeline = StructuralFaultPipeline(config, ground_control=control)
            # The keyword is output_dir. This call said output_root, so every Run AI,
            # Start and "Run Full Pipeline" click failed with a TypeError seconds in.
            result = pipeline.run(
                image_dir=images, output_dir=output, progress_callback=progress,
                should_cancel=should_cancel,
            )
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            payload["ground_control"] = getattr(pipeline.reconstructor, "control_report", None)
            if control and not hasattr(pipeline.reconstructor, "control_report"):
                payload.setdefault("warnings", []).append(
                    "Ground control points were not used: only the COLMAP engine fits the "
                    "model to them, and this run used "
                    f"{type(pipeline.reconstructor).__name__}.")
            summary = json.loads(Path(payload["summary_json_path"]).read_text(encoding="utf-8"))
            # Model detections become findings the reviewer can accept or reject and the
            # report can print; heuristic screening output stays in the run summary.
            try:
                payload.update(desktop_ops.findings_from_pipeline(root, project_id, summary))
            except Exception as exc:  # noqa: BLE001 - never lose a finished run
                payload["findings_error"] = f"{type(exc).__name__}: {exc}"
            payload["reconstruction_status"] = summary.get("reconstruction", {}).get("status", "ok")
            # The pipeline reconstructs too; its ortho, DSM and camera track belong on the
            # map exactly as a standalone reconstruction's do. They used to be left on
            # disk, so a full run ended with only the photo points visible.
            payload["layers_registered"] = _register_reconstruction_layers(
                session, summary.get("reconstruction") or {})
            # Which optional detectors ran as heuristics and so were left out of the
            # health score -- said in the finished-job message, not only in the report.
            components = (summary.get("health_scoring", {}) or {}).get("components", {}) or {}
            payload["heuristic_images_not_scored"] = dict(
                components.get("excluded_heuristic_images", {}) or {})
            payload["detectors_run"] = {
                name: sorted({str(r.get(f"{name}_model_used", "not_run"))
                              for r in summary.get("images", []) if isinstance(r, dict)})
                for name in ("crack", "structural", "solar", "metal")}
            # Solar and metal refuse rather than guess when no model reads this imagery;
            # the finished-job message names the refusal and its reason.
            # Every distinct reason, each with its own count: a mixed folder refuses for
            # different reasons, and one reason for all of them would be wrong.
            from core.reporting import refusal_reasons

            refused: dict[str, dict[str, Any]] = {}
            for name in ("solar", "metal"):
                reasons = refusal_reasons(summary.get("images", []), name)
                if reasons:
                    refused[name] = {
                        "images": sum(r["images"] for r in reasons),
                        "reasons": reasons,
                        "reason": " | ".join(f"{r['images']} image(s): {r['reason']}"
                                             if len(reasons) > 1 else r["reason"]
                                             for r in reasons),
                    }
            payload["detectors_refused"] = refused
            session.audit("pipeline_complete", {
                "run_dir": Path(payload.get("run_dir", "")).name,
                "report": Path(payload.get("report_markdown_path", "")).name,
                "findings_created": payload.get("findings_created", 0),
            })
            return payload

        return ok(job_id=self._jobs.submit("pipeline", work))

    @guard
    def prelabel_image(self, image_path: str,
                       model_key: str = "structural_multiclass_detector") -> dict[str, Any]:
        """Run an installed detector on one image; its detections become findings in review.

        The desktop half of POST /projects/{id}/annotations/prelabel, which worked and
        was reachable from no screen.
        """
        from app import desktop_ops

        path = self._resolve_image(image_path)
        if path is None:
            return fail(f"No such image: {image_path}")
        try:
            result = desktop_ops.prelabel_image(
                self._session.project_root(), str(self._session.project_id()),
                str(path), model_key)
        except ValueError as exc:
            return fail(str(exc))
        self._session.audit("prelabel_created", {"model_key": model_key, "count": result["count"]})
        return ok(**result)

    @guard
    def run_model(self, model_key: str, image_path: str) -> dict[str, Any]:
        """Run any installed model on one image and show what it said.

        Every installed model is reachable here, including the six that had no path at
        all. The result carries the registry's scope note, because several of them are
        valid on one kind of imagery only and the result is where a user reads that.
        """
        from core.detection import load_image
        from core.model_runner import ModelRunRefused, run_registered_model, save_overlay

        path = self._resolve_image(image_path)
        if path is None:
            return fail(f"No such image: {image_path}")
        try:
            result = run_registered_model(load_image(path), str(model_key))
        except ModelRunRefused as exc:
            return fail(str(exc))
        out_dir = self._session.project_root() / "analysis" / "models" / str(model_key)
        overlay_path = save_overlay(result, out_dir, path.stem)
        self._session.audit("model_run", {"model_key": model_key, "image": str(path)})
        return ok(**result.to_dict(), image=str(path), overlay_path=overlay_path,
                  overlay_data_uri=self._data_uri(result.overlay) if result.overlay is not None else "")

    @guard
    def grade_corrosion(self, image_path: str) -> dict[str, Any]:
        """Grade corrosion severity per pixel, or refuse with the reason.

        There is no heuristic grade to fall back to. The grade is only meaningful on a
        close-range photo of a corroded surface; the scope note says so and travels
        with the result.
        """
        return self.run_model("corrosion_severity_segmentation", image_path)

    @guard
    def run_landcover(self, orthomosaic_path: str = "", allow_cpu: bool = False) -> dict[str, Any]:
        """Land-cover segmentation of a georeferenced orthomosaic, as a background job.

        Refuses on CPU unless asked, because a survey-sized orthomosaic on a CPU takes
        long enough to look hung. The result lists which classes the model's holdout
        says are usable; the others are still drawn and are marked as not usable.
        """
        source = orthomosaic_path or ""
        if not source:
            for layer in self._session.layers.values():
                candidate = str(getattr(layer, "path", "") or "")
                if "orthomosaic" in candidate.lower() and candidate.lower().endswith((".tif", ".tiff")):
                    source = candidate
                    break
        if not source:
            return fail("No orthomosaic: run a reconstruction or add an orthomosaic layer first.")
        if not Path(source).is_file():
            return fail(f"Orthomosaic not found: {source}")
        output = self._session.project_root() / "analysis" / "landcover"
        session = self._session

        def work(progress: Callable[[int, str], None], should_cancel: Callable[[], bool], **_: Any) -> dict[str, Any]:
            from core.model_runner import run_landcover

            progress(5, "Running land-cover model")
            result = run_landcover(source, output, allow_cpu=bool(allow_cpu))
            polygons = result.get("polygons_path")
            if polygons and Path(polygons).exists():
                session.register_vector_layer(polygons, name="Land cover (model)")
            session.audit("landcover_complete", {"model_key": result["model_key"]})
            return result

        return ok(job_id=self._jobs.submit("landcover", work))

    def _resolve_image(self, image_path: str) -> Path | None:
        """An absolute path, or a name inside the active dataset."""
        raw = str(image_path or "").strip()
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute() and self._session.active_dataset_dir:
            candidate = Path(self._session.active_dataset_dir) / raw
        return candidate if candidate.is_file() else None

    @staticmethod
    def _data_uri(image: Any, max_size: int = 1200) -> str:
        import base64

        import cv2

        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / float(max(height, width))
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        okay, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not okay:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")

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


def _parse_pylon_elements(text: str) -> list[dict[str, Any]]:
    """"crossarm@32, insulator@28:4" -> [{name, height_m[, radius_m]}, ...]."""
    elements: list[dict[str, Any]] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, rest = chunk.partition("@")
        height, _, radius = rest.partition(":")
        entry: dict[str, Any] = {"name": name.strip(), "height_m": float(height) if height.strip() else 0.0}
        if radius.strip():
            entry["radius_m"] = float(radius)
        elements.append(entry)
    return elements


def _centroid(ring: list[list[float]]) -> list[float]:
    return [sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)]


def _command_payload(result: Any, count: int) -> dict[str, Any]:
    """A CommandResult as the UI reads it, with how many items it carried."""
    payload = result.to_dict() if hasattr(result, "to_dict") else {"message": str(result)}
    payload.setdefault("success", bool(getattr(result, "success", False)))
    payload.setdefault("message", "")
    payload["count"] = count
    return payload


def _plan_to_dict(plan: Any) -> dict[str, Any]:
    """Serialize a MissionPlan dataclass without depending on a to_dict method."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(plan):
        return asdict(plan)
    if isinstance(plan, dict):
        return plan
    return {key: getattr(plan, key) for key in dir(plan) if not key.startswith("_") and not callable(getattr(plan, key))}
