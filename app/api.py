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
        return ok(path=self._session.terrain_source_path)

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
        from mission.planner import MissionPlanner

        templates = [
            "grid", "double_grid", "corridor", "facade", "tower_mapping", "solar_inspection",
            "orbit", "panorama", "bubble_360", "waypoints", "linear_inspection", "lateral_capture",
            "roof_inspection", "magnetic_mapping", "linked_mission",
        ]
        return ok(templates=templates, planner=MissionPlanner.__name__)

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
        )

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
        import webview

        if self._window is None:
            return fail("Window is not ready.")
        filters = tuple(extensions) if extensions else ("All files (*.*)",)
        result = self._window.create_file_dialog(webview.OPEN_DIALOG, file_types=filters)
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
