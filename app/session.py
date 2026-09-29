"""Application state: projects, missions, datasets, vehicle link, and GIS layers.

This is the Qt-free replacement for the old `ui.workspace.AppSession`. It owns the
SQLite project store and mediates every call into `core/` and `mission/`, so the UI
layer holds no domain logic of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any

from .store import ProjectStore

# A vehicle that has said nothing for this long is not "connected" in any sense a pilot
# cares about. ArduPilot streams several messages a second, so three seconds of silence
# is a lost link, not a slow one.
TELEMETRY_STALE_AFTER_S = 3.0

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VehicleLink:
    """Current vehicle connection. `driver` is always reported truthfully to the UI."""

    connected: bool = False
    driver: str = ""
    uri: str = ""
    last_error: str = ""
    connected_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "driver": self.driver,
            "uri": self.uri,
            "last_error": self.last_error,
            "connected_utc": self.connected_utc,
            "is_simulated": _is_simulated(self.driver),
        }


def _is_simulated(driver: str) -> bool:
    from core.drone import SIMULATED_DRIVERS

    return str(driver or "").lower() in SIMULATED_DRIVERS


def json_safe(value: Any) -> Any:
    """``value`` with every non-finite float replaced by None.

    Python's json writes NaN and Infinity as bare tokens, which JSON.parse rejects. One
    raster whose nodata is NaN, carried in a layer's metadata, made every get_state and
    list_layers reply unreadable to the UI, so the open project looked closed.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        items = [json_safe(v) for v in value]
        return tuple(items) if type(value) is tuple else items
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return json_safe(value.item()) if getattr(value, "ndim", 0) == 0 else json_safe(value.tolist())
    return value


@dataclass
class MapLayer:
    """One entry in the GIS layer tree."""

    id: str
    name: str
    kind: str  # raster | vector | basemap | pointcloud
    path: str = ""
    visible: bool = True
    opacity: float = 1.0
    crs_epsg: int | None = None
    bounds_lonlat: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "visible": self.visible,
            "opacity": self.opacity,
            "crs_epsg": self.crs_epsg,
            "bounds_lonlat": json_safe(self.bounds_lonlat),
            "metadata": json_safe(self.metadata),
        }


class AppSession:
    """Shared, offline-first application state."""

    def __init__(self, store: ProjectStore | None = None):
        self.store = store or ProjectStore()
        self.mission_plan: Any = None
        self.mission_plan_dict: dict[str, Any] = {}
        self.aoi_polygon: list[list[float]] = []
        # Surveyed control points, once imported. Empty is the honest default: a survey
        # with no control has no measured accuracy.
        self.ground_control: list = []

        from core.flight_log import FlightLog

        # One log per vehicle connection, begun by connect_vehicle. Recording starts as
        # soon as telemetry arrives, because a log that has to be armed separately is one
        # somebody forgets to arm. It used to be one log per application session, so a
        # mock connection followed by a real one exported a single "flight" that started
        # in San Francisco and ended in Ohio -- 3,477 km, with the mock's battery.
        self.flight_log = FlightLog()
        # Earlier connections' logs, archived rather than discarded when a new
        # connection starts, so reconnecting never throws a flight away.
        self.archived_flight_logs: list[dict[str, Any]] = []
        # Whether the planned mission is known to be on the vehicle: set only when the
        # vehicle acknowledged the upload, cleared by a new plan or a new connection.
        self.mission_uploaded: bool = False
        # Manual preflight checks the operator has confirmed, by check id, for the
        # current connection and plan.
        self.preflight_acknowledged: set[str] = set()
        # What the planner warned about the current plan (flat-ground assumption, more
        # than one battery, a route it had to guess...). Preflight's Mission Valid reads
        # it: a plan the planner itself qualified is not simply "validated".
        self.mission_plan_warnings: list[str] = []
        self.no_fly_polygons: list[list[list[float]]] = []
        # The last capture match (Api.match_captures), read by Request Reflight.
        self.capture_report: dict[str, Any] | None = None
        self.terrain_source_path: str = ""
        # Which geocoding backend place search uses. Switchable to a self-hosted
        # Nominatim or to "offline" so no query ever leaves the machine.
        self.geocoding_provider: str = "nominatim"
        # Restored below from the project's most recent import. The field is in memory,
        # so before this every launch opened with no dataset selected even when the
        # project plainly had one -- and every dataset-dependent action refused until the
        # user re-imported a folder the application already knew about.
        self.active_dataset_dir: str = ""
        self.layers: dict[str, MapLayer] = {}
        self.vehicle = VehicleLink()
        self._drone_client: Any = None
        from core.flight import FlightManager

        # The live flight session: commands.jsonl, telemetry.jsonl and a flight state
        # per connection. Owned by this session rather than the module singleton so two
        # sessions (or two tests) never share an aircraft.
        self.flight_manager = FlightManager()
        self.display_epsg: int = 4326

    # -- projects --------------------------------------------------------

    def ensure_active_project(self) -> dict[str, Any]:
        """Return the active project, creating a default one on first run."""
        project = self.store.get_active_project()
        if project:
            return project
        projects = self.store.list_projects()
        if projects:
            self.store.set_active_project(int(projects[0]["id"]))
            return self.store.get_active_project() or projects[0]
        created = self.store.create_project(name="Default Project", description="Created automatically")
        self.store.set_active_project(int(created["id"]))
        return self.store.get_active_project() or created

    def create_project(self, name: str, root_dir: str = "", description: str = "") -> dict[str, Any]:
        project = self.store.create_project(name=name, root_dir=root_dir or None, description=description)
        # Through set_active_project rather than the store, so the project being left
        # takes its mission, layers, area of interest and dataset with it. A new project
        # used to open showing the previous one's plan and photo points on its map.
        self.set_active_project(int(project["id"]))
        self.audit("project_created", {"name": name})
        return self.store.get_active_project() or project

    def set_active_project(self, project_id: int) -> dict[str, Any]:
        self.store.set_active_project(int(project_id))
        # Project-scoped state must not leak between projects.
        self.mission_plan = None
        self.mission_plan_dict = {}
        self.mission_plan_warnings = []
        self.mission_uploaded = False
        self.preflight_acknowledged = set()
        self.layers.clear()
        self.active_dataset_dir = ""
        # The drawn area and no-fly zones are held in memory, not per project, so they
        # followed the operator into the next project and planned its mission over the
        # last one's site.
        self.aoi_polygon = []
        self.no_fly_polygons = []
        self.capture_report = None
        # Clearing is right -- the old project's layers are not this one's -- but leaving
        # it there meant opening a project with a finished reconstruction showed an empty
        # layer tree. Restore what this project has on disk.
        self.restore_active_dataset()
        self.restore_layers()
        return self.store.get_active_project() or {}

    def project_id(self) -> int:
        return int(self.ensure_active_project()["id"])

    def project_root(self) -> Path:
        project = self.ensure_active_project()
        root = project.get("root_dir") or ""
        path = Path(root) if root else Path.cwd() / "projects" / f"project_{project['id']}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def announce_finished_job(self, job: dict[str, Any]) -> None:
        """Record a finished job as a notification against the open project.

        Registered with the job manager once, at startup. A reconstruction is minutes
        long; the operator has switched workspace by the time it lands, and without this
        the result appears in a panel nobody is looking at.
        """
        from core.notifications import NotificationCentre, describe_job

        root = self.project_root()
        if not root:
            return
        level, title, detail = describe_job(job)
        NotificationCentre(Path(root)).notify(
            title, detail, level=level, subject_kind="job", subject_id=str(job.get("id", "")))

    def audit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        try:
            self.store.append_audit_event(self.project_id(), event_type, payload or {})
        except Exception:  # noqa: BLE001 - auditing must never break a user action
            pass

    # -- layers ----------------------------------------------------------

    def add_layer(self, layer: MapLayer) -> dict[str, Any]:
        # The same file registered twice -- importing a folder again, re-running a
        # match -- replaced nothing and appended a second row, so the layer panel listed
        # "images photo points" twice and the map drew it twice. Same file, same layer.
        if layer.path:
            for existing in list(self.layers.values()):
                if existing.path == layer.path and existing.kind == layer.kind:
                    layer.id = existing.id
                    layer.visible = existing.visible
                    layer.opacity = existing.opacity
        self.layers[layer.id] = layer
        return layer.to_dict()

    def remove_layer(self, layer_id: str) -> bool:
        return self.layers.pop(layer_id, None) is not None

    def set_layer_visibility(self, layer_id: str, visible: bool) -> bool:
        layer = self.layers.get(layer_id)
        if layer is None:
            return False
        layer.visible = bool(visible)
        return True

    def set_layer_opacity(self, layer_id: str, opacity: float) -> bool:
        layer = self.layers.get(layer_id)
        if layer is None:
            return False
        layer.opacity = float(max(0.0, min(1.0, opacity)))
        return True

    def layer_list(self) -> list[dict[str, Any]]:
        return [layer.to_dict() for layer in self.layers.values()]

    def register_raster_layer(self, path: str | Path, name: str = "", kind: str = "raster") -> dict[str, Any]:
        """Add a GeoTIFF to the layer tree, reading its real CRS and bounds.

        Rasters without a CRS are still added but flagged, so the UI can say why they
        cannot be placed on the map instead of dropping them silently.
        """
        from core import geo

        source = Path(path)
        layer_id = f"{kind}:{source.stem}:{len(self.layers)}"
        layer = MapLayer(id=layer_id, name=name or source.stem, kind=kind, path=str(source))
        try:
            _data, meta = geo.read_geotiff(source)
            layer.crs_epsg = meta.get("epsg")
            bounds = meta.get("bounds") or []
            if layer.crs_epsg and len(bounds) == 4:
                lon, lat = geo.projected_to_wgs84([bounds[0], bounds[2]], [bounds[1], bounds[3]], int(layer.crs_epsg))
                layer.bounds_lonlat = [float(lon[0]), float(lat[0]), float(lon[1]), float(lat[1])]
            layer.metadata = {k: v for k, v in meta.items() if k != "transform"}
        except Exception as exc:  # noqa: BLE001
            layer.metadata = {"error": f"Could not read georeferencing: {exc}"}
        return self.add_layer(layer)

    def register_vector_layer(self, path: str | Path, name: str = "") -> dict[str, Any]:
        """Add a GeoJSON file to the layer tree, computing its extent."""
        source = Path(path)
        layer_id = f"vector:{source.stem}:{len(self.layers)}"
        layer = MapLayer(id=layer_id, name=name or source.stem, kind="vector", path=str(source), crs_epsg=4326)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            coords = list(_iter_coordinates(payload))
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                layer.bounds_lonlat = [min(lons), min(lats), max(lons), max(lats)]
            layer.metadata = {"feature_count": len(payload.get("features", []))}
        except Exception as exc:  # noqa: BLE001
            layer.metadata = {"error": f"Could not read vector file: {exc}"}
        return self.add_layer(layer)

    # -- datasets --------------------------------------------------------

    def import_dataset(self, folder: str | Path, name: str = "") -> dict[str, Any]:
        """Register an image folder, recording geotag coverage up front."""
        from core import geo

        source = Path(folder)
        if not source.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {source}")
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No supported images found in {source}")

        fixes = geo.collect_gps_fixes(images)
        metadata: dict[str, Any] = {
            "image_count": len(images),
            "geotagged_count": len(fixes),
            "imported_utc": _now(),
        }
        if fixes:
            latitudes = [f.latitude for f in fixes.values()]
            longitudes = [f.longitude for f in fixes.values()]
            metadata["bounds_lonlat"] = [min(longitudes), min(latitudes), max(longitudes), max(latitudes)]
            metadata["suggested_epsg"] = geo.auto_utm_epsg_for_fixes(fixes)
            metadata["center_lonlat"] = [sum(longitudes) / len(longitudes), sum(latitudes) / len(latitudes)]

        camera = geo.read_exif_camera(images[0])
        metadata["camera"] = f"{camera.make} {camera.model}".strip()

        # Importing the same folder again appended another identical row, so Load Flight
        # and the dataset panels listed one flight four times. The same folder with the
        # same images is the same dataset; a folder whose contents changed is recorded
        # again, because its metadata is then genuinely different.
        existing = next((
            row for row in self.store.list_datasets(self.project_id())
            if row.get("path") == str(source)
            and json.loads(row.get("metadata_json") or "{}").get("image_count") == len(images)
        ), None)
        entry = existing or self.store.save_dataset_entry(
            project_id=self.project_id(),
            name=name or source.name,
            path=str(source),
            metadata=metadata,
        )
        self.active_dataset_dir = str(source)
        self.audit("dataset_imported", {"folder": str(source), "images": len(images),
                                        "already_recorded": existing is not None})

        if fixes:
            features = [
                geo.point_feature(fix.longitude, fix.latitude, {"image": image_name}, alt=fix.altitude_m)
                for image_name, fix in fixes.items()
            ]
            track_path = self.project_root() / "layers" / f"{source.name}_photos.geojson"
            geo.write_geojson(track_path, features)
            self.register_vector_layer(track_path, name=f"{source.name} photo points")

        return {**(entry or {}), "metadata": metadata}

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.store.list_datasets(self.project_id())

    def restore_active_dataset(self) -> str:
        """Reselect the newest dataset this project imported, if it is still on disk.

        Called at startup. A path that has since been deleted or unplugged is skipped
        rather than restored, because a selected dataset that does not exist turns every
        later refusal into a confusing one.
        """
        if self.active_dataset_dir:
            return self.active_dataset_dir
        try:
            entries = self.store.list_datasets(self.project_id())
        except Exception:  # noqa: BLE001 - a fresh store has no project yet
            return ""
        for entry in entries:
            candidate = str(entry.get("path") or "")
            if candidate and Path(candidate).is_dir():
                self.active_dataset_dir = candidate
                return candidate
        return ""

    # Reconstruction writes these names, and they are the products worth putting back.
    RESTORABLE_PRODUCTS = (
        ("orthomosaic.tif", "raster", "Orthomosaic"),
        ("dsm.tif", "raster", "DSM"),
        ("dtm.tif", "raster", "DTM"),
        ("dsm_hillshade.tif", "raster", "DSM hillshade"),
        ("camera_track.geojson", "vector", "Camera positions"),
    )

    def restore_layers(self) -> list[dict[str, Any]]:
        """Put this project's reconstruction products back in the layer tree.

        `self.layers` is an in-memory dict that is cleared whenever the active project
        changes and never written anywhere, so a reconstruction registered its
        orthomosaic, DSM, DTM, hillshade and camera track -- and all five vanished the
        moment the application was closed or another project was opened. The files were
        still on disk the whole time; only the application's knowledge of them was
        thrown away, which is the worst version of this because the operator can see the
        folder and cannot see the layer.

        Rediscovering from disk rather than persisting the registry keeps one source of
        truth: a product that has been deleted does not come back as a broken row.
        """
        try:
            root = self.project_root()
        except Exception:  # noqa: BLE001 - no project open yet
            return []

        restored: list[dict[str, Any]] = []
        known = {layer.path for layer in self.layers.values()}
        for directory in sorted(root.glob("**/reconstruction")):
            for filename, kind, label in self.RESTORABLE_PRODUCTS:
                path = directory / filename
                if not path.exists() or str(path) in known:
                    continue
                try:
                    if kind == "vector":
                        restored.append(self.register_vector_layer(path, name=label))
                    else:
                        restored.append(self.register_raster_layer(path, name=label, kind=kind))
                except Exception:  # noqa: BLE001 - an unreadable product is not fatal
                    continue
        # CAD drawings and placed images live in overlays/, and vanished from the layer
        # tree on every restart in exactly the way the products did.
        overlay_dir = root / "overlays"
        if overlay_dir.is_dir():
            for path in sorted(overlay_dir.glob("*.geojson")):
                if str(path) in known:
                    continue
                try:
                    restored.append(self.register_vector_layer(path, name=f"CAD: {path.stem}"))
                except Exception:  # noqa: BLE001
                    continue
            for path in sorted(overlay_dir.glob("*.tif")):
                if str(path) in known:
                    continue
                try:
                    restored.append(self.register_raster_layer(path, name=f"Overlay: {path.stem}"))
                except Exception:  # noqa: BLE001
                    continue
        return restored

    # -- vehicle ---------------------------------------------------------

    def connect_vehicle(self, uri: str, driver: str = "pymavlink") -> dict[str, Any]:
        """Connect over the requested driver. Never silently substitutes the mock."""
        from core.drone import create_drone_client

        self.disconnect_vehicle()
        driver = str(driver or "pymavlink").strip().lower()
        try:
            client = create_drone_client(driver)
            client.connect(uri) if hasattr(client, "connect") else None
            self._drone_client = client
            self.vehicle = VehicleLink(connected=True, driver=driver, uri=uri, connected_utc=_now())
            self.audit("vehicle_connected", {"driver": driver, "uri": uri})
        except Exception as exc:  # noqa: BLE001
            self.vehicle = VehicleLink(connected=False, driver=driver, uri=uri, last_error=str(exc))
            return self.vehicle.to_dict()

        self.mission_uploaded = False
        self.preflight_acknowledged = set()
        self._begin_flight_log(driver, uri)
        self._begin_flight_session(client, driver, uri)
        return self.vehicle.to_dict()

    def _begin_flight_log(self, driver: str, uri: str) -> None:
        """Start a fresh log for this connection, archiving the previous one.

        The previous log is written to disk rather than dropped: a pilot who flies,
        reconnects and then goes to export must still find the flight they flew.
        """
        from core.flight_log import FlightLog, export_all

        previous = self.flight_log
        if previous is not None and previous.samples:
            try:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                stem = previous.default_stem()
                files = export_all(previous, self.project_root() / "flights", stem=f"{stem}_{stamp}")
                self.archived_flight_logs.append({"files": files, "summary": previous.summary()})
            except Exception:  # noqa: BLE001 - archiving must never block a connection
                pass
        self.flight_log = FlightLog(
            driver=driver,
            connection_uri=uri,
            is_simulated=_is_simulated(driver),
        )

    def _begin_flight_session(self, client: Any, driver: str, uri: str) -> None:
        """Open the FlightManager session: command and telemetry logs for this link."""
        manager = self.flight_manager
        try:
            root = self.project_root()
            manager.start_session(
                project_root=root,
                project_id=str(self.project_id()),
                mission_id=str((self.mission_plan_dict or {}).get("template", "")),
                drone_id=f"{driver}:{uri}",
                drone=client,
            )
            manager.add_telemetry_listener(self._record_telemetry)
        except Exception as exc:  # noqa: BLE001 - the link itself is still usable
            self.vehicle.last_error = f"Flight session log unavailable: {exc}"

    def disconnect_vehicle(self) -> None:
        manager = getattr(self, "flight_manager", None)
        if manager is not None and manager.active_session() is not None:
            try:
                manager.remove_telemetry_listener(self._record_telemetry)
                manager.end_session(reason="disconnected")
            except Exception:  # noqa: BLE001
                pass
        client = self._drone_client
        if client is not None:
            try:
                if hasattr(client, "disconnect"):
                    client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._drone_client = None
        self.vehicle = VehicleLink()
        self.mission_uploaded = False
        # "Camera checked by eye" was about the aircraft that was on the link.
        self.preflight_acknowledged = set()

    def _record_telemetry(self, data: Any) -> None:
        """Append one sample to this connection's flight log, if it is new and live.

        A snapshot the vehicle has not refreshed is the same observation read twice,
        and a stale one is not an observation of now at all; recording either would put
        points in the log that the aircraft never reported.
        """
        try:
            age = data.age_s() if hasattr(data, "age_s") else 0.0
            if age > TELEMETRY_STALE_AFTER_S:
                return
            self.flight_log.record(data)
        except Exception:  # noqa: BLE001
            # Logging must never be the reason a pilot loses their telemetry display.
            pass

    def telemetry(self) -> dict[str, Any]:
        """Live telemetry, or an explicit not-connected / link-lost marker."""
        if self._drone_client is None:
            return {"connected": False, "reason": "No vehicle connected."}
        try:
            data = self._drone_client.get_telemetry()
        except Exception as exc:  # noqa: BLE001
            return {"connected": False, "reason": f"Telemetry read failed: {exc}"}
        payload = data.to_dict() if hasattr(data, "to_dict") else dict(getattr(data, "__dict__", {}))
        payload["driver"] = self.vehicle.driver
        payload["is_simulated"] = _is_simulated(self.vehicle.driver)

        # Staleness, measured rather than assumed. The bridge sets connected=True on the
        # first message and nothing ever cleared it, so an aircraft that flew out of
        # range read "connected, AUTO, armed, 77 %" indefinitely. The last known values
        # are kept -- they are where the aircraft was last seen -- but labelled as such.
        age = data.age_s() if hasattr(data, "age_s") else 0.0
        payload["telemetry_age_s"] = round(age, 1)
        if age > TELEMETRY_STALE_AFTER_S:
            payload["connected"] = False
            payload["stale"] = True
            payload["link_lost"] = True
            payload["reason"] = (
                f"Link lost: nothing heard from the vehicle for {age:.0f} s. The values "
                "shown are the last ones it reported, not where it is now."
            )
            return payload

        payload["connected"] = True
        payload["stale"] = False
        # Recorded here as well as by the flight session's poll, so the log holds what
        # the pilot saw; the log drops a snapshot it has already recorded.
        self._record_telemetry(data)
        return payload


def _iter_coordinates(payload: Any):
    """Yield every [lon, lat] pair in a GeoJSON structure, at any nesting depth."""
    if isinstance(payload, dict):
        if "coordinates" in payload:
            yield from _iter_coordinates(payload["coordinates"])
        for key in ("features", "geometry", "geometries"):
            if key in payload:
                yield from _iter_coordinates(payload[key])
    elif isinstance(payload, (list, tuple)):
        if len(payload) >= 2 and all(isinstance(v, (int, float)) for v in payload[:2]):
            yield [float(payload[0]), float(payload[1])]
        else:
            for item in payload:
                yield from _iter_coordinates(item)
