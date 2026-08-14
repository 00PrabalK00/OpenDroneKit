"""Application state: projects, missions, datasets, vehicle link, and GIS layers.

This is the Qt-free replacement for the old `ui.workspace.AppSession`. It owns the
SQLite project store and mediates every call into `core/` and `mission/`, so the UI
layer holds no domain logic of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from .store import ProjectStore

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
            "is_simulated": self.driver in {"mock", "sitl"},
        }


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
            "bounds_lonlat": self.bounds_lonlat,
            "metadata": self.metadata,
        }


class AppSession:
    """Shared, offline-first application state."""

    def __init__(self, store: ProjectStore | None = None):
        self.store = store or ProjectStore()
        self.mission_plan: Any = None
        self.mission_plan_dict: dict[str, Any] = {}
        self.aoi_polygon: list[list[float]] = []

        from core.flight_log import FlightLog

        # One log per session. Recording starts as soon as telemetry is read, because a
        # log that has to be armed separately is one somebody forgets to arm.
        self.flight_log = FlightLog()
        self.no_fly_polygons: list[list[list[float]]] = []
        self.terrain_source_path: str = ""
        # Which geocoding backend place search uses. Switchable to a self-hosted
        # Nominatim or to "offline" so no query ever leaves the machine.
        self.geocoding_provider: str = "nominatim"
        self.active_dataset_dir: str = ""
        self.layers: dict[str, MapLayer] = {}
        self.vehicle = VehicleLink()
        self._drone_client: Any = None
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
        self.store.set_active_project(int(project["id"]))
        self.audit("project_created", {"name": name})
        return self.store.get_active_project() or project

    def set_active_project(self, project_id: int) -> dict[str, Any]:
        self.store.set_active_project(int(project_id))
        # Project-scoped state must not leak between projects.
        self.mission_plan = None
        self.mission_plan_dict = {}
        self.layers.clear()
        self.active_dataset_dir = ""
        return self.store.get_active_project() or {}

    def project_id(self) -> int:
        return int(self.ensure_active_project()["id"])

    def project_root(self) -> Path:
        project = self.ensure_active_project()
        root = project.get("root_dir") or ""
        path = Path(root) if root else Path.cwd() / "projects" / f"project_{project['id']}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def audit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        try:
            self.store.append_audit_event(self.project_id(), event_type, payload or {})
        except Exception:  # noqa: BLE001 - auditing must never break a user action
            pass

    # -- layers ----------------------------------------------------------

    def add_layer(self, layer: MapLayer) -> dict[str, Any]:
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

        entry = self.store.save_dataset_entry(
            project_id=self.project_id(),
            name=name or source.name,
            path=str(source),
            metadata=metadata,
        )
        self.active_dataset_dir = str(source)
        self.audit("dataset_imported", {"folder": str(source), "images": len(images)})

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

    # -- vehicle ---------------------------------------------------------

    def connect_vehicle(self, uri: str, driver: str = "mavlink") -> dict[str, Any]:
        """Connect over the requested driver. Never silently substitutes the mock."""
        from core.drone import create_drone_client

        self.disconnect_vehicle()
        try:
            client = create_drone_client(driver)
            client.connect(uri) if hasattr(client, "connect") else None
            self._drone_client = client
            self.vehicle = VehicleLink(connected=True, driver=driver, uri=uri, connected_utc=_now())
            self.audit("vehicle_connected", {"driver": driver, "uri": uri})
        except Exception as exc:  # noqa: BLE001
            self.vehicle = VehicleLink(connected=False, driver=driver, uri=uri, last_error=str(exc))
        return self.vehicle.to_dict()

    def disconnect_vehicle(self) -> None:
        client = self._drone_client
        if client is not None:
            try:
                if hasattr(client, "disconnect"):
                    client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._drone_client = None
        self.vehicle = VehicleLink()

    def telemetry(self) -> dict[str, Any]:
        """Live telemetry, or an explicit not-connected marker."""
        if self._drone_client is None:
            return {"connected": False, "reason": "No vehicle connected."}
        try:
            data = self._drone_client.get_telemetry()
        except Exception as exc:  # noqa: BLE001
            return {"connected": False, "reason": f"Telemetry read failed: {exc}"}
        payload = data.to_dict() if hasattr(data, "to_dict") else dict(getattr(data, "__dict__", {}))
        payload["connected"] = True
        payload["driver"] = self.vehicle.driver
        payload["is_simulated"] = self.vehicle.driver in {"mock", "sitl"}

        # Record as we go. A telemetry stream is transient, and the moment it is wanted
        # is after the flight, when nobody can go back and start recording.
        try:
            self.flight_log.record(data)
        except Exception:  # noqa: BLE001
            # Logging must never be the reason a pilot loses their telemetry display.
            pass

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
