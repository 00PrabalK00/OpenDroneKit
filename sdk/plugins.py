"""Versioned plugin registry for every extension point promised by the spec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import importlib
import importlib.metadata
import json
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping


PLUGIN_API_VERSION = "1"


class PluginKind(str, Enum):
    DRONE = "drone"
    CAMERA = "camera"
    PAYLOAD = "payload"
    MISSION_TYPE = "mission_type"
    ENGINE = "engine"
    MODEL = "model"
    EXPORTER = "exporter"
    REPORT_TEMPLATE = "report_template"
    MAP_PROVIDER = "map_provider"


@dataclass(frozen=True)
class PluginSpec:
    kind: PluginKind
    name: str
    factory: Callable[..., Any]
    api_version: str = PLUGIN_API_VERSION
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Plugin name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("Plugin factory must be callable.")
        if self.api_version != PLUGIN_API_VERSION:
            raise ValueError(
                f"Plugin {self.name!r} targets API {self.api_version}; "
                f"this SDK supports {PLUGIN_API_VERSION}."
            )

    def describe(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["factory"] = f"{self.factory.__module__}:{self.factory.__qualname__}"
        payload["metadata"] = dict(self.metadata)
        return payload


def _load_dotted_factory(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Plugin factory must use module.path:attribute syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"Plugin factory {reference!r} is not callable.")
    return factory


class PluginRegistry:
    """Thread-safe explicit registry; importing the SDK performs no discovery or I/O."""

    def __init__(self) -> None:
        self._plugins: dict[tuple[PluginKind, str], PluginSpec] = {}
        self._lock = RLock()

    def register(self, spec: PluginSpec, *, replace: bool = False) -> PluginSpec:
        key = (spec.kind, spec.name.casefold())
        with self._lock:
            if key in self._plugins and not replace:
                raise ValueError(
                    f"Plugin {spec.kind.value}:{spec.name} is already registered."
                )
            self._plugins[key] = spec
        return spec

    def unregister(self, kind: PluginKind | str, name: str) -> None:
        key = (PluginKind(kind), name.casefold())
        with self._lock:
            if key not in self._plugins:
                raise KeyError(f"Unknown plugin {key[0].value}:{name}.")
            del self._plugins[key]

    def get(self, kind: PluginKind | str, name: str) -> PluginSpec:
        key = (PluginKind(kind), name.casefold())
        with self._lock:
            try:
                return self._plugins[key]
            except KeyError as exc:
                raise KeyError(f"Unknown plugin {key[0].value}:{name}.") from exc

    def create(self, kind: PluginKind | str, name: str, **configuration: Any) -> Any:
        return self.get(kind, name).factory(**configuration)

    def list(self, kind: PluginKind | str | None = None) -> list[PluginSpec]:
        selected = PluginKind(kind) if kind is not None else None
        with self._lock:
            values = [
                spec for (plugin_kind, _), spec in self._plugins.items()
                if selected is None or plugin_kind == selected
            ]
        return sorted(values, key=lambda spec: (spec.kind.value, spec.name.casefold()))

    def load_manifest(self, path: str | Path, *, replace: bool = False) -> list[PluginSpec]:
        """Load explicitly selected local plugin factories from a versioned JSON file."""

        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        version = str(payload.get("api_version", ""))
        if version != PLUGIN_API_VERSION:
            raise ValueError(
                f"Plugin manifest targets API {version or 'unspecified'}; "
                f"expected {PLUGIN_API_VERSION}."
            )
        rows = payload.get("plugins")
        if not isinstance(rows, list):
            raise ValueError("Plugin manifest must contain a plugins list.")
        loaded: list[PluginSpec] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Every plugin manifest row must be an object.")
            spec = PluginSpec(
                kind=PluginKind(str(row["kind"])),
                name=str(row["name"]),
                factory=_load_dotted_factory(str(row["factory"])),
                api_version=version,
                description=str(row.get("description", "")),
                metadata=dict(row.get("metadata") or {}),
            )
            self.register(spec, replace=replace)
            loaded.append(spec)
        return loaded

    def discover_entry_points(self, *, replace: bool = False) -> list[PluginSpec]:
        """Load installed ``opendronekit.plugins`` entry points only when requested."""

        entry_points = importlib.metadata.entry_points()
        selected = (
            entry_points.select(group="opendronekit.plugins")
            if hasattr(entry_points, "select")
            else entry_points.get("opendronekit.plugins", [])
        )
        loaded: list[PluginSpec] = []
        for entry_point in selected:
            value = entry_point.load()
            spec = value() if callable(value) and not isinstance(value, PluginSpec) else value
            if not isinstance(spec, PluginSpec):
                raise TypeError(
                    f"Entry point {entry_point.name!r} did not return PluginSpec."
                )
            self.register(spec, replace=replace)
            loaded.append(spec)
        return loaded


registry = PluginRegistry()
