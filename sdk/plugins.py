"""Versioned plugin registry for every extension point promised by the spec.

The public import path is :mod:`opendronekit.plugins`, which re-exports this module and
adds the per-kind contracts. ``sdk.plugins`` keeps working for code written against it.

Contract version
----------------
``PLUGIN_API_LEVEL`` is ``"MAJOR.MINOR"``. A plugin declares the level it was written
against and is accepted when the major numbers are equal and its minor is not newer than
the host's. A plugin for ``1.0`` therefore loads on a ``1.1`` host; a plugin for ``1.2``
or ``2`` is refused at load time, with the reason, instead of failing mid-job.

Failure isolation
-----------------
:meth:`PluginRegistry.discover` loads every installed ``opendronekit.plugins`` entry point
and *never raises* for a broken one: an import error, a factory that is not callable, an
incompatible API level, a duplicate name or a bad manifest is recorded as a
:class:`PluginFailure` and the plugin is skipped. That is failure isolation, not a
security sandbox -- plugin code runs with the permissions of the OpenDroneKit process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
from threading import RLock
import traceback
from typing import Any, Callable, Iterable, Mapping


#: Major contract number. Kept as the historic string so ``api_version="1"`` still loads.
PLUGIN_API_VERSION = "1"
#: The full level this host implements. 1.1 added manifests, versions, report sections
#: and failure-isolated discovery; everything a 1.0 plugin relied on is unchanged.
PLUGIN_API_LEVEL = "1.1"
ENTRY_POINT_GROUP = "opendronekit.plugins"

_LEVEL = re.compile(r"^\s*(\d+)(?:\.(\d+))?\s*$")
_VERSION = re.compile(r"^[0-9][0-9A-Za-z.+!-]*$")


class PluginKind(str, Enum):
    DRONE = "drone"
    CAMERA = "camera"
    PAYLOAD = "payload"
    #: A mission template: turns an area and parameters into planner options.
    MISSION_TYPE = "mission_type"
    ENGINE = "engine"
    #: A detector model: image in, detections with a model key and digest out.
    MODEL = "model"
    #: A mission exporter: a planned mission written to a file format.
    EXPORTER = "exporter"
    REPORT_TEMPLATE = "report_template"
    #: A report section: one extra block built from the report's own context.
    REPORT_SECTION = "report_section"
    MAP_PROVIDER = "map_provider"


def api_compatibility(requested: str) -> tuple[bool, str]:
    """Whether a plugin written against ``requested`` runs on this host, and why not.

    >>> api_compatibility("1")[0], api_compatibility("1.1")[0], api_compatibility("2")[0]
    (True, True, False)
    """
    wanted = _LEVEL.match(str(requested or ""))
    host = _LEVEL.match(PLUGIN_API_LEVEL)
    assert host is not None
    if not wanted:
        return False, (f"api_version {requested!r} is not MAJOR or MAJOR.MINOR; "
                       f"this host implements {PLUGIN_API_LEVEL}.")
    major, minor = int(wanted.group(1)), int(wanted.group(2) or 0)
    host_major, host_minor = int(host.group(1)), int(host.group(2) or 0)
    if major != host_major:
        return False, (f"targets plugin API {requested}; this host implements "
                       f"{PLUGIN_API_LEVEL} (major versions differ).")
    if minor > host_minor:
        return False, (f"targets plugin API {requested}, newer than this host's "
                       f"{PLUGIN_API_LEVEL}; update OpenDroneKit.")
    return True, ""


@dataclass(frozen=True)
class PluginSpec:
    kind: PluginKind
    name: str
    factory: Callable[..., Any]
    api_version: str = PLUGIN_API_VERSION
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: The plugin's own release, e.g. "0.3.1". Filled from its manifest when omitted.
    version: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PluginKind):
            object.__setattr__(self, "kind", PluginKind(self.kind))
        if not str(self.name).strip():
            raise ValueError("Plugin name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("Plugin factory must be callable.")
        compatible, why = api_compatibility(self.api_version)
        if not compatible:
            raise ValueError(f"Plugin {self.name!r} {why}")
        if self.version and not _VERSION.match(str(self.version)):
            raise ValueError(f"Plugin {self.name!r} has an unreadable version {self.version!r}.")

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.name}"

    def describe(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["factory"] = f"{self.factory.__module__}:{self.factory.__qualname__}"
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class PluginManifest:
    """What one installed distribution provides: who it is and which plugins it adds.

    An ``opendronekit.plugins`` entry point should resolve to one of these (or to a
    zero-argument callable returning one). The manifest's ``api_version`` applies to
    every plugin in it, and its ``version`` fills any plugin that does not state one.
    """

    name: str
    version: str
    api_version: str
    plugins: tuple[PluginSpec, ...]
    description: str = ""
    homepage: str = ""
    license: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Plugin manifest name cannot be empty.")
        if not str(self.version).strip() or not _VERSION.match(str(self.version)):
            raise ValueError(f"Plugin manifest {self.name!r} needs a version such as '0.1.0'.")
        compatible, why = api_compatibility(self.api_version)
        if not compatible:
            raise ValueError(f"Plugin package {self.name!r} {why}")
        plugins = tuple(self.plugins)
        if not plugins:
            raise ValueError(f"Plugin manifest {self.name!r} lists no plugins.")
        filled = []
        for spec in plugins:
            if not isinstance(spec, PluginSpec):
                raise TypeError(f"Plugin manifest {self.name!r} lists {spec!r}, not a PluginSpec.")
            if spec.version:
                filled.append(spec)
            else:
                filled.append(PluginSpec(spec.kind, spec.name, spec.factory, spec.api_version,
                                         spec.description, spec.metadata, self.version))
        object.__setattr__(self, "plugins", tuple(filled))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "api_version": self.api_version,
            "description": self.description, "homepage": self.homepage,
            "license": self.license,
            "plugins": [{"kind": s.kind.value, "name": s.name, "version": s.version,
                         "description": s.description} for s in self.plugins],
        }


@dataclass(frozen=True)
class PluginFailure:
    """A plugin that was skipped, and exactly why."""

    source: str          # entry point "name = module:attr", or the plugin key
    stage: str           # load | manifest | register | create | call
    error: str           # "ExceptionType: message"
    distribution: str = ""
    detail: str = ""     # the last frames of the traceback, for the plugin's author

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DiscoveryReport:
    """The result of one discovery pass: what loaded, and what was skipped."""

    loaded: list[PluginSpec] = field(default_factory=list)
    manifests: list[PluginManifest] = field(default_factory=list)
    failures: list[PluginFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": PLUGIN_API_LEVEL,
            "entry_point_group": ENTRY_POINT_GROUP,
            "loaded": [{"kind": s.kind.value, "name": s.name, "version": s.version,
                        "api_version": s.api_version, "description": s.description,
                        "package": str(s.metadata.get("package", ""))} for s in self.loaded],
            "packages": [m.describe() for m in self.manifests],
            "failed": [f.to_dict() for f in self.failures],
        }


def short_traceback(exc: BaseException, frames: int = 3) -> str:
    """The last few frames of ``exc``'s traceback, for a plugin author to read."""
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(lines[-(frames * 2 + 1):]).strip()


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _load_dotted_factory(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Plugin factory must use module.path:attribute syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"Plugin factory {reference!r} is not callable.")
    return factory


def installed_entry_points(group: str = ENTRY_POINT_GROUP) -> list[Any]:
    """Every installed entry point in ``group``; nothing is imported yet."""
    found = importlib.metadata.entry_points()
    if hasattr(found, "select"):
        return list(found.select(group=group))
    return list(found.get(group, []))  # pragma: no cover - Python < 3.10


class PluginRegistry:
    """Thread-safe explicit registry; importing the SDK performs no discovery or I/O."""

    def __init__(self) -> None:
        self._plugins: dict[tuple[PluginKind, str], PluginSpec] = {}
        self._lock = RLock()
        #: Failures from the most recent discovery.
        self.failures: list[PluginFailure] = []

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

    def has(self, kind: PluginKind | str, name: str) -> bool:
        with self._lock:
            return (PluginKind(kind), str(name).casefold()) in self._plugins

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

    def clear(self) -> None:
        with self._lock:
            self._plugins.clear()
            self.failures = []

    def load_manifest(self, path: str | Path, *, replace: bool = False) -> list[PluginSpec]:
        """Load explicitly selected local plugin factories from a versioned JSON file.

        An explicit manifest is something the operator chose, so a bad one raises: the
        mistake is theirs to see immediately. Installed entry points go through
        :meth:`discover`, which never raises for a plugin's fault.
        """

        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        version = str(payload.get("api_version", ""))
        compatible, why = api_compatibility(version)
        if not version or not compatible:
            raise ValueError(
                f"Plugin manifest targets API {version or 'unspecified'}; "
                f"expected {PLUGIN_API_LEVEL}. {why}".strip()
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
                version=str(row.get("version") or payload.get("version") or ""),
            )
            self.register(spec, replace=replace)
            loaded.append(spec)
        return loaded

    def _register_quietly(self, spec: PluginSpec, source: str, distribution: str,
                          report: DiscoveryReport, replace: bool) -> None:
        try:
            self.register(spec, replace=replace)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not stop the rest
            report.failures.append(PluginFailure(source, "register", describe_error(exc),
                                                 distribution))
            return
        report.loaded.append(spec)

    def discover(self, *, replace: bool = False,
                 entry_points: Iterable[Any] | None = None) -> DiscoveryReport:
        """Load installed ``opendronekit.plugins`` entry points, skipping broken ones.

        Each entry point may resolve to a :class:`PluginManifest`, a :class:`PluginSpec`,
        a list of either, or a zero-argument callable returning one of those. Anything
        that raises or does not fit is recorded in the report's ``failures`` and skipped;
        this method does not raise for a plugin's fault.
        """

        report = DiscoveryReport()
        candidates = list(entry_points) if entry_points is not None else installed_entry_points()
        for entry_point in candidates:
            source = f"{getattr(entry_point, 'name', '?')} = {getattr(entry_point, 'value', '?')}"
            distribution = ""
            dist = getattr(entry_point, "dist", None)
            if dist is not None:
                try:
                    distribution = f"{dist.metadata['Name']} {dist.version}"
                except Exception:  # noqa: BLE001
                    distribution = ""
            try:
                value = entry_point.load()
            except BaseException as exc:  # noqa: BLE001 - a SystemExit at import too
                if isinstance(exc, KeyboardInterrupt):
                    raise
                report.failures.append(PluginFailure(source, "load", describe_error(exc),
                                                     distribution, short_traceback(exc)))
                continue
            try:
                if callable(value) and not isinstance(value, (PluginSpec, PluginManifest)):
                    value = value()
                items = value if isinstance(value, (list, tuple)) else [value]
                for item in items:
                    if isinstance(item, PluginManifest):
                        report.manifests.append(item)
                        for spec in item.plugins:
                            meta = dict(spec.metadata)
                            meta.setdefault("package", item.name)
                            spec = PluginSpec(spec.kind, spec.name, spec.factory,
                                              spec.api_version, spec.description, meta,
                                              spec.version)
                            self._register_quietly(spec, source, distribution, report, replace)
                    elif isinstance(item, PluginSpec):
                        self._register_quietly(item, source, distribution, report, replace)
                    else:
                        raise TypeError(
                            f"Entry point {getattr(entry_point, 'name', '?')!r} gave "
                            f"{type(item).__name__}, not a PluginManifest or PluginSpec.")
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, KeyboardInterrupt):
                    raise
                report.failures.append(PluginFailure(source, "manifest", describe_error(exc),
                                                     distribution, short_traceback(exc)))
        self.failures = list(report.failures)
        return report

    def discover_entry_points(self, *, replace: bool = False) -> list[PluginSpec]:
        """Load installed ``opendronekit.plugins`` entry points only when requested.

        Kept for code written against API 1.0. Broken plugins are skipped and listed in
        ``self.failures`` rather than raising; use :meth:`discover` for the full report.
        """

        return self.discover(replace=replace).loaded


registry = PluginRegistry()
