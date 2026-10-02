"""The OpenDroneKit plugin API (contract level 1.1).

Plugins are ordinary Python distributions that declare an entry point in the
``opendronekit.plugins`` group. The entry point resolves to a :class:`PluginManifest`
(name, version, ``api_version``, and the plugins it provides)::

    # pyproject.toml of the plugin
    [project.entry-points."opendronekit.plugins"]
    corridor = "odk_corridor_template:manifest"

    # odk_corridor_template/__init__.py
    from opendronekit.plugins import PluginKind, PluginManifest, PluginSpec

    manifest = PluginManifest(
        name="odk-corridor-template", version="0.1.0", api_version="1.1",
        plugins=(PluginSpec(PluginKind.MISSION_TYPE, "corridor", CorridorTemplate),),
    )

Four kinds have a contract the host calls into; each is a small duck-typed object built
by the spec's factory:

==================  ===========================================================
Kind                What the built object must provide
==================  ===========================================================
``mission_type``    ``plan_options(aoi_lonlat, **params) -> dict`` (planner options;
                    may replace ``aoi_lonlat`` and must name a built-in ``template``)
``model``           ``model_key``, ``model_sha256`` and
                    ``detect(image_bgr) -> list[dict]`` (``label``, ``score``,
                    ``bbox_xyxy``)
``exporter``        ``suffix`` and ``export(plan: dict, path: Path) -> str | list[str]``
``report_section``  ``title`` and ``build(context: dict) -> dict`` (``paragraphs``
                    and/or ``table``)
==================  ===========================================================

The other kinds (drone, camera, payload, engine, report_template, map_provider) are
registered and listed the same way; their host integration is the factory call.

Failure isolation: discovery never raises for a broken plugin (it is listed under
``failed`` with the reason), and a plugin that raises or returns something outside its
contract raises :class:`PluginError` to the caller that asked for it -- never past it.
Output is checked, not trusted: an exporter that claims a file it did not write, a
detector with no model digest or a box outside the image, are refused.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterable, Protocol, runtime_checkable

from sdk.plugins import (
    ENTRY_POINT_GROUP,
    PLUGIN_API_LEVEL,
    PLUGIN_API_VERSION,
    DiscoveryReport,
    PluginFailure,
    PluginKind,
    PluginManifest,
    PluginRegistry,
    PluginSpec,
    api_compatibility,
    describe_error,
    installed_entry_points,
    registry,
    short_traceback,
)

from .errors import PluginError

__all__ = [
    "ENTRY_POINT_GROUP", "PLUGIN_API_LEVEL", "PLUGIN_API_VERSION",
    "DetectorPlugin", "DiscoveryReport", "ExporterPlugin", "MissionTemplatePlugin",
    "PluginDetections", "PluginError", "PluginFailure", "PluginHost", "PluginKind",
    "PluginManifest", "PluginRegistry", "PluginSpec", "ReportSectionPlugin",
    "api_compatibility", "host", "installed_entry_points", "load_plugins",
    "plugin_status", "registry",
]


# ── Contracts ─────────────────────────────────────────────────────────────────

@runtime_checkable
class MissionTemplatePlugin(Protocol):
    """``mission_type``: a parameterised mission over the built-in planner."""

    def plan_options(self, aoi_lonlat: list[list[float]], **params: Any) -> dict[str, Any]:
        """Planner options for this area: ``template`` (a built-in one), altitudes,
        overlaps... and optionally a replacement ``aoi_lonlat``."""


@runtime_checkable
class DetectorPlugin(Protocol):
    """``model``: a detector that names the exact weights that produced its output."""

    model_key: str
    model_sha256: str

    def detect(self, image_bgr: Any) -> list[dict[str, Any]]:
        """Detections as dicts with ``label``, ``score`` (0..1) and ``bbox_xyxy``."""


@runtime_checkable
class ExporterPlugin(Protocol):
    """``exporter``: writes a planned mission to one file format."""

    suffix: str

    def export(self, plan: dict[str, Any], path: Path) -> str | list[str]:
        """Write ``plan`` to ``path`` (or to parts beside it); return what was written."""


@runtime_checkable
class ReportSectionPlugin(Protocol):
    """``report_section``: one block of a report, built from the report's context."""

    title: str

    def build(self, context: dict[str, Any]) -> dict[str, Any]:
        """``{"paragraphs": [...], "table": {"header": [...], "rows": [[...]]}}``."""


CONTRACTS: dict[PluginKind, tuple[str, ...]] = {
    PluginKind.MISSION_TYPE: ("plan_options",),
    PluginKind.MODEL: ("detect", "model_key", "model_sha256"),
    PluginKind.EXPORTER: ("export", "suffix"),
    PluginKind.REPORT_SECTION: ("build", "title"),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PluginDetections:
    """What a detector plugin found, with the identity of the weights that found it."""

    plugin: str
    plugin_version: str
    model_key: str
    model_sha256: str
    detections: tuple[dict[str, Any], ...]
    image_size: tuple[int, int]          # (width, height)

    @property
    def model_used(self) -> str:
        return f"plugin:{self.plugin}"

    def to_dict(self) -> dict[str, Any]:
        return {"plugin": self.plugin, "plugin_version": self.plugin_version,
                "model_used": self.model_used, "model_key": self.model_key,
                "model_sha256": self.model_sha256, "image_size": list(self.image_size),
                "detections": [dict(d) for d in self.detections]}


# ── Host ──────────────────────────────────────────────────────────────────────

@dataclass
class _State:
    report: DiscoveryReport | None = None
    call_failures: list[PluginFailure] = field(default_factory=list)


class PluginHost:
    """Loads plugins once, builds them on demand and checks what they return.

    One process-wide instance, :data:`host`, backs the SDK, the desktop API's plugin
    listing and Environment Capabilities. Tests make their own.
    """

    def __init__(self, plugin_registry: PluginRegistry | None = None) -> None:
        self.registry = plugin_registry if plugin_registry is not None else PluginRegistry()
        self._state = _State()
        self._lock = RLock()

    # -- discovery --------------------------------------------------------

    def load(self, *, refresh: bool = False,
             entry_points: Iterable[Any] | None = None) -> DiscoveryReport:
        """Discover installed plugins (once; ``refresh=True`` rescans)."""
        with self._lock:
            if self._state.report is not None and not refresh and entry_points is None:
                return self._state.report
            if refresh or entry_points is not None:
                self.registry.clear()
                self._state = _State()
            self._state.report = self.registry.discover(entry_points=entry_points)
            return self._state.report

    def status(self) -> dict[str, Any]:
        """Loaded and failed plugins, for Environment Capabilities and ``list_plugins``."""
        report = self.load()
        payload = report.to_dict()
        payload["failed"] = payload["failed"] + [f.to_dict() for f in self._state.call_failures]
        payload["kinds"] = {kind.value: list(CONTRACTS.get(kind, ())) for kind in PluginKind}
        return payload

    def names(self, kind: PluginKind | str) -> list[str]:
        self.load()
        return [spec.name for spec in self.registry.list(kind)]

    # -- building and calling --------------------------------------------

    def _record(self, failure: PluginFailure) -> None:
        with self._lock:
            self._state.call_failures.append(failure)
            del self._state.call_failures[:-50]

    def _fail(self, spec_key: str, stage: str, exc: BaseException | str) -> PluginError:
        reason = exc if isinstance(exc, str) else describe_error(exc)
        detail = "" if isinstance(exc, str) else short_traceback(exc)
        self._record(PluginFailure(spec_key, stage, reason, detail=detail))
        return PluginError(spec_key, stage, reason)

    def spec(self, kind: PluginKind | str, name: str) -> PluginSpec:
        self.load()
        kind = PluginKind(kind)
        try:
            return self.registry.get(kind, name)
        except KeyError:
            installed = ", ".join(self.names(kind)) or "none"
            raise PluginError(f"{kind.value}:{name}", "lookup",
                              f"no {kind.value} plugin named {name!r} is installed "
                              f"(installed: {installed}).") from None

    def build(self, kind: PluginKind | str, name: str, **config: Any) -> tuple[PluginSpec, Any]:
        """The plugin object, built by its factory and checked against its contract."""
        spec = self.spec(kind, name)
        try:
            instance = spec.factory(**config)
        except Exception as exc:  # noqa: BLE001 - a plugin's fault is reported, not raised raw
            raise self._fail(spec.key, "create", exc) from exc
        missing = [attr for attr in CONTRACTS.get(spec.kind, ()) if not hasattr(instance, attr)]
        if missing:
            raise self._fail(spec.key, "contract",
                             f"the {spec.kind.value} contract needs {', '.join(missing)}, "
                             f"which {type(instance).__name__} does not provide.")
        return spec, instance

    def _call(self, spec: PluginSpec, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise self._fail(spec.key, "call", exc) from exc

    # -- mission templates ------------------------------------------------

    def mission_options(self, name: str, aoi_lonlat: list[list[float]],
                        **params: Any) -> dict[str, Any]:
        """Planner options from a mission-template plugin, validated."""
        from mission.mission_types import SPECIAL_MISSION_TYPES
        from mission.planner import TEMPLATE_ALIASES

        spec, template = self.build(PluginKind.MISSION_TYPE, name)
        options = self._call(spec, template.plan_options,
                             [list(map(float, p[:2])) for p in aoi_lonlat], **params)
        if not isinstance(options, dict):
            raise self._fail(spec.key, "output", f"plan_options returned "
                             f"{type(options).__name__}, not a dict of planner options.")
        options = dict(options)
        base = str(options.get("template", "")).strip().lower()
        if base not in TEMPLATE_ALIASES and base not in SPECIAL_MISSION_TYPES:
            raise self._fail(spec.key, "output",
                             f"plan_options must name a built-in planner template, got "
                             f"{options.get('template')!r}.")
        area = options.get("aoi_lonlat", aoi_lonlat)
        try:
            area = [[float(p[0]), float(p[1])] for p in area]
        except (TypeError, ValueError, IndexError) as exc:
            raise self._fail(spec.key, "output", f"aoi_lonlat is not a list of [lon, lat]: {exc}")
        if len(area) < 3 or any(not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in area):
            raise self._fail(spec.key, "output",
                             "aoi_lonlat needs three or more [lon, lat] points in range.")
        options["aoi_lonlat"] = area
        options["template"] = base
        options["_plugin"] = {"kind": spec.kind.value, "name": spec.name, "version": spec.version}
        return options

    # -- exporters --------------------------------------------------------

    def export(self, name: str, plan: dict[str, Any], directory: Path | str,
               stem: str = "mission") -> list[Path]:
        """Run an exporter plugin; return the files it actually wrote."""
        spec, exporter = self.build(PluginKind.EXPORTER, name)
        suffix = str(getattr(exporter, "suffix", "") or "")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise self._fail(spec.key, "contract", f"suffix must look like '.ext', got {suffix!r}.")
        target = Path(directory) / f"{stem}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        result = self._call(spec, exporter.export, copy.deepcopy(plan), target)
        claimed = result if isinstance(result, (list, tuple)) else [result]
        written: list[Path] = []
        for item in claimed:
            path = Path(str(item))
            if not path.is_file() or path.stat().st_size == 0:
                raise self._fail(spec.key, "output",
                                 f"export reported {path} but no non-empty file is there.")
            written.append(path)
        if not written:
            raise self._fail(spec.key, "output", "export returned no files.")
        return written

    # -- detectors ----------------------------------------------------------

    def detect(self, name: str, image_bgr: Any) -> PluginDetections:
        """Run a detector plugin and refuse output that could not be traced or placed."""
        spec, detector = self.build(PluginKind.MODEL, name)
        key = str(getattr(detector, "model_key", "") or "").strip()
        digest = str(getattr(detector, "model_sha256", "") or "").strip().lower()
        if not key or not _SHA256.match(digest):
            raise self._fail(spec.key, "provenance",
                             "a detector plugin must name its model_key and the sha256 of "
                             "the weights that ran; without them its output cannot be "
                             "traced and is refused.")
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        raw = self._call(spec, detector.detect, image_bgr)
        if not isinstance(raw, (list, tuple)):
            raise self._fail(spec.key, "output", f"detect returned {type(raw).__name__}, not a list.")
        clean: list[dict[str, Any]] = []
        for index, det in enumerate(raw):
            try:
                label = str(det["label"]).strip()
                score = float(det["score"])
                x1, y1, x2, y2 = (float(v) for v in det["bbox_xyxy"])
            except (KeyError, TypeError, ValueError) as exc:
                raise self._fail(spec.key, "output", f"detection {index} is malformed: "
                                 f"{describe_error(exc)}")
            if not label or not (0.0 <= score <= 1.0):
                raise self._fail(spec.key, "output",
                                 f"detection {index} needs a label and a score in 0..1.")
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise self._fail(spec.key, "output",
                                 f"detection {index} box {[x1, y1, x2, y2]} is not inside the "
                                 f"{width}x{height} image.")
            row = {k: v for k, v in dict(det).items()
                   if k not in ("label", "score", "bbox_xyxy")}
            row.update(label=label, score=score, bbox_xyxy=[x1, y1, x2, y2])
            clean.append(row)
        return PluginDetections(spec.name, spec.version, key, digest, tuple(clean),
                                (width, height))

    # -- report sections ----------------------------------------------------

    def report_section(self, name: str, context: dict[str, Any]) -> dict[str, Any]:
        """A report block from a report-section plugin, given a copy of the context."""
        spec, section = self.build(PluginKind.REPORT_SECTION, name)
        safe = {k: v for k, v in context.items() if not str(k).startswith("_")}
        block = self._call(spec, section.build, copy.deepcopy(safe))
        if not isinstance(block, dict):
            raise self._fail(spec.key, "output", f"build returned {type(block).__name__}, not a dict.")
        paragraphs = block.get("paragraphs") or []
        table = block.get("table")
        if not isinstance(paragraphs, list) or not all(isinstance(p, str) for p in paragraphs):
            raise self._fail(spec.key, "output", "paragraphs must be a list of strings.")
        if table is not None:
            if not isinstance(table, dict) or not isinstance(table.get("rows", []), list):
                raise self._fail(spec.key, "output", "table must be {'header': [...], 'rows': [[...]]}.")
            table = {"header": [str(h) for h in table.get("header") or []],
                     "rows": [[str(c) for c in row] for row in table.get("rows") or []]}
        if not paragraphs and not (table and table["rows"]):
            raise self._fail(spec.key, "output", "the section has no paragraphs and no table rows.")
        out: dict[str, Any] = {
            "title": str(block.get("title") or getattr(section, "title", "") or spec.name),
            "paragraphs": list(paragraphs) + [
                f"Section written by the report-section plugin {spec.name} {spec.version}".strip()
                + "."],
        }
        if table:
            out["table"] = table
        return out


#: The process-wide host used by the SDK and the desktop API.
host = PluginHost(registry)


def load_plugins(*, refresh: bool = False) -> DiscoveryReport:
    """Discover installed ``opendronekit.plugins`` entry points (cached)."""
    return host.load(refresh=refresh)


def plugin_status() -> dict[str, Any]:
    """Loaded and failed plugins as plain data (what Environment Capabilities shows)."""
    return host.status()
