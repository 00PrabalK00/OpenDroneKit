"""Workspaces and projects: the entry point of the OpenDroneKit SDK.

The SDK drives the same code the desktop application's buttons do. A :class:`Workspace`
holds an in-process instance of the desktop API (``app.api.Api``) bound to its own
project store, so a script and the application refuse the same things for the same
reasons and write the same files. Nothing here re-implements planning, reconstruction,
detection, measurement or reporting.

    >>> import opendronekit as odk
    >>> ws = odk.Workspace("C:/odk-data")                       # doctest: +SKIP
    >>> project = ws.create_project("Quarry survey")            # doctest: +SKIP
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Sequence

from .errors import JobFailed, OdkError, Refused
from .results import (
    CrackDetection,
    Dataset,
    Detections,
    Measurement,
    MissionSummary,
    ProjectInfo,
    Reconstruction,
    Report,
)

LonLat = Sequence[float]
Progress = Callable[[int, str], None]

_TERMINAL = {"done", "failed", "cancelled"}


def _unwrap(reply: dict[str, Any]) -> dict[str, Any]:
    """An ``Api`` reply as its payload, or :class:`Refused` with the application's reason."""
    if not isinstance(reply, dict):
        raise OdkError(f"Unexpected reply from OpenDroneKit: {reply!r}")
    if not reply.get("ok", False):
        raise Refused(str(reply.get("error") or "OpenDroneKit refused without a reason."),
                      payload={k: v for k, v in reply.items() if k not in ("ok", "error")})
    return {k: v for k, v in reply.items() if k != "ok"}


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip()).strip("_")
    return cleaned or "project"


class Workspace:
    """A folder holding OpenDroneKit projects, opened for scripting.

    ``data_dir`` is the same kind of folder as the application's data root
    (``ODK_DATA_DIR``): the project index lives in ``<data_dir>/app_state/projects.db``
    and new projects in ``<data_dir>/projects/<name>``. Pointing a workspace at the
    application's own data root opens the user's real projects; point it at a scratch
    folder for experiments.
    """

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        if data_dir is None:
            from core.paths import data_root

            data_dir = data_root()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._store = ProjectStore(self.data_dir / "app_state" / "projects.db")
        self._session = AppSession(self._store)
        self._api = Api(self._session)
        self._active: int | None = None
        # One Api serves every Project object of this workspace, and the Api's mission,
        # dataset and layers belong to the active project, so calls are serialised.
        self._lock = threading.RLock()

    # -- context management ---------------------------------------------------

    def close(self) -> None:
        """Release the project index. The workspace cannot be used afterwards."""
        with self._lock:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- projects -------------------------------------------------------------

    def create_project(self, name: str, *, root: str | os.PathLike[str] | None = None,
                       description: str = "") -> "Project":
        """Create a project; refuses a name that already exists in this workspace."""
        target = Path(root) if root else self.data_dir / "projects" / _safe_name(name)
        with self._lock:
            payload = _unwrap(self._api.create_project(name, str(target), description))
            project = payload["project"]
            self._active = int(project["id"])
            return Project(self, int(project["id"]))

    def open_project(self, project: int | str) -> "Project":
        """Open a project by numeric id or by (case-insensitive) name."""
        rows = self.list_projects()
        if isinstance(project, int) or str(project).isdigit():
            match = [p for p in rows if p.id == int(project)]
        else:
            match = [p for p in rows if p.name.casefold() == str(project).strip().casefold()]
        if not match:
            names = ", ".join(p.name for p in rows) or "none"
            raise Refused(f"No project {project!r} in {self.data_dir} (projects: {names}).")
        return Project(self, match[0].id)

    def list_projects(self) -> list[ProjectInfo]:
        """Every project in this workspace."""
        with self._lock:
            rows = _unwrap(self._api.list_projects())["projects"]
        return [ProjectInfo(int(r["id"]), str(r.get("name", "")), Path(str(r.get("root_dir", ""))),
                            str(r.get("description", ""))) for r in rows]

    # -- environment ----------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """What this installation can actually do (the Environment Capabilities data)."""
        with self._lock:
            return _unwrap(self._api.capabilities())["capabilities"]

    def mission_templates(self) -> list[str]:
        """Built-in planner templates plus installed mission-template plugins."""
        from .plugins import PluginKind, host

        with self._lock:
            payload = _unwrap(self._api.mission_templates())
        builtin = list(payload.get("templates", [])) + list(payload.get("special_templates", []))
        return builtin + [f"plugin:{n}" for n in host.names(PluginKind.MISSION_TYPE)]

    # -- internals ------------------------------------------------------------

    def _activate(self, project_id: int) -> Any:
        """Make ``project_id`` the API's active project (caller holds the lock)."""
        if self._active != project_id or self._store.get_active_project_id() != project_id:
            _unwrap(self._api.set_active_project(project_id))
            self._active = project_id
        return self._api


class MissionPlan:
    """A planned mission: its summary, its GeoJSON and the full plan, ready to export."""

    def __init__(self, project: "Project", plan: dict[str, Any], summary: MissionSummary,
                 geojson: dict[str, Any]) -> None:
        self.project = project
        self.plan = plan
        self.summary = summary
        self.geojson = geojson

    def __repr__(self) -> str:
        s = self.summary
        return (f"MissionPlan(template={s.template!r}, waypoints={s.waypoints}, "
                f"distance_m={s.distance_m}, duration_min={s.duration_min})")

    def save(self, name: str = "", note: str = "") -> dict[str, Any]:
        """Save this plan as a new version in the project's mission history."""
        with self.project.workspace._lock:
            self.project._api()
            session = self.project.workspace._session
            # The history stores the session's current plan; make it this one, so saving
            # an older MissionPlan object never records a newer plan under its name.
            session.mission_plan_dict = dict(self.plan)
            return _unwrap(self.project.workspace._api.save_mission(name, note))["version"]

    @staticmethod
    def builtin_formats() -> list[str]:
        """The mission export formats built into OpenDroneKit (plugins not included)."""
        from mission import exporters

        return list(exporters.EXPORTERS)

    def export(self, formats: Iterable[str] | None = None, *,
               directory: str | os.PathLike[str] | None = None,
               stem: str | None = None) -> dict[str, list[Path]]:
        """Write the mission in each format; returns ``{format: [files]}``.

        Built-in formats are those of :meth:`builtin_formats`; any other name is looked
        up as an installed exporter plugin. A format that fails raises (built-in) or
        raises :class:`PluginError` (plugin) -- a file is never reported that was not
        written.
        """
        from mission import exporters

        from .plugins import host

        chosen = list(formats) if formats is not None else self.builtin_formats()
        target = Path(directory) if directory else self.project.root / "missions"
        target.mkdir(parents=True, exist_ok=True)
        stem = stem or "mission_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        written: dict[str, list[Path]] = {}
        for name in chosen:
            entry = exporters.EXPORTERS.get(name)
            if entry is None:
                written[name] = host.export(name, self.plan, target, stem)
                continue
            writer, suffix = entry
            parts: dict[str, str] = {}
            exporters.record_written(parts, name, writer(target / f"{stem}{suffix}", self.plan))
            files = [Path(p) for p in parts.values()]
            missing = [p for p in files if not p.is_file()]
            if missing:
                raise OdkError(f"{name} export reported {missing[0]} but it was not written.")
            written[name] = files
        with self.project.workspace._lock:
            self.project._api()
            self.project.workspace._session.audit(
                "mission_exported", {"formats": chosen, "directory": str(target), "via": "sdk"})
        return written


class Project:
    """One project in a :class:`Workspace`. Obtain it from the workspace, not directly."""

    def __init__(self, workspace: Workspace, project_id: int) -> None:
        self.workspace = workspace
        self.id = int(project_id)

    def __repr__(self) -> str:
        return f"Project(id={self.id}, name={self.info.name!r})"

    def _api(self) -> Any:
        return self.workspace._activate(self.id)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.workspace._lock:
            return _unwrap(getattr(self._api(), method)(*args, **kwargs))

    # -- identity -----------------------------------------------------------

    @property
    def info(self) -> ProjectInfo:
        """This project's row in the workspace index (refused if it was deleted)."""
        for row in self.workspace.list_projects():
            if row.id == self.id:
                return row
        raise Refused(f"Project {self.id} no longer exists.")

    @property
    def name(self) -> str:
        """The project's name."""
        return self.info.name

    @property
    def root(self) -> Path:
        """The project's folder on disk."""
        return self.info.root

    def audit_log(self, limit: int = 200) -> list[dict[str, Any]]:
        """The project's audit trail, newest first (up to ``limit`` events)."""
        return self._call("audit_log", limit)["events"]

    # -- missions -----------------------------------------------------------

    def plan_mission(self, aoi_lonlat: Sequence[LonLat], *, template: str = "grid",
                     **options: Any) -> MissionPlan:
        """Plan a mission over an area of interest.

        ``aoi_lonlat`` is a polygon as ``[[lon, lat], ...]``. ``template`` is a built-in
        planner template (see :meth:`Workspace.mission_templates`) or
        ``"plugin:<name>"`` for an installed mission-template plugin, which is then
        called with ``aoi_lonlat`` and ``options`` and returns the planner options.
        Other ``options`` are the planner's: ``altitude_m``, ``front_overlap_pct``,
        ``side_overlap_pct``, ``speed_m_s``, ``camera``, ``gimbal_tilt_deg`` ...
        An unknown template is refused rather than silently planned as a grid.
        """
        from mission.mission_types import SPECIAL_MISSION_TYPES
        from mission.planner import TEMPLATE_ALIASES

        from .plugins import host

        plugin_info: dict[str, str] = {}
        area = [[float(p[0]), float(p[1])] for p in aoi_lonlat]
        name = str(template).strip()
        if name.lower().startswith("plugin:"):
            opts = host.mission_options(name.split(":", 1)[1], area, **options)
            plugin_info = {k: str(v) for k, v in opts.pop("_plugin").items()}
            area = opts.pop("aoi_lonlat")
        else:
            key = name.lower()
            if key not in TEMPLATE_ALIASES and key not in SPECIAL_MISSION_TYPES:
                raise Refused(f"Unknown mission template {template!r}. Use one of "
                              f"{', '.join(self.workspace.mission_templates())}.")
            opts = dict(options, template=key)
        with self.workspace._lock:
            api = self._api()
            _unwrap(api.set_aoi(area))
            payload = _unwrap(api.plan_mission(opts))
            plan = dict(self.workspace._session.mission_plan_dict)
        if plugin_info:
            plan["plugin"] = plugin_info
        s = payload.get("summary") or {}
        summary = MissionSummary(
            template=str(s.get("template", opts.get("template", ""))),
            waypoints=int(s.get("waypoints", 0)),
            distance_m=float(s.get("distance_m", 0.0)),
            duration_min=float(s.get("duration_min", 0.0)),
            gsd_cm=float(s.get("gsd_cm", 0.0) or 0.0),
            altitude_m=float(s.get("altitude_m", 0.0) or 0.0),
            warnings=tuple(str(w) for w in payload.get("warnings") or []),
            plugin=plugin_info,
        )
        return MissionPlan(self, plan, summary, dict(payload.get("geojson") or {}))

    # -- datasets -----------------------------------------------------------

    def import_dataset(self, folder: str | os.PathLike[str], name: str = "") -> Dataset:
        """Register an image folder with the project and make it the active dataset."""
        payload = self._call("import_dataset", str(Path(folder).resolve()), name)["dataset"]
        meta = payload.get("metadata") or {}
        return Dataset(str(payload.get("name") or Path(folder).name), Path(str(folder)).resolve(),
                       meta.get("image_count"), meta.get("geotagged_count"), payload)

    # -- processing ---------------------------------------------------------

    def reconstruct(self, images: str | os.PathLike[str] | None = None, *,
                    engine: str = "auto", profile: str = "standard",
                    max_image_size: int | None = None, dense: bool | None = None,
                    use_gpu: bool = True, epsg: int | None = None,
                    output_dir: str | os.PathLike[str] | None = None,
                    timeout_s: float | None = None, progress: Progress | None = None,
                    poll_s: float = 0.5) -> Reconstruction:
        """Run a reconstruction and wait for it.

        ``images`` defaults to the active dataset (:meth:`import_dataset`). ``engine``
        is ``auto``, ``colmap`` or ``custom``. Raises :class:`Refused` before starting
        (no images, a busy output folder...) and :class:`JobFailed` if the job fails;
        a :class:`TimeoutError` cancels the job.
        """
        options: dict[str, Any] = {"engine": engine, "profile": profile, "use_gpu": use_gpu}
        if images is not None:
            options["image_dir"] = str(Path(images).resolve())
        if max_image_size:
            options["max_image_size"] = int(max_image_size)
        if dense is not None:
            options["dense"] = bool(dense)
        if epsg:
            options["epsg"] = int(epsg)
        if output_dir:
            options["output_dir"] = str(Path(output_dir).resolve())
        job_id = self._call("run_reconstruction", options)["job_id"]
        job = self._wait(job_id, timeout_s, progress, poll_s)
        result = dict(job.get("result") or {})
        return Reconstruction(
            engine=str(result.get("engine", "")),
            output_dir=Path(options.get("output_dir") or self.root / "reconstruction"),
            registered_images=int(result.get("registered_images") or 0),
            frame_count=int(result.get("frame_count") or 0),
            reprojection_error_px=result.get("reprojection_error_px"),
            crs_epsg=result.get("crs_epsg"),
            geo_rmse_m=result.get("geo_rmse_m"),
            orthomosaic_path=str(result.get("orthomosaic_path") or ""),
            dsm_path=str(result.get("dsm_path") or ""),
            point_cloud_path=str(result.get("point_cloud_path") or ""),
            warnings=tuple(str(w) for w in result.get("warnings") or []),
            raw=result,
        )

    def _wait(self, job_id: str, timeout_s: float | None, progress: Progress | None,
              poll_s: float) -> dict[str, Any]:
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        last = None
        while True:
            with self.workspace._lock:
                job = _unwrap(self.workspace._api.job_status(job_id))["job"]
            seen = (job.get("percent"), job.get("message"))
            if progress and seen != last:
                progress(int(job.get("percent") or 0), str(job.get("message") or ""))
                last = seen
            if job.get("status") in _TERMINAL:
                if job["status"] != "done":
                    raise JobFailed(job)
                return job
            if deadline is not None and time.monotonic() > deadline:
                with self.workspace._lock:
                    self.workspace._api.cancel_job(job_id)
                raise TimeoutError(f"Job {job_id} did not finish within {timeout_s} s; "
                                   "it was asked to cancel.")
            time.sleep(poll_s)

    # -- detection ----------------------------------------------------------

    def detect_cracks(self, image: str | os.PathLike[str], *, use_model: bool = True,
                      min_area_px: int = 30) -> CrackDetection:
        """Segment cracks on one photograph; the mask and overlay are saved in the project
        under ``analysis/cracks/<image stem>/<source>/``.

        With ``use_model`` the installed ``crack_segmentation`` model runs; when it is
        not installed the classical heuristic runs instead and the result says
        ``source="heuristic"``. Pass ``use_model=False`` to ask for the heuristic.
        """
        import cv2

        from core.detection import detect_cracks, load_image

        path = Path(image).resolve()
        if not path.is_file():
            raise Refused(f"No such image: {image}")
        pixels = load_image(path)
        result = detect_cracks(pixels, min_area_px=min_area_px, use_model=use_model)
        fallback_reason = ""
        if result.model_used == "onnx_fallback_heuristic":
            # The model is installed but its forward pass failed. Try once more (a
            # transient out-of-memory on a busy machine is the usual cause); if it
            # still fails, keep the heuristic mask, labelled as such, with the cause.
            result = detect_cracks(pixels, min_area_px=min_area_px, use_model=use_model)
            if result.model_used == "onnx_fallback_heuristic":
                fallback_reason = _crack_model_failure(pixels)
        source = "model" if str(result.model_used).startswith("onnx:") else "heuristic"
        # One folder per source: a heuristic run never overwrites a model's mask.
        out = self.root / "analysis" / "cracks" / path.stem / source
        out.mkdir(parents=True, exist_ok=True)
        mask_path, overlay_path = out / "crack_mask.png", out / "crack_overlay.jpg"
        if not cv2.imwrite(str(mask_path), result.mask) or not cv2.imwrite(str(overlay_path), result.overlay):
            raise OdkError(f"Could not write the crack mask under {out}.")
        with self.workspace._lock:
            self._api()
            self.workspace._session.audit("crack_detection", {
                "image": str(path), "model_used": result.model_used, "via": "sdk"})
        return CrackDetection(path, source, result.model_used, result.model_key,
                              result.model_sha256, int(result.crack_pixels),
                              float(result.crack_ratio), float(result.estimated_length_px),
                              float(result.estimated_max_width_px), mask_path, overlay_path,
                              fallback_reason)

    def run_model(self, model_key: str, image: str | os.PathLike[str]) -> Detections:
        """Run any installed registry model on one image (refused if not installed)."""
        payload = self._call("run_model", str(model_key), str(Path(image).resolve()))
        output = payload.get("output") or {}
        return Detections(Path(str(payload.get("image", image))), str(payload.get("model_used", "")),
                          str(payload.get("model_key", "")), str(payload.get("model_sha256", "")),
                          tuple(output.get("detections") or ()), str(payload.get("scope", "")),
                          str(payload.get("overlay_path", "")),
                          {k: v for k, v in payload.items() if k != "overlay_data_uri"})

    def detect(self, image: str | os.PathLike[str], *, detector: str) -> Detections:
        """Run a detector plugin (``detector="<plugin name>"``) on one image."""
        from core.detection import load_image

        from .plugins import host

        path = Path(image).resolve()
        if not path.is_file():
            raise Refused(f"No such image: {image}")
        found = host.detect(detector, load_image(path))
        with self.workspace._lock:
            self._api()
            self.workspace._session.audit("plugin_detection", {
                "image": str(path), "plugin": found.plugin, "model_key": found.model_key,
                "model_sha256": found.model_sha256, "count": len(found.detections)})
        return Detections(path, found.model_used, found.model_key, found.model_sha256,
                          found.detections, raw=found.to_dict())

    # -- measurement ----------------------------------------------------------

    def _dataset_name(self, image: str | os.PathLike[str]) -> str:
        active = self.workspace._session.active_dataset_dir
        if not active:
            raise Refused("Import the photograph's dataset first (Project.import_dataset).")
        path = Path(image)
        if not path.is_absolute():
            return str(path)
        try:
            return str(path.resolve().relative_to(Path(active).resolve()))
        except ValueError:
            raise Refused(f"{path} is not in the active dataset {active}.") from None

    @staticmethod
    def _measurement(kind: str, payload: dict[str, Any]) -> Measurement:
        record = payload.get("measurement") or {}
        value = payload.get("value_m", payload.get("value", record.get("value")))
        unit = payload.get("unit") or record.get("unit") or ("m2" if kind == "area" else "m")
        if "value_m" in payload:
            unit = "m2" if kind == "area" else "m"
        return Measurement(str(payload.get("measurement_id") or record.get("id") or ""), kind,
                           float(value), str(unit), str(payload.get("caveat", "")), payload)

    def measure_image(self, kind: str, image: str | os.PathLike[str],
                      pixels: Sequence[Sequence[float]], *,
                      scale_mm_per_px: float | None = None,
                      reference_pixels: Sequence[Sequence[float]] | None = None,
                      reference_length_mm: float | None = None,
                      label: str = "") -> Measurement:
        """A distance or area on one photograph of the active dataset.

        A photograph has no scale of its own: give ``scale_mm_per_px`` or a reference of
        known length (``reference_pixels`` + ``reference_length_mm``). Without either
        the measurement is refused.
        """
        name = self._dataset_name(image)
        payload = self._call("measure_on_image", kind, name, [list(p) for p in pixels],
                             scale_mm_per_px, [list(p) for p in reference_pixels or []] or None,
                             reference_length_mm, label)
        return self._measurement(kind, payload)

    def measure_crack(self, image: str | os.PathLike[str], mask: str | os.PathLike[str], *,
                      scale_mm_per_px: float | None = None,
                      reference_pixels: Sequence[Sequence[float]] | None = None,
                      reference_length_mm: float | None = None, label: str = "") -> Measurement:
        """Crack length (mm) from a crack mask over a dataset photograph; needs a scale."""
        name = self._dataset_name(image)
        payload = self._call("measure_crack_on_image", name, str(mask), scale_mm_per_px,
                             [list(p) for p in reference_pixels or []] or None,
                             reference_length_mm, label)
        return self._measurement("crack_length", payload)

    def measure_map(self, kind: str, coordinates_lonlat: Sequence[LonLat], *,
                    raster: str | os.PathLike[str] = "", **options: Any) -> Measurement:
        """Measure on the project's georeferenced rasters (the map's Measure tool).

        ``kind``: distance, area, perimeter, slope, profile, volume, cut_fill or
        stockpile. ``raster`` defaults to the project's orthomosaic (or DSM).
        """
        payload = self._call("measure_drawn", kind, [list(map(float, p[:2])) for p in coordinates_lonlat],
                             str(raster or ""), dict(options))
        if kind in {"distance", "area", "perimeter"}:
            return self._measurement(kind, payload)
        value = payload.get("value_m3", payload.get("value", 0.0))
        return Measurement(str(payload.get("measurement_id", "")), kind,
                           float(value or 0.0), str(payload.get("unit", "")), "", payload)

    def measurements(self) -> list[dict[str, Any]]:
        """Every measurement recorded in this project, as stored."""
        return self._call("list_measurements")["measurements"]

    # -- reports ------------------------------------------------------------

    def report_readiness(self, report_type: str = "standard") -> dict[str, Any]:
        """The report checklist for ``report_type``: what the project holds, what is
        missing and the warnings the report would carry.
        """
        return self._call("report_readiness", "", report_type)

    def generate_report(self, *, title: str = "Inspection report", report_type: str = "standard",
                        formats: Sequence[str] = ("html",), author: str = "",
                        organization: str = "", client: str = "", summary: str = "",
                        recommendations: Sequence[str] = (),
                        sections: Sequence[str] | None = None,
                        plugin_sections: Sequence[str] = ()) -> Report:
        """Write a report into ``<project>/reports`` from what the project holds.

        Refused (with the checklist) when the project has nothing for this report type
        to be about. ``plugin_sections`` appends sections built by installed
        report-section plugins; one that fails is left out with a warning, never filled.
        """
        from app import report_builder
        from core.report_engine import sections_for

        with self.workspace._lock:
            api = self._api()
            try:
                config = api._report_config(title, report_type, author, organization, client,
                                            "", list(formats), summary, list(recommendations),
                                            list(sections or []))
            except ValueError as exc:
                raise Refused(str(exc)) from exc
            if plugin_sections:
                config.sections = sections_for(config.report_type, config.sections) + [
                    f"plugin:{name}" for name in plugin_sections]
            try:
                result = report_builder.generate(self.workspace._session, config)
            except ValueError as exc:
                raise Refused(str(exc)) from exc
        raw = result.to_dict()
        return Report(str(raw["id"]), str(raw["title"]), str(raw["report_type"]),
                      Path(raw["html_path"]), Path(raw["pdf_path"]) if raw.get("pdf_path") else None,
                      Path(raw["docx_path"]) if raw.get("docx_path") else None,
                      Path(raw["output_dir"]), int(raw.get("findings") or 0),
                      tuple(raw.get("warnings") or ()), raw)


def _crack_model_failure(pixels: Any) -> str:
    """Why the installed crack model could not produce a mask, in words."""
    from core.detection import _run_onnx_segmentation
    from core.models import get_model_spec, model_status

    status = model_status("crack_segmentation")
    spec = get_model_spec("crack_segmentation")
    if not status.get("exists") or spec is None:
        return "The crack_segmentation model is not installed."
    try:
        _run_onnx_segmentation(
            pixels, Path(str(status["path"])),
            threshold=float(getattr(spec, "score_threshold", 0.5) or 0.5),
            input_size=int(getattr(spec, "input_size", 512) or 512), raise_errors=True)
    except Exception as exc:  # noqa: BLE001 - the cause is reported, not hidden
        return f"The crack_segmentation model failed: {type(exc).__name__}: {exc}"
    return "The crack_segmentation model failed twice and then ran; the heuristic mask was kept."
