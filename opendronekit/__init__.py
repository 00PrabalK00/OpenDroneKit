"""OpenDroneKit Python SDK.

A stable, typed layer over the OpenDroneKit core: open or create a project, plan and
export a mission, run a reconstruction, detect, measure and write a report -- through
the same code the desktop application runs::

    import opendronekit as odk

    with odk.Workspace("odk-data") as ws:
        project = ws.create_project("Aukerman")
        plan = project.plan_mission(aoi, template="grid", altitude_m=70)
        plan.export(["qgc_plan", "kml"])
        project.import_dataset("odm_aukerman/images")
        recon = project.reconstruct(engine="auto")
        report = project.generate_report(title="Aukerman survey")

Versioning: ``__version__`` follows semantic versioning. Everything listed in
``__all__`` (and the names documented in ``docs/sdk/reference.md``) is the public API;
a breaking change to it bumps the major version. Modules and names starting with an
underscore, and everything outside the ``opendronekit`` package, are internal.

The plugin API is :mod:`opendronekit.plugins` (contract level ``PLUGIN_API_LEVEL``).
"""

from __future__ import annotations

from .errors import JobFailed, OdkError, PluginError, Refused
from .plugins import PLUGIN_API_LEVEL, load_plugins, plugin_status
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
from .workspace import MissionPlan, Project, Workspace

__version__ = "1.0.0"

__all__ = [
    "CrackDetection",
    "Dataset",
    "Detections",
    "JobFailed",
    "Measurement",
    "MissionPlan",
    "MissionSummary",
    "OdkError",
    "PLUGIN_API_LEVEL",
    "PluginError",
    "Project",
    "ProjectInfo",
    "Reconstruction",
    "Refused",
    "Report",
    "Workspace",
    "__version__",
    "load_plugins",
    "plugin_status",
]
