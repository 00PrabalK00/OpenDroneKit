# Tutorial 3: write, install and use a plugin

This tutorial writes a small OpenDroneKit plugin from scratch: a mission **exporter** that
writes the planned flight path as GeoJSON, which any GIS opens. You will:

1. write the package (two files);
2. install it with `pip install -e`;
3. see it listed in Environment Capabilities;
4. plan a mission over the Aukerman survey area and export it through the plugin;
5. see what OpenDroneKit does with a plugin that is broken.

The full contract for every plugin kind is in [docs/plugins.md](../plugins.md); two more
complete examples live in `examples/plugins/`. `tests/test_tutorials.py` writes the files
below, installs the package in editable mode and runs the `python` blocks, so the code is
known to work.

## What you need

* OpenDroneKit, run from the repository root.
* The 8 Aukerman photographs from [tutorial 1](01-aukerman-end-to-end.md) (step 0 there),
  used here only for the survey area.

## 1. The package

A plugin is an ordinary Python distribution with an entry point in the
`opendronekit.plugins` group. The entry point names a `PluginManifest`: the package's
name, version, the plugin API level it was written against, and the plugins it provides.

```toml file=odk-geojson-path/pyproject.toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "odk-geojson-path"
version = "0.1.0"
description = "OpenDroneKit plugin: export the planned flight path as GeoJSON."
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = []

[project.entry-points."opendronekit.plugins"]
geojson_path = "odk_geojson_path:manifest"

[tool.setuptools]
packages = ["odk_geojson_path"]
```

The exporter contract is two members: `suffix` (the file extension) and
`export(plan, path)`, which writes the file and returns what it wrote. OpenDroneKit checks
the claim: a path that is missing or empty fails the export instead of being reported.

```python file=odk-geojson-path/odk_geojson_path/__init__.py
"""Export an OpenDroneKit mission's flight path as a GeoJSON LineString."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opendronekit.plugins import PluginKind, PluginManifest, PluginSpec


class GeoJsonPathExporter:
    suffix = ".path.geojson"

    def export(self, plan: dict[str, Any], path: Path) -> str:
        waypoints = plan.get("waypoints") or []
        if len(waypoints) < 2:
            raise ValueError("A flight path needs at least two waypoints.")
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[float(w[0]), float(w[1])] for w in waypoints]},
            "properties": {
                "waypoints": len(waypoints),
                "altitude_m": plan.get("altitude_m"),
                "altitude_frame": plan.get("altitude_frame", "relative_to_home"),
                "template": plan.get("template") or plan.get("mission_type"),
            },
        }
        Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}),
                              encoding="utf-8")
        return str(path)


manifest = PluginManifest(
    name="odk-geojson-path",
    version="0.1.0",
    api_version="1.1",
    description="Flight path as GeoJSON.",
    license="MIT",
    plugins=(PluginSpec(PluginKind.EXPORTER, "geojson_path", GeoJsonPathExporter,
                        api_version="1.1", description="Flight path as a GeoJSON LineString."),),
)
```

`api_version="1.1"` is the contract level the plugin was written against. A host accepts a
plugin with the same major number and a minor number that is not newer than its own, so
this plugin keeps loading on later 1.x hosts and is refused (with that reason) by a 2.x
host rather than failing halfway through a job.

## 2. Install it

Into the environment that runs OpenDroneKit:

```bash
pip install -e odk-geojson-path
```

`-e` (editable) means edits to `odk_geojson_path/__init__.py` take effect the next time
OpenDroneKit starts, without reinstalling.

## 3. See it listed

```python
import json
import os
from pathlib import Path

import opendronekit as odk

work = Path(os.environ.get("ODK_TUTORIAL_DIR", "odk-tutorial"))
samples = Path(os.environ.get("ODK_SAMPLE_DATA", "sample_data"))

odk.load_plugins(refresh=True)
ws = odk.Workspace(work / "plugin-data")
plugins = ws.capabilities()["plugins"]
print("plugin API level:", plugins["api_version"])
for p in plugins["loaded"]:
    print(f"loaded: {p['kind']} {p['name']} {p['version']} (package {p['package']})")
for f in plugins["failed"]:
    print(f"failed: {f}")
```

This is the Environment Capabilities data of the desktop API (`capabilities()["plugins"]`),
which the application's Plugins panel also reads through `list_plugins`: a user can see
which plugins are installed, which were skipped, and why.

## 4. Use it

Plan a grid over the Aukerman survey area and export it with the plugin, next to a built-in
format.

```python
project = ws.create_project("Aukerman (plugin export)")
meta = project.import_dataset(samples / "odm_aukerman" / "images").raw["metadata"]
west, south, east, north = meta["bounds_lonlat"]
plan = project.plan_mission([[west, south], [east, south], [east, north], [west, north]],
                            template="grid", altitude_m=60)

files = plan.export(["geojson_path", "kml"], directory=work / "plugin-mission", stem="aukerman")
path_file = files["geojson_path"][0]
feature = json.loads(path_file.read_text(encoding="utf-8"))["features"][0]
print("wrote", path_file.name, "with", len(feature["geometry"]["coordinates"]), "vertices;",
      "the plan has", plan.summary.waypoints, "waypoints")
```

The exporter receives a copy of the plan, so it cannot change the mission other exporters
see.

## 5. When a plugin is broken

A plugin that fails to import, declares an incompatible API level, or reuses a name is
listed under `failed` and skipped; the rest load and OpenDroneKit keeps running. Here a
private host is handed one good and one broken entry point, so the demonstration does not
touch your installation:

```python
from importlib.metadata import EntryPoint

from opendronekit.plugins import PluginError, PluginHost, PluginRegistry

demo = PluginHost(PluginRegistry())
report = demo.load(entry_points=[
    EntryPoint("geojson_path", "odk_geojson_path:manifest", "opendronekit.plugins"),
    EntryPoint("broken", "odk_no_such_module:manifest", "opendronekit.plugins"),
])
print("loaded:", [spec.name for spec in report.loaded])
for failure in report.failures:
    print("skipped:", failure.source, "at", failure.stage, "-", failure.error)

try:
    plan.export(["no_such_format"], directory=work / "plugin-mission")
except PluginError as exc:
    print("refused:", exc)
ws.close()
```

A plugin that raises while it runs, or returns something outside its contract (a file it
did not write, a detection box outside the image, a detector that does not name the
weights it ran), fails only the call that asked for it, with `PluginError`. This is failure
isolation, not a security sandbox: plugin code runs with the permissions of the
OpenDroneKit process, so install plugins you trust.

## Next

* [docs/plugins.md](../plugins.md): the contracts for mission templates, detector models,
  exporters and report sections.
* `examples/plugins/odk-strip-template`: a mission template that turns a centreline into a
  corridor survey.
* `examples/plugins/odk-csv-exporter`: an exporter plus a report section that appears in
  generated reports.
