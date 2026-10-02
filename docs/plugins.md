# Plugins

OpenDroneKit loads extensions from ordinary Python packages. A plugin package declares an
entry point in the `opendronekit.plugins` group; when OpenDroneKit starts it discovers
every installed one, checks that each was written for a compatible contract, and lists
what loaded (and what did not, with the reason) in its Environment Capabilities data.

This page is the reference. [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) covers the rules every
plugin has to live by; [tutorial 3](tutorials/03-custom-plugin.md) builds one step by step;
`examples/plugins/` holds two complete, tested packages. The generated API reference for
`opendronekit.plugins` is in [sdk/reference.md](sdk/reference.md#opendronekitplugins).

## Contract level

The plugin API has its own version, `opendronekit.plugins.PLUGIN_API_LEVEL`, currently
**1.1**. A plugin and its manifest state the level they were written against
(`api_version`). The host accepts a plugin when the **major** numbers are equal and the
plugin's **minor** number is not newer than the host's:

| Plugin `api_version` | 1.1 host |
|---|---|
| `"1"`, `"1.0"`, `"1.1"` | loads |
| `"1.2"` | refused: newer than the host, update OpenDroneKit |
| `"2"`, `"2.0"` | refused: major versions differ |

A refusal is recorded as a failure with that reason; it never stops OpenDroneKit.
`opendronekit.plugins.api_compatibility(level)` returns `(ok, reason)` if you want to check
in your own tests.

What changed in 1.1: manifests (`PluginManifest`), a `version` on every plugin, the
`report_section` kind, contract checks on plugin output, and discovery that never raises.
Everything a 1.0 plugin relied on is unchanged.

## Packaging

```toml file=pyproject.toml
[project]
name = "odk-my-plugins"
version = "0.1.0"
dependencies = []          # OpenDroneKit is the host; do not depend on it

[project.entry-points."opendronekit.plugins"]
my_plugins = "odk_my_plugins:manifest"
```

The entry point resolves to a `PluginManifest`. It may also resolve to a single
`PluginSpec`, a list of either, or a zero-argument callable that returns one of those.

```python
from opendronekit.plugins import PluginKind, PluginManifest, PluginSpec

manifest = PluginManifest(
    name="odk-my-plugins",        # the distribution, shown in listings
    version="0.1.0",              # the package's release; fills plugins that state none
    api_version="1.1",            # the contract level, for every plugin listed
    description="What the package adds.",
    license="MIT",
    homepage="https://example.org/odk-my-plugins",
    plugins=(
        PluginSpec(PluginKind.EXPORTER, "my_format", MyExporter, api_version="1.1",
                   description="One line for the capability listing."),
    ),
)
```

`PluginSpec(kind, name, factory, api_version="1", description="", metadata={}, version="")`:

* `name` is unique within its kind (case-insensitive). A second package claiming the same
  name is refused; it does not replace the first.
* `factory` is any callable that returns the plugin object: usually the class itself.
  The host calls it with no arguments each time the plugin is needed.

Install into the environment that runs OpenDroneKit (`pip install -e <folder>` while you
develop). Discovery runs once per process; `opendronekit.load_plugins(refresh=True)` or the
API call `list_plugins(refresh=True)` rescans.

## The four contracts

The host calls into four kinds. Each is a small duck-typed object; the `Protocol` classes
in `opendronekit.plugins` (`MissionTemplatePlugin`, `DetectorPlugin`, `ExporterPlugin`,
`ReportSectionPlugin`) document them for type checkers. A built object missing a required
member fails with `PluginError` at stage `contract`.

### Mission template (`mission_type`)

```python
class CorridorTemplate:
    def plan_options(self, aoi_lonlat: list[list[float]], **params) -> dict:
        ...
        return {"template": "grid", "altitude_m": 60, "front_overlap_pct": 80,
                "aoi_lonlat": corridor_polygon}
```

`plan_options` turns the user's area and parameters into options for the **built-in**
planner: it must name a built-in `template` and may replace `aoi_lonlat` (three or more
`[lon, lat]` points, in range). The planner then does the geometry, the estimates and the
safety checks, so a template plugin cannot produce a mission the planner would refuse.
Use it with `project.plan_mission(aoi, template="plugin:<name>", **params)`; the plan
records which plugin and version produced it. Example:
`examples/plugins/odk-strip-template`.

### Detector model (`model`)

```python
class MyDetector:
    model_key = "my_detector_v3"
    model_sha256 = "<64 hex chars: the sha256 of the weights file that runs>"

    def detect(self, image_bgr) -> list[dict]:
        return [{"label": "crack", "score": 0.91, "bbox_xyxy": [x1, y1, x2, y2]}]
```

A detector must name the weights that produced its output. Without a `model_key` and a
well-formed `model_sha256` its output is refused (stage `provenance`): an untraceable
detection cannot go into a report. Each detection needs a non-empty `label`, a `score` in
0..1 and a box inside the image; anything else fails the call (stage `output`). Extra keys
are kept. Run it with `project.detect(image, detector="<name>")`; the result carries
`model_used="plugin:<name>"`, the key and the digest, and the run is written to the audit
log.

### Exporter (`exporter`)

```python
class MyExporter:
    suffix = ".myformat"

    def export(self, plan: dict, path) -> str | list[str]:
        ...
        return str(path)
```

`export` receives a deep copy of the planned mission (`plan["waypoints"]` is
`[[lon, lat, alt], ...]`, plus the planner's other fields) and the target path
`<directory>/<stem><suffix>`. It returns the file or files it wrote. Each one is checked
on disk: a path that is missing or empty fails the export, so a format is never reported
written when it was not. Use it with `plan.export(["<name>"])`, alongside the built-in
formats. Example: `examples/plugins/odk-csv-exporter`.

### Report section (`report_section`)

```python
class MySection:
    title = "Site access"

    def build(self, context: dict) -> dict:
        return {"paragraphs": ["..."],
                "table": {"header": ["Gate", "Code"], "rows": [["North", "4411"]]}}
```

`build` receives a copy of the report's context (project, datasets, missions, findings,
measurements, reconstructions; private keys removed) and returns paragraphs, a table, or
both. Ask for it with `project.generate_report(plugin_sections=["<name>"])`. The section
ends with a line naming the plugin and its version. If the plugin is missing, raises or
returns something unusable, the report is still written: the section says it was left
out and why, and the report carries a warning. Nothing is filled in on the plugin's
behalf.

### The other kinds

`drone`, `camera`, `payload`, `engine`, `report_template` and `map_provider` plugins are
discovered, version-checked and listed the same way; the host's integration for them is
calling the factory (`opendronekit.plugins.host.build(kind, name)`).

## A broken plugin is reported and skipped

Discovery never raises because of a plugin. Each problem becomes a `PluginFailure`
(`source`, `stage`, `error`, `distribution`, and the last traceback frames in `detail`):

| Stage | Cause |
|---|---|
| `load` | the entry point's module fails to import, or the attribute is missing |
| `manifest` | the entry point gave something that is not a manifest or spec, or the manifest is invalid (incompatible `api_version`, no version, no plugins) |
| `register` | the name is already taken in that kind |
| `create` / `contract` | at use: the factory raised, or the object lacks a contract member |
| `call` / `output` / `provenance` | at use: the plugin raised, or returned output outside its contract |

Load-time failures are listed and the plugin is skipped; the rest load. Use-time failures
raise `opendronekit.PluginError` to the caller that asked for that plugin and are added to
the failure list; nothing else is affected.

This is failure isolation, not a security sandbox. Plugin code runs inside the
OpenDroneKit process with its permissions. Install plugins you trust.

## Seeing what is installed

* The desktop API: `capabilities()["plugins"]` (the Environment Capabilities data) and
  `list_plugins(refresh=False)` give the API level, each loaded plugin (kind, name,
  version, package) and each failure with its reason.
* The desktop application's **Plugins** panel (Developer workspace) reads `list_plugins`.
* The SDK:

```python
import opendronekit as odk

status = odk.plugin_status()
print(status["api_version"], [p["name"] for p in status["loaded"]], status["failed"])
```

`status["kinds"]` lists, for each kind, the members its contract requires.

## Testing a plugin

Test the artefact, not the call. `tests/test_example_plugins.py` is a template: it installs
the example packages with `pip install -e` into a scratch prefix, discovers them through
`importlib.metadata` exactly as an installed plugin is found, plans a real mission through
the SDK and checks the exported rows against the plan's waypoints. For unit tests without
installing, hand a private host the entry points directly:

```python
from importlib.metadata import EntryPoint

from opendronekit.plugins import PluginHost, PluginRegistry

host = PluginHost(PluginRegistry())
report = host.load(entry_points=[EntryPoint("mine", "odk_my_plugins:manifest",
                                             "opendronekit.plugins")])
```
