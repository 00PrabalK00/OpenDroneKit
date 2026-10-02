# Plugin guide

Extend OpenDroneKit without forking it. Plugins register a factory against a kind; the
toolkit calls the factory when it needs one.

The full reference for writing, packaging and installing a plugin (the
`opendronekit.plugins` entry point group, the manifest, the per-kind contracts and how
broken plugins are reported) is [plugins.md](plugins.md). This guide covers the rules.

## Kinds

`sdk/plugins.py` defines what can be plugged in (import it as `opendronekit.plugins`):

| Kind | Extends |
|---|---|
| `drone` | A new airframe or autopilot behind the drone abstraction. |
| `camera` | A camera not in the camera database. |
| `payload` | A payload with its own capture commands. |
| `mission_type` | A new mission template (contract: `plan_options`). |
| `engine` | A reconstruction or processing engine. |
| `model` | A detector model outside the shipped registry (contract: `model_key`, `model_sha256`, `detect`). |
| `exporter` | A mission output format (contract: `suffix`, `export`). |
| `report_template` | A report layout. |
| `report_section` | One extra section in a report (contract: `title`, `build`). |
| `map_provider` | A basemap or tile source. |

## Writing one

```python
from sdk.plugins import PluginKind, PluginSpec, PluginRegistry

def make_exporter(**config):
    class ShapefileExporter:
        def export(self, features, destination):
            ...
    return ShapefileExporter()

SPEC = PluginSpec(
    kind=PluginKind.EXPORTER,
    name="shapefile",
    factory=make_exporter,
    description="ESRI Shapefile export.",
)

registry = PluginRegistry()
registry.register(SPEC)
```

A factory may also be given as a dotted reference, `module.path:callable`, which is how
plugins are loaded from configuration without importing them eagerly.

## The API version check

The contract has a level, `PLUGIN_API_LEVEL` (currently `1.1`). Every plugin and every
manifest states the level it was written against in `api_version`. A plugin is accepted
when the major number equals the host's and its minor number is not newer: a `1.0`
plugin loads on a `1.1` host, a `1.2` or `2` plugin is refused. `PluginSpec` and
`PluginManifest` refuse to construct otherwise. This is intentional and it is not
negotiable at runtime.

A plugin built against a contract the host does not implement does not fail at
registration -- it fails later, in the middle of a job, having already been handed data
it does not understand. The version check turns that into an error at load time, where
it is one line to read instead of a corrupted output to diagnose.

If you see this error, the plugin needs updating for the current contract. Do not
override the version to silence it.

## A broken plugin is skipped, never fatal

Installed plugins are discovered through the `opendronekit.plugins` entry point group.
A plugin that fails to import, has an incompatible level, repeats a name already taken
or returns something that is not a manifest is listed under `failed` (with the stage and
the error) in Environment Capabilities and `list_plugins`, and the rest load. A plugin
that raises while it runs, or returns output outside its contract, fails the one call
that asked for it with `PluginError`. This is failure isolation, not a security sandbox:
plugin code runs with the permissions of the OpenDroneKit process.

## What a plugin cannot do

- **Bypass provenance.** A `model` plugin still has to supply a key and a digest for
  anything it detects; the host refuses its output without them, and refuses boxes that
  fall outside the image.
- **Claim a file it did not write.** An exporter's reported files are checked on disk.
- **Bypass a refusal.** If the toolkit refuses a measurement because the reconstruction
  has arbitrary scale, a plugin exporter does not get a version of that measurement
  that skips the check.
- **Silently replace a shipped capability.** Registering a name that already exists is
  an error, not an override.

These are the same rules the built-in code follows. A plugin that could route around
them would make every guarantee in this repository conditional on which plugins happen
to be loaded.

## Testing your plugin

Register it, build it, and exercise the real path rather than the factory:

```python
registry = PluginRegistry()
registry.register(SPEC)
exporter = registry.create(PluginKind.EXPORTER, "shapefile")
```

A factory that returns an object is not evidence the object works. Test the output the
same way the toolkit tests its own: assert on the artefact, not on the call succeeding.
