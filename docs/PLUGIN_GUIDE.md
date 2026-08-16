# Plugin guide

Extend OpenDroneKit without forking it. Plugins register a factory against a kind; the
toolkit calls the factory when it needs one.

## Kinds

`sdk/plugins.py` defines what can be plugged in:

| Kind | Extends |
|---|---|
| `drone` | A new airframe or autopilot behind the drone abstraction. |
| `camera` | A camera not in the camera database. |
| `payload` | A payload with its own capture commands. |
| `mission_type` | A new mission template. |
| `engine` | A reconstruction or processing engine. |
| `model` | An inference model outside the shipped registry. |
| `exporter` | An output format. |
| `report_template` | A report layout. |
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

`PluginSpec` refuses to construct if its `api_version` does not match the SDK's
`PLUGIN_API_VERSION`. This is intentional and it is not negotiable at runtime.

A plugin built against an older contract does not fail at registration — it fails later,
in the middle of a job, having already been handed data it does not understand. The
version check turns that into an error at load time, where it is one line to read
instead of a corrupted output to diagnose.

If you see this error, the plugin needs updating for the current contract. Do not
override the version to silence it.

## What a plugin cannot do

- **Bypass provenance.** A `model` plugin still has to supply a key and a digest for
  anything it detects; the asset and detection paths refuse instances without them.
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
