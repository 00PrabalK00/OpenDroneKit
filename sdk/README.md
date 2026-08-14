# OpenDroneKit developer SDK

The Python package at `sdk` is a stable, thin layer over the same mission planner,
drone protocols and REST resources used by OpenDroneKit itself.

```python
from sdk import MissionPlanRequest, OpenDroneKitClient, plan_mission

plan = plan_mission(MissionPlanRequest(
    mission_name="Quarry survey",
    polygon_lonlat=[[77.59, 12.97], [77.60, 12.97], [77.60, 12.98]],
    altitude_m=60,
))

client = OpenDroneKitClient("http://127.0.0.1:8000", token="shown-once-token")
job = client.submit_job(7, kind="reconstruction", dataset_id=12)
finished = client.wait_for_job(job.id)
```

## Plugin points

`PluginKind` defines the nine supported points: drone, camera, payload, mission type,
processing/reconstruction engine, model, exporter, report template and map provider.
Plugins target API version `1` and may be registered directly, loaded from an explicit
JSON manifest, or discovered on request through the `opendronekit.plugins` Python entry
point group. Importing the SDK never scans plugins or performs network I/O.

An explicit manifest looks like:

```json
{
  "api_version": "1",
  "plugins": [{
    "kind": "map_provider",
    "name": "district-wmts",
    "factory": "my_odk_plugins.maps:create_provider",
    "description": "District-hosted WMTS tiles"
  }]
}
```

Plugin code runs with the permissions of the OpenDroneKit process. Install only code
you trust; the registry is an extension contract, not a security sandbox.
