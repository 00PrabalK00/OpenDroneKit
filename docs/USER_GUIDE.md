# User guide

Plan a survey, fly it, process it, measure it. This guide is written around what the
toolkit will and will not tell you, because that is where surveys go wrong.

## 1. Create a project and set an area

```python
from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

session = AppSession(store=ProjectStore("projects.db"))
session.create_project("bhopal-warehouse", root_dir="./site")
api = Api(session)

api.set_aoi([[77.40, 23.25], [77.41, 23.25], [77.41, 23.26], [77.40, 23.26]])
```

## 2. Plan

```python
result = api.plan_mission({"altitude_m": 60.0, "speed_m_s": 8.0})
```

**Read `result["warnings"]` before you read anything else.** The plan is not wrong when
it warns; it is telling you what it assumed. The one that matters most:

> No terrain model loaded: altitudes are relative to a flat plane at the launch
> elevation. Over sloping ground the true height above surface will differ.

A flat-earth plan and a terrain-following plan contain the same waypoints with the same
altitude numbers. The only difference is what those numbers mean. Over a slope, that
difference is the aircraft.

### Choosing a mission type

| You want | Use | Notes |
|---|---|---|
| An orthomosaic / map | `grid` | Nadir. Fastest. |
| A better map, fewer holes | `double_grid` | Cross-hatch. ~2× the flying. |
| A **3D model of buildings** | `mapping_3d` | Cross-hatch **plus oblique rings** at -45° and -60°. |
| A facade | `facade_mapping` | Stand-off from the wall. |
| A structure from all sides | `closed_loop`, `orbit` | Continuous, no corner stops. |
| A corridor, road, rail | `linear_inspection` | |
| Panels | `solar_inspection` | |
| Tower, pylon, turbine | `tower_mapping`, `pylon_inspection`, `wind_turbine` | |

Use `mapping_3d` rather than `double_grid` whenever the deliverable includes walls. A
nadir survey sees vertical surfaces at a grazing angle or not at all, and the model it
produces still *has* walls — stretched texture over guessed geometry. It looks finished,
which is why nobody queries it. The oblique bands cost about 13% more flight time on a
400 m site.

### Terrain

```python
api._session.terrain_source_path = "dem.tif"
api.plan_mission({"altitude_m": 60.0, "terrain_follow": True})
```

If the DEM cannot be read, or does not cover the area, you get a plan **and a warning**
saying so. It never silently reverts.

## 3. Fly

Missions export to the usual formats, or upload directly over MAVLink. The upload
prepends the home item MAVLink reserves at sequence 0, so what you planned is what the
aircraft stores. Download returns your plan, not the vehicle's bookkeeping.

## 4. Process

```bash
python run_pipeline.py --images ./site/images --engine colmap --output ./site/out
```

Before a long job, ask what this machine can produce:

```python
api.reconstruction_capabilities()   # dense cloud needs CUDA COLMAP
api.size_reconstruction_job(image_count=1200)
api.check_spatial_reference(image_paths, gcp_count=0)
```

`check_spatial_reference` is the one people skip and regret. Structure-from-motion
recovers geometry only up to a similarity transform. Without geotags or ground control,
your model has **arbitrary position, rotation and scale**. It renders. It meshes. Every
distance in it is wrong by an unknown factor. The toolkit tells you which of the three
modes you are in — `georeferenced`, `control_referenced`, or `arbitrary` — and refuses
measurements in the last.

## 5. Measure

```python
api.measure_area(polygon)
api.measure_volume(polygon, base="lowest_point")
api.find_ponding(dsm_path, vertical_accuracy_m=0.05)
api.compare_surveys(before, after)
```

Two rules the measurement code enforces on your behalf:

- **Vertical accuracy is required, not optional.** Ponding and deformation refuse
  without it and report nothing shallower than twice it. A 3 cm depression measured by
  a survey with 5 cm accuracy is not a puddle, it is noise.
- **Volume needs a stated base.** "The volume of that pile" is not a question until
  someone says what it sits on.

## 6. Detect

```python
api.run_detection(image_path, model="crack_presence_classifier")
```

Every result carries the model key, its sha256 and a confidence. Read the model's
registry description before acting on its output — each one names its weak class and
its scope. Examples that matter in practice:

- `crack_presence_classifier` (balanced accuracy 0.958) says *whether* a tile has a
  crack. `crack_segmentation` says *where* — and is the weaker model.
- `solar_thermal_anomaly_classifier` is worst at **Soiling** (0.367 recall), which is
  often the class you care about most.
- `rail_obstacle_detector` misses roughly one obstacle in four. An empty result means
  the model found nothing. **It does not mean the corridor is clear.**

## Demo mode

```python
api.demo_workflow()
```

Explores the whole workflow with no hardware. Every artefact is stamped
`synthetic: true` recursively, sites are at Null Island and timestamps at the epoch, so
a demo result cannot be mistaken for a survey even if a single finding is lifted out of
it. Measurement calls refuse demo inputs outright.
