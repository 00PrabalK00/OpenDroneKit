# API guide

Two APIs, for two different things.

## Which one you want

| | `app.api.Api` | `services.api` (HTTP) |
|---|---|---|
| Shape | Python object, one method per capability | FastAPI application |
| Users | Single, local | Multi-user, organisations, roles |
| State | One `AppSession` | PostgreSQL / SQLite |
| Use for | Desktop shell, scripts, notebooks | Hub deployment, web client |

The desktop `Api` is the reference surface: every capability in the feature registry is
reachable through it. The HTTP service wraps a subset for multi-user deployment.

## Desktop API

```python
from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

session = AppSession(store=ProjectStore("projects.db"))
session.create_project("site", root_dir="./site")
api = Api(session)
```

Every method returns a dict with an `ok` boolean. **Failures carry `error`, not
`reason`** — a detail worth knowing before you write a handler against it:

```python
result = api.plan_mission({"altitude_m": 60.0})
if not result["ok"]:
    print(result["error"])
```

Successful results may also carry `warnings`. A warning does not make the result
invalid; it states what the result assumed. Ignoring warnings is how a flat-earth plan
gets flown over a slope.

Representative methods:

```python
api.set_aoi(polygon)
api.plan_mission(options)                 # options["mode"] selects the template
api.reconstruction_capabilities()         # what this machine can produce
api.check_spatial_reference(paths, gcp_count=0)
api.size_reconstruction_job(image_count=1200)
api.plan_job_chunks(image_count=5000)
api.find_ponding(dsm, vertical_accuracy_m=0.05)
api.compare_surveys(before, after)
api.build_asset_inventory(instances, model_key=..., model_sha256=..., crs="EPSG:32643")
api.asset_taxonomy("power")
api.demo_workflow()
```

## HTTP API

```bash
uvicorn services.api.main:app --reload
```

Interactive documentation is generated at `/docs`, and it is the authority — the routes
below are the shape, not a contract to code against blind.

| Area | Router |
|---|---|
| System | `/health`, `/health/live`, `/health/ready`, `/metrics` |
| Auth | `/auth` |
| Organisations | `/organizations` |
| Projects, datasets, processing | `projects`, `datasets`, `processing` |
| Inspection, annotations | `inspection`, `annotations` |
| Sharing, fleet, resources, events | `sharing`, `fleet`, `resources`, `events` |

### Health endpoints are not interchangeable

- `/health/live` — the process is up. Use for **liveness**.
- `/health/ready` — dependencies (database, object store) are reachable. Use for
  **readiness**.

Pointing a readiness probe at the liveness endpoint sends traffic to a pod that cannot
reach its database, and restarting that pod — which is what a failing liveness probe
does — will not fix a database that is down.

### What `/health` tells you about spatial storage

The health payload reports the database backend honestly:

```json
{"backend": "postgresql", "postgis": true,
 "geometry_storage": "geojson_text", "native_geometry_columns": false}
```

`postgis: true` means the extension is available. It does **not** mean your geometry is
natively indexed — every geometry column in the current schema is text holding GeoJSON,
on both backends, and spatial filtering happens in Python. Size your workloads
accordingly. This field used to report `native_geometry` whenever the extension
answered, which was a claim about the server rather than about the schema.

### Uploads

Dataset uploads are resumable, and upload paths are contained: a key that would escape
the storage root is rejected rather than normalised. Do not rely on the client to
sanitise paths — the server does not trust it either.

### Processing jobs and AI jobs

`POST /projects/{id}/jobs` accepts four kinds, each with a worker behind it:

| kind | needs | produces |
|---|---|---|
| `reconstruction` | `dataset_id` | rasters, mesh, point cloud under `runs/job_<id>/`, with `crs_epsg` |
| `detection` | `dataset_id`, `options.model_key` (or `"heuristic"`) | unreviewed defects attributed to the model file, severity `unassessed` until a reviewer grades them |
| `measurement` | `options.measurement`, `options.source_job_id` (a finished reconstruction), `polygon_xy` (volume) or `pixels` | a Measurement in the raster's projected CRS |
| `report` | nothing | a PDF and JSON snapshot of accepted findings (`include_unreviewed` to add the rest) |

Artifacts are storage-relative references; fetch one with
`GET /jobs/{id}/artifacts/{ref}`. A job finishes as `done`, `failed` or `cancelled`.

`POST /projects/{id}/ai-jobs` runs one registered model over uploaded images:
`input_ref` is `dataset:<id>` or `dataset:<id>/frame:<file>`. The request is refused
with the reason if the model is unknown, cannot answer the task, or its weights are
not installed; nothing is substituted. `ODK_AI_WORKER=external` leaves jobs for
`python -m services.worker.ai_worker`, and `ODK_JOB_BACKEND=celery` sends processing
jobs to a Celery worker. `GET /ai-models` lists each model a job may name, per task,
with the reason when it cannot run on this server. A job that fails says why (for
example, running out of memory) rather than reporting an empty result. Events from a
worker in another process reach WebSocket clients too, because the live-event buffer
is the shared database.

In the browser Hub, the "Data and jobs" panel does the same: create a dataset, upload
images (chunked and checksummed), submit any job kind or an AI job, then watch,
cancel and download the results.

Without `ODK_DATABASE_URL` and `ODK_STORAGE_PATH` the server keeps its database and
files in a per-user directory (`ODK_HUB_HOME` overrides it), not the directory it was
started from. Browser pages on `127.0.0.1`/`localhost` may call the API from any port;
set `ODK_CORS_LOOPBACK=0` to turn that off.

## Webhooks and realtime

Webhooks carry a secret; only its prefix is ever returned after creation. Each
subscribable event is sent by the action it names (a defect created or reviewed, an
upload completed, a job or AI job finished, a report generated, a share or mission
created, a flight logged, telemetry published), after that action commits. The same
events go to the organisation's WebSocket stream. Neither is required — the toolkit is
offline-first, and nothing calls out unless you configure it to.
