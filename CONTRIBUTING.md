# Contributing to OpenDroneKit

Thanks for wanting to help. OpenDroneKit produces numbers people act on: a volume that
gets invoiced, a defect area that decides a repair, a corridor someone believes is clear.
Most of the rules below exist because of that, and two of them are not negotiable:

1. **Never present synthesised output as measurement** ([the no-fabrication rule](#the-no-fabrication-rule)).
2. **A feature is done when its tests pass, not when someone says so** ([the feature registry](#the-feature-registry-done-is-computed-not-claimed)).

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems
go through [SECURITY.md](SECURITY.md), not public issues.

## Contents

- [Set up a development environment](#set-up-a-development-environment)
- [Run the application](#run-the-application)
- [Run the tests](#run-the-tests)
- [The feature registry: done is computed, not claimed](#the-feature-registry-done-is-computed-not-claimed)
- [The no-fabrication rule](#the-no-fabrication-rule)
- [Model weights and datasets](#model-weights-and-datasets)
- [Where things go](#where-things-go)
- [UI contributions](#ui-contributions)
- [Code style](#code-style)
- [Commits, branches and pull requests](#commits-branches-and-pull-requests)
- [Documentation](#documentation)

## Set up a development environment

The reference environment is the `odk` conda environment defined in `environment.yml`
at the repository root: Python 3.11 plus everything in `requirements.txt`, which stays
the single list of packages.

```bash
git clone https://github.com/00PrabalK00/OpenDroneKit.git
cd OpenDroneKit
conda env create -f environment.yml     # creates the "odk" env (Python 3.11)
conda activate odk
python main.py --check-deps             # names anything missing or too old
```

After `requirements.txt` changes, run `conda env update -f environment.yml`. Any other
Python 3.11+ environment works too (`pip install -r requirements.txt`); the code uses
`X | Y` type syntax and `tomllib`, so 3.10 and older will not.

Optional pieces, and what you lose without them, are listed in
[docs/INSTALLATION.md](docs/INSTALLATION.md#optional-dependencies-and-what-you-lose-without-each).
`torch` is **not** a runtime dependency (inference loads ONNX through `cv2.dnn`); you
only need it for the training tests and the training rig:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Node 20 is needed for the tests that execute the web UI in Node
(`tests/test_workspace_ui.py` and friends); CI installs it for the same reason.

## Run the application

```bash
python main.py                  # classic single-screen UI (the default)
python main.py --ui cockpit     # the fourteen-workspace cockpit (docs/UI_GUIDE.md)
python main.py --ui classic     # explicit default
python main.py --debug          # with the webview's developer tools
ODK_UI=cockpit python main.py   # same as --ui; --ui wins if both are set
```

An unknown `--ui` / `ODK_UI` value is refused, shown, and written to
`final_toolkit_outputs/logs/startup-error.log`. On Windows, `OpenDroneKit.bat` runs
`--check-deps` first and passes its arguments through.

The desktop shell serves its UI from a loopback-only server and starts the Hub REST
service in-process on `127.0.0.1` (`ODK_HUB=0` disables it). Headless entry points:

```bash
python run_pipeline.py --help
python run_pipeline.py --images <dataset_dir> --engine colmap --output final_toolkit_outputs
uvicorn services.api.main:app --reload
```

All local state lives under one data root, `final_toolkit_outputs/` unless
`ODK_DATA_DIR` says otherwise (see `core/paths.py`). The test suite points
`ODK_DATA_DIR` at a temporary folder for every test, so tests never touch your projects.

## Run the tests

`pytest.ini` sets `testpaths = tests`, `-q` and `--strict-markers`.

### Fast loop

Run the files that cover what you touched. A typical inner loop takes seconds:

```bash
python -m pytest tests/test_honesty.py tests/test_exporters.py tests/test_mission_constraints.py
python -m pytest tests/test_documentation_set.py          # after editing docs or the README
python -m pytest tests/test_exporters.py -k wpl           # one area, by keyword
python -m pytest tests/test_honesty.py -rs                # -rs prints why anything skipped
```

### Full suite

```bash
python -m pytest
python -m pytest --durations=15     # what CI runs, plus the slowest tests
```

The suite collects about 3,600 tests and takes **roughly 80 minutes** on the development
machine (reconstruction, dense-stereo and real-model tests dominate). Run it before you
open a pull request that touches `core/`, `mission/`, `app/` or `services/`; CI runs it on
every pull request either way.

**Skips are expected, and a skip is not a pass.** Tests skip when an optional dependency,
service, data set or model is absent. Each skip names what it needs; `-rs` lists them.

### What makes tests run instead of skip

| Variable | Effect |
|---|---|
| `ODK_MODEL_WEIGHTS_DIR` | A second folder to look for trained `.onnx` weights in (after `models/`). Use it from a worktree or second checkout. |
| `ODK_MODELS_DIR` | The models folder the survey-pack tests use (default `models`). |
| `ODK_SURVEY_IMAGERY` | Where the gitignored Aukerman survey frames live (default `training/data/aukerman_subset`). |
| `ODK_COLMAP` | Path to a `colmap` executable, for the dense-MVS tests (CUDA build required; see INSTALLATION.md). |
| `ODK_TEST_POSTGIS` | SQLAlchemy URL of a live PostGIS; enables `tests/test_postgis_spatial.py`. |
| `ODK_BROKER_URL` | Celery/Redis broker for the live broker tests (default `redis://localhost:6379/0`, so a local Redis is picked up automatically). |
| `ODK_TEST_S3_ENDPOINT` | Endpoint of a live S3-compatible store; enables the live tests in `tests/test_storage.py`. |
| `ODK_GUI_TESTS=1` | Opens real WebView2 windows (Windows desktop session only). |

### Services in Docker

These are the commands the maintainers use; each was run against the stated images.

PostGIS (the CI `spatial` job uses the same image):

```bash
docker run -d --name odk-postgis -e POSTGRES_PASSWORD=odk -e POSTGRES_DB=odk \
    -p 55432:5432 postgis/postgis:16-3.4
ODK_TEST_POSTGIS=postgresql+psycopg://postgres:odk@127.0.0.1:55432/odk \
    python -m pytest tests/test_postgis_spatial.py -rs
```

Redis (the Celery broker tests find it on the default URL):

```bash
docker run -d --name odk-redis -p 6379:6379 redis:7-alpine
python -m pytest tests/test_celery_broker.py tests/test_worker_tasks.py -rs
```

S3. Any S3-compatible server works. The project uses SeaweedFS, which is also what the
Docker Compose stack and the Helm chart deploy (MinIO no longer publishes public images;
see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)). It is a single container. The live tests
create a bucket of their own with the `ODK_S3_*` keys and remove it afterwards:

```bash
docker run -d --name odk-s3 -p 8333:8333 chrislusf/seaweedfs:latest server -s3
ODK_TEST_S3_ENDPOINT=http://127.0.0.1:8333 \
ODK_S3_ACCESS_KEY=x ODK_S3_SECRET_KEY=y python -m pytest tests/test_storage.py -rs
```

(With no identities configured, SeaweedFS accepts any key. Do not deploy it like that;
the compose stack writes an identity from `ODK_S3_ACCESS_KEY`/`ODK_S3_SECRET_KEY`.)
`ODK_TEST_DOCKER_COMPOSE=1` additionally lets `tests/test_compose.py` boot the compose
object store itself and round-trip the storage layer through it.

The Celery worker picks the solo pool on Windows (Celery's prefork pool fails every task
there) and prefork elsewhere; `ODK_WORKER_POOL` overrides it.

On Windows PowerShell, set variables with `$env:ODK_TEST_POSTGIS = "..."` before the
`python -m pytest` line instead of the `VAR=value command` prefix.

### SITL: flight tests against real ArduPilot

`tests/sitl/` flies the production planner, exporter and MAVLink client against
ArduPilot Copter-4.5.7. The tests are opt-in: a plain `pytest` collects and skips them,
and only the exact marker expression `-m sitl` starts a simulator.

The simplest way to run them on any OS is the container CI uses. Building it compiles
ArduPilot from source and takes 15-25 minutes the first time:

```bash
docker build -f infrastructure/docker/Dockerfile.sitl -t opendronekit-sitl:local .
docker run --rm -v "$PWD:/workspace" opendronekit-sitl:local \
    python3 -m pytest -m sitl tests/sitl -v -p no:cacheprovider
```

(The image runs as an unprivileged user; `-p no:cacheprovider` stops pytest trying to
write its cache into your checkout. In Git Bash on Windows, prefix the `docker run` with
`MSYS_NO_PATHCONV=1`.) To run SITL natively on Linux or WSL instead, see
[docs/SITL.md](docs/SITL.md); `python tools/sitl/launch.py --check` reports whether a
simulator can be found without starting one.

The suite is three tests (mission upload round trip, arm/takeoff/waypoints/RTL, and the
full lifecycle through the `Api`). Run with the container above under Docker Desktop on
the Windows development machine, all three pass (3 passed, 0 skipped, about 90 s). Add
`-v "<dir>:/out"` and `--junit-xml=/out/sitl-report.xml` to keep the report for
`tools/feature_status.py --extra-report`.

### What CI runs

`.github/workflows/ci.yml` has four jobs: **Test suite** (the whole suite, with CPU torch
so training tests run), **PostGIS spatial** and **SITL flight tests** (each fails if its
tests *skipped* rather than ran), and **Feature status**, which merges the spatial and
SITL junit reports and runs `tools/feature_status.py --strict`. All four must be green.

## The feature registry: done is computed, not claimed

`docs/features/registry.py` lists every specified capability (168 rows today). Each row
has acceptance `criteria`, a `claimed` status, and the `tests` that prove it.
`tools/feature_status.py` runs the suite and gives each row the **lower** of what it
claims and what the evidence supports:

```
not_started -> in_progress -> implemented -> verified
```

A row is `verified` only when every test it names actually passed in that run. A
skipped test is no evidence. A selector that matches nothing downgrades the row. Passing
tests never promote a row above what its author claimed.

When you add or change a capability:

1. Write tests that check the acceptance criteria against **real inputs**, not a mock of
   the thing under test. Assert on the artefact (the file written, the number measured),
   not on a call returning.
2. Add or update the row in `docs/features/registry.py`, naming those tests by pytest
   node id or prefix (`tests/test_geo.py::TestUmeyamaSimilarity`). If the work is
   honestly incomplete, leave it at `in_progress` and say what is missing in `notes`.
3. Check nothing was downgraded:

   ```bash
   python tools/feature_status.py --strict     # runs the full suite; non-zero on any downgrade
   python tools/feature_status.py --no-tests   # claims only, instant (every row reads as unproven)
   ```

4. **Never edit `docs/FEATURES.md` by hand.** It is generated by
   `python tools/feature_status.py --markdown`. That command refuses to run on a machine
   that lacks evidence for some rows (no SITL container, no live PostGIS), because writing
   the document there would demote rows CI proves. It needs the junit reports from those
   runs:

   ```bash
   python tools/feature_status.py --markdown --strict \
       --extra-report sitl-report.xml --extra-report spatial-report.xml
   ```

   Every CI run publishes a generated `features-md` artifact, plus `sitl-report` and
   `spatial-report`. If you cannot regenerate it, say so in the pull request and leave the
   file alone.

A pull request that lowers any row's computed status will fail the **Feature status**
job. If a downgrade is the honest outcome (the feature really is broken), lower the
`claimed` value in the registry in the same change and explain why.

## The no-fabrication rule

**Refusal over fabrication.** Every capability either produces a result it can account
for, or refuses and says why. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#the-rule)
explains the reasoning. In practice:

- **Never present synthesised, placeholder, default or heuristic output as a
  measurement.** No jittered points called a dense cloud, no constant used as a failure
  probability, no zero printed where nothing was measured, no assumed overlap used to lay
  out images.
- **Refuse honestly.** If the inputs cannot support an answer (no geotags, a degree-based
  CRS for an area, a missing model, a dependency that is not installed), return a refusal
  that names the reason and what would fix it. In the desktop `Api`, failures carry
  `error`, and assumptions go in `warnings`.
- **Label what is not measured.** A heuristic result reports `model_used: heuristic`,
  never a model name. Example data is stamped `synthetic: true` and shown under the
  EXAMPLE DATA banner. A simulated aircraft is labelled SIMULATED.
- **Say what an empty result means.** A detector that finds nothing has found nothing;
  it has not shown the asset is sound. Say so where a reader would otherwise assume it.
- **Nothing leaves the machine unasked.** No network call the user did not configure.

When you **remove** a fabrication, add a regression test to `tests/test_honesty.py` (or a
focused test file named for the behaviour, like `test_a_refused_report_says_why.py`) that
fails loudly if it comes back. Fabrications are cheap to reintroduce by accident; the test
is how the next contributor finds out.

Reviewers will reject a change that makes a number look better by making it less true,
however small.

## Model weights and datasets

Trained weights are **not** in the repository. `*.onnx`, `*.pt`, `*.ckpt`, everything
under `models/` except `README.md`, `model_registry.json` and
`manifests/model_provenance.json`, `training/data/` and `final_toolkit_outputs/` are all
gitignored. Do not force-add them.

- Install weights only through the training rig (`python -m training.register ...`).
  It refuses an export whose ONNX-vs-torch parity check failed and records the real
  sha256, source run, datasets and licences in `models/manifests/model_provenance.json`.
  The runtime hashes the installed file and reports a mismatch if it was replaced.
- Record the true source, licence and sha256 of every weight file you register.
- Do not redistribute weights whose source licence is absent or incompatible with
  distribution. ELPV is CC BY-NC-SA 4.0, so models trained on it must not be shipped
  commercially.
- Each dataset's licence is recorded in the catalogue (`python -m
  training.datasets.download --list`) and in `training/data/manifest.json` on download.
  Keep that record when you add a dataset.
- Quote model quality from a held-out evaluation, the number at the shipped threshold,
  with per-class figures where a mean would hide a dead class. A rejected model's
  numbers go in the registry with the reason, not in the bin.
- Credentials (`KAGGLE_API_TOKEN`, `ROBOFLOW_API_KEY`, `ODK_SECRET_KEY`, S3 keys) belong
  in your environment or an uncommitted `.env`, never in the repository.

## Where things go

`app/` uses `mission/` and `core/`; `core/` never imports `app/`; `mission/` imports
nothing else from the toolkit, so the mission engine stays usable as a library.

| To add | Start here | Notes |
|---|---|---|
| A mission template | `mission/planner.py`: `TEMPLATE_ALIASES`, `TEMPLATE_LABELS`, `TEMPLATE_DEFAULT_GIMBAL_DEG`, and the template's branch in `MissionPlanner.generate` | The application's template list is derived from `TEMPLATE_ALIASES` (`tests/test_template_inventory_is_derived.py`). Unknown primitive kinds silently normalise to `grid`; non-template kinds go in `_NON_TEMPLATE_PRIMITIVE_KINDS`. Tests: `tests/test_mission_types.py`, `tests/test_special_mission_types.py`. |
| A mission export format | `mission/exporters.py`: write `export_<format>(path, mission)` and add it to `EXPORTERS` | Read the exported file back in the test and check the poses, speed, fences and limits survived (`tests/test_mission_exports_carry_the_plan.py`, `tests/test_export_parity.py`). |
| A defect detector | `core/detection.py`, a key in `models/model_registry.json`, weights via `training/` | Report `model_used` truthfully, carry the model identity, and refuse rather than fall back where a heuristic cannot answer the question (see `grade_corrosion_severity`). |
| A plugin | `sdk/plugins.py` and [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) | Plugins cannot bypass provenance or refusals, and cannot replace a shipped name. A new `PluginKind` must be documented (`tests/test_documentation_set.py` checks). |
| A desktop capability | `app/api.py` (one method per capability, returning a dict with `ok`) | Then reach it from the UI; a capability nobody can reach is not done. |
| An HTTP endpoint | `services/api/routers/` | See [docs/API_GUIDE.md](docs/API_GUIDE.md). |

## UI contributions

The desktop UI is HTML/JS under `app/web/`, rendered by the OS webview (Edge WebView2 on
Windows) with MapLibre GL for maps. There is no bundler and no framework build step.

- `index.html` + `js/app.js` is the classic shell; `workspace.html` + `js/workspace/` is
  the cockpit; `hub.html` + `js/hub*.js` is the browser Hub.
- Colour, spacing and type come from `app/web/css/tokens.css`. Use its tokens; do not add
  new hex values in component CSS. Colour carries meaning (blue interaction, green
  healthy, amber warning, red error/safety-critical, purple thermal/semantic only).
- New cockpit workspaces are an object in `WORKSPACES` in `js/workspace/workspaces.js`
  ([docs/UI_GUIDE.md](docs/UI_GUIDE.md#extending-it)); `tests/test_workspace_ui.py`
  mounts every workspace and renders every panel in Node.
- A panel with no data source shows its title and a one-line description, never sample
  numbers. A panel whose read failed says so, rather than showing an empty state that
  looks like a real answer.
- Brand assets live in `app/web/assets/brand/`; rasters are generated by
  `tools/brand_assets.py`, so edit the SVG sources, not the PNGs.
- Include before/after screenshots in the pull request for any visible change.

## Code style

There is no enforced formatter or linter configuration; match the code around you.

- `from __future__ import annotations` at the top of every module, type hints on
  public functions, 4-space indents, lines mostly under ~100 characters.
- Plain descriptive names (`extra_weights_root`, `secret_is_deployment_grade`). Test
  names are sentences about behaviour: `test_an_empty_model_result_is_not_replaced_by_the_heuristic`.
- Comments and docstrings explain **why**, often the failure that made the code this
  way. That is the house style: if you fixed a subtle bug, leave the reason next to the
  fix so nobody "simplifies" it back.
- Error messages are for the operator: say what was refused, why, and what would fix it.
- `except Exception` needs a reason in a `# noqa: BLE001 - ...` comment, and must never
  swallow the failure that matters.

## Commits, branches and pull requests

**Commit messages** have a subject that says what now behaves differently, and a body
that explains what was wrong, why, and how it was verified. A `type(scope):` prefix is
common (`fix(missions):`, `fix(reports):`, `docs:`, `test:`, `feat(cockpit):`) but
optional. Examples from the history:

```
fix(launch): --ui and ODK_UI agree, and every doc says what opens

`ODK_UI=Cockpit` opened the cockpit while `--ui Cockpit` was refused, and a
blank ODK_UI was refused as a UI named ''. The flag is now read
case-insensitively like the variable, and a blank variable counts as unset.
...
```

```
fix: classic map says what it shows with no network, and enlarges the deepest cached tiles
```

Keep commits logical: one behaviour change with its tests and docs together.

**Branches and pull requests.**

1. Fork the repository (or branch, if you have write access) from `main`. Name the branch
   for its area and intent, e.g. `fix/missions-standoff` or `docs/sitl-windows`.
2. Keep the pull request to one concern. Large mechanical changes go separately from
   behaviour changes.
3. Fill in the pull request template: tests, registry, honesty, docs, screenshots.
4. CI must be green. A red **Feature status** job means a capability lost its evidence;
   it is not a flaky check to be re-run until it passes.
5. The maintainer reviews and merges. Expect questions about any number the change
   produces: where it comes from and what it does not mean.

Issues and pull requests are welcome for bugs, refusals that are wrong or unhelpful,
missing documentation, and new capabilities. For larger work, open an issue first so the
approach can be agreed before you write it.

## Documentation

- If behaviour changes, update the guide that describes it in the same pull request.
  The guides in `docs/` are part of the product: [Installation](docs/INSTALLATION.md),
  [Architecture](docs/ARCHITECTURE.md), [User](docs/USER_GUIDE.md),
  [Pilot](docs/PILOT_GUIDE.md), [Plugin](docs/PLUGIN_GUIDE.md), [API](docs/API_GUIDE.md),
  [Deployment](docs/DEPLOYMENT.md), [UI](docs/UI_GUIDE.md), [SITL](docs/SITL.md).
- `tests/test_documentation_set.py` and several README tests check specific claims
  against the code (failure keys, endpoints, installed models, plugin kinds). Run them
  after editing docs.
- State limits plainly. "Not implemented" and "verified only in CI" are useful sentences;
  a confident sentence that is no longer true is not.
- Add user-visible changes to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md).

## Licence

OpenDroneKit is MIT-licensed ([LICENSE](LICENSE)). By contributing you agree that your
contribution is released under the same licence. Do not contribute code, weights or data
you do not have the right to release under it.
