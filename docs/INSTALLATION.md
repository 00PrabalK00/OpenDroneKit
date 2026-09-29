# Installation

OpenDroneKit runs offline by default. Nothing here calls home, and every optional
dependency below is optional because the toolkit refuses the corresponding capability
rather than degrading it silently.

## Requirements

| Component | Version | Why |
|---|---|---|
| Python | 3.11+ | The codebase uses `X \| Y` type syntax and `tomllib`. |
| pip | any recent | — |
| Git | any | Only for cloning. |

Everything else is a Python package.

## Install

The reference environment is the `odk` conda environment in `environment.yml`: Python
3.11 plus everything in `requirements.txt` (which stays the single package list).

```bash
git clone <repository-url> OpenDroneKit
cd OpenDroneKit
conda env create -f environment.yml      # or, in any Python 3.11+ env:
pip install -r requirements.txt
```

Check that the environment is complete. This reads `requirements.txt` and names every
package that is missing or too old; the launcher runs it before every start:

```bash
python main.py --check-deps
```

Then verify with the test suite. It is the installation check worth trusting, because
it exercises the code rather than the import graph:

```bash
python -m pytest
```

A clean run reports several hundred passes and a handful of skips. **Skips are
expected** and are not failures: they mark capabilities whose optional dependency is
absent on this machine (see below).

## Run it

On Windows, double-click `OpenDroneKit.bat`. It starts in the toolkit folder whatever
folder it is run from, uses `ODK_PYTHON` (a python.exe), else `.venv\`, else the conda
environment named by `ODK_CONDA_ENV` (default `odk`), runs `--check-deps`, and offers
to install anything missing before opening the window. Startup errors are shown in a
message box and written to `final_toolkit_outputs/logs/startup-error.log`. Arguments
are passed through (`OpenDroneKit.bat --ui cockpit`).

Desktop shell from a terminal:

```bash
python main.py                 # classic single-screen UI (default)
python main.py --ui cockpit    # the fourteen-workspace cockpit (docs/UI_GUIDE.md)
ODK_UI=cockpit python main.py  # the same, from the environment (--ui wins if both are set)
```

An unknown interface name is refused: the window does not open, and the reason is shown
and written to `final_toolkit_outputs/logs/startup-error.log`.

The desktop shell also starts the Hub REST service in-process on 127.0.0.1 (port 8000
when free, otherwise any free port; `ODK_HUB=0` disables it), so the toolbar's **Hub**
link works without a separate server. Its **Sign in…** button logs in or creates a local
account.

Headless processing of an image set:

```bash
python run_pipeline.py --images <dataset_dir> --engine colmap --output final_toolkit_outputs
```

Web API on its own (see `docs/DEPLOYMENT.md` for anything beyond a laptop):

```bash
uvicorn services.api.main:app --reload
```

## Where data is kept

All local state lives under one data root, resolved to an absolute path at startup
(`core/paths.py`), so it does not matter which folder the app is started from:

| Path under the data root | What |
|---|---|
| `app_state/projects.db` | desktop projects, mission versions, audit log |
| `projects/<name>/` | default folder for a new project |
| `opendronekit.db` | fleet, share links, webhooks, Hub accounts (SQLite unless `ODK_DATABASE_URL` is set) |
| `workspace/` | report engine index and generated reports |
| `storage/tiles/` | offline basemap tiles (`ODK_STORAGE_PATH` moves `storage/`) |
| `logs/` | startup error log |

The data root is `ODK_DATA_DIR` when set, otherwise `final_toolkit_outputs/` inside the
toolkit folder -- where earlier versions already wrote when launched from Explorer, so
existing projects are found.

## Offline basemap

Online basemaps need a network. Before going to a site with no signal, frame it on the
map and press **Cache view** (or *View > Cache Visible Map for Offline Use*): the
satellite tiles for that view and three zoom levels deeper are downloaded into the
cache. The **Offline cache** basemap serves them from disk; if an online basemap cannot
be reached at all, the map switches to the cache by itself and says so. Areas that were
never cached stay visibly empty.

## Optional dependencies, and what you lose without each

This is the part worth reading. The toolkit is built so that a missing dependency
produces a refusal with a reason, never a quietly worse answer.

| Missing | What stops working | What you get instead |
|---|---|---|
| `open3d` | Poisson meshing (`pr.mesh`) | A warning in the run's output and no mesh file. The point cloud, DSM and orthomosaic are unaffected. |
| A CUDA COLMAP that runs on this GPU | Dense point clouds | Sparse cloud only, reported up front by `Api.reconstruction_capabilities()` with the reason. The sparse cloud is never inflated to look dense. See [Dense reconstruction on RTX 50-series](#dense-reconstruction-on-rtx-50-series). |
| `rasterio` | GeoTIFF reading, terrain following from DEMs | Those tests skip; planning falls back to flat earth **with a warning on every plan**. |
| `onnxruntime` / `opencv-python` | Model inference | Detection capabilities refuse rather than returning empty results. |
| `pymavlink` | Flight control, telemetry | Planning and processing are unaffected. |
| `torch`, `ultralytics` | Training only | Nothing at runtime; the shipped models are ONNX. |
| PostgreSQL + PostGIS | Multi-user deployment | SQLite, with geometry stored as GeoJSON text. See the note in `docs/DEPLOYMENT.md` — this is also true *with* PostGIS today. |

## GPU

No GPU is required to run OpenDroneKit. A GPU matters for two things:

- **Training** your own models (`docs/../training/`), which is optional.
- **Dense reconstruction**, which needs CUDA COLMAP specifically. Check what this
  machine can do before starting a long job:

```python
from app.api import Api
Api(session).reconstruction_capabilities()
```

The same answer is on screen under Environment Capabilities ("Dense MVS"), with the
reason when it is "no". It comes from running patch-match once on a two-image test
scene with the COLMAP that will be used, not from the binary existing.

### Dense reconstruction on RTX 50-series

**Symptom.** Dense MVS reports "no" with `the provided PTX was compiled with an
unsupported toolchain`, and reconstructions warn "Dense MVS skipped" and build their
products from the sparse cloud.

**Cause.** The COLMAP build's GPU kernels were compiled with a newer CUDA toolkit
than the installed NVIDIA driver can load. The reason on screen names both versions,
read from the binary and from `nvidia-smi`. On the development machine (RTX 5060
Laptop, driver 592.15, CUDA 13.1):

| COLMAP build | Built with | Dense MVS on that machine |
|---|---|---|
| Official 4.1.0, 4.1.1, 4.2.0 Windows CUDA archives | CUDA 13.2 | Fails: toolkit newer than the driver |
| Official 4.0.4 Windows CUDA archive | CUDA 12.9.1 | **Works** (measured, see below) |
| conda-forge `colmap=4.2.0` `cuda_129` build (win-64), in its own env | CUDA 12.9 | **Works**, no JIT delay (probe 2 s); needs `openimageio` added, see below |
| pip `pycolmap` (Windows) | no CUDA | Sparse only; `pycolmap-cuda12` wheels are Linux-only |

**Fix, any of:**

1. **Update the NVIDIA driver** to one whose `nvidia-smi` header shows
   `CUDA Version: 13.2` or higher. The installed COLMAP then works unchanged. (Not
   tested here: the development machine's driver was not changed.)
2. **Use COLMAP 4.0.4** and point OpenDroneKit at it. Nothing is installed into the
   Python environment:

   ```powershell
   # about 430 MB
   Invoke-WebRequest https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip -OutFile colmap-4.0.4-cuda.zip
   Expand-Archive colmap-4.0.4-cuda.zip "$env:LOCALAPPDATA\COLMAP-4.0.4"
   setx ODK_COLMAP "$env:LOCALAPPDATA\COLMAP-4.0.4\bin\colmap.exe"
   ```

   Start a new terminal (or restart the app) so `ODK_COLMAP` is seen, then check
   Environment Capabilities.
3. **Use conda-forge's COLMAP 4.2.0 CUDA 12.9 build in a separate conda env.** It has
   kernels for the RTX 50-series, so there is no first-run JIT delay. Do not install it
   into the OpenDroneKit env; only `ODK_COLMAP` points at it. Its package does not
   pull in OpenImageIO, which `colmap.exe` needs (without it the exe exits with
   0xC0000135 and prints nothing), so add it explicitly:

   ```powershell
   conda create -y -n colmap-cuda -c conda-forge --override-channels "colmap=4.2.0=cuda_129*" openimageio
   setx ODK_COLMAP "$env:USERPROFILE\miniconda3\envs\colmap-cuda\Library\bin\colmap.exe"
   ```

   Create the env on an NTFS drive: on a non-NTFS drive the install fails in
   `qt-main` with `[Errno 22] Invalid argument`.

**The first dense run is slow, once.** The 4.0.4 kernels predate the RTX 50-series,
so the driver JIT-compiles them on first use: the capability probe took 292 s the
first time and 10 s afterwards. The driver caches the compiled kernels in
`%APPDATA%\NVIDIA\ComputeCache`. Do not set `CUDA_CACHE_DISABLE`; if the probe is slow
every time, raise `CUDA_CACHE_MAXSIZE` (bytes, e.g. `setx CUDA_CACHE_MAXSIZE
4294967296`). The probe waits up to `ODK_DENSE_PROBE_TIMEOUT_S` seconds (default 900).
A probe that runs out of time is reported as "Not decided" and is not cached, so the
next check tries again.

**Measured with 4.0.4 on the development machine.** Environment Capabilities:
Dense MVS yes. Run Reconstruction on 20 Aukerman frames at 2400 px (COLMAP engine,
`standard` profile): 20 of 20 registered, 672,441 fused dense points, DSM 0.32 m/px,
ortho 0.0625 m/px, EPSG:32617. Without dense MVS, the same 20 frames at full
resolution gave 4,893 sparse points and a 3.76 m/px DSM. Dense MVS took about 25
minutes of that run.
`tests/test_dense_mvs_runs.py` repeats a 10-frame version when `ODK_DENSE_COLMAP` names
a working `colmap.exe`; it passes with both the 4.0.4 archive and the conda-forge
4.2.0 build.

## Windows notes

The project is developed and tested on Windows as well as Linux. Two things bite:

- **Page file size.** Training locally can fail with `WinError 1455` or
  `DataLoader worker exited unexpectedly`. Both are the same underlying limit: each
  worker process loads its own copy of torch's CUDA libraries. Set `num_workers: 0` in
  the training config, or raise the page file.
- **Console encoding.** Set `PYTHONIOENCODING=utf-8` before reading logs that contain
  non-ASCII characters, or `cp1252` will raise on output that is perfectly valid.

## Model weights

Weights are **not** in the repository — they are large binaries and `.onnx` is
gitignored. `models/model_registry.json` is tracked and records, for every model, the
path, labels, input size, published metrics and a **sha256 digest**.

The digest is verified at load. A model whose file does not match its recorded digest
is refused rather than used, because published metrics describe a specific file and not
a filename.

To see what is installed on this machine:

```bash
python -m training.register --list
```
