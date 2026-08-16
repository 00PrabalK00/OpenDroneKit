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

```bash
git clone <repository-url> OpenDroneKit
cd OpenDroneKit
pip install -r requirements.txt
```

Verify with the test suite. It is the only installation check worth trusting, because
it exercises the code rather than the import graph:

```bash
python -m pytest
```

A clean run reports several hundred passes and a handful of skips. **Skips are
expected** and are not failures: they mark capabilities whose optional dependency is
absent on this machine (see below).

## Run it

Desktop shell:

```bash
python main.py
```

Headless processing of an image set:

```bash
python run_pipeline.py --images <dataset_dir> --engine colmap --output final_toolkit_outputs
```

Web API (see `docs/DEPLOYMENT.md` for anything beyond a laptop):

```bash
uvicorn services.api.main:app --reload
```

## Optional dependencies, and what you lose without each

This is the part worth reading. The toolkit is built so that a missing dependency
produces a refusal with a reason, never a quietly worse answer.

| Missing | What stops working | What you get instead |
|---|---|---|
| `open3d` | Poisson meshing (`pr.mesh`) | A warning in the run's output and no mesh file. The point cloud, DSM and orthomosaic are unaffected. |
| CUDA-enabled COLMAP | Dense point clouds | Sparse cloud only, reported up front by `Api.reconstruction_capabilities()`. The sparse cloud is never inflated to look dense. |
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
