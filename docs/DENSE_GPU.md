# Dense 3D on modern GPUs (RTX 50-series / Blackwell)

Reconstructions used to stop at the sparse cloud on the RTX 5060 development laptop:
the COLMAP builds that were installed could not run patch-match stereo on a
Blackwell (sm_120) GPU. This page records what was measured, which prebuilt binaries
work, how OpenDroneKit finds and tests them, and what a real dense run of the
Aukerman survey produced.

Nothing on this page is shipped with OpenDroneKit. COLMAP (BSD) and OpenMVS (AGPL-3.0)
are separate programs the operator downloads; OpenDroneKit runs them as executables.

## The machine

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, sm_120, 8 GB) |
| Driver | 592.15, `nvidia-smi` header `CUDA Version: 13.1` |
| CUDA toolkit | not installed (`nvcc` absent, no `NVIDIA GPU Computing Toolkit`) |
| Visual Studio | `C:\Program Files\Microsoft Visual Studio\18` present (not needed: no source build was required) |

A CUDA program runs on this card if it carries sm_120 machine code, or PTX that the
driver can JIT, **and** was built with a CUDA toolkit no newer than the driver
supports (13.1 here).

## Prebuilt binaries tested

Download URLs and sha256 (verified against the GitHub release asset digests). The
binaries live outside the repository, under `../odk-tools/`
on the development machine; none are committed.

| Archive | URL | sha256 | Built with | sm_120 in binary | Result on RTX 5060 |
|---|---|---|---|---|---|
| COLMAP 4.0.4 Windows CUDA | https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip | `f4496c3dd9ffccfb4bfda225401b738b545068302ab6487f5539b5b28861b143` | CUDA 12.9 (`cudart64_12.dll`) | yes: colmap.exe embeds code objects built by `nvcc 12.9.86 -arch sm_120` and `-arch sm_100`, and PTX `.target sm_120` / `sm_90` (most of the fatbin is compressed, so this is a lower bound) | **Works.** Patch-match probe passes; full Aukerman dense run below. |
| COLMAP 4.2.1 Windows CUDA | https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip | `e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881` | CUDA 13.2 | yes (`sm_75` .. `sm_120`) | **Fails**: `patch_match_stereo` exits `0xC0000409` right after `PatchMatch::Run`. Toolkit 13.2 is newer than the driver's 13.1. Works after a driver update to one reporting CUDA >= 13.2 (not tested here). |
| OpenMVS 2.4.0 Windows (CPU) | https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip | `0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167` | no CUDA | n/a | **Works** (CPU densify, mesh, texture). |
| OpenMVS 2.4.0 Windows CUDA | https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64_CUDA.7z | `6aac6b14ef478e501d2514cd1d74ed20e659b53b3bc2835d45a473ddfb921621` | CUDA 13.1 | **no** (highest is `sm_89`) | GPU densify fails: `CUDA error ... UtilCUDADevice.h:54: named symbol not found (code 500)`. Its CPU path (`--cuda-device -2`) works, and OpenDroneKit falls back to it automatically. |

Earlier measurements on the same machine (docs/INSTALLATION.md, "Dense reconstruction
on RTX 50-series"): official 4.1.0/4.1.1/4.2.0 archives (CUDA 13.2) fail the same way
as 4.2.1; conda-forge `colmap=4.2.0` `cuda_129` works.

**Conclusion: no source build is needed.** The official COLMAP 4.0.4 CUDA archive
already contains sm_120 code built with CUDA 12.9, which driver 592.15 loads. (An
earlier note in docs/INSTALLATION.md measured a 292 s first probe and 10 s afterwards,
consistent with the driver JIT-compiling some kernels from PTX once; the compute cache
in `%APPDATA%\NVIDIA\ComputeCache` keeps the result.) A
from-source build (CUDA 12.8+ toolkit, `CMAKE_CUDA_ARCHITECTURES=120`) would only be
needed if 4.0.4 were unusable; it was not attempted, and no C: space was used.

GPU OpenMVS is the one thing not available from a prebuilt: its 2.4.0 CUDA archive has
no Blackwell kernels. See "What remains" below.

## Install (Windows)

```powershell
# COLMAP 4.0.4, about 430 MB; anywhere, no admin rights needed
Invoke-WebRequest https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip -OutFile colmap-4.0.4.zip
(Get-FileHash colmap-4.0.4.zip -Algorithm SHA256).Hash   # f4496c3d...b143
Expand-Archive colmap-4.0.4.zip D:\tools\colmap-4.0.4

# OpenMVS 2.4.0 (optional engine), 22 MB
Invoke-WebRequest https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip -OutFile openmvs-2.4.0.zip
(Get-FileHash openmvs-2.4.0.zip -Algorithm SHA256).Hash  # 0c31660c...2167
Expand-Archive openmvs-2.4.0.zip D:\tools\openmvs-2.4.0
```

Then point OpenDroneKit at them, either way:

* **Settings** (`%APPDATA%`/user config `settings.json`): `colmap_binary` (the exe or
  the extracted folder), `openmvs_dir`, `dense_engine` (`colmap` or `openmvs`). From
  the desktop API: `set_colmap_binary(path)`, `set_openmvs_dir(path)`,
  `set_dense_engine(name)`; each re-tests and returns the capability answer.
* **Environment** (wins over the settings): `ODK_COLMAP`, `ODK_OPENMVS_DIR`.

## How capability is decided

`core.reconstruction_colmap.engine_capabilities()` (shown under Environment
Capabilities and returned by `Api.capabilities()` / `Api.reconstruction_capabilities()`):

| Key | Meaning |
|---|---|
| `colmap_binary`, `colmap_binary_source` | the exe used and where it came from: `env:ODK_COLMAP`, `settings:colmap_binary`, `PATH` or a default install location |
| `colmap_cuda_toolkit` | CUDA release the binary was built with (read from the binary) |
| `dense_stereo`, `dense_stereo_reason` | whether patch-match **ran** on a two-view synthetic scene with this binary (cached per binary + driver) and why not |
| `openmvs`, `openmvs_gpu`, `openmvs_reason`, `openmvs_version`, `openmvs_cuda_toolkit`, `openmvs_dir`, `openmvs_configured_by` | the same for OpenMVS: GPU densify is tried, then CPU, on a synthetic plane whose recovered depth must be 2.00 +- 0.10 |
| `dense_engines`, `dense_engine` | which engines work, and which one a run will use |

Rules the code keeps:

* A binary existing is not proof; the probe has to produce a depth map / point cloud.
* A configured path that does not exist is reported as such; no other COLMAP is used
  silently in its place.
* Crashes are named (`0xC0000409 fail-fast`, `0xC0000135 DLL not found`, ...), and a
  toolkit newer than the driver is named as the cause, with the fix.
* If the chosen engine cannot run, the reconstruction warns `Dense MVS skipped: <reason>`
  and every product is built from the sparse cloud, with the reduced raster resolution
  that implies. Nothing is densified synthetically.
* OpenMVS `DensifyPointCloud` is always run with `--tower-mode 0`: its default appends a
  cylinder of synthetic points when it guesses the scene is a tower.
* The OpenMVS textured mesh (`textured_mesh_obj_path`, `texture_image_path`) is reported
  only when the files exist. The Poisson mesh from COLMAP clouds stays vertex-coloured.

## Real run: Aukerman (77 images, ODM sample dataset)

Data: `odk-testdata/odm_aukerman/images` (77 x 6000x4000 Sony DSC JPEGs). Command:

```
python tools/dense_benchmark.py <images> <out> --engine colmap|openmvs|sparse \
    --colmap <colmap-4.0.4/bin/colmap.exe> [--openmvs <openmvs dir>]
```

Evidence: `../odk-testdata/_outputs/dense-gpu/`
(`<run>/summary.json`, `<run>.log`, products under `<run>/`).

RESULTS_TABLE

## What remains

REMAINING

## ODM as an engine (not integrated)

ODM was not available on this machine (no Docker image, no install), so it is
documented, not wired. It would plug in the same way the cloud engine does:

* `core/reconstruction_remote.py` already submits a survey to a server and maps its
  products back into a `ReconstructionResult`. A NodeODM backend would be a second
  `RemoteReconstructor`-shaped class: `POST /task/new` with the images and options
  (`dsm: true`, `pc-quality`, `gcp` from `core.gcp.read_odm_gcp_file`'s source file),
  poll `/task/{uuid}/info`, then download `all.zip` and read `odm_dem/dsm.tif`,
  `odm_orthophoto/odm_orthophoto.tif`, `odm_georeferencing/odm_georeferenced_model.laz`
  and `odm_texturing/odm_textured_model_geo.obj`.
* `build_reconstructor("odm", endpoint=...)` would return it, and
  `engine_capabilities()` would report it only after `GET /info` answers.
* ODM's own GPU path uses OpenSfM + OpenMVS; on Blackwell it has the same sm_120 question
  as OpenMVS above, answered by its image's CUDA build.
