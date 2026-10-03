# Windows installer and auto-update

OpenDroneKit can be installed on Windows from a self-contained installer: a bundled
Python runtime with every requirement, a per-user install (no administrator rights),
Start-menu and desktop shortcuts, and an uninstaller. Model weights are not in the
installer; a first-run downloader fetches them only from a published location and only
when they match the registry's sha256. An opt-in update check reads GitHub Releases and
never installs anything by itself.

**Current state: the installer is UNSIGNED.** The project has no code-signing
certificate yet. The signing step is wired (Azure Trusted Signing or signtool) and skips
itself with a clear message; every unsigned output says UNSIGNED in its file name,
version info, manifest and first wizard page.

| Piece | Where |
|---|---|
| Build pipeline | `tools/installer/build.py`, `config.json`, `requirements-lock.txt` |
| Installer script | `tools/installer/OpenDroneKit.iss` (Inno Setup 6) |
| Signing step | `tools/installer/sign.py` |
| Manifest | `tools/installer/manifest.py` (`install_manifest.json` in the install) |
| Installed entry point | `tools/installer/launch.py` (copied to the install root) |
| Headless start-up check | `app/selftest.py` (`main.py --self-test`, `launch.py --self-test`) |
| Weights downloader | `core/model_download.py` |
| Update check | `core/updates.py` |
| GPU detection | `core/gpu.py` (Environment Capabilities: `gpu_detection`, `gpu_summary`) |
| End-to-end smoke | `tools/installer/smoke_install.py` |
| Tests | `tests/test_installer_pipeline.py`, `tests/test_installer_runtime.py` |

## Why an embeddable Python, not a frozen build

The runtime is python.org's **embeddable CPython 3.11.9 (amd64)** with the app's
packages installed into `runtime\Lib\site-packages`, and the app as plain source in
`odk\`. PyInstaller / Nuitka / cx_Freeze were rejected for this stack:

- **pywebview + WebView2** loads .NET through pythonnet (`clr_loader`), which locates
  `Python.Runtime.dll` and the WinForms/WebView2 assemblies at run time by path.
  Freezers need per-version hooks for this and break when pythonnet or pywebview move
  files; an ordinary site-packages layout is exactly what those packages are tested in.
- **onnxruntime** loads its provider DLLs (`onnxruntime_providers_*.dll`) by path next to
  the package; **rasterio / pyproj** wheels carry GDAL and PROJ with their data
  directories (`gdal_data`, `proj_data`) found relative to the package. Freezers relocate
  these and need data-file hooks; one missed file is a run-time failure, not a build error.
- **open3d / pycolmap / OpenCV** are large binary wheels with their own DLL directories;
  leaving them as installed wheels keeps `os.add_dll_directory` logic intact.
- The install is **auditable**: the payload is `git archive HEAD` plus wheels pinned by
  sha256, and `install_manifest.json` lists every file with its sha256. The self-test
  re-hashes the whole install.
- The embeddable distribution itself is pinned by sha256 in `config.json` (python.org
  publishes only an MD5 for 3.11.9; the pinned sha256 was computed from the download and
  the file's MD5 matches the published one).

The cost is size (no tree-shaking) and that `.py` sources are visible, which is fine for
an MIT project.

Packages: `requirements-lock.txt` is the closure of `requirements.txt` at exactly the
versions the test suite runs on (read from the `odk` environment's installed metadata,
no index re-resolution), every entry with `--hash=sha256:`. Two deliberate differences
from the dev environment: the **CPU build of onnxruntime** replaces `onnxruntime-gpu`
(the GPU builds need a matching CUDA/cuDNN install an installer cannot assume), and
**torch is never bundled** (optional, multi-GB). `opencv-python` replaces the headless
build because the desktop app is not headless.

## Building

Requirements: Windows 10 1809+ x64, Python 3.11 (the runtime's minor version; the `odk`
conda env), git. Inno Setup 6 is installed automatically **only if absent** — with
`winget install JRSoftware.InnoSetup --scope user`, or `choco install innosetup` when
winget is unavailable; the build logs which. `--no-tool-install` turns that off.

```
set TMP=E:\odk-build\tmp & set TEMP=E:\odk-build\tmp & set PIP_CACHE_DIR=E:\odk-build\pipcache
python tools\installer\build.py --out E:\odk-build             :: build from HEAD
python tools\installer\build.py --out E:\odk-build --relock    :: re-pin the lock first
python tools\installer\build.py --out E:\odk-build --skip-installer   :: stage only
```

Steps (each refuses rather than guesses): verify the embeddable zip's sha256 → download
and install the locked wheels with `--require-hashes` → extract `git archive HEAD` minus
`tests/`, `.github/`, `.claude/`, `.continuum/` (the build is the commit, not the working
tree) → byte-compile the app and **all of site-packages** as *unchecked-hash* `.pyc` with
the embedded interpreter, and import numpy, onnxruntime, rasterio and webview with it →
prune the named folders in `payload_prune` (today only `share/jupyter`: JupyterLab
front-end assets ipywidgets installs for notebooks, which the desktop app never runs) →
**refuse any payload path longer than `max_relative_path` (190)** so no installed path
passes Windows' 260-character MAX_PATH → **refuse the build if any weight, credential or
vendor SDK file is in the payload** (`*.onnx`, `*.pt`, `.env`, `*.pfx`, `kaggle.json`,
`google*.html`, DJI thermal SDK libraries, …; the three exemptions in
`forbidden_payload_allow` are named third-party files, each with its reason) → write
`install_manifest.json` → compile the installer → sign or mark UNSIGNED → write
`SHA256SUMS.txt` and `build-report.json`.

Why unchecked-hash bytecode: pip writes timestamp-checked `.pyc` files. Inno Setup does
not restore every source's mtime exactly (identical files are stored once), so the first
smoke install had Python rewrite 9 `__init__` `.pyc` files on import and the self-test
correctly flagged the install as modified. Unchecked-hash `.pyc` files are never
rewritten, and the launcher also sets `sys.dont_write_bytecode`, so the installed tree
stays byte-identical to the manifest (and works read-only under Program Files).

Output: `dist\OpenDroneKit-<version>-win64-setup-UNSIGNED.exe` and `dist\SHA256SUMS.txt`.
Put the build root on a drive with room: the staged runtime is about 1.5 GB and the
wheel cache several hundred MB.

## What the installer does

- Installs per user into `%LOCALAPPDATA%\Programs\OpenDroneKit` with
  `PrivilegesRequired=lowest`; the first page still offers "install for all users",
  which elevates. Silent: `setup.exe /VERYSILENT /CURRENTUSER [/DIR=...] [/TASKS=desktopicon]`.
- Start menu: OpenDroneKit, Download AI models, Self-test, Uninstall OpenDroneKit.
  Optional desktop shortcut. Uninstall entry under HKCU.
- Warns after install if the Edge WebView2 runtime is missing (Windows 11 has it).
- User state stays outside the install: `ODK_DATA_DIR` defaults to
  `%LOCALAPPDATA%\OpenDroneKit\data`, `ODK_MODEL_WEIGHTS_DIR` to
  `%LOCALAPPDATA%\OpenDroneKit\models`. Uninstall removes the install folder,
  shortcuts and registry entry and leaves projects, settings and weights alone.

## First-run model weights

`launch.py --download-models --list | --all | <key>` (also the "Download AI models"
shortcut and an unchecked post-install option; `Api.model_downloads` /
`Api.download_model_weights` in the app).

- A model is downloadable only when a location is **published**: a `download` entry on
  its row in `models/model_registry.json`, or in the JSON file named by
  `ODK_MODEL_SOURCES`. Nothing invents a URL.
- **No weights are hosted yet**, so no row has a `download` entry and every missing
  model reports *"not available for download yet: no download URL has been published
  for this model"*. A test pins that no registry row carries an invented URL.
- A row with no recorded sha256 is refused before any byte is fetched. Downloaded bytes
  that do not hash to the registry sha256 are deleted and refused; they never reach the
  models folder. ONNX graphs with an external-data sidecar are verified per file and as
  a set (the same set hash `core.models.model_identity` computes).

## Auto-update (opt-in)

- Off by default (`AppSettings.update_check_opt_in`); with it off, nothing touches the
  network unless the user explicitly asks (`Api.check_for_update(explicit=True)`).
- Reads `releases/latest` of **00PrabalK00/OpenDroneKit**. A newer release is *offered*
  only when it publishes a sha256 for the installer — GitHub's asset `digest` field or a
  `SHA256SUMS` asset — and the two agree when both exist. A release without a digest is
  reported but not offered.
- `download_update` saves the installer to `<data dir>\updates` (installed default
  `%LOCALAPPDATA%\OpenDroneKit\data\updates`),
  verifies its sha256, and returns the path. It never runs the installer, never passes
  silent flags and never schedules anything; installing is the user's decision.
- Version comparison: numeric parts compare as numbers (`1.2.10 > 1.2.9`), trailing
  zeros are ignored, a release sorts after its pre-releases (`1.0.0rc1 < 1.0.0`), a
  leading `v` and `+build` metadata are ignored, and a tag that is not a version
  (`nightly`) is rejected rather than treated as newer.

To publish a release the update check will offer: attach the installer and the
`SHA256SUMS.txt` the build wrote (GitHub also records an asset digest itself).

## GPU detection

Environment Capabilities (`Api.capabilities`) carries `gpu_detection` and
`gpu_summary`. It reports two things separately: GPUs **present** (Win32_VideoController,
`nvidia-smi` with driver and memory) and GPU inference **usable** (onnxruntime execution
providers in this build, torch CUDA). The installed build ships CPU onnxruntime, so on a
machine with an NVIDIA GPU it says the GPU is present and that this build runs
inference on the CPU — which is what it does.

## Signing

`tools/installer/sign.py`, configured only by environment variables (no secret ever in
the repo):

- **Azure Trusted Signing**: `ODK_AZURE_SIGNING_DLIB` (path to
  `Azure.CodeSigning.Dlib.dll`) and `ODK_AZURE_SIGNING_METADATA` (metadata.json with
  `Endpoint`, `CodeSigningAccountName`, `CertificateProfileName`); credentials from the
  Azure identity chain.
- **Certificate + signtool**: `ODK_SIGN_CERT_THUMBPRINT` (cert in a store) or
  `ODK_SIGN_PFX` + `ODK_SIGN_PFX_PASSWORD`. `ODK_SIGNTOOL` if signtool is not found in
  the Windows SDK.

When configured, Inno Setup signs the installer and the uninstaller (`SignTool=` /
`SignedUninstaller=yes`) with SHA-256 and an RFC 3161 timestamp. Without configuration
the build prints `UNSIGNED: no signing certificate is configured ... Signing skipped.`
and marks every output UNSIGNED. The registry row `inf.installer.signing` stays
`implemented` until a signed build has been produced and verified.

## Verification on this machine

See "Evidence" below; reproduce with:

```
python tools\installer\build.py --out E:\odk-build
python tools\installer\smoke_install.py --installer E:\odk-build\dist\OpenDroneKit-0.1.0-win64-setup-UNSIGNED.exe --work E:\odk-build\smoke
```

The smoke run installs silently per user into `<work>\install`, checks the HKCU
uninstall entry and both shortcut sets, runs `runtime\python.exe launch.py --self-test`
with a **scrubbed environment** (PATH is System32 only, no PYTHONPATH/PYTHONHOME/conda
variables, so it can only succeed on what the installer put down), runs the model
downloader's `--list`, then uninstalls and checks that the folder, shortcuts and
registry entry are gone. The self-test imports pywebview, loads .NET through pythonnet,
finds the WebView2 runtime, imports onnxruntime / rasterio / pyproj / cv2 / pycolmap /
open3d, constructs the `Api` the window binds and calls `capabilities`,
`model_downloads`, `update_settings` and `check_for_update` (opt-in off, no network),
then re-hashes every file in `install_manifest.json`. No window is opened.

## Evidence

Filled in from the end-to-end run (build, install, self-test, uninstall) of the final commit.

## What the owner must supply

1. **A code-signing identity**: an Azure Trusted Signing account + certificate profile
   (set the two `ODK_AZURE_SIGNING_*` variables and `az login`), or an OV/EV
   certificate (thumbprint or pfx). Then rebuild; the output loses the UNSIGNED label.
2. **Hosted weights**: upload each model's file(s) somewhere stable (e.g. a GitHub
   release asset) and add a `download` entry to its row in `models/model_registry.json`
   (or an `ODK_MODEL_SOURCES` file). The sha256 is already in the registry; the
   downloader will refuse anything that does not match it.
3. **A GitHub release** of 00PrabalK00/OpenDroneKit with the installer and its
   `SHA256SUMS.txt` attached, for the update check to have something to offer.
