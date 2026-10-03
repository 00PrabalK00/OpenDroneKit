# Beta testing OpenDroneKit v0.1.0-beta

Thanks for trying the first public beta. This page tells you what the beta is for, how
to install it, a 30-minute test run on public data, what not to do, and how to tell us
what broke.

## Who this beta is for

- People who plan, process or inspect drone surveys (mapping, roofs, solar, roads,
  structures) and want to try an offline, open-source toolkit on a Windows PC.
- Developers who want to read the code, run the tests and report where it is wrong.

It is **not** ready for paid client work or for flying aircraft. Version 1.0 is still
being built; this beta is how we find what breaks on machines other than ours.

## Install

You need a 64-bit Windows 10 (1809 or later) or Windows 11 PC. No GPU is required. A
CUDA-capable NVIDIA GPU with a CUDA build of COLMAP enables dense reconstruction; without
one, reconstruction runs sparse and says so.

### Option A: the Windows installer (when it is published)

1. Download the installer from the GitHub Release `v0.1.0-beta`. Its name has the form
   `OpenDroneKit-<version>-win64-setup-UNSIGNED.exe`, where `<version>` is the Python
   package version (`0.1.0b0` for this beta), so expect
   `OpenDroneKit-0.1.0b0-win64-setup-UNSIGNED.exe`. The release page lists its exact
   file name, size and SHA-256.
2. Check the hash before running it:

   ```
   certutil -hashfile <installer file name> SHA256
   ```

   It must match the SHA-256 on the release page. If it does not, do not run it, and
   tell us.
3. If the file name contains `-UNSIGNED`, Windows SmartScreen will warn that the
   publisher is unknown. That build has no code-signing certificate; the hash check in
   step 2 is how you know the file is the one we published.
4. Run the installer. It installs per user (into `%LOCALAPPDATA%\Programs\OpenDroneKit`)
   and needs no administrator rights.
5. Model weights are not inside the installer. Run **Download AI models** from the
   Start menu. It fetches only the weights published with this release and installs a
   file only if it matches the SHA-256 recorded in `models/model_registry.json`.

**Which models you can get.** Of the nine registered models, this beta publishes the
weights of two: `crack_presence_classifier` and `solar_thermal_anomaly_classifier`. The
other seven are not distributed, because their base models or training data carry
licences that do not allow it (AGPL-3.0, non-commercial or research-only terms);
[WEIGHTS_LICENSES.md](../WEIGHTS_LICENSES.md) gives the reason for each. Without a
model's weights the app still runs, and that detector says it has no model instead of
returning empty results.

If the release has no installer attached yet, use option B.

### Option B: from source

```
git clone https://github.com/00PrabalK00/OpenDroneKit.git
cd OpenDroneKit
git checkout v0.1.0-beta
conda env create -f environment.yml     # creates the "odk" environment, Python 3.11
conda activate odk
python main.py --check-deps             # names anything missing or too old
```

Then double-click `OpenDroneKit.bat`, or run `python main.py --ui cockpit`.

Model weights: download the two published `.onnx` files from the `v0.1.0-beta` release
page and put them where the registry expects them:

| Release asset | Put it at |
|---|---|
| `crack_presence_classifier.onnx` | `models/structural/crack_presence_classifier.onnx` |
| `solar_thermal_anomaly_cls.onnx` | `models/solar/solar_thermal_anomaly_cls.onnx` |

(Or keep them in another folder with the same layout and point `ODK_MODEL_WEIGHTS_DIR`
at it.) If your checkout has `core/model_download.py`, `python -m core.model_download
--all` downloads and verifies them for you. Either way,
`python -m training.register --list` shows which models are installed and whether each
file matches its recorded SHA-256; a file that does not match is refused, not used.
Full details: [INSTALLATION.md](INSTALLATION.md).

## A guided 30-minute test

The steps use the cockpit (`--ui cockpit`). Workspaces are along the top row; the verbs
for each workspace are in the second row; every action reports in the bottom right. If
you click something and **no message appears**, that is a bug: please report it.

Write down, as you go, anything that surprises you, takes longer than you expect, or
refuses with a message you don't understand. Note the time each step took.

### 0. Get the test data (544 MB download)

The OpenDroneMap **Aukerman** survey: 77 geotagged nadir photos of a park in Ohio,
released under CC0 1.0 (public domain dedication).

- Download: <https://github.com/OpenDroneMap/odm_data_aukerman/archive/refs/heads/master.zip>
- Our copy is 543,539,933 bytes with SHA-256
  `7ea900fe1ff4a420cf47de0cd9b0780f5b78448e6b744e7b627ae33bcdf8887e`, archive of commit
  `4e8031630f4193494c79b1c1d3524108826d1ba9` (the zip comment carries the commit id).
  GitHub generates branch archives on request and does not promise byte-identical zips,
  so a different zip hash does not by itself mean the data is wrong.

Unzip it. The check that matters is per file:
[docs/beta/aukerman.sha256](beta/aukerman.sha256) lists the SHA-256 of all 77 images
and the licence file at that commit. In PowerShell, from inside
`odm_data_aukerman-master/`:

```
Get-Content <path to>\aukerman.sha256 | ForEach-Object { $h, $p = $_ -split ' \*'; if ((Get-FileHash $p).Hash -ne $h) { "MISMATCH $p" } }
```

No output means every file matches. (In Git Bash: `sha256sum -c <path to>/aukerman.sha256`.)
If a file mismatches, mention it in your feedback.

To keep the run inside 30 minutes, copy the **first 20 images** of
`odm_data_aukerman-master/images/` into a folder of their own (for example
`aukerman20/`). Use all 77 if you have time to spare.

You will also want one photo of cracked concrete for the AI step. The repository has
one: `tests/fixtures/concrete_cracked_public_domain.jpg` (Wikimedia Commons, public
domain, SHA-256 `e020561bb21af625a19e820cbd75ce8246505fd4de68dd0ad6a12b57ba28e6ba`).
Installer users can download it from the repository on GitHub, or use any photo of a
crack of their own.

### 1. Open and create a project (2 minutes)

1. Start OpenDroneKit (Start menu, or `OpenDroneKit.bat --ui cockpit`).
2. **Projects**: `New Project`, choose a folder, name it `aukerman-beta`.
3. **Projects**: `Import`, choose your `aukerman20/` folder.

Check: the project becomes active and the images are listed as the active dataset.

### 2. Plan a grid mission (5 minutes)

1. **Mission Planning**: with the **Area of interest** drawing tool on the map, click out
   a polygon over the park (the images are around 41.30 N, 81.75 W). The message should
   read "Area of interest set" with its size in hectares.
2. Choose the `grid` mission type, set altitude and overlap on the right, then `Plan`.
3. Read the warnings first. With no terrain model loaded the plan warns that altitudes
   are relative to a flat plane at the launch elevation. That warning is correct.

Check: waypoints, flight time and photo count appear, and changing altitude and pressing
`Plan` again changes them.

### 3. Export to QGroundControl, DJI and Litchi (3 minutes)

1. **Mission Planning**: `Save`, then `Export`.
2. Look in the project's `missions/` folder. You should find a QGroundControl `.plan`
   and `.waypoints`, a DJI WPML `.kmz`, a Litchi `.csv` (several parts if the mission
   has more waypoints than Litchi accepts), plus `.kml` and `.geojson`.
3. If you have QGroundControl, Litchi Mission Hub or DJI Pilot 2, open the files there
   **to look at them only**. Do not upload them to an aircraft (see below).

Check: every format is written, or the export says which one failed and why.

### 4. Reconstruct (10 minutes or more, depending on the PC)

1. **Processing**: `Process`. Confirm.
2. Watch the job in the queue. Without a CUDA build of COLMAP, reconstruction is sparse
   and the run says so; that is expected, not a bug.

Check: the run reports how many images registered, the reprojection error, the
coordinate system (Aukerman lands in UTM zone 17N, EPSG:32617) and an orthomosaic and
DSM. Note how long it took and what your CPU/GPU is.

### 5. Measure (3 minutes)

1. **Measurements**: `Distance`, then draw a line across a path or field.
2. `Area`, then draw a polygon around a field, for its area and perimeter.

Check: the results are in metres, carry the project's coordinate system, and appear in
Measurement History. A measurement refused because the model is not georeferenced is a
correct refusal; tell us if it happens on Aukerman, because Aukerman is geotagged.

### 6. Generate a report (3 minutes)

1. **Reports**: `Generate`.
2. `Open Report`, or open the PDF it writes.

Check: either a report that matches what you did, or a refusal with a checklist of what
is missing. A checklist is the report engine declining to emit empty sections, not a
crash.

### 7. Crack detection on a sample image (3 minutes)

Model runs pick their image from the active dataset, so give the crack photo a dataset
of its own. Do this after the report, because importing makes the new folder the
active dataset.

1. Put `concrete_cracked_public_domain.jpg` (or your own crack photo) alone in a new
   folder, for example `crack-sample/`.
2. **Projects**: `Import`, choose `crack-sample/`.
3. **AI Inspection**: `Run Model`, choose `crack_presence_classifier`, then choose the
   image. (`crack_segmentation` also appears if you have obtained its weights yourself;
   they are not published with this beta.)

Check: the result shows the model's output beside its sha256 and the registry's scope
note, and an overlay is written under the project's `analysis/models/` folder. Only
installed, verified models are offered; with no weights installed the list is empty,
which is correct. `crack_presence_classifier` was trained on close-range tiles and
missed about one cracked tile in fourteen at validation, so "uncracked" is not proof
of a sound surface ([model card](model_cards/crack_presence_classifier.md)).

## What NOT to do

- **Do not fly real aircraft with this beta.** Do not upload exported missions to a real
  drone, and do not connect the Flight workspace to a real vehicle. Mission exports and
  flight control have been tested against specifications and simulators, not in the
  air with your aircraft, your firmware and your site. Use a simulator (see
  [SITL.md](SITL.md)) if you want to try the flight side.
- Do not rely on any number from this beta for client deliverables, safety decisions or
  structural assessments.
- Do not post private site imagery, coordinates or tokens in public issues.

## Report bugs and feedback

Open an issue on GitHub: <https://github.com/00PrabalK00/OpenDroneKit/issues/new/choose>

- **Beta feedback**: the 30-minute test above, step by step: what worked, what didn't,
  how long it took, and on what hardware.
- **Bug report**: something crashed, gave a wrong number, presented something as
  measured that was not, or refused when it should not have.
- **Feature request**: something you need for v1.
- Security problems: report privately as described in `SECURITY.md`, never in a public
  issue.

Useful things to attach: the version (the installer's file name, or
`git rev-parse --short HEAD` for a source install), Windows version, CPU/GPU, the exact
message from the bottom right, and the startup error log if startup failed (`logs/startup-error.log` under the data folder;
for a source install that is `final_toolkit_outputs/logs/`).

## Privacy

Everything runs on your PC. OpenDroneKit needs no account, has no telemetry and no
analytics, and sends nothing about you or your data anywhere. The UI is served on
127.0.0.1 only. The things that do use the network are ones you ask for: online
basemap tiles (which the map fetches from the tile provider while you browse it),
downloading model weights, the installer's update check if you switch it on (it is off
by default), and anything you configure yourself, such as a cloud reconstruction
endpoint. Your projects, images and reports stay in the local data folder. Nothing you do in the beta reaches us unless you put it in an issue.
