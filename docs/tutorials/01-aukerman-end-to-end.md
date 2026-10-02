# Tutorial 1: Aukerman park, end to end

This tutorial runs one survey all the way through the OpenDroneKit Python SDK, on real
photographs from the OpenDroneMap
[Aukerman park dataset](https://github.com/OpenDroneMap/odm_data_aukerman) (CC0, Sony
DSC-WX220, geotagged). It covers:

1. creating a project and importing a photo folder;
2. planning a survey mission over the photographed area and exporting it;
3. reconstructing the site (COLMAP) and reading what the reconstruction says about itself;
4. measuring on the georeferenced orthomosaic;
5. writing a report.

To keep it fast it uses 8 of the 77 photographs. The code runs as written and is
tested: `tests/test_tutorials.py` extracts every `python` block on this page and runs
them in order.

## What you need

* OpenDroneKit installed (`docs/INSTALLATION.md`) and run from the repository root, so
  `import opendronekit` works.
* COLMAP for step 3. Without it the reconstruction is refused, with the reason (see
  "When something is missing" below).
* About 60 MB of disk space and a network connection for the download.

## 0. Fetch the photographs

```bash
python tools/fetch_sample_data.py fetch odm_aukerman --dest sample_data --limit 8
python tools/fetch_sample_data.py verify odm_aukerman --dest sample_data --limit 8
```

The script downloads from the publisher's GitHub repository at a pinned commit and checks
every file against the sha256 pinned in `tools/sample_data/manifests/odm_aukerman.sha256`.
`verify` should print `9 ok, 0 missing, 0 mismatched` (8 photographs plus the licence).

## 1. A workspace, a project and a dataset

A `Workspace` is a folder of projects, the same kind as the desktop application's data
folder. Use a scratch folder while you experiment.

```python
import os
from pathlib import Path

import opendronekit as odk

samples = Path(os.environ.get("ODK_SAMPLE_DATA", "sample_data"))
images = samples / "odm_aukerman" / "images"
work = Path(os.environ.get("ODK_TUTORIAL_DIR", "odk-tutorial"))

print("OpenDroneKit SDK", odk.__version__)
ws = odk.Workspace(work / "aukerman-data")
project = ws.create_project("Aukerman park")
dataset = project.import_dataset(images)
print(f"{dataset.image_count} photos, {dataset.geotagged_count} geotagged")

meta = dataset.raw["metadata"]
west, south, east, north = meta["bounds_lonlat"]
print("camera:", meta.get("camera"), "| suggested CRS: EPSG", meta.get("suggested_epsg"))
```

`import_dataset` reads the EXIF of every photograph: `bounds_lonlat` is the footprint of
the camera positions, not of the ground they saw.

## 2. Plan and export a mission

Plan a grid over the photographed area, padded by about 40 m, at 70 m above take-off.

```python
pad = 0.0005   # degrees, about 40 m at this latitude
aoi = [[west - pad, south - pad], [east + pad, south - pad],
       [east + pad, north + pad], [west - pad, north + pad]]
plan = project.plan_mission(aoi, template="grid", altitude_m=70,
                            front_overlap_pct=75, side_overlap_pct=70)
s = plan.summary
print(f"{s.waypoints} waypoints, {s.distance_m:.0f} m, {s.duration_min:.1f} min, "
      f"GSD {s.gsd_cm:.2f} cm/px")
for warning in s.warnings:
    print("planner warning:", warning)

files = plan.export(["qgc_plan", "kml"], directory=work / "mission")
for fmt, paths in files.items():
    print(fmt, "->", [p.name for p in paths])
plan.save("Aukerman grid", note="tutorial 1")
```

`plan.export` returns only files that were actually written. `MissionPlan.builtin_formats()`
lists every built-in format; any other name is looked up as an installed exporter plugin
(tutorial 3 writes one).

## 3. Reconstruct

```python
recon = project.reconstruct(engine="auto", timeout_s=3600,
                            progress=lambda pct, msg: None)
print(f"engine {recon.engine}: {recon.registered_images}/{recon.frame_count} photos registered")
print("georeferenced:", recon.georeferenced, "| EPSG", recon.crs_epsg,
      "| camera-position RMSE (m):", recon.geo_rmse_m)
print("orthomosaic:", recon.orthomosaic_path)
for warning in recon.warnings:
    print("-", warning)
```

Read the warnings. They are the reconstruction describing its own limits: photographs
that did not register (and are left out of every product), a dense step that could not
run on this machine, raster resolution relaxed because a sparse cloud cannot fill a finer
grid. On 8 photographs expect 6 to 8 registered images and a georeferenced sparse model;
the exact counts depend on your COLMAP build and on whether SIFT runs on the GPU.

`recon.geo_rmse_m` is how far the reconstructed camera positions sit from the GPS tags
after alignment. It describes the consumer GPS as much as the model; treat it as the
accuracy floor, not a survey-grade figure.

## 4. Measure on the orthomosaic

Measurements on the map are made in the raster's projected CRS. A point outside the
mapped area is refused, so measure near the middle of the registered photographs.

```python
cx, cy = meta["center_lonlat"]
d = 0.0002   # degrees
line = project.measure_map("distance", [[cx - d, cy], [cx + d, cy]])
print(f"distance: {line.value:.2f} {line.unit}")
square = project.measure_map("area", [[cx - d, cy - d], [cx + d, cy - d],
                                      [cx + d, cy + d], [cx - d, cy + d]])
print(f"area: {square.value:.1f} {square.unit}")
for caveat in (square.raw.get("measurement") or {}).get("caveats", []):
    print("caveat:", caveat)
```

Both numbers come from the orthomosaic's georeferencing, not from the lon/lat you typed:
the points are converted to raster pixels and measured there, so they inherit the
model's accuracy (step 3).

## 5. Write the report

```python
report = project.generate_report(title="Aukerman park survey (tutorial)",
                                 report_type="standard", formats=["html"],
                                 summary="Eight-photo demonstration run of the OpenDroneKit SDK.")
print("report:", report.html_path)
for warning in report.warnings:
    print("report warning:", warning)
ws.close()
```

The report says what the project does not hold (for example "No findings recorded", which
is not the same as a clean inspection) instead of leaving a section looking complete.

## When something is missing

Every SDK call returns a real result or raises `odk.Refused` with the application's own
reason. For example, without COLMAP the reconstruction refuses before it starts, and
measuring before anything was reconstructed says "This project has no georeferenced raster
to measure on. Run Process first, or add a GeoTIFF layer." Catch `odk.OdkError` to handle
every refusal and failure in one place:

```python
try:
    odk.Workspace(work / "aukerman-data").open_project("No such project")
except odk.Refused as exc:
    print("refused:", exc.reason)
```

## Next

* [Tutorial 2](02-crack-detection.md): crack detection on a road-damage photograph.
* [Tutorial 3](03-custom-plugin.md): write, install and use your own plugin.
* The full API: [docs/sdk/reference.md](../sdk/reference.md).
