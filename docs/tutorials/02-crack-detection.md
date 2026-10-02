# Tutorial 2: crack detection on a road-damage photograph

This tutorial runs OpenDroneKit's crack segmentation on a real drone photograph of a road
from [RDD2022](https://doi.org/10.6084/m9.figshare.21431547) (Arya et al. 2022, CC BY 4.0)
and checks the result against the dataset's own annotation. It covers:

1. detecting cracks on one photograph and reading which model (or not) produced the mask;
2. comparing the mask with the ground-truth box a human drew;
3. why a crack *length in millimetres* is refused on this photograph, and what makes it
   possible.

The code runs as written: `tests/test_tutorials.py` extracts every `python` block and runs
it.

## What you need

* OpenDroneKit, run from the repository root.
* Optional: the `crack_segmentation` model. With it the mask comes from a trained
  segmentation network; without it OpenDroneKit runs its classical image-processing
  heuristic and **says so** (`source == "heuristic"`). Install models with the Models
  screen of the application, or point `ODK_MODEL_WEIGHTS_DIR` at a folder that holds
  `structural/crack_segmentation.onnx`.

## 0. Fetch the photographs

```bash
python tools/fetch_sample_data.py fetch road_damage_rdd2022_subset --dest sample_data
```

The 200-image subset is about 51 MB. RDD2022 is published as one 13 GB archive; the script
reads only the pinned photographs out of it with HTTP range requests and checks each
against its sha256. Credit the authors when you use the images (CC BY 4.0).

## 1. Detect cracks

`China_Drone_000011.jpg` is a 512 x 512 top-down photograph of a road with one annotated
longitudinal crack (class `D00`).

```python
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

import opendronekit as odk

samples = Path(os.environ.get("ODK_SAMPLE_DATA", "sample_data"))
rdd = samples / "road_damage_rdd2022_subset" / "China_Drone" / "train"
photo = rdd / "images" / "China_Drone_000011.jpg"
work = Path(os.environ.get("ODK_TUTORIAL_DIR", "odk-tutorial"))

ws = odk.Workspace(work / "cracks-data")
project = ws.create_project("Road cracks")
found = project.detect_cracks(photo)

print("source:", found.source, "| model:", found.model_used)
if found.source == "model":
    print("weights:", found.model_key, found.model_sha256[:16] + "...")
print(f"crack pixels: {found.crack_pixels} ({found.crack_ratio:.2%} of the image)")
print(f"skeleton length {found.length_px:.0f} px, max width {found.max_width_px:.1f} px")
print("mask:", found.mask_path)
print("overlay:", found.overlay_path)
```

`found.source` is the honest answer to "where did this mask come from". A model result
names the weights (`model_key`, `model_sha256`) so a report can say exactly what ran;
heuristic output is labelled screening, never a model result.

## 2. Compare with the annotation

RDD2022 annotates damage with boxes (Pascal VOC), not masks, so the fair question is how
much of the predicted crack lies inside the boxes a person drew.

```python
annotation = ET.parse(rdd / "annotations" / "xmls" / (photo.stem + ".xml"))
boxes = []
for obj in annotation.iter("object"):
    name = obj.findtext("name")
    if name in {"D00", "D10", "D20"}:          # longitudinal, transverse, alligator crack
        b = obj.find("bndbox")
        boxes.append((name, *(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))))
print("annotated cracks:", boxes)

def against_annotation(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    inside = np.zeros_like(mask)
    rows_reached = rows_total = 0
    for _, x1, y1, x2, y2 in boxes:
        inside[y1:y2, x1:x2] = True
        rows_reached += int(mask[y1:y2, x1:x2].any(axis=1).sum())
        rows_total += y2 - y1
    share_inside = int((mask & inside).sum()) / max(int(mask.sum()), 1)
    return share_inside, rows_reached / max(rows_total, 1)

inside_share, reach = against_annotation(found.mask_path)
print(f"{found.source}: {reach:.0%} of the annotated crack's length is covered by the mask; "
      f"{inside_share:.0%} of the mask lies inside the annotated box")

screen = project.detect_cracks(photo, use_model=False)      # the classical heuristic
inside_share, reach = against_annotation(screen.mask_path)
print(f"{screen.source}: {reach:.0%} of the annotated crack's length is covered by the mask; "
      f"{inside_share:.0%} of the mask lies inside the annotated box")
```

Two numbers, because a box is coarse ground truth. *Coverage* asks whether the mask follows
the annotated crack along its length (rows of the box that contain mask pixels). *Inside
share* asks how much of the mask the annotation accounts for. Open the overlay image before
reading too much into the second number: on this photograph the model also marks a long
horizontal line along the edge of a patched area, which the annotator did not box. Whether
that is a crack or a patch seam is an inspector's call; the tool reports what it found and
which model found it, and does not grade itself against a box that was never drawn.

The heuristic responds to any thin dark line (joints, shadows, texture), which is why its
output is labelled `heuristic` and never presented as a model result.

## 3. A length in millimetres needs a scale

A photograph has no scale of its own. Asking for a crack length in millimetres without one
is refused, with the reason:

```python
project.import_dataset(photo.parent)
try:
    project.measure_crack(photo, found.mask_path)
except odk.Refused as exc:
    print("refused:", exc.reason)
```

RDD2022 publishes no ground sampling distance, so this tutorial stops at pixels. On your
own photographs, give the scale you know (`scale_mm_per_px=`, from the flight's GSD), or
mark a reference of known length in the photograph (`reference_pixels=[[x1, y1], [x2, y2]],
reference_length_mm=...`), and `measure_crack` returns the length in millimetres with the
scale it used recorded next to it.

```python
ws.close()
```

## Next

* [Tutorial 3](03-custom-plugin.md): add your own detector, exporter or report section as a
  plugin.
* `Project.run_model(model_key, image)` runs any installed registry model (for example the
  structural multi-class detector) and returns its boxes with the weights' sha256.
