# Model card: relative_depth_dav2_small

| | |
|---|---|
| Registry key | `relative_depth_dav2_small` |
| Kind | `onnx_relative_depth`, `output_semantics: "relative"` |
| Release asset | `relative_depth_dav2_small.onnx` (v0.1.0-beta) |
| sha256 | `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c` |
| Size | 99,060,839 bytes (fp32, single file, no external data) |
| Weights licence | **Apache-2.0** (see `docs/model_licences/relative_depth_dav2_small.md`) |
| Units | **relative (unitless). Never metres.** |

## Task

Relative depth from one photo, to help a person read structure: occlusion order, a bulge
or sag that the shading hides, which member is in front of which. The output is *relative
inverse depth*: larger is nearer. It is defined only up to an unknown scale and shift that
differ from photo to photo, so no pixel has a distance.

**Do not use it for** clearance, crack depth, spall depth, bulge height, stockpile volume
or any other measurement. OpenDroneKit has no API that converts this output to metres, and
the tests pin that. It is a viewing aid, not a finding; nothing it produces goes to Review.

## Model

- Depth Anything V2, ViT-S encoder + DPT head (24.8M parameters). Yang et al.,
  arXiv:2406.09414.
- Original weights: `depth-anything/Depth-Anything-V2-Small` @ `03876f8651c73a60fe4c2c48294e09fcb6838fcf`.
- ONNX: `onnx-community/depth-anything-v2-small` @ `4472b7362082ad9968fee890ca0f1e5aca36b93d`,
  `onnx/model.onnx`, shipped byte for byte (its sha256 is the Hugging Face LFS id).
  OpenDroneKit did not export, modify or fine-tune it, and has not itself compared this
  graph against the PyTorch checkpoint.
- Only Small is Apache-2.0. Base, Large and Giant are CC BY-NC 4.0 and are not shipped.

## Input and output (`core/depth.py`)

- BGR photo, resized bicubic so the short side is 518 with aspect kept and both sides
  rounded to a multiple of 14; RGB in 0-1; ImageNet mean/std normalisation.
- Output `[1, H', W']` relative inverse depth, upsampled bicubic back to the photo's size.
- Display: per-photo 2nd-98th percentile stretch, inferno colour map, colour bar labelled
  "nearer / farther (relative)" with no numbers.
- Saved by `Api.relative_depth`: `analysis/depth/<stem>.npy` (float32),
  `<stem>_relative16.png` (16-bit display stretch, 0 farthest, 65535 nearest),
  `<stem>_overlay.png` and `<stem>.json` stating the values are relative.
- CPU time about 2 s per photo on the development machine.

## Publisher's metrics (not measured by OpenDroneKit)

Zero-shot relative depth, affine-invariant (scale and shift aligned), as published in
arXiv:2406.09414 and checked against v2 of the paper on 2026-10-03:

| Benchmark | AbsRel (lower is better) | δ1 (higher is better) | Source |
|---|---|---|---|
| KITTI | 0.078 | 0.936 | Table 2 |
| NYU-D | 0.053 | 0.973 | Table 2 |
| DA-2K (relative ordering accuracy) | 95.3% | | Table 3 |

These are the publisher's numbers on street and indoor scenes. None of them is drone
inspection imagery.

## Measured by OpenDroneKit

### Agreement with SfM depth (not ground truth)

Evidence: `docs/holdout/relative_depth_dav2_small.json` (per-photo rows). Method (`tools/eval_relative_depth.py sfm`): each 3-D point of an existing COLMAP sparse
reconstruction is projected into each registered photo, giving sparse depth in the
reconstruction's own arbitrary units. Scale and shift are fitted in inverse depth on a
random half of each photo's points (seed 0) and AbsRel and δ1 (< 1.25) are scored on the
other half; the table gives the mean over photos. The reconstructions are the existing
OpenDroneKit runs in `odk-testdata/_outputs/recon/` (no new reconstruction was run).

A flat-scene baseline (every point at the fit half's mean inverse depth) is reported beside
the model. Nadir survey photos have little depth range, so AbsRel and δ1 alone flatter any
predictor; the baseline and the rank correlation show what the network adds.

| Flight (open data) | Photos (N) | Points scored | AbsRel | δ1 | Spearman ρ | Flat-scene AbsRel | Flat-scene δ1 | Depth range p95/p5 |
|---|---|---|---|---|---|---|---|---|
| ODM bellus (`odm_data_bellus` @ 36f8086), nadir, Canon S110 | 106 of 122 registered | 28,113 | 0.029 | 0.9995 | 0.78 | 0.063 | 0.980 | 1.27 |
| ODM sheffield_cross (@ 143f556), oblique ~-75°, DJI Phantom 3 | 172 of 172 | 412,477 | 0.012 | 0.9999 | 0.98 | 0.082 | 0.994 | 1.36 |

Read it as: on these flights the relative map, once given the right scale and shift,
cuts the error of a flat-scene guess roughly in half (bellus) and sevenfold (sheffield), and ranks point
depths in the same order as photogrammetry (ρ 0.78 and 0.98). δ1 is near 1 for both the
model and the baseline because the depth range is small; it does not separate them.

SfM depth is itself an estimate (sparse tie points with their own noise). These numbers
say how well the relative map agrees with photogrammetry on these flights, not how
accurate either is. The training data (62M+ web images) cannot be checked for these two
public ODM datasets; overlap is unlikely but not excluded.

### PARITY of int8 against fp32 (not accuracy)

`onnx/model_int8.onnx` (27,258,801 bytes, sha256 `01aa7a23…6b9001`) against fp32, raw
network-resolution maps, 20 photos evenly spaced from
`odk-testdata/bridge_inspection_h20t_boga_cayi/B1_Wide`: Spearman ρ mean 0.978 (min
0.862); AbsRel after scale-shift alignment mean 0.031 (max 0.068); 5.3 s vs 2.2 s per
photo on this CPU. int8 is not at parity and is slower, so **fp32 ships**. The int8 file
stays in the local weights folder for this measurement and is not released.

## Limits

- Zero-shot on drone imagery; never trained on it.
- Sky, water, glass and thin wires are frequently wrong, as for any monocular model.
- Scale and shift change from photo to photo: two photos' maps cannot be compared
  value for value, and neither can tiles of one photo.
