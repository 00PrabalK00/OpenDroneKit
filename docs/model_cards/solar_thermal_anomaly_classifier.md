# Model card: solar_thermal_anomaly_classifier

| | |
|---|---|
| Registry key | `solar_thermal_anomaly_classifier` |
| Release asset | `solar_thermal_anomaly_cls.onnx` (v0.1.0-beta) |
| sha256 | `6933a09abff99bf8d4e2619a50a64498432d2877b9abd55ccab7a87ac357fd42` |
| Size | 44,721,172 bytes |
| Weights licence | MIT, with the training-data attribution below (see `WEIGHTS_LICENSES.md`) |

## Task

Per-module infrared anomaly classification into 12 classes: Cell, Cell-Multi, Cracking,
Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi, No-Anomaly, Offline-Module, Shadowing,
Soiling, Vegetation. It classifies a crop of one PV module that something else has already
located. It is **not** a localiser and cannot find modules in a survey image.

## Architecture and input

- ResNet18 (torchvision), initialised from torchvision's ImageNet-1K weights
  (`training/configs/solar_thermal_cls.yaml`: `model_id: resnet18`, `pretrained: true`),
  run `solar_thermal_cls`, 26 epochs (early-stopped), class-weighted loss.
- ONNX opset 17. Input `images`: float32 `[n, 3, 96, 96]`, RGB scaled to 0-1, then ImageNet
  mean/std normalisation applied **outside** the graph (registry `preprocessing`:
  `rgb_0_1`, `imagenet`). Output `logits`: `[n, 12]` in the label order of the registry row.

## Training data

| Dataset | Licence | Attribution |
|---|---|---|
| Raptor Maps InfraredSolarModules, https://github.com/RaptorMaps/InfraredSolarModules | MIT (repository licence) | Raptor Maps, InfraredSolarModules; published at the ICLR 2020 AI for Earth Sciences workshop. |

20,000 infrared module crops of 24x40 px, upsampled to 96 px. The corpus holds 10,000
No-Anomaly crops against 249 Hot-Spot and 204 Soiling.

## Metrics (validation split, as recorded in the registry)

Balanced accuracy selected the checkpoint. Read per-class recall, not the mean.

| | Value |
|---|---|
| Balanced accuracy | 0.724 |
| Plain accuracy | 0.815 |

Recall by class: Diode 0.969, Diode-Multi 0.957, No-Anomaly 0.926, Shadowing 0.809,
Vegetation 0.722, Hot-Spot-Multi 0.719, Offline-Module 0.707, Hot-Spot 0.650,
Cracking 0.646, Cell 0.635, Cell-Multi 0.537, Soiling 0.367.

These are validation figures, not a separate test-split measurement.

## Known weaknesses

- **Soiling is the weak class**: recall 0.367 and precision 0.344 on 30 validation samples.
  About two in three soiled modules are missed and about a third of soiling calls are wrong.
  Do not report soiling findings from this model without saying so.
- Hot-Spot recall is 0.650: about one hot spot in three is missed.
- The source crops are 24x40 px. Upsampling adds no detail that was not captured.
- Raptor Maps imagery only. Transfer to other thermal cameras, altitudes and radiometric
  settings is unmeasured, and it has not been measured on Indian sites.
- This is a review aid, not a substitute for an IEC TS 62446-3 thermographic inspection.
