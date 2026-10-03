# Model weights: licences for v0.1.0-beta

OpenDroneKit's source code is MIT. The trained model weights are **not** covered by that
licence automatically: each one inherits terms from the base model it was fine-tuned from
and from the data it was trained on. This file records that chain for every installed
model in `models/model_registry.json`, and which weights v0.1.0-beta publishes.

Weights are never committed to this repository. Published weights are GitHub Release
assets of `v0.1.0-beta`, and the first-run downloader (`core/model_download.py`) installs
a file only if it hashes to the sha256 recorded in the registry.

This is an engineering record, not legal advice. Where a chain was unclear, the model was
treated as blocked.

## Published in v0.1.0-beta

| Registry key | Asset | Bytes | sha256 |
|---|---|---|---|
| `crack_presence_classifier` | `crack_presence_classifier.onnx` | 44,701,863 | `92a4d1425b60a8ff96aa31de0ae404325f5a1ab47edc01f5b3acef061da34d74` |
| `solar_thermal_anomaly_classifier` | `solar_thermal_anomaly_cls.onnx` | 44,721,172 | `6933a09abff99bf8d4e2619a50a64498432d2877b9abd55ccab7a87ac357fd42` |

Both are released under the MIT licence, with the attribution notices below kept with
them. Model cards: `docs/model_cards/`.

### crack_presence_classifier

- Base model: torchvision ResNet18, ImageNet-1K weights. torchvision is BSD 3-Clause
  (https://github.com/pytorch/vision/blob/main/LICENSE).
- Data: SDNET2018, CC BY 4.0 (https://digitalcommons.usu.edu/all_datasets/48/,
  https://doi.org/10.15142/T3TD19); Concrete Crack Images for Classification (Özgenel and
  Gönenç Sorguç), CC BY 4.0 (https://data.mendeley.com/datasets/5y9wdsg2zt/2), used via the
  Kaggle mirror `arunrk7/surface-crack-detection`.
- Attribution required by CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/):
  - Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image
    dataset for machine learning applications. Utah State University.
    https://doi.org/10.15142/T3TD19
  - Özgenel, Ç.F., Gönenç Sorguç, A. "Performance Comparison of Pretrained Convolutional
    Neural Networks on Crack Detection in Buildings", ISARC 2018, Berlin.

### solar_thermal_anomaly_classifier

- Base model: torchvision ResNet18, ImageNet-1K weights, BSD 3-Clause as above.
- Data: Raptor Maps InfraredSolarModules, MIT
  (https://github.com/RaptorMaps/InfraredSolarModules). Keep Raptor Maps' copyright and
  MIT notice with the weights.

### Caveat that applies to both

torchvision's documentation says: "The pre-trained models provided in this library may
have their own licenses or terms and conditions derived from the dataset used for
training. It is your responsibility to determine whether you have permission to use the
models for your use case." (https://docs.pytorch.org/vision/stable/models.html). The
ImageNet terms of access restrict the *images* to "non-commercial research and educational
purposes" (https://image-net.org/download.php). Whether those terms reach weights
initialised from ImageNet-trained backbones is not settled. Fine-tuned ImageNet backbones
are widely redistributed, so this was not treated as a blocker. A commercial user with a
low tolerance for risk should make their own assessment.

## Not published: blocked

| Registry key | Base model and licence | Training data and licence | Blocker |
|---|---|---|---|
| `structural_multiclass_detector` | Ultralytics YOLO11x, AGPL-3.0 or Enterprise | CODEBRIM via Roboflow mirror (`codebrim_structural`) | Two: (1) the original CODEBRIM record is "Other (Non-Commercial)" (https://zenodo.org/records/2620293). The Roboflow mirror's CC BY 4.0 label cannot relicense it. (2) AGPL-3.0 weights. |
| `crack_segmentation` | NVIDIA SegFormer `nvidia/mit-b5`, NVIDIA Source Code License (non-commercial) | crack_segmentation_kaggle (licence "see Kaggle page", aggregating CFD, Crack500, GAPs384, DeepCrack, Rissbilder, Volker), DeepCrack (MIT), CrackForest ("research use") | Two: (1) the SegFormer licence is non-commercial; (2) the data licences are research-only or unstated. |
| `solar_cell_defect_detector` | Ultralytics YOLO11l, AGPL-3.0 or Enterprise | PVEL-AD | Two: (1) PVEL-AD is distributed for research behind a signed request form (https://github.com/binyisu/PVEL-AD). The repository's Apache-2.0 covers code, and no dataset licence that allows redistributing derived weights was found. (2) AGPL-3.0 weights. |
| `rail_obstacle_detector` | Ultralytics YOLO11l, AGPL-3.0 or Enterprise | UAV-RSOD obstacle archive, CC BY 4.0 (https://zenodo.org/records/12606374) | AGPL-3.0 only. The data is clean. |
| `rail_corridor_segmentation` | NVIDIA SegFormer `nvidia/mit-b2`, NVIDIA Source Code License (non-commercial) | UAV-RSOD segmentation archive, CC BY 4.0 (https://zenodo.org/records/12606374) | NVIDIA non-commercial licence only. The data is clean. |
| `corrosion_severity_segmentation` | NVIDIA SegFormer `nvidia/mit-b2`, NVIDIA Source Code License (non-commercial) | Virginia Tech Corrosion Condition State, CC0 1.0 (https://doi.org/10.7294/16624663.v2) | NVIDIA non-commercial licence only. The data is clean. |
| `shared_semantic_landcover` | DINOv2 ViT-B/14; code and weights Apache-2.0 (https://github.com/facebookresearch/dinov2) | SpaceNet 7 (CC BY-SA 4.0), OpenEarthMap CC BY and CC BY-SA regions, **and** OpenEarthMap CC BY-NC-SA 4.0 regions, admitted by commit 8e06ab5c1 (`--include-noncommercial`; see https://open-earth-map.org/attribution.html). Gorakhpur (xBD/Maxar, CC BY-NC-SA 4.0) is test-only. | Non-commercial ShareAlike training data, mixed with CC BY-SA data. CC BY-SA 4.0 and CC BY-NC-SA 4.0 are not compatible with each other, so even a non-commercial release has no single licence that satisfies both. |

### What the licences say

- Ultralytics (https://www.ultralytics.com/license): "All Ultralytics YOLO trained models
  fall under the AGPL-3.0 License by default." Ultralytics says software that uses them
  must be released under AGPL-3.0, unless the user holds an Enterprise licence.
- NVIDIA SegFormer (https://github.com/NVlabs/SegFormer/blob/master/LICENSE, which the
  `nvidia/mit-b2` and `nvidia/mit-b5` model cards point to): "Work" includes derivative
  works. It may be redistributed only under the same licence. "The Work and any derivative
  works thereof only may be used or intended for use non-commercially", where
  non-commercially means "for research or evaluation purposes only".

### Options for the owner

1. **AGPL models** (`rail_obstacle_detector`; the other two are also data-blocked):
   - (a) Publish them as separate AGPL-3.0 assets. ODK is open source and MIT code may be
     combined into an AGPL work. However, any distribution or network service that bundles
     these weights would then carry AGPL obligations, and that has to be stated to users.
   - (b) Buy an Ultralytics Enterprise licence.
   - (c) Retrain on a permissively licensed detector (for example a torchvision detection
     model, BSD-3) using the training code already in the repository.
2. **SegFormer models** (`rail_corridor_segmentation`, `corrosion_severity_segmentation`;
   `crack_segmentation` is also data-blocked):
   - (a) Publish them under the NVIDIA Source Code License, marked research and evaluation
     only. `docs/INDIA_FIRST_AI_PLAN.md` records a standing decision that non-commercial
     weights are acceptable in this repository. A public release that ships beside an MIT
     app needs that decision confirmed for distribution, not only for use.
   - (b) Retrain on a permissive encoder. DINOv2 (Apache-2.0) is already in the stack, and
     torchvision segmentation backbones are BSD-3. The data for all three non-crack models
     is CC BY 4.0 or CC0, so retraining would clear them completely.
3. **shared_semantic_landcover**: retrain without the CC BY-NC-SA regions. That is the
   default corpus build without `--include-noncommercial`. Then publish under CC BY-SA 4.0,
   because SpaceNet 7 and some OpenEarthMap regions are ShareAlike.
4. **crack_segmentation, solar_cell_defect_detector, structural_multiclass_detector**: these
   need clean data as well as a new base model. For crack segmentation, train only on
   corpora with a stated permissive licence (DeepCrack is MIT). PVEL-AD and CODEBRIM need
   written permission from their authors, or replacing.

Until one of these options is taken, users can still run the blocked models from weights
they obtain themselves. The registry rows and their sha256 checks are unchanged.
