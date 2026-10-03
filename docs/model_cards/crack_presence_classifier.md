# Model card: crack_presence_classifier

| | |
|---|---|
| Registry key | `crack_presence_classifier` |
| Release asset | `crack_presence_classifier.onnx` (v0.1.0-beta) |
| sha256 | `92a4d1425b60a8ff96aa31de0ae404325f5a1ab47edc01f5b3acef061da34d74` |
| Size | 44,701,863 bytes |
| Weights licence | MIT, with the training-data attribution below (see `WEIGHTS_LICENSES.md`) |

## Task

Binary tile classification: is there a crack in this image tile? It answers that and nothing
else. It is **not** a localiser. Do not use it to say where a crack is, how long it is or how
wide. `crack_segmentation` is the model for crack extent; the intended pipeline is that this
model triages tiles and the segmenter measures the ones it flags.

## Architecture and input

- ResNet18 (torchvision), initialised from torchvision's ImageNet-1K weights
  (`training/configs/crack_cls.yaml`: `model_id: resnet18`, `pretrained: true`), fine-tuned
  for 20 epochs on a Kaggle P100 (run `crack_cls`).
- ONNX opset 17. Input `images`: float32 `[batch, 3, 224, 224]`, RGB scaled to 0-1.
  ImageNet normalisation is inside the graph. Output `probabilities`: `[batch, 2]`, softmax
  in the graph, class order `cracked`, `uncracked`.
- ONNX/PyTorch parity: max abs diff 1.19e-07; loads in `cv2.dnn`
  (`models/manifests/model_provenance.json`).

## Training data

The `crack_cls` corpus (`training/datasets/prepare.py`, task `crack_cls`):

| Dataset | Licence | Attribution |
|---|---|---|
| SDNET2018 (Utah State University) | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ (stated at https://digitalcommons.usu.edu/all_datasets/48/) | Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning applications. Utah State University. https://doi.org/10.15142/T3TD19 |
| Concrete Crack Images for Classification (Kaggle mirror `arunrk7/surface-crack-detection`) | CC BY 4.0 (stated at https://data.mendeley.com/datasets/5y9wdsg2zt/2) | Özgenel, Ç.F., Gönenç Sorguç, A. "Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings", ISARC 2018, Berlin. |

Tiles are close-range photographs of concrete and pavement surfaces.

## Metrics (validation split, as recorded in the registry)

14,399 held-out tiles. Balanced accuracy selected the checkpoint.

| | Value |
|---|---|
| Balanced accuracy | 0.9575 |
| Plain accuracy | 0.9659 |
| cracked: recall / precision (4,322 tiles) | 0.931 / 0.954 |
| uncracked: recall / precision (10,077 tiles) | 0.981 / 0.971 |

These are validation figures, not a separate test-split measurement.

## Known weaknesses

- Unmeasured on aerial imagery at survey altitude. The corpus is close-range tiles.
- The corpus holds far more cracked tiles relative to sound surface than any real
  structure, so the precision figure will fall on a survey that is mostly sound surface.
- About one cracked tile in fourteen is missed at validation (recall 0.931). A tile it
  passes as uncracked is not proof that the surface is sound.
- A high score says nothing about the quality of any crack geometry downstream.
- Not measured on Indian sites.
