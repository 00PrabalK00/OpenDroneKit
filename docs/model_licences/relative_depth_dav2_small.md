# Licence: relative_depth_dav2_small (Depth Anything V2 Small)

| | |
|---|---|
| Weights licence | **Apache-2.0** |
| Full text | [`LICENSES/relative_depth_dav2_small.txt`](../../LICENSES/relative_depth_dav2_small.txt), also released as `relative_depth_dav2_small__LICENSE.txt` beside the weights |
| Licence source | https://github.com/DepthAnything/Depth-Anything-V2#license and the `license: apache-2.0` tag on https://huggingface.co/depth-anything/Depth-Anything-V2-Small |
| Use | commercial use allowed, under Apache-2.0's notice and attribution terms |
| Released file | `relative_depth_dav2_small.onnx`, 99,060,839 bytes, sha256 `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c` |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence,
Apache-2.0, as published by their authors.

## Attribution

Depth Anything V2 (ViT-S, "Small") by Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao,
Xiaogang Xu, Jiashi Feng and Hengshuang Zhao. "Depth Anything V2", arXiv:2406.09414 (2024).
https://github.com/DepthAnything/Depth-Anything-V2

## Provenance of the file

- Original weights: `depth-anything/Depth-Anything-V2-Small` @ `03876f8651c73a60fe4c2c48294e09fcb6838fcf`.
- ONNX graph: `onnx-community/depth-anything-v2-small` @ `4472b7362082ad9968fee890ca0f1e5aca36b93d`,
  `onnx/model.onnx`. The released file is that file **byte for byte**, renamed only. Its sha256
  equals the Hugging Face LFS object id of `onnx/model.onnx` at that revision.
- Not modified, not fine-tuned, not quantised by OpenDroneKit.

## Only Small is Apache-2.0

The Depth Anything V2 README states that **only the Small model** is Apache-2.0. Base, Large
and Giant are **CC BY-NC 4.0** (non-commercial). OpenDroneKit ships Small only. Never put a
bigger checkpoint under this key: the registry's sha256 refuses any other file, and
`tests/test_relative_depth.py` pins the source repo to the Small model.

## Training data

The paper reports 595K synthetic images with exact depth labels (teacher training) and
more than 62M real images pseudo-labelled by the teacher (student training). The source
datasets carry mixed licences. OpenDroneKit redistributes none of the data; only the
weights are shipped. Details: arXiv:2406.09414, section 4.
