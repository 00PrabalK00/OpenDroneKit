# Licence: interactive_efficientsam_ti (EfficientSAM-Ti)

| | |
|---|---|
| Weights licence | **Apache-2.0** |
| Full text | [`LICENSES/interactive_efficientsam_ti.txt`](../../LICENSES/interactive_efficientsam_ti.txt), also released as `interactive_efficientsam_ti__LICENSE.txt` beside the weights |
| Licence source | https://huggingface.co/opencv/image_segmentation_efficientsam (README: "All files in this directory are licensed under Apache 2.0 License", with the LICENSE file in the repo) and https://github.com/yformer/EfficientSAM/blob/main/LICENSE (Apache-2.0) |
| Use | commercial use allowed, under Apache-2.0's notice and attribution terms |
| Released file | `interactive_efficientsam_ti.onnx`, 48,312,857 bytes, sha256 `4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77` |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence,
Apache-2.0.

## Attribution

EfficientSAM-Ti, from "EfficientSAM: Leveraged Masked Image Pretraining for Efficient
Segment Anything" by Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao,
Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman
Krishnamoorthi and Vikas Chandra, CVPR 2024, arXiv:2312.00863.
https://github.com/yformer/EfficientSAM. ONNX export by the OpenCV Model Zoo
(`opencv/image_segmentation_efficientsam`).

## Provenance of the file

- `opencv/image_segmentation_efficientsam` @ `1e791988b34b2098736003057da0f3e6bc826259`, file
  `image_segmentation_efficientsam_ti_2025april.onnx`. Released **byte for byte**, renamed only.
  Its sha256 equals the value printed in that repo's README.
- The 2025april export takes up to six prompt points, including a box as two points. The
  2024may export (single point only) is not used.
- At run time the graph is cut in memory at the image-encoder output so the encoder runs once
  per image; the file on disk is never modified.
- The int8 file in the same repo (`..._2025april_int8.onnx`, sha256
  `5ecc8d59a2802c32246e68553e1cf8ce74cf74ba707b84f206eb9181ff774b4e`) was measured but is not
  released; see the model card.

## Training data

Per the paper: SAMI pretraining on ImageNet-1K with SAM ViT-H features as targets, then
fine-tuning on SA-1B (Meta's SA-1B Dataset Research License). OpenDroneKit redistributes
none of the data; only the weights are shipped.
