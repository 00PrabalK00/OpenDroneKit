# Licence: road_damage_rfdetr_small

| | |
|---|---|
| Weights licence | **Apache-2.0** |
| Full text | [`LICENSES/road_damage_rfdetr_small.txt`](../../LICENSES/road_damage_rfdetr_small.txt) |
| Licence source | https://huggingface.co/dronefreak/rdd2022-rfdetr-small |
| Use | commercial use allowed, under Apache-2.0's notice and attribution terms |
| Released file | `road_damage_rfdetr_small.onnx`, 114,253,423 bytes, sha256 `38d4799494058741c01590d5a919f182f1644af58f12b378f6b2161e77f2c504` |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence, as
published by their authors.

## Attribution

Weights: dronefreak/rdd2022-rfdetr-small (DetectionBench), base roboflow/rf-detr (Apache-2.0; Robinson et al., ICLR 2026). Data: Arya, Maeda, Ghosh, Toshniwal, Sekimoto, 'RDD2022: A multi-national image dataset for automatic Road Damage Detection', arXiv:2209.08538, 2022.

## Provenance of the file

- Original: `dronefreak/rdd2022-rfdetr-small` @ `51718250c97c40e7ddf2f586695a441fa19fb38b`, `checkpoint_best_total.pth` (sha256 `19e446593695f2ae4335114b6f5a419b06352b31ddda01c357ed4d3c0351f33b`, equal to the HF LFS id) and `config.json`.
- Converted to ONNX by OpenDroneKit with `rfdetr` 1.11.1 (`RFDETRSmall(pretrain_weights=...).export(opset_version=17)`, 512 x 512, CPU). Weights not retrained or quantised. Parity: see the model card.

## Training data

RDD2022, CC BY 4.0 per the authors' Figshare release (10.6084/m9.figshare.21431547). The HF mirror dronefreak/RDD2022 states CC BY-SA 4.0; the original's terms are used. Images are not redistributed. (https://doi.org/10.6084/m9.figshare.21431547)

The HF mirror `dronefreak/RDD2022` (and DetectionBench's adapter) state CC BY-SA 4.0 for the images. OpenDroneKit follows the original authors' Figshare release, which is CC BY 4.0, and attributes Arya et al. 2022. No RDD2022 images are shipped.
