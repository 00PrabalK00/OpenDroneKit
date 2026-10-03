# Licence: interactive_sam21_tiny (SAM 2.1 Hiera-Tiny)

| | |
|---|---|
| Weights licence | **Apache-2.0** |
| Full text | [`LICENSES/interactive_sam21_tiny.txt`](../../LICENSES/interactive_sam21_tiny.txt), also released as `interactive_sam21_tiny__LICENSE.txt` beside the weights |
| Licence source | the `license: apache-2.0` tag on https://huggingface.co/facebook/sam2.1-hiera-tiny and https://github.com/facebookresearch/sam2/blob/main/LICENSE |
| Use | commercial use allowed, under Apache-2.0's notice, attribution and change-notice terms |
| Released files | `interactive_sam21_tiny__vision_encoder.onnx` 109,405,503 bytes, sha256 `8e9e38d722d78bdeee2648fc06c981c713109fe729544fa825dffb913ff39119`; `interactive_sam21_tiny__mask_decoder.onnx` 16,996,922 bytes, sha256 `8232ada3f39d475bfb0a61e0cfaff4b69ce7991f3ee357a13d0182c776e48dc1`; `interactive_sam21_tiny__preprocessor_config.json` 683 bytes, sha256 `6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d` |

MIT covers the OpenDroneKit code only. The weights are shipped under their own licence,
Apache-2.0, as published by Meta.

## Attribution

SAM 2.1 Hiera-Tiny, from "SAM 2: Segment Anything in Images and Videos" by Nikhila Ravi,
Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr,
Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev
Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár and Christoph
Feichtenhofer (Meta FAIR), arXiv:2408.00714 (2024). Copyright Meta Platforms, Inc. and
affiliates. https://github.com/facebookresearch/sam2

## Provenance of the files

- Source checkpoint: `facebook/sam2.1-hiera-tiny` @ `de431c4043854a71d8101e17995dfe596bf101a5`,
  `model.safetensors` (sha256 `48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`).
- **Converted by OpenDroneKit** with `tools/export_sam21_onnx.py`: transformers `Sam2Model`
  loaded from that checkpoint, exported with the TorchScript ONNX exporter at opset 17, fp32,
  weights folded into each graph. Not retrained, not quantised. Only the image-mode path
  (image encoder; prompt encoder plus mask decoder) is exported; the video memory modules are
  not. This is the change notice Apache-2.0 section 4(b) asks for.
- `interactive_sam21_tiny__preprocessor_config.json` is the source repo's
  `preprocessor_config.json`, byte for byte, renamed.
- Cross-check only, **not shipped**: `onnx-community/sam2.1-hiera-tiny-ONNX` @
  `814a066640debee5a91e70aa401fb8e17e030503`. That repo has no model card and no licence tag,
  which is why OpenDroneKit re-exported from the Apache-2.0 source instead of shipping it. Our
  graphs match it to max |diff| 1.1e-05 on the encoder features and 9.5e-06 on the mask logits.

## Training data

SAM 2 was trained on SA-1B (Meta's SA-1B Dataset Research License), SA-V (CC BY 4.0) and
Meta-internal video data, per the SAM 2 paper. OpenDroneKit redistributes none of the data;
only the weights are shipped. The SA-1B research licence applies to the images, not to the
Apache-2.0 weights; it is recorded here so a user can make their own judgement.
