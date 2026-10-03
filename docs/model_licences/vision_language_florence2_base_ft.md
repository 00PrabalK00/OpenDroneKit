# Licence: vision_language_florence2_base_ft (Florence-2-base-ft)

| | |
|---|---|
| Weights licence | **MIT** (Microsoft's own MIT licence for the weights, with its copyright notice) |
| Full text | [`LICENSES/vision_language_florence2_base_ft.txt`](../../LICENSES/vision_language_florence2_base_ft.txt), also released as `vision_language_florence2_base_ft__LICENSE.txt` beside the weights |
| Licence source | https://huggingface.co/microsoft/Florence-2-base-ft/blob/main/LICENSE; `license: mit` on https://huggingface.co/onnx-community/Florence-2-base-ft |
| Use | commercial use allowed; keep Microsoft's copyright and permission notice with the files |
| Released graphs (int8) | `vision_language_florence2_base_ft__vision_encoder_int8.onnx` 93,746,478 bytes, sha256 `d7876c1ab0f7ec11998942ca189e99a775c5a4a912b813c7745d0f6fa9343487`; `__embed_tokens_int8.onnx` 39,390,433 bytes, `6b2258db1c8ee9b160576ccde3cd3814d83a2edaed0dd1c6ca9ff3c38fa62214`; `__encoder_model_int8.onnx` 43,651,493 bytes, `f4ad7a68f1fb875d3bcf735ea14a7021b7ba7e83baf7cf10289881b4ed6d9b85`; `__decoder_model_merged_int8.onnx` 98,177,697 bytes, `f22f52f980c33df0efa15932c2f3db6d9d3595ce6387eca938b8cfe23dc4c641` |
| Released companions | `__tokenizer.json`, `__vocab.json`, `__merges.txt`, `__added_tokens.json`, `__config.json`, `__generation_config.json`, `__preprocessor_config.json`, `__tokenizer_config.json`, `__special_tokens_map.json` (sha256 of each in `models/model_registry.json`) |

MIT covers the OpenDroneKit code; the weights carry their own MIT licence and Microsoft's
copyright notice, which travel with them.

## Attribution

Florence-2 by Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael
Zeng, Ce Liu and Lu Yuan (Microsoft). "Florence-2: Advancing a Unified Representation for
a Variety of Vision Tasks", CVPR 2024, arXiv:2311.06242. Copyright (c) Microsoft
Corporation. ONNX conversion by onnx-community (Hugging Face).

## Provenance of the files

- Original weights: `microsoft/Florence-2-base-ft` @ `f6c1a258`.
- ONNX graphs: `onnx-community/Florence-2-base-ft` @ `e88a44eaf3791a35eae0c5a47b3dbcd36e67eb6f`,
  `onnx/{vision_encoder,embed_tokens,encoder_model,decoder_model_merged}_int8.onnx`. Each
  released file is that file **byte for byte**, renamed only (key prefixed).
- Not modified, fine-tuned or re-quantised by OpenDroneKit. The fp32 graphs (about 1.09 GB)
  are kept off the release, for parity measurement only.

## Training data

FLD-5B (126M images, 5.4B annotations), built by Microsoft and not released; its terms are
unknown. OpenDroneKit redistributes no training data, only the MIT weights.
