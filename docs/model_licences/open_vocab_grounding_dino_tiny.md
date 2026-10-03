# Licence: open_vocab_grounding_dino_tiny (Grounding DINO Tiny)

| | |
|---|---|
| Weights licence | **Apache-2.0** |
| Full text | [`LICENSES/open_vocab_grounding_dino_tiny.txt`](../../LICENSES/open_vocab_grounding_dino_tiny.txt), also released as `open_vocab_grounding_dino_tiny__LICENSE.txt` beside the weights |
| Licence source | https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE; `license: apache-2.0` on https://huggingface.co/IDEA-Research/grounding-dino-tiny and on https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX |
| Use | commercial use allowed, under Apache-2.0's notice and attribution terms |
| Released graph | `open_vocab_grounding_dino_tiny.onnx` (int8), 203,824,481 bytes, sha256 `3bff430de583461ab3c1e8b99b19508f4fb238bf0fea0cde2c45f840a0082a26` |
| Released companions | `open_vocab_grounding_dino_tiny__tokenizer.json`, `__vocab.txt`, `__config.json`, `__preprocessor_config.json`, `__tokenizer_config.json`, `__special_tokens_map.json` (sha256 of each in `models/model_registry.json`) |

MIT covers the OpenDroneKit code only. The weights ship under their own licence,
Apache-2.0, as published by IDEA Research.

## Attribution

Grounding DINO by Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang,
Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu and Lei Zhang (IDEA Research). "Grounding
DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024,
arXiv:2303.05499. Copyright 2023 - present, IDEA Research. ONNX conversion by
onnx-community (Hugging Face).

## Provenance of the files

- Original weights: `IDEA-Research/grounding-dino-tiny` @ `a2bb814d`.
- ONNX graph: `onnx-community/grounding-dino-tiny-ONNX` @ `ff690b0a8050566c290287545bd059350f3e9096`,
  `onnx/model_int8.onnx`. The released file is that file **byte for byte**, renamed only.
  The tokenizer and configs are likewise unmodified copies with the key prefixed.
- Not modified, fine-tuned or re-quantised by OpenDroneKit. The fp32 `onnx/model.onnx`
  (718.8 MB, sha256 `267c8b00d269dc2dda0d8592812d3027abd5fb59f1cabf8e3d56f8fdf07f68ec`)
  is kept off the release, for parity measurement only.

## Training data (flag, does not block shipping)

Pre-training used Objects365, GoldG (Flickr30k Entities, GQA) and Cap4M. **The Objects365
images are offered for academic purposes only** (https://www.objects365.org/); its
annotations are CC BY 4.0. The weights themselves are Apache-2.0 and no training data is
redistributed. The flag is recorded so that users can make their own judgement.
