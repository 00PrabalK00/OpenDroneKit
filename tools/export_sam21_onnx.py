"""Re-export SAM 2.1 Hiera-Tiny to ONNX from the Apache-2.0 source checkpoint.

Why re-export instead of shipping onnx-community/sam2.1-hiera-tiny-ONNX: that repo
carries no model card and no licence tag. These graphs are built here, from
facebook/sam2.1-hiera-tiny (Apache-2.0) through transformers' Sam2Model, so the
provenance of every shipped byte is the Apache source plus this script.

Two graphs, both fp32 with their weights folded in (no external-data sidecar):

vision_encoder.onnx
    pixel_values [1,3,1024,1024] (RGB, /255, ImageNet mean/std, direct square resize --
    the publisher's Sam2ImageProcessor does not keep aspect or pad)
    -> image_embeddings_0 [1,32,256,256], image_embeddings_1 [1,64,128,128],
       image_embeddings_2 [1,256,64,64]
    Equivalent to Sam2Model.get_image_embeddings (no-memory embedding added; the decoder's
    conv_s0 / conv_s1 projections pre-applied).

mask_decoder.onnx
    image_embeddings_0..2, point_coords [1,1,N,2] float (1024-square space),
    point_labels [1,1,N] int64 (1 positive, 0 negative, 2 box top-left, 3 box bottom-right)
    -> iou_scores [1,1,3], low_res_masks [1,1,3,256,256] (logits), object_score_logits [1,1,1]
    A box is two points labelled 2 and 3, which is exactly how Sam2PromptEncoder embeds
    input_boxes; the encoder appends the (0,0, label -1) padding point itself.
    Always multimask (three candidates); the caller picks the highest predicted IoU.

Usage:
    python tools/export_sam21_onnx.py --source <dir of facebook/sam2.1-hiera-tiny> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import torch
    from transformers import Sam2Model

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--keep-encoder", action="store_true",
                    help="reuse an encoder graph already exported to --out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = Sam2Model.from_pretrained(args.source).eval().float()

    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            e = self.m.get_image_embeddings(pixel_values)
            return e[0], e[1], e[2]

    class Decoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, e0, e1, e2, point_coords, point_labels):
            pos = self.m.get_image_wide_positional_embeddings()
            sparse, dense = self.m.prompt_encoder(
                input_points=point_coords, input_labels=point_labels,
                input_boxes=None, input_masks=None)
            low, iou, _, obj = self.m.mask_decoder(
                image_embeddings=e2, image_positional_embeddings=pos,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True, high_resolution_features=[e0, e1])
            return iou, low, obj

    enc, dec = Encoder(model).eval(), Decoder(model).eval()
    x = torch.rand(1, 3, 1024, 1024)
    with torch.no_grad():
        e0, e1, e2 = enc(x)
        pts = torch.tensor([[[[300.0, 400.0], [600.0, 700.0], [500.0, 500.0]]]])
        lbl = torch.tensor([[[2, 3, 1]]], dtype=torch.int64)
        enc_path = out / "vision_encoder.onnx"
        dec_path = out / "mask_decoder.onnx"
        if not (args.keep_encoder and enc_path.is_file()):
            torch.onnx.export(enc, (x,), str(enc_path), input_names=["pixel_values"],
                              output_names=["image_embeddings_0", "image_embeddings_1",
                                            "image_embeddings_2"],
                              opset_version=args.opset, dynamo=False, do_constant_folding=True)
        torch.onnx.export(dec, (e0, e1, e2, pts, lbl), str(dec_path),
                          input_names=["image_embeddings_0", "image_embeddings_1",
                                       "image_embeddings_2", "point_coords", "point_labels"],
                          output_names=["iou_scores", "low_res_masks", "object_score_logits"],
                          dynamic_axes={"point_coords": {2: "num_points"},
                                        "point_labels": {2: "num_points"}},
                          opset_version=args.opset, dynamo=False, do_constant_folding=True)

    # Fold any external data back into the graph (both are far below 2 GB).
    import onnx

    for p in (enc_path, dec_path):
        m = onnx.load(str(p), load_external_data=True)
        onnx.save_model(m, str(p), save_as_external_data=False)
        for stray in p.parent.glob(p.name + ".data"):
            stray.unlink()

    # Same-process check against the torch modules, on prompts of other lengths.
    import onnxruntime as ort

    se = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    sd = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    oe = se.run(None, {"pixel_values": x.numpy()})
    print("encoder max|diff|", [float(np.abs(a - b.numpy()).max()) for a, b in zip(oe, (e0, e1, e2))])
    for coords, labels in (([[512.0, 512.0]], [1]),
                           ([[100.0, 120.0], [700.0, 650.0]], [2, 3]),
                           ([[100.0, 120.0], [700.0, 650.0], [400, 400], [200, 300]], [2, 3, 1, 0])):
        c = torch.tensor([[coords]], dtype=torch.float32)
        l = torch.tensor([[labels]], dtype=torch.int64)
        with torch.no_grad():
            ti, tl, to = dec(e0, e1, e2, c, l)
        oi, ol, oo = sd.run(None, {"image_embeddings_0": oe[0], "image_embeddings_1": oe[1],
                                    "image_embeddings_2": oe[2], "point_coords": c.numpy(),
                                    "point_labels": l.numpy()})
        print(f"decoder n={len(labels)} max|iou diff| {np.abs(oi - ti.numpy()).max():.2e} "
              f"max|logit diff| {np.abs(ol - tl.numpy()).max():.2e}")
    for p in (enc_path, dec_path):
        print(p.name, p.stat().st_size, sha256(p))


if __name__ == "__main__":
    main()
