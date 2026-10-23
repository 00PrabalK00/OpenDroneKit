"""ONNX-vs-reference parity for the two U2 models (a parity check, never an accuracy figure).

    python tools/parity_open_vocab.py gdino    --graph <onnx> [--ref <hf dir>] --out <json>
    python tools/parity_open_vocab.py florence --onnx-dir <dir> [--precision int8|fp32] --out <json>

Grounding DINO: the reference is ``IDEA-Research/grounding-dino-tiny`` in transformers.
Both graphs get the *same* ``pixel_values`` (our 800x800 preprocessing) and the same
token ids, so the comparison isolates the graph; separately, our numpy post-processing
is compared with ``GroundingDinoProcessor.post_process_grounded_object_detection`` on
the reference's own outputs, and our caption tokens with the processor's.

Florence-2: the reference is ``florence-community/Florence-2-base-ft`` in transformers,
greedy decoding (``num_beams=1``) with the generation config's no-repeat-3, from the same
``pixel_values``. Reported: token-sequence agreement and the parsed text/regions.

torch and transformers are used here only; neither is a runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHTS = Path(os.environ.get("ODK_MODEL_WEIGHTS_DIR", "../odk-weights"))
TESTDATA = Path(os.environ.get("ODK_TESTDATA_DIR", "../odk-testdata"))


def sample_images() -> list[tuple[Path, str]]:
    thermal = TESTDATA / "thermal_pv_panel_detection_mavic3t" / "Thermal PV Panel Detection Dataset for UAV Inspection"
    rdd = TESTDATA / "road_damage_rdd2022_subset"
    out: list[tuple[Path, str]] = []
    t = sorted((thermal / "test" / "images").glob("*.jpg"))[:1]
    out += [(p, "solar panel.") for p in t]
    india = sorted((rdd / "India" / "train" / "images").glob("*.jpg"))[:2]
    out += [(p, "car. crack. pothole.") for p in india]
    drone = sorted((rdd / "China_Drone" / "train" / "images").glob("*.jpg"))[:1]
    out += [(p, "crack.") for p in drone]
    return out


def gdino(args) -> dict:
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    from core.open_vocab import GroundingDinoOnnx, encode_caption, post_process_grounded

    d = WEIGHTS / "open_vocab_grounding_dino_tiny"
    ref_dir = Path(args.ref or WEIGHTS / "_reference" / "IDEA-Research__grounding-dino-tiny")
    runner = GroundingDinoOnnx(Path(args.graph), d / "open_vocab_grounding_dino_tiny__tokenizer.json",
                               d / "open_vocab_grounding_dino_tiny__preprocessor_config.json", threads=4)
    processor = AutoProcessor.from_pretrained(str(ref_dir))
    model = GroundingDinoForObjectDetection.from_pretrained(str(ref_dir)).eval()
    rows = []
    for img_path, prompt in sample_images():
        im = cv2.imread(str(img_path))
        enc = encode_caption(runner.tokenizer, prompt)
        hf_ids = processor.tokenizer(enc.caption)["input_ids"]
        pv = runner.preprocess(im)
        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(pv), input_ids=torch.from_numpy(enc.input_ids),
                        attention_mask=torch.from_numpy(enc.attention_mask),
                        token_type_ids=torch.from_numpy(enc.token_type_ids),
                        pixel_mask=torch.ones((1, 800, 800), dtype=torch.long))
        ref_logits = out.logits[0].numpy(); ref_boxes = out.pred_boxes[0].numpy()
        onnx_logits, onnx_boxes = runner.raw(im, enc)
        L = enc.input_ids.shape[1]
        p_ref = 1 / (1 + np.exp(-ref_logits[:, :L])); p_onx = 1 / (1 + np.exp(-onnx_logits[:, :L]))
        h, w = im.shape[:2]
        hf_pp = processor.post_process_grounded_object_detection(
            out, input_ids=torch.from_numpy(enc.input_ids), threshold=0.35, text_threshold=0.25,
            target_sizes=[(h, w)])[0]
        ours_pp = post_process_grounded(ref_logits, ref_boxes, enc.input_ids[0].tolist(),
                                        box_threshold=0.35, text_threshold=0.25, image_size_hw=(h, w),
                                        decode=runner._decode)
        pp_box_diff = (float(np.abs(np.asarray([o["bbox"] for o in ours_pp]) - hf_pp["boxes"].numpy()).max())
                       if len(ours_pp) == len(hf_pp["boxes"]) and len(ours_pp) else 0.0)
        hf_labels = list(hf_pp.get("text_labels", hf_pp.get("labels")))
        # Graph parity at the operating point: same kept queries? matched F1 at IoU 0.5.
        onnx_pp = post_process_grounded(onnx_logits, onnx_boxes, enc.input_ids[0].tolist(),
                                        box_threshold=0.35, text_threshold=0.25, image_size_hw=(h, w),
                                        decode=runner._decode)
        rows.append({
            "image": img_path.name, "prompt": prompt,
            "tokens_equal_hf_processor": list(enc.input_ids[0]) == list(hf_ids),
            "max_abs_prob_diff": float(np.abs(p_ref - p_onx).max()),
            "max_abs_box_diff": float(np.abs(ref_boxes - onnx_boxes).max()),
            "postprocess_count_ref_vs_ours": [len(hf_pp["boxes"]), len(ours_pp)],
            "postprocess_max_box_diff_px": pp_box_diff,
            "postprocess_labels_equal": [o["phrase"] for o in ours_pp] == [str(x) for x in hf_labels],
            "kept_ref_vs_onnx": [len(ours_pp), len(onnx_pp)],
            "kept_box_f1_iou50": _f1([o["bbox"] for o in ours_pp], [o["bbox"] for o in onnx_pp]),
            "hf_labels": [str(x) for x in hf_labels][:8],
            "our_labels": [o["phrase"] for o in ours_pp][:8],
        })
        print(json.dumps(rows[-1]), flush=True)
    return {"model": "open_vocab_grounding_dino_tiny", "graph": str(args.graph), "reference": str(ref_dir),
            "rows": rows}


def _f1(a: list, b: list, thr: float = 0.5) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    from tools.eval_open_vocab import iou

    m = iou(np.asarray(a, float), np.asarray(b, float))
    used = set(); tp = 0
    for i in range(len(a)):
        j = int(np.argmax(m[i]))
        if m[i, j] >= thr and j not in used:
            used.add(j); tp += 1
    return 2 * tp / (len(a) + len(b))


def florence(args) -> dict:
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    from core.model_files import role_of
    from core.vision_language import FlorenceOnnx, post_process, task_prompt

    key = "vision_language_florence2_base_ft"
    ref_dir = Path(args.ref or WEIGHTS / "_reference" / "florence-community__Florence-2-base-ft")
    if args.precision == "fp32":
        src = WEIGHTS / key / "_src" / "onnx"
        files = {role_of(p.name, key): p for p in (WEIGHTS / key).iterdir() if p.is_file() and p.suffix != ".onnx"}
        for role in ("vision_encoder", "embed_tokens", "encoder_model", "decoder_model_merged"):
            files[role] = src / f"{role}.onnx"
    else:
        files = {role_of(p.name, key): p for p in (WEIGHTS / key).iterdir() if p.is_file()}
    runner = FlorenceOnnx(files, threads=4)
    processor = AutoProcessor.from_pretrained(str(ref_dir))
    model = Florence2ForConditionalGeneration.from_pretrained(str(ref_dir)).eval()
    rows = []
    tasks = [("<CAPTION>", ""), ("<OD>", ""), ("<CAPTION_TO_PHRASE_GROUNDING>", "a car")]
    for img_path, _ in sample_images()[:3]:
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        pv = runner.preprocess(im)
        for task, text in tasks:
            prompt = task_prompt(task, text)
            ids = runner.tokenizer.encode(prompt).ids
            with torch.no_grad():
                gen = model.generate(input_ids=torch.tensor([ids]), pixel_values=torch.from_numpy(pv),
                                     max_new_tokens=256, num_beams=1, do_sample=False,
                                     no_repeat_ngram_size=3)
            ref_seq = gen[0].tolist()
            ours = runner.generate(im, prompt)
            ref_text = processor.batch_decode([ref_seq], skip_special_tokens=False)[0]
            our_text = runner.tokenizer.decode(ours, skip_special_tokens=False)
            ref_pp = processor.post_process_generation(ref_text, task=task, image_size=(w, h))
            mine_on_ref = post_process(task, ref_text, (w, h))
            rows.append({
                "image": img_path.name, "task": task, "text": text,
                "sequences_equal": ref_seq == ours,
                "first_divergence": next((i for i, (x, y) in enumerate(zip(ref_seq, ours)) if x != y),
                                         None if len(ref_seq) == len(ours) else min(len(ref_seq), len(ours))),
                "ref_len": len(ref_seq), "ours_len": len(ours),
                "ref_text": ref_text[:300], "our_text": our_text[:300],
                "ref_post": json.loads(json.dumps(ref_pp, default=str))[task] if task in ref_pp else ref_pp,
                "our_post_on_ref_text": mine_on_ref,
            })
            print(json.dumps(rows[-1], default=str)[:1500], flush=True)
    return {"model": key, "precision": args.precision, "reference": str(ref_dir), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["gdino", "florence"])
    ap.add_argument("--graph", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.which == "gdino" and not args.graph:
        args.graph = str(WEIGHTS / "open_vocab_grounding_dino_tiny" / "open_vocab_grounding_dino_tiny.onnx")
    result = gdino(args) if args.which == "gdino" else florence(args)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
