"""Pack the shared-semantic corpus into something that can be uploaded.

The built corpus references 14.19 GB of source imagery scattered across two downloads:
1,423 SpaceNet 7 mosaic tiles with GeoJSON building polygons, and 1,456 OpenEarthMap
tiles with raster class labels. Training on a hosted GPU means getting that data to the
host, and 14 GB over a home connection is the slowest part of the whole exercise.

This packs it to roughly a tenth of the size by doing two things:

  * Images are re-encoded as JPEG. This is LOSSY and it is recorded as such in the
    manifest, because a metric from a packed corpus is not directly comparable with one
    from the originals and nobody should have to guess which they are looking at.
  * Labels are pre-rasterised through read_semantic_sample -- the same function the
    trainer uses -- so the GeoJSON rasterisation, the class remap and the no-data
    masking all happen once, here, rather than needing the source georeferencing that
    JPEG cannot carry.

That second point is what makes the pack faithful rather than approximate: the labels
written out are exactly the arrays the trainer would have computed, so the only
difference between training on this and training on the originals is JPEG.

    python tools/pack_semantic_corpus.py --out training/data/packed/shared_semantic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JPEG_QUALITY = 92


def pack(corpus_path: Path, out_dir: Path, *, quality: int = JPEG_QUALITY,
         limit: int = 0) -> dict:
    from PIL import Image

    from training.semantic_tiles import IGNORE_INDEX, read_semantic_sample, sample_gsd

    Image.MAX_IMAGE_PIXELS = None
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    samples = corpus["samples"][: limit or None]

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    packed: list[dict] = []
    failures: list[dict] = []
    ignored_pixels = 0
    total_pixels = 0

    for index, sample in enumerate(samples, start=1):
        stem = f"{index:05d}_{sample['source']}"
        try:
            image, label, _ = read_semantic_sample(sample)
        except Exception as exc:  # noqa: BLE001 - recorded, never silently skipped
            failures.append({"id": sample.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            continue

        rgb = np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        image_path = images_dir / f"{stem}.jpg"
        Image.fromarray(rgb).save(image_path, format="JPEG", quality=quality, subsampling=0)

        label_array = np.asarray(label)
        if label_array.max(initial=0) > 255:
            failures.append({"id": sample.get("id"), "error": "label ids exceed uint8"})
            continue
        label_path = labels_dir / f"{stem}.png"
        # PNG, not JPEG: a lossy label is a corrupted label. One wrong pixel value is a
        # different class, not a slightly different colour.
        Image.fromarray(label_array.astype(np.uint8), mode="L").save(label_path)

        ignored_pixels += int((label_array == IGNORE_INDEX).sum())
        total_pixels += int(label_array.size)

        # Measured HERE, from the georeferenced original, because JPEG cannot carry a
        # geotransform and the trainer needs the ground sample distance to bring every
        # source to one scale. Without this the loader finds no CRS on a packed corpus
        # and the scale harmonisation silently does nothing -- a rented GPU spending
        # hours reproducing the exact defect the option exists to fix.
        gsd = sample_gsd(sample)
        if gsd is None:
            failures.append({"id": sample.get("id"), "error": "no ground sample distance"})
            continue

        packed.append({
            **{k: v for k, v in sample.items()
               if k not in {"image", "label", "label_format", "class_map",
                            "class_id", "background_id"}},
            "image": f"images/{image_path.name}",
            "label": f"labels/{label_path.name}",
            "label_format": "raster_class_ids",
            "gsd_m": round(float(gsd), 6),
        })
        if index % 200 == 0:
            print(f"  packed {index}/{len(samples)}", flush=True)

    manifest = {
        **{k: v for k, v in corpus.items() if k != "samples"},
        "samples": packed,
        "packing": {
            "image_format": "jpeg",
            "jpeg_quality": quality,
            "lossy": True,
            "label_format": "png_lossless",
            "labels_pre_rasterised": True,
            "gsd_recorded": True,
            "source_corpus": str(corpus_path),
            "packed_samples": len(packed),
            "failed_samples": len(failures),
            "failures": failures[:50],
            "ignored_pixel_fraction": round(ignored_pixels / total_pixels, 6) if total_pixels else 0.0,
            "reading_note": (
                "Images in this corpus are JPEG re-encodings of the originals, so a "
                "metric measured here is not directly comparable with one measured on "
                "the source TIFFs. Labels are lossless PNG and were rasterised through "
                "the trainer's own read_semantic_sample, so GeoJSON rasterisation, class "
                "remapping and no-data masking are already applied. Each sample carries gsd_m, "
                "measured from the georeferenced original, because the trainer resamples "
                "every source to one ground scale and JPEG cannot carry a geotransform."
            ),
        },
    }
    (out_dir / "corpus.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/pack_semantic_corpus.py")
    parser.add_argument("--corpus", type=Path,
                        default=REPO_ROOT / "training/data/prepared/shared_semantic/corpus.json")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "training/data/packed/shared_semantic")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--limit", type=int, default=0, help="Pack only the first N samples.")
    args = parser.parse_args(argv)

    manifest = pack(args.corpus, args.out, quality=args.quality, limit=args.limit)
    packing = manifest["packing"]
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(json.dumps({
        "packed": packing["packed_samples"],
        "failed": packing["failed_samples"],
        "ignored_pixel_fraction": packing["ignored_pixel_fraction"],
        "bytes": size,
        "gigabytes": round(size / 1e9, 3),
    }, indent=2))
    if packing["failed_samples"]:
        print(f"\n{packing['failed_samples']} samples failed; see corpus.json packing.failures",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
