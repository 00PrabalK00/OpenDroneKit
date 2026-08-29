"""Refuse a packed corpus that would silently train at the wrong scale.

The pack is JPEG, and JPEG carries no georeferencing. The trainer resamples every source
to one ground scale, and to do that it needs each sample's ground sample distance -- which
the packer measures from the georeferenced original and records as gsd_m.

If that field is missing the loader finds no CRS, measures nothing, and falls back to
cropping a fixed pixel count from sources spanning 0.20 to 4.78 metres per pixel. That is
the exact defect the run exists to correct, and it would look completely normal for hours
on a machine being paid for by the hour.

So this runs before training starts, on the box, and exits non-zero rather than letting
that happen.

    python tools/check_packed_corpus.py /workspace/shared_semantic_v3/corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def check(corpus_path: Path) -> int:
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if not samples:
        print(f"{corpus_path} has no samples.", file=sys.stderr)
        return 1

    missing = [s.get("id", "?") for s in samples if not s.get("gsd_m")]
    if missing:
        print(
            f"{len(missing)} of {len(samples)} sample(s) carry no gsd_m, "
            f"e.g. {missing[:3]}. Repack with a packer that records it, or this run "
            "trains at the mixed scale it is meant to fix.",
            file=sys.stderr,
        )
        return 1

    distances = sorted(float(s["gsd_m"]) for s in samples)
    splits: dict[str, int] = {}
    for sample in samples:
        splits[str(sample.get("split"))] = splits.get(str(sample.get("split")), 0) + 1

    print(
        f"{len(samples)} samples, gsd {distances[0]:.2f}-{distances[-1]:.2f} m/px, "
        "all recorded"
    )
    print("  splits: " + ", ".join(f"{k}={v}" for k, v in sorted(splits.items())))

    # The holdout is the whole point of the exercise. If it is not in test, the number
    # this run produces is not a holdout number.
    held = [s for s in samples if str(s.get("group", "")).startswith("spacenet7::L15-14")]
    if held:
        elsewhere = {str(s.get("split")) for s in held} - {"test"}
        if elsewhere:
            print(
                f"India holdout tiles appear in {sorted(elsewhere)}, not only test. "
                "Any score from this corpus would be measured on data the model saw.",
                file=sys.stderr,
            )
            return 1
        print(f"  India holdout: {len(held)} samples, all in test")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/check_packed_corpus.py")
    parser.add_argument("corpus", type=Path)
    return check(parser.parse_args(argv).corpus)


if __name__ == "__main__":
    raise SystemExit(main())
