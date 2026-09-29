"""Coverage validation must not hold every full-resolution frame at once.

CoverageValidator.validate kept a list of every decoded frame that nothing ever read,
about 54 MB per 18 MP drone photo. The per-image work only needs a reduced grey frame
and a thumbnail, so memory grew with survey size for no reason -- roughly 11 GB for
200 frames, which is a survey the CLI then could not finish.

This runs the real validator on real full-resolution survey frames and bounds its
peak allocation by a few frames' worth, which the old code exceeded on six frames.
"""

from __future__ import annotations

from pathlib import Path
import tracemalloc

import pytest

from conftest import survey_imagery_dir

FRAMES = 6


def test_peak_memory_does_not_grow_with_the_number_of_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from core.coverage_validation import CoverageValidator

    source = survey_imagery_dir()
    frames = sorted(source.glob("*.JPG"))[:FRAMES] if source.is_dir() else []
    if len(frames) < FRAMES:
        pytest.skip(f"Survey imagery not present at {source} (gitignored).")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    frame_bytes = first.nbytes
    del first

    tracemalloc.start()
    try:
        report = CoverageValidator().validate(frames, tmp_path / "coverage")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(report["images"]) == FRAMES
    # One frame decoding at a time plus working buffers. Keeping all six decoded frames
    # measured 360 MB against about 75 MB with only the reduced copies kept.
    assert peak < 3 * frame_bytes, (
        f"peak {peak / 1e6:.0f} MB for {FRAMES} frames of {frame_bytes / 1e6:.0f} MB: "
        "full-resolution frames are being retained"
    )
