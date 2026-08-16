"""Sizing a reconstruction before it runs.

The failure being prevented: a job that runs for hours, exhausts memory during bundle
adjustment, and dies having produced nothing. Everything looked fine at the start
because nothing checked.

The tests that matter are the ones about not lying -- an unreadable resource probe must
not read as "plenty of room", and chunks must overlap or the merge has nothing to work
with.
"""

from __future__ import annotations

import pytest

from core.job_sizing import (
    EXHAUSTIVE_MATCH_LIMIT,
    MEMORY_PER_IMAGE_MB,
    chunk_images,
    size_job,
)


class TestSizing:
    def test_a_small_job_runs_in_one_pass(self) -> None:
        estimate = size_job(50, memory_mb=32000, disk_mb=500000)
        assert estimate.can_run_as_one_job
        assert not estimate.chunking_required

    def test_a_job_too_big_for_memory_is_chunked(self) -> None:
        estimate = size_job(2000, memory_mb=8000, disk_mb=5_000_000)
        assert not estimate.fits_in_memory
        assert estimate.chunking_required
        assert estimate.recommended_chunk_size < 2000

    def test_a_job_too_big_for_disk_is_flagged(self) -> None:
        estimate = size_job(1000, memory_mb=64000, disk_mb=1000)
        assert not estimate.fits_on_disk
        assert any("free on the work volume" in w for w in estimate.warnings)

    def test_the_quadratic_matching_limit_is_explained(self) -> None:
        # A user needs to know WHY thousands of images is different in kind, not just
        # that it is slower.
        estimate = size_job(EXHAUSTIVE_MATCH_LIMIT + 500, memory_mb=64000, disk_mb=5_000_000)
        assert any("square of the count" in w for w in estimate.warnings)

    def test_zero_images_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            size_job(0)

    def test_the_estimate_says_it_is_rough(self) -> None:
        # An estimate presented as a budget invites someone to plan against it.
        payload = size_job(100, memory_mb=32000, disk_mb=500000).to_dict()
        assert "order of magnitude" in payload["estimate_basis"]


class TestUnknownResources:
    def test_an_unreadable_memory_probe_is_declared(self) -> None:
        # Silence must not read as "plenty of room": the verdict is marked unverified
        # rather than presented as a pass.
        estimate = size_job(100, memory_mb=0, disk_mb=500000)
        assert any("unverified" in w for w in estimate.warnings)

    def test_an_unreadable_disk_probe_is_declared(self) -> None:
        estimate = size_job(100, memory_mb=32000, disk_mb=0)
        assert any("disk verdict is unverified" in w for w in estimate.warnings)


class TestChunking:
    def test_a_small_capture_is_one_chunk(self) -> None:
        assert len(chunk_images([f"i{n}" for n in range(20)], 50)) == 1

    def test_chunks_overlap_so_they_can_be_merged(self) -> None:
        """Overlap is the whole point.

        Chunks reconstructed independently share no geometry. Without images appearing
        in both, merging produces several disconnected models rather than one survey.
        """
        images = [f"i{n}" for n in range(100)]
        chunks = chunk_images(images, 30, overlap=10)
        assert len(chunks) > 1
        for first, second in zip(chunks, chunks[1:]):
            shared = set(first) & set(second)
            assert shared, "consecutive chunks share no images; the sub-models cannot be tied"

    def test_every_image_appears_somewhere(self) -> None:
        images = [f"i{n}" for n in range(97)]
        seen = {image for chunk in chunk_images(images, 25, overlap=5) for image in chunk}
        assert seen == set(images), "chunking silently dropped images"

    def test_overlap_must_be_smaller_than_the_chunk(self) -> None:
        # Otherwise the window never advances and this loops forever.
        with pytest.raises(ValueError, match="never advance"):
            chunk_images([f"i{n}" for n in range(50)], 10, overlap=10)

    def test_a_nonsense_chunk_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            chunk_images(["a", "b"], 0)

    def test_chunk_size_tracks_available_memory(self) -> None:
        small = size_job(5000, memory_mb=8000, disk_mb=5_000_000)
        large = size_job(5000, memory_mb=64000, disk_mb=5_000_000)
        assert large.recommended_chunk_size >= small.recommended_chunk_size
        # Never above the matching limit, however much memory there is.
        assert large.recommended_chunk_size <= EXHAUSTIVE_MATCH_LIMIT
