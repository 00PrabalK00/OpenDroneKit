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


class TestTheVerdictAgreesWithItself:
    """Audit: Size Job said "Does not fit as one job: 8 images should run as a single
    job ... Run in chunks of about 8 images" on a machine with 573 MB free. The fit test
    deducted a 2 GB margin from memory that is already net of the OS, and the note and
    the verdict were worked out from different conditions."""

    def test_a_job_within_available_memory_fits_and_says_memory_is_tight(self) -> None:
        estimate = size_job(8, memory_mb=573, disk_mb=500000)
        assert estimate.fits_in_memory and estimate.can_run_as_one_job
        assert not estimate.chunking_required and estimate.recommended_chunk_size == 8
        assert estimate.note.startswith("8 images should run as a single job")
        assert any(w.startswith("Memory is tight") for w in estimate.warnings)

    def test_a_job_past_available_memory_is_split_into_smaller_chunks(self) -> None:
        estimate = size_job(20, memory_mb=700, disk_mb=500000)
        assert not estimate.fits_in_memory and not estimate.can_run_as_one_job
        assert estimate.chunking_required
        assert 1 <= estimate.recommended_chunk_size < 20
        assert estimate.note.startswith("This job should be split")

    def test_a_job_short_of_disk_does_not_claim_it_should_run(self) -> None:
        estimate = size_job(50, memory_mb=32000, disk_mb=1000)
        assert not estimate.can_run_as_one_job
        assert "should run as a single job" not in estimate.note
        assert "not on disk" in estimate.note

    @pytest.mark.parametrize("images", [1, 8, 40, 299, 301, 2000])
    @pytest.mark.parametrize("memory", [0, 5, 36, 44, 45, 100, 573, 2500, 8000, 64000])
    @pytest.mark.parametrize("disk", [0, 500, 5_000_000])
    def test_note_verdict_and_chunk_never_contradict(self, images, memory, disk) -> None:
        e = size_job(images, memory_mb=memory, disk_mb=disk)
        assert e.note.startswith("This job should be split") == e.chunking_required
        assert ("should run as a single job" in e.note) == e.can_run_as_one_job
        if not e.fits_one_image:
            # Nothing fits: no chunk is offered as if it would.
            assert e.recommended_chunk_size == 0 and not e.chunking_required
            assert not e.can_run_as_one_job
            assert e.note.startswith("This machine cannot run this job now")
            assert not any("Run in chunks" in w for w in e.warnings)
            return
        assert e.chunking_required == (e.recommended_chunk_size < images)
        if memory > 0:
            # Whatever chunk is recommended must itself fit in what is free.
            assert e.recommended_chunk_size * MEMORY_PER_IMAGE_MB <= memory
        if not e.fits_in_memory:
            assert e.chunking_required, "a job that does not fit was not split"

    @pytest.mark.parametrize("images,memory", [(24, 36.0), (1, 30.0), (1, 5.0)])
    def test_less_memory_than_one_image_refuses_instead_of_chunking(self, images, memory
                                                                    ) -> None:
        """Audit: 24 images at 36 MB free said "chunks of about 1 fit" though one image
        is estimated at 45 MB, and 1 image at 30 MB said "Run in chunks of about 1"."""
        e = size_job(images, memory_mb=memory, disk_mb=500000)
        assert not e.fits_one_image and e.recommended_chunk_size == 0
        assert "fit" not in e.note.split(":")[0] or "cannot" in e.note
        assert any("splitting the job cannot help" in w for w in e.warnings)
        assert e.to_dict()["fits_one_image"] is False
