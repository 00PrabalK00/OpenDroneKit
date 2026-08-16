"""Whether this machine can finish this reconstruction, asked before it starts.

The failure being prevented is specific and expensive: a job that runs for three hours,
exhausts memory during bundle adjustment or dense matching, and dies having produced
nothing. Everything about it looked fine at the start, because nothing checked.

Feature matching is the binding constraint and it is quadratic-ish in image count -- an
exhaustive matcher compares every pair, so doubling the images roughly quadruples the
work. Past a few hundred images that stops being a scheduling detail and becomes the
reason a job cannot finish at all, which is why chunking is recommended by size rather
than offered as a preference.

The estimates here are deliberately rough and labelled as such. A wrong estimate that
says "this will not fit" costs a user one conversation; no estimate costs them an
afternoon.

    from core.job_sizing import size_job
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Rough working-set cost per image during matching and bundle adjustment, at a typical
# 20 MP frame. Measured from observed peak usage rather than derived: features, their
# descriptors, and the match graph all scale with image count.
MEMORY_PER_IMAGE_MB = 45.0

# Descriptors dominate on disk, and the sparse model plus depth maps follow.
DISK_PER_IMAGE_MB = 120.0

# Above this, exhaustive matching stops being viable and the job needs vocabulary-tree
# or sequential matching plus chunking. Not a hard limit -- a large machine can push
# past it -- which is why it drives a recommendation rather than a refusal.
EXHAUSTIVE_MATCH_LIMIT = 300

# Keep this much memory free for the OS and everything else.
MEMORY_HEADROOM_MB = 2048.0


@dataclass
class JobEstimate:
    """What this job will need, and whether this machine has it."""

    image_count: int
    estimated_memory_mb: float
    estimated_disk_mb: float
    available_memory_mb: float
    available_disk_mb: float
    fits_in_memory: bool
    fits_on_disk: bool
    recommended_chunk_size: int
    chunking_required: bool
    warnings: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def can_run_as_one_job(self) -> bool:
        return self.fits_in_memory and self.fits_on_disk and not self.chunking_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_count": self.image_count,
            "estimated_memory_mb": round(self.estimated_memory_mb, 1),
            "estimated_disk_mb": round(self.estimated_disk_mb, 1),
            "available_memory_mb": round(self.available_memory_mb, 1),
            "available_disk_mb": round(self.available_disk_mb, 1),
            "fits_in_memory": self.fits_in_memory,
            "fits_on_disk": self.fits_on_disk,
            "can_run_as_one_job": self.can_run_as_one_job,
            "chunking_required": self.chunking_required,
            "recommended_chunk_size": self.recommended_chunk_size,
            "warnings": list(self.warnings),
            "note": self.note,
            "estimate_basis": (
                "Rough, from per-image working-set and disk costs at a typical 20 MP "
                "frame. Treat as an order of magnitude, not a budget."
            ),
        }


def available_memory_mb() -> float:
    """Free system memory, or 0.0 when it cannot be determined."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024 * 1024)
    except Exception:  # noqa: BLE001 - psutil absent or unavailable on this platform
        return 0.0


def available_disk_mb(path: str = ".") -> float:
    """Free disk on the volume holding ``path``, or 0.0 when unknown."""
    try:
        import shutil

        return float(shutil.disk_usage(path).free) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def size_job(
    image_count: int,
    *,
    work_dir: str = ".",
    memory_mb: float | None = None,
    disk_mb: float | None = None,
) -> JobEstimate:
    """Estimate what this reconstruction needs and whether it will fit.

    Resources are probed rather than assumed, but can be supplied for planning a job on
    a different machine than the one asking.
    """
    if image_count <= 0:
        raise ValueError("A reconstruction needs at least one image.")

    memory = available_memory_mb() if memory_mb is None else float(memory_mb)
    disk = available_disk_mb(work_dir) if disk_mb is None else float(disk_mb)

    needed_memory = image_count * MEMORY_PER_IMAGE_MB
    needed_disk = image_count * DISK_PER_IMAGE_MB

    warnings: list[str] = []
    # A probe that returned nothing must not read as "plenty of room".
    if memory <= 0:
        warnings.append(
            "Could not read available memory (psutil missing or unsupported here), so "
            "the memory verdict below is unverified."
        )
        fits_memory = True
    else:
        fits_memory = needed_memory + MEMORY_HEADROOM_MB <= memory

    if disk <= 0:
        warnings.append("Could not read free disk space, so the disk verdict is unverified.")
        fits_disk = True
    else:
        fits_disk = needed_disk <= disk

    usable_memory = max(memory - MEMORY_HEADROOM_MB, 0.0)
    by_memory = int(usable_memory // MEMORY_PER_IMAGE_MB) if usable_memory > 0 else image_count
    chunk = max(1, min(EXHAUSTIVE_MATCH_LIMIT, by_memory or image_count))
    chunking_required = image_count > chunk

    if not fits_memory:
        warnings.append(
            f"Estimated peak memory {needed_memory:.0f} MB plus {MEMORY_HEADROOM_MB:.0f} MB "
            f"headroom exceeds the {memory:.0f} MB available. Run in chunks of about "
            f"{chunk} images, or the job will fail partway through with nothing to show."
        )
    if not fits_disk:
        warnings.append(
            f"Estimated {needed_disk:.0f} MB of intermediates exceeds the {disk:.0f} MB "
            "free on the work volume. Free space or point the job elsewhere."
        )
    if image_count > EXHAUSTIVE_MATCH_LIMIT:
        warnings.append(
            f"{image_count} images is past the ~{EXHAUSTIVE_MATCH_LIMIT} where exhaustive "
            "matching stops being viable: it compares every pair, so cost grows with the "
            "square of the count. Use sequential or vocabulary-tree matching."
        )

    if chunking_required:
        note = (
            f"This job should be split. {image_count} images at roughly "
            f"{MEMORY_PER_IMAGE_MB:.0f} MB each exceeds what this machine can hold in one "
            f"pass; chunks of about {chunk} fit, with overlap between chunks so the "
            "sub-models can be merged."
        )
    else:
        note = (
            f"{image_count} images should run as a single job: an estimated "
            f"{needed_memory:.0f} MB of memory and {needed_disk:.0f} MB of intermediates "
            "against what is available here."
        )

    return JobEstimate(
        image_count=image_count,
        estimated_memory_mb=needed_memory,
        estimated_disk_mb=needed_disk,
        available_memory_mb=memory,
        available_disk_mb=disk,
        fits_in_memory=fits_memory,
        fits_on_disk=fits_disk,
        recommended_chunk_size=chunk,
        chunking_required=chunking_required,
        warnings=warnings,
        note=note,
    )


def chunk_images(image_paths: list[str], chunk_size: int, *, overlap: int = 10
                 ) -> list[list[str]]:
    """Split a capture into overlapping chunks.

    Overlap is not optional. Chunks reconstructed independently have no shared geometry,
    so merging them needs images appearing in both to tie the sub-models together;
    without it the result is several disconnected models rather than one survey.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap {overlap} must be smaller than the chunk size {chunk_size}, "
            "otherwise the chunks never advance."
        )
    if len(image_paths) <= chunk_size:
        return [list(image_paths)]

    step = chunk_size - overlap
    chunks: list[list[str]] = []
    start = 0
    while start < len(image_paths):
        chunk = image_paths[start:start + chunk_size]
        if len(chunk) <= overlap and chunks:
            # A tail shorter than the overlap adds no new geometry; fold it back.
            chunks[-1].extend(p for p in chunk if p not in chunks[-1])
            break
        chunks.append(list(chunk))
        if start + chunk_size >= len(image_paths):
            break
        start += step
    return chunks
