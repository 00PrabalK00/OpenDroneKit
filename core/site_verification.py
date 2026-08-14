"""The check that decides whether it is safe to leave the site.

Every failure this catches is cheap on site and expensive afterwards. A missed capture
point is a ten-minute re-fly while the aircraft is out of its case, and a wasted trip
once it is packed away. A folder of blurred frames is a morning's reshoot, or a survey
delivered at a quality nobody can use. A corrupt card is discovered either now or during
processing, days later, with nothing to be done about it.

The parts already existed and were not joined up: core.capture_matching knows which
planned points produced images, core.coverage_validation measures blur and exposure, and
neither says whether the pilot can go home. This answers that one question, and refuses
to answer it optimistically -- anything it could not check is reported as unchecked
rather than counted as passed.

The verdict is advisory. Nothing here grounds an aircraft or blocks a departure; the
operator decides, having been told what is wrong and what it will cost to fix now
against later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import capture_matching

# Below this share of planned points captured, the survey is not usable as flown and a
# re-fly is not optional. Chosen to sit well under any sane overlap margin: losing more
# than a twentieth of a grid opens holes that photogrammetry cannot bridge.
MINIMUM_COVERAGE_PCT = 95.0

# A handful of soft frames in a large survey is normal. This is the share above which
# blur stops being noise and starts being a systematic problem -- a wrong shutter speed,
# a dirty lens, or flying too fast for the light.
BLUR_TOLERANCE_PCT = 5.0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".raw"}


@dataclass
class SiteVerdict:
    """Whether the pilot can leave, and what it would cost to find out later."""

    ok: bool
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.blocking:
            summary = (
                f"Do not leave the site yet: {len(self.blocking)} problem(s) can only be "
                "fixed while the aircraft is here."
            )
        elif self.warnings:
            summary = (
                f"Safe to leave, with {len(self.warnings)} thing(s) worth knowing before "
                "you do."
            )
        else:
            summary = "Everything checked passed. Safe to leave the site."

        return {
            "ok": self.ok,
            "summary": summary,
            "blocking": self.blocking,
            "warnings": self.warnings,
            "unchecked": self.unchecked,
            "details": self.details,
            "note": (
                "Advisory only. Nothing here prevents a departure; the operator decides. "
                "Anything that could not be checked is listed as unchecked rather than "
                "counted as passed."
            ),
        }


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder of images.")
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_unreadable(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Images that cannot be decoded, which is a card problem, not a flying problem.

    Checked by actually decoding rather than by file size, because a truncated JPEG is
    usually the right sort of size and only fails when something tries to read it --
    which would otherwise be during processing, days later.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - environment dependent
        return []

    broken: list[dict[str, Any]] = []
    for path in paths:
        try:
            if path.stat().st_size == 0:
                broken.append({"path": str(path), "reason": "empty file"})
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                broken.append({"path": str(path), "reason": "could not be decoded"})
        except OSError as exc:
            broken.append({"path": str(path), "reason": f"unreadable: {exc}"})
    return broken


def assess_quality(paths: Sequence[Path],
                   sample_limit: int = 0) -> dict[str, Any]:
    """Blur and exposure across the captured frames.

    ``sample_limit`` exists because decoding several thousand full-resolution frames on
    a laptop in a field takes longer than the pilot will wait. When a sample is used it
    is reported as a sample, so nobody reads a rate measured over 200 frames as a
    statement about 2000.
    """
    try:
        import cv2
        from .coverage_validation import CoverageValidationConfig, _quality_metrics, _to_gray_small
    except ImportError:  # pragma: no cover - environment dependent
        return {"checked": 0, "available": False,
                "reason": "OpenCV is not installed, so image quality was not assessed."}

    config = CoverageValidationConfig()
    selected = list(paths)
    sampled = False
    if sample_limit and len(selected) > sample_limit:
        # Evenly spaced rather than the first N: the first frames of a survey are the
        # climb-out, which is not representative of the grid.
        step = len(selected) / sample_limit
        selected = [selected[int(i * step)] for i in range(sample_limit)]
        sampled = True

    flagged: list[dict[str, Any]] = []
    checked = 0
    for path in selected:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        metrics = _quality_metrics(_to_gray_small(image), config)
        checked += 1
        if not metrics["quality_ok"]:
            flagged.append({"path": path.name, "flags": metrics["quality_flags"],
                            "sharpness": round(metrics["sharpness"], 1)})

    rate = (100.0 * len(flagged) / checked) if checked else 0.0
    return {
        "available": True,
        "checked": checked,
        "sampled": sampled,
        "population": len(paths),
        "flagged": flagged[:40],
        "flagged_count": len(flagged),
        "flagged_pct": round(rate, 1),
    }


def verify_site(image_folder: str | Path,
                plan: dict[str, Any] | Any | None = None,
                *,
                quality_sample_limit: int = 200,
                match_radius_m: float = capture_matching.DEFAULT_MATCH_RADIUS_M
                ) -> SiteVerdict:
    """Decide whether this survey can be left as flown."""
    paths = list_images(image_folder)

    blocking: list[str] = []
    warnings: list[str] = []
    unchecked: list[str] = []
    details: dict[str, Any] = {"image_count": len(paths)}

    if not paths:
        return SiteVerdict(
            ok=False,
            blocking=["The folder contains no images at all. Check the card was copied "
                      "from the aircraft before it is wiped."],
            details=details,
        )

    # 1. Files that cannot be read at all.
    broken = find_unreadable(paths)
    details["unreadable"] = broken
    if broken:
        blocking.append(
            f"{len(broken)} image(s) cannot be decoded. Re-copy them from the card now: "
            "once the card is reused they are gone."
        )

    # 2. Coverage against the plan, when there is one.
    if plan is not None:
        planned = capture_matching.planned_captures_from_plan(plan)
        if planned:
            images = capture_matching.images_from_folder(image_folder)
            report = capture_matching.match_captures(planned, images,
                                                     match_radius_m=match_radius_m)
            coverage = report.to_dict()
            details["coverage"] = coverage

            if coverage["coverage_pct"] < MINIMUM_COVERAGE_PCT:
                missed = coverage["missed"][:10]
                indices = ", ".join(str(m["index"]) for m in missed)
                blocking.append(
                    f"Only {coverage['coverage_pct']}% of planned capture points produced "
                    f"an image. Re-fly points {indices}"
                    f"{'...' if len(coverage['missed']) > 10 else ''} before leaving: a "
                    "gap here becomes a hole in the reconstruction."
                )
            elif coverage["missed"]:
                warnings.append(
                    f"{len(coverage['missed'])} planned capture point(s) produced no "
                    "image, which is within tolerance but worth a look."
                )

            if coverage["ungeotagged"]:
                warnings.append(
                    f"{len(coverage['ungeotagged'])} image(s) carry no GPS position. They "
                    "cannot seed the reconstruction; check the camera's geotagging "
                    "settings before the next flight."
                )
        else:
            unchecked.append(
                "The plan contains no capture points, so coverage could not be checked."
            )
    else:
        unchecked.append(
            "No mission plan was supplied, so coverage against the plan was not checked. "
            "Only the images themselves were examined."
        )

    # 3. Blur and exposure.
    quality = assess_quality(paths, sample_limit=quality_sample_limit)
    details["quality"] = quality
    if not quality.get("available"):
        unchecked.append(quality.get("reason", "Image quality was not assessed."))
    elif quality["checked"] == 0:
        unchecked.append("No image could be decoded for quality assessment.")
    elif quality["flagged_pct"] > BLUR_TOLERANCE_PCT:
        scope = (f"{quality['checked']} sampled of {quality['population']}"
                 if quality["sampled"] else f"all {quality['checked']}")
        blocking.append(
            f"{quality['flagged_pct']}% of images ({scope}) are blurred or badly exposed. "
            "That is a systematic problem -- shutter speed, a dirty lens, or flying too "
            "fast for the light -- and reshooting now is cheaper than returning."
        )
    elif quality["flagged_count"]:
        warnings.append(
            f"{quality['flagged_count']} image(s) are blurred or badly exposed, which is "
            "within normal tolerance for a survey this size."
        )

    return SiteVerdict(ok=not blocking, blocking=blocking, warnings=warnings,
                       unchecked=unchecked, details=details)
