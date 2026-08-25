"""Turn boxes a user drew into a detection corpus, or refuse and say what is wrong.

`custom_training.py` already does this for image-level labels: a photograph and the class
someone gave it. That covers "is there a crack in this tile" and nothing else. A user who
wants to find WHERE the defect is has to draw a box, and until this module existed there
was nowhere for such a box to go -- the capability was for someone who already had YOLO
label files.

The refusals are the substance, and they are the same argument as the classification
builder: a corpus is built once and its metric is read many times. By the time anyone
notices that a class had six boxes, or that the same photograph sat in train and val
under two names, the number has been quoted. So this refuses rather than warns.

What is checked here and not there, because boxes can be wrong in ways a class label
cannot:

  * A box outside the image, or with zero area. Usually a click that was meant to be a
    drag. Written to a label file it becomes a target the model cannot ever match.
  * An image with no boxes at all. In detection that is a legitimate negative sample,
    so it is ACCEPTED -- but only when the caller says so explicitly, because the far
    more common cause is an image someone opened and forgot to label, and silently
    training on it teaches the model the defect is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.custom_training import (
    MIN_SAMPLES_PER_CLASS,
    MIN_VALIDATION_PER_CLASS,
    CorpusRefused,
    split_for,
)

DEFAULT_SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass(frozen=True)
class Box:
    """One drawn region, in normalised image coordinates.

    Normalised on purpose. A box in pixels is meaningless once the image is resized,
    and every trainer here resizes; storing pixels would make the corpus depend on the
    resolution of the screen it was drawn on.
    """

    label: str
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise CorpusRefused("A box needs a class label.")
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value != value:  # NaN
                raise CorpusRefused(f"Box {name} must be a number, got {value!r}.")
        if self.width <= 0 or self.height <= 0:
            raise CorpusRefused(
                f"Box for {self.label!r} has zero area ({self.width}x{self.height}). "
                "This is usually a click that was meant to be a drag."
            )
        if self.x < 0 or self.y < 0 or self.x + self.width > 1.0000001 or \
                self.y + self.height > 1.0000001:
            raise CorpusRefused(
                f"Box for {self.label!r} falls outside the image "
                f"(x={self.x}, y={self.y}, w={self.width}, h={self.height}). "
                "Coordinates are normalised, so every edge must lie within 0..1."
            )

    def to_yolo(self, class_index: int) -> str:
        """YOLO's centre-based line. The trainers read this format directly."""
        return (
            f"{class_index} {self.x + self.width / 2:.6f} {self.y + self.height / 2:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass
class LabelledRegion:
    """One image and every box drawn on it."""

    path: Path
    boxes: list[Box] = field(default_factory=list)
    # A deliberate negative: an image a user looked at and confirmed has nothing to find.
    # Distinct from an unlabelled image, which is an accident.
    confirmed_empty: bool = False
    digest: str = ""

    def content_digest(self) -> str:
        if not self.digest:
            self.digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        return self.digest


def _classes_of(regions: Sequence[LabelledRegion]) -> list[str]:
    return sorted({box.label for region in regions for box in region.boxes})


def _deduplicate(regions: Sequence[LabelledRegion]) -> tuple[list[LabelledRegion], int]:
    """Drop images whose BYTES are already present.

    Same reasoning as the classification builder: leakage arrives as one photograph
    saved twice, and a filename comparison misses precisely that case. The first
    occurrence wins, so the boxes kept are the ones drawn on the copy seen first.
    """
    seen: dict[str, LabelledRegion] = {}
    duplicates = 0
    for region in regions:
        digest = region.content_digest()
        if digest in seen:
            duplicates += 1
            continue
        seen[digest] = region
    return list(seen.values()), duplicates


def build_detection_corpus(
    regions: Sequence[LabelledRegion],
    output_dir: str | Path,
    *,
    salt: str = "custom-boxes-v1",
    ratios: Mapping[str, float] = DEFAULT_SPLITS,
    allow_empty_images: bool = False,
) -> dict[str, Any]:
    """Write a YOLO-layout corpus from drawn boxes, or refuse with the reason.

    Returns a report rather than only the paths, because the counts are what tell the
    user whether the thing they just built can support the number a trainer will print.
    """
    if not regions:
        raise CorpusRefused("No labelled images were supplied.")

    unlabelled = [
        region for region in regions
        if not region.boxes and not region.confirmed_empty
    ]
    if unlabelled and not allow_empty_images:
        names = ", ".join(Path(r.path).name for r in unlabelled[:3])
        raise CorpusRefused(
            f"{len(unlabelled)} image(s) carry no boxes and were not marked as "
            f"deliberately empty ({names}). An image someone opened and forgot to "
            "label teaches the model the defect is absent. Mark them empty on purpose "
            "or remove them."
        )

    missing = [region for region in regions if not Path(region.path).is_file()]
    if missing:
        raise CorpusRefused(
            f"{len(missing)} labelled image(s) are not on disk; first: {missing[0].path}"
        )

    kept, duplicates = _deduplicate(regions)
    classes = _classes_of(kept)
    if not classes:
        raise CorpusRefused(
            "Every image is empty, so there is nothing to detect. A corpus of negatives "
            "trains a model that answers 'nothing here' to everything and scores well "
            "doing it."
        )

    box_counts = {name: 0 for name in classes}
    for region in kept:
        for box in region.boxes:
            box_counts[box.label] += 1
    thin = {name: count for name, count in box_counts.items()
            if count < MIN_SAMPLES_PER_CLASS}
    if thin:
        detail = ", ".join(f"{name}: {count}" for name, count in sorted(thin.items()))
        raise CorpusRefused(
            f"Too few boxes to learn from ({detail}). At least {MIN_SAMPLES_PER_CLASS} "
            "per class, or the model memorises them and the metric describes nothing."
        )

    assigned: dict[str, list[LabelledRegion]] = {name: [] for name in ratios}
    for region in kept:
        assigned[split_for(region.content_digest(), salt=salt, ratios=ratios)].append(region)

    validation_boxes = {name: 0 for name in classes}
    for region in assigned.get("val", []):
        for box in region.boxes:
            validation_boxes[box.label] += 1
    unmeasurable = {name: count for name, count in validation_boxes.items()
                    if count < MIN_VALIDATION_PER_CLASS}
    if unmeasurable:
        detail = ", ".join(f"{name}: {count}" for name, count in sorted(unmeasurable.items()))
        raise CorpusRefused(
            f"Not enough held-out boxes to measure ({detail}). Below "
            f"{MIN_VALIDATION_PER_CLASS} the recall for that class is an accident of "
            "which images landed in validation, not a property of the model."
        )

    root = Path(output_dir)
    index = {name: position for position, name in enumerate(classes)}
    counts: dict[str, dict[str, int]] = {}
    for split, members in assigned.items():
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        split_counts = {name: 0 for name in classes}
        for region in members:
            source = Path(region.path)
            target = images_dir / source.name
            if not target.exists():
                target.write_bytes(source.read_bytes())
            lines = [box.to_yolo(index[box.label]) for box in region.boxes]
            (labels_dir / f"{source.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            for box in region.boxes:
                split_counts[box.label] += 1
        counts[split] = {"images": len(members), **split_counts}

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                f"nc: {len(classes)}",
                "names:",
                *[f"  {position}: {name}" for name, position in index.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = {
        "root": str(root),
        "data_yaml": str(data_yaml),
        "classes": classes,
        "counts": counts,
        "total_images": len(kept),
        "total_boxes": sum(box_counts.values()),
        "boxes_per_class": box_counts,
        "duplicates_removed": duplicates,
        "empty_images": sum(1 for region in kept if not region.boxes),
        "reading_note": (
            "Counts are of boxes drawn, not of defects in the world. A model trained "
            "here can only be as good as the labelling, and its metric describes these "
            "images and the sites they came from."
        ),
    }
    (root / "corpus_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def regions_from_payload(payload: Iterable[Mapping[str, Any]]) -> list[LabelledRegion]:
    """Build regions from what the labelling UI posts, refusing malformed geometry."""
    regions: list[LabelledRegion] = []
    for entry in payload:
        path = str(entry.get("path", "")).strip()
        if not path:
            raise CorpusRefused("Every labelled image needs a path.")
        boxes = [
            Box(
                label=str(raw.get("label", "")),
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                width=float(raw.get("width", 0.0)),
                height=float(raw.get("height", 0.0)),
            )
            for raw in entry.get("boxes", []) or []
        ]
        regions.append(
            LabelledRegion(
                path=Path(path),
                boxes=boxes,
                confirmed_empty=bool(entry.get("confirmed_empty", False)),
            )
        )
    return regions
