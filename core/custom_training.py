"""Turn a user's labelled images into a corpus, or refuse and say what is missing.

A user who labels their own defects is doing the most valuable and most fragile thing in
this toolkit. Valuable because a model trained on their site beats a general one.
Fragile because the failure modes are silent: a corpus can be built, trained on, and
reported without anyone noticing that one class had four examples, or that the same
photograph sits in both training and validation, or that the number everyone is quoting
came from eleven images.

So this refuses more than it builds. Every refusal below corresponds to a way a corpus
produces a model whose reported metric means nothing:

  * A class with too few examples cannot be learned, and a class with no validation
    examples cannot be measured. Both are refused rather than warned about, because a
    warning at build time is invisible by the time the metric is read.
  * The same image in two splits is leakage. The model is scored on what it memorised,
    the number is inflated, and nothing about the run looks wrong. Detected by content
    digest, not filename -- a copy under another name is the common case.
  * A single-class corpus is refused for classification. A classifier with one class
    scores 100 per cent by answering the only thing it knows.

Splitting is deterministic and STRATIFIED: each class is divided separately, so every
class keeps its share of every split. A whole-corpus hash is deterministic but not
proportional, and on the small corpora a user actually labels that bites -- a 30-image
class drew ONE validation example in testing, which would have been refused as "label
more" when the images were sufficient and the split was not. Ordering is by salted digest
so the assignment is stable across runs and machines.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class CorpusRefused(ValueError):
    """The labelled data cannot produce a corpus anyone should train on."""


# Below this a class cannot be learned in any meaningful sense. It is deliberately not
# a warning: a corpus is built once and its metric is read many times, and by then the
# warning has scrolled away.
MIN_SAMPLES_PER_CLASS = 10
# A class needs at least this many held-out examples for its recall to mean anything.
# One validation example gives a recall of either 0.0 or 1.0.
MIN_VALIDATION_PER_CLASS = 3

DEFAULT_SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass
class LabelledImage:
    """One user-supplied image and the label they gave it."""

    path: Path
    label: str
    digest: str = ""

    def content_digest(self) -> str:
        """Hash of the bytes, not the name.

        Leakage usually arrives as the same photograph saved twice under different
        names, so a filename comparison misses exactly the case that matters.
        """
        if not self.digest:
            self.digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        return self.digest


@dataclass
class CorpusReport:
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    duplicates_removed: int = 0
    total_samples: int = 0
    classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "counts": self.counts,
            "total_samples": self.total_samples,
            "duplicates_removed": self.duplicates_removed,
            "warnings": self.warnings,
            "reading_note": (
                "Counts are of labelled images, not of defects in the world. A model "
                "trained here can only be as good as the labelling, and its metric "
                "describes these images and the sites they came from."
            ),
        }


def split_for(digest: str, *, salt: str, ratios: Mapping[str, float] = DEFAULT_SPLITS) -> str:
    """Deterministic split from the image's content digest.

    Keyed on content rather than an index, so adding photographs to a corpus does not
    reshuffle what was previously held out -- which would quietly invalidate every
    metric measured before the addition.
    """
    total = sum(ratios.values())
    if total <= 0:
        raise CorpusRefused("Split ratios must sum to more than zero.")
    position = int(hashlib.sha1(f"{salt}/{digest}".encode("utf-8")).hexdigest(), 16) % 10_000
    threshold = 0.0
    for name, ratio in ratios.items():
        threshold += (ratio / total) * 10_000
        if position < threshold:
            return name
    return list(ratios)[-1]


def stratified_split(
    samples: Sequence[LabelledImage],
    *,
    salt: str,
    ratios: Mapping[str, float] = DEFAULT_SPLITS,
) -> dict[str, list[LabelledImage]]:
    """Split within each class, so every class keeps its share of every split.

    A plain hash over the whole corpus is deterministic but not proportional, and on the
    small corpora a user actually labels the variance bites: in testing, a 30-image class
    received ONE validation example under a whole-corpus hash. The recall of that class
    would then be 0.0 or 1.0, and the corpus would be refused for a reason -- "label
    more" -- that misdiagnoses the problem. The images were sufficient; the split was not.

    Ordering is by digest rather than by filename or arrival, so the assignment is stable
    across runs and machines, and adding images reshuffles only within the affected class.
    """
    by_class: dict[str, list[LabelledImage]] = {}
    for sample in samples:
        by_class.setdefault(sample.label, []).append(sample)

    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise CorpusRefused("Split ratios must sum to more than zero.")

    splits: dict[str, list[LabelledImage]] = {name: [] for name in ratios}
    names = list(ratios)
    for label in sorted(by_class):
        ordered = sorted(by_class[label], key=lambda s: hashlib.sha1(
            f"{salt}/{s.content_digest()}".encode("utf-8")).hexdigest())
        count = len(ordered)
        # Largest-remainder allocation, so the counts sum to exactly `count` rather than
        # losing or duplicating a sample to rounding.
        exact = [(ratios[name] / total_ratio) * count for name in names]
        base = [int(value) for value in exact]
        remainder = count - sum(base)
        for index in sorted(range(len(names)), key=lambda i: exact[i] - base[i], reverse=True)[:remainder]:
            base[index] += 1
        cursor = 0
        for name, take in zip(names, base):
            splits[name].extend(ordered[cursor:cursor + take])
            cursor += take
    return splits


def _deduplicate(samples: Sequence[LabelledImage]) -> tuple[list[LabelledImage], int]:
    """Drop byte-identical repeats, keeping the first.

    A duplicate is not merely wasted space: left in, it lands in two splits and becomes
    leakage. Removing it here is why the leakage check further down can be strict.
    """
    seen: dict[str, LabelledImage] = {}
    removed = 0
    conflicting: list[str] = []
    for sample in samples:
        digest = sample.content_digest()
        existing = seen.get(digest)
        if existing is None:
            seen[digest] = sample
            continue
        removed += 1
        if existing.label != sample.label:
            conflicting.append(f"{Path(existing.path).name} / {Path(sample.path).name}")
    if conflicting:
        # Same pixels, two labels. Guessing which is right would bake a labelling
        # mistake into the corpus and every model trained on it.
        raise CorpusRefused(
            "The same image appears with different labels: "
            + "; ".join(conflicting[:5])
            + ". Resolve the disagreement before building a corpus from it."
        )
    return list(seen.values()), removed


def build_custom_corpus(
    samples: Iterable[LabelledImage],
    *,
    salt: str,
    ratios: Mapping[str, float] = DEFAULT_SPLITS,
    min_per_class: int = MIN_SAMPLES_PER_CLASS,
    min_validation_per_class: int = MIN_VALIDATION_PER_CLASS,
) -> tuple[dict[str, list[LabelledImage]], CorpusReport]:
    """Assign labelled images to splits, refusing anything unmeasurable."""
    materialised = list(samples)
    if not materialised:
        raise CorpusRefused("No labelled images were supplied.")

    for sample in materialised:
        if not str(sample.label).strip():
            raise CorpusRefused(f"{Path(sample.path).name} has no label.")
        if not Path(sample.path).is_file():
            raise CorpusRefused(f"{sample.path} does not exist.")

    unique, removed = _deduplicate(materialised)

    per_class = Counter(sample.label for sample in unique)
    if len(per_class) < 2:
        raise CorpusRefused(
            "A corpus needs at least two classes. A model trained on one class answers "
            "the only thing it knows and scores perfectly while learning nothing."
        )

    thin = {name: count for name, count in per_class.items() if count < min_per_class}
    if thin:
        raise CorpusRefused(
            "These classes have too few examples to learn: "
            + ", ".join(f"{name} ({count})" for name, count in sorted(thin.items()))
            + f". At least {min_per_class} each are needed. Label more, or merge the "
            "class into one it belongs with."
        )

    splits = stratified_split(unique, salt=salt, ratios=ratios)

    validation_counts = Counter(s.label for s in splits.get("val", []))
    unmeasurable = [
        name for name in per_class
        if validation_counts.get(name, 0) < min_validation_per_class
    ]
    if unmeasurable:
        raise CorpusRefused(
            "These classes have too few validation examples to be measured: "
            + ", ".join(sorted(unmeasurable))
            + f". At least {min_validation_per_class} each are needed. A class with one "
            "held-out example has a recall of either 0.0 or 1.0, which is not a "
            "measurement."
        )

    # Leakage is impossible after deduplication by digest, and asserted anyway: this is
    # the failure that inflates a number without looking like anything is wrong.
    seen_digests: dict[str, str] = {}
    for split_name, entries in splits.items():
        for sample in entries:
            digest = sample.content_digest()
            if digest in seen_digests and seen_digests[digest] != split_name:
                raise CorpusRefused(
                    f"{Path(sample.path).name} appears in both {seen_digests[digest]} "
                    f"and {split_name}. The model would be scored on what it memorised."
                )
            seen_digests[digest] = split_name

    report = CorpusReport(
        counts={
            split_name: dict(sorted(Counter(s.label for s in entries).items()))
            for split_name, entries in splits.items()
        },
        duplicates_removed=removed,
        total_samples=len(unique),
        classes=sorted(per_class),
    )
    if removed:
        report.warnings.append(
            f"{removed} duplicate image(s) removed. Left in they would have landed in "
            "two splits and inflated the score."
        )
    smallest = min(per_class.values())
    largest = max(per_class.values())
    if largest >= smallest * 10:
        # Not refused: imbalance is a fact of defect data, and refusing it would block
        # the realistic case. Named, because a mean metric will hide the rare class.
        report.warnings.append(
            f"Class balance is {largest}:{smallest} between the largest and smallest "
            "class. Read per-class recall rather than any averaged figure -- the rare "
            "class is usually the one the model exists to find."
        )
    if report.total_samples < 100:
        report.warnings.append(
            f"{report.total_samples} images is a small corpus. Treat any metric from it "
            "as a first indication rather than a settled result."
        )
    return splits, report
