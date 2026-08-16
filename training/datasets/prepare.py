"""Normalise the downloaded datasets into the two layouts the trainers consume.

Every raw dataset arrives in its own shape: DeepCrack ships a nested zip of paired
image/label PNGs, CrackForest ships BSDS ``.seg`` run-length text, ELPV ships a
whitespace-separated CSV of defect probabilities, and the Roboflow exports already
sit in YOLO form. This module converts each of them into one of:

``segmentation``
    ``<out>/<task>/<split>/images/<id>.png`` plus ``masks/<id>.png``, where the mask
    is single-channel 0/255. Nothing is resized or tiled here -- the trainer does
    its own cropping, so prepared data stays lossless.

``classification``
    ``<out>/<task>/<split>/<class_name>/<id>.png``.

``detection``
    ``<out>/<task>/<split>/images`` + ``labels`` in YOLO txt form, with a merged
    ``data.yaml`` listing the union of class names across the merged sources.

Splits are deterministic: a dataset that carries its own train/test division keeps
it, and anything else is bucketed by a salted SHA-1 of the sample id so re-running
prepare never shuffles a sample across the train/val boundary and silently leaks
validation data into a later training run.

    python -m training.datasets.prepare --list
    python -m training.datasets.prepare crack solar
    python -m training.datasets.prepare all --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from xml.etree import ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

try:  # Pillow is a hard requirement of the project, but keep the failure legible.
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "training.datasets.prepare needs Pillow. Install with: pip install pillow"
    ) from exc

from .registry import DATASETS

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
PREPARED_ROOT = DATA_ROOT / "prepared"
WORK_ROOT = DATA_ROOT / "_work"

# Held-out fractions for datasets that do not ship their own split. The remainder
# goes to train.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.10

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------------------
# deterministic splitting
# --------------------------------------------------------------------------


def deterministic_split(sample_id: str, *, salt: str) -> str:
    """Assign a sample to train/val/test from a hash of its id.

    Hashing rather than shuffling means the assignment survives re-runs, added
    samples, and a different filesystem ordering. Without that, a second prepare
    pass could move a sample from val into train and quietly invalidate every
    metric produced before it.
    """
    digest = hashlib.sha1(f"{salt}/{sample_id}".encode("utf-8")).hexdigest()
    # 8 hex chars -> 32 bits is far more resolution than the fractions need.
    position = int(digest[:8], 16) / 0xFFFFFFFF
    if position < TEST_FRACTION:
        return "test"
    if position < TEST_FRACTION + VAL_FRACTION:
        return "val"
    return "train"


# --------------------------------------------------------------------------
# sample records
# --------------------------------------------------------------------------


@dataclass
class SegSample:
    """One image with a binary mask, ready to be written out."""

    sample_id: str
    image_path: Path
    mask: np.ndarray | Path
    split: str | None = None


@dataclass
class ClsSample:
    sample_id: str
    image_path: Path
    class_name: str
    split: str | None = None


@dataclass
class DetSample:
    sample_id: str
    image_path: Path
    label_path: Path | None
    class_names: tuple[str, ...]
    split: str | None = None


@dataclass
class PreparedTask:
    """Accumulates counts for the manifest."""

    task: str
    kind: str
    root: Path
    counts: dict[str, int] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    licenses: dict[str, str] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in IMAGE_SUFFIXES:
            yield path


def _binarise(array: np.ndarray) -> np.ndarray:
    """Collapse an arbitrary label image to a 0/255 uint8 mask.

    Ground-truth masks in these corpora are variously binary 0/1, 0/255, greyscale
    with antialiased edges, or RGB. Anything above mid-grey counts as foreground.
    """
    if array.ndim == 3:
        array = array[..., :3].max(axis=2)
    return ((array > 127) * 255).astype(np.uint8)


def _safe_extract(archive: Path, destination: Path) -> Path:
    """Extract a zip, refusing entries that would escape the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    with zipfile.ZipFile(archive) as zf:
        resolved_root = destination.resolve()
        for member in zf.namelist():
            target = (destination / member).resolve()
            if not str(target).startswith(str(resolved_root)):
                raise ValueError(f"Archive {archive.name} contains an unsafe path: {member}")
        zf.extractall(destination)
    marker.write_text("ok", encoding="utf-8")
    return destination


def _write_segmentation(samples: Iterable[SegSample], task: PreparedTask, salt: str) -> None:
    for sample in samples:
        split = sample.split or deterministic_split(sample.sample_id, salt=salt)
        image_dir = task.root / split / "images"
        mask_dir = task.root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
            rgb.save(image_dir / f"{sample.sample_id}.png")
            size = rgb.size

        if isinstance(sample.mask, Path):
            with Image.open(sample.mask) as raw:
                mask_array = _binarise(np.asarray(raw))
        else:
            mask_array = _binarise(sample.mask)

        mask_image = Image.fromarray(mask_array, mode="L")
        if mask_image.size != size:
            # Nearest neighbour: a binary mask must not gain interpolated grey edges.
            mask_image = mask_image.resize(size, Image.NEAREST)
        mask_image.save(mask_dir / f"{sample.sample_id}.png")

        task.counts[split] = task.counts.get(split, 0) + 1


def _write_classification(samples: Iterable[ClsSample], task: PreparedTask, salt: str) -> None:
    """Sort images into per-class split folders.

    Files already in a format the loaders read are copied rather than re-encoded.
    SDNET2018 alone is 56k JPEG tiles; transcoding those to PNG costs hours of CPU
    and multiplies the on-disk size for no benefit, since classification training
    never needs lossless pixels.
    """
    classes: set[str] = set()
    for sample in samples:
        split = sample.split or deterministic_split(sample.sample_id, salt=salt)
        out_dir = task.root / split / sample.class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = sample.image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            shutil.copy2(sample.image_path, out_dir / f"{sample.sample_id}{suffix}")
        else:
            with Image.open(sample.image_path) as img:
                img.convert("RGB").save(out_dir / f"{sample.sample_id}.png")

        classes.add(sample.class_name)
        task.counts[split] = task.counts.get(split, 0) + 1
    task.class_names = sorted(classes)


def _voc_to_yolo(
    sample: DetSample,
    merged: list[str],
    index_of: dict[str, int],
) -> list[str]:
    """Convert one Pascal VOC annotation to YOLO lines, dropping unlisted classes.

    RDD2022 and PVEL-AD both ship VOC xml rather than YOLO text. VOC boxes are absolute
    pixel corners and YOLO wants a normalised centre and size, so the image dimensions
    have to come from the annotation's own ``<size>`` element -- reading them from the
    image file instead would silently disagree wherever the two differ, and a box
    normalised against the wrong width lands somewhere else entirely.

    A class not in ``sample.class_names`` is dropped. That is the mechanism by which the
    long-tail PVEL-AD defects and the country-specific RDD2022 damage types leave the
    corpus, so it has to stay a drop and never a fallback to class zero.
    """
    if not sample.label_path or not sample.label_path.exists():
        return []
    try:
        root = ElementTree.parse(sample.label_path).getroot()
    except ElementTree.ParseError:
        return []

    size = root.find("size")
    width = float(size.findtext("width") or 0) if size is not None else 0.0
    height = float(size.findtext("height") or 0) if size is not None else 0.0
    if width <= 0 or height <= 0:
        return []

    allowed = set(sample.class_names)
    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in allowed:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin") or 0)
            ymin = float(box.findtext("ymin") or 0)
            xmax = float(box.findtext("xmax") or 0)
            ymax = float(box.findtext("ymax") or 0)
        except ValueError:
            continue
        # Some RDD2022 boxes run a pixel or two past the stated image edge.
        xmin, xmax = max(0.0, min(xmin, xmax)), min(width, max(xmin, xmax))
        ymin, ymax = max(0.0, min(ymin, ymax)), min(height, max(ymin, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        if name not in index_of:
            merged.append(name)
            merged.sort()
            index_of.clear()
            index_of.update({value: i for i, value in enumerate(merged)})
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{index_of[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _write_detection(samples: Iterable[DetSample], task: PreparedTask, salt: str) -> None:
    """Write YOLO-format detection data, remapping class ids into a merged space.

    Each source Roboflow export numbers its classes from zero against its own
    ``data.yaml``. Merging two exports without remapping would silently relabel
    every box in the second one, so class indices are rewritten against the union
    of names as sources are consumed.

    Those indices are assigned in sorted name order, and that detail is the whole
    point. They used to be assigned in encounter order -- whatever sequence the
    filesystem happened to yield samples in -- which meant the same adapter over the
    same data produced different class numbering on different machines. PVEL-AD built
    on Linux put finger at 1 and crack at 2; the same command on Windows swapped them.
    Weights trained against one corpus and evaluated against the other reported 0.002
    and 0.014 for exactly those two classes and sensible numbers for the other six,
    which reads as "the model cannot detect its two most common defects" rather than
    as a numbering bug. Sorting makes the mapping a property of the class names alone,
    so any machine that prepares this corpus agrees on what class 1 means.
    """
    merged: list[str] = sorted(task.class_names)
    index_of = {name: i for i, name in enumerate(merged)}

    for sample in samples:
        split = sample.split or deterministic_split(sample.sample_id, salt=salt)
        image_dir = task.root / split / "images"
        label_dir = task.root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(sample.image_path, image_dir / f"{sample.sample_id}{sample.image_path.suffix}")

        lines: list[str] = []
        if sample.label_path is not None and sample.label_path.suffix.lower() == ".xml":
            lines = _voc_to_yolo(sample, merged, index_of)
        elif sample.label_path is not None and sample.label_path.exists():
            for raw in sample.label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.split()
                if len(parts) < 5:
                    continue
                local_index = int(float(parts[0]))
                if local_index >= len(sample.class_names):
                    continue
                name = sample.class_names[local_index]
                if not name:
                    # A class the dataset declared and the adapter dropped: a duplicate
                    # label under a second name, a negative class, or an export artefact.
                    continue
                if name not in index_of:
                    # A name the task did not declare. Appending it here would make the
                    # numbering depend on which sample happened to introduce it, which
                    # is the non-determinism this function exists to avoid.
                    merged.append(name)
                    merged.sort()
                    index_of = {value: i for i, value in enumerate(merged)}
                lines.append(" ".join([str(index_of[name]), *parts[1:]]))
        (label_dir / f"{sample.sample_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        task.counts[split] = task.counts.get(split, 0) + 1

    task.class_names = merged
    yaml_lines = [
        f"path: {task.root.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(merged)}",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(merged)],
    ]
    (task.root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# per-dataset adapters
# --------------------------------------------------------------------------


def adapt_deepcrack(raw: Path) -> Iterator[SegSample]:
    """DeepCrack: 539 paired image/label PNGs inside a nested zip.

    The repository checkout only contains ``dataset/DeepCrack.zip``; the actual
    pairs live one level deeper, split into train_img/train_lab and
    test_img/test_lab. That native split is preserved -- the published benchmark
    numbers are reported against it, so re-splitting would make our metrics
    incomparable with the literature.
    """
    archive = raw / "dataset" / "DeepCrack.zip"
    if not archive.exists():
        return
    extracted = _safe_extract(archive, WORK_ROOT / "deepcrack")

    for image_dir_name, label_dir_name, split in (
        ("train_img", "train_lab", "train"),
        ("test_img", "test_lab", "test"),
    ):
        image_dir = extracted / image_dir_name
        label_dir = extracted / label_dir_name
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for image_path in _iter_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.png"
            if not label_path.exists():
                matches = list(label_dir.glob(f"{image_path.stem}.*"))
                if not matches:
                    continue
                label_path = matches[0]
            # DeepCrack's own train split is large enough to carve a val set from;
            # the published test set is left untouched.
            resolved = split
            if split == "train":
                bucket = deterministic_split(image_path.stem, salt="deepcrack-val")
                resolved = "val" if bucket == "val" else "train"
            yield SegSample(
                sample_id=f"deepcrack_{image_path.stem}",
                image_path=image_path,
                mask=label_path,
                split=resolved,
            )


def _parse_bsds_seg(path: Path, *, crack_label: int = 1) -> np.ndarray | None:
    """Decode a BSDS ``.seg`` run-length file into a binary mask.

    Format: a text header terminated by a bare ``data`` line, then rows of
    ``<label> <row> <col_start> <col_end>`` with both column bounds inclusive.
    CrackForest uses label 1 for crack (verified: ~1.2% of pixels on 001.seg).
    """
    width = height = 0
    in_data = False
    mask: np.ndarray | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not in_data:
            if stripped == "data":
                if not width or not height:
                    return None
                mask = np.zeros((height, width), dtype=np.uint8)
                in_data = True
                continue
            parts = stripped.split()
            if len(parts) == 2 and parts[0] == "width":
                width = int(parts[1])
            elif len(parts) == 2 and parts[0] == "height":
                height = int(parts[1])
            continue

        parts = stripped.split()
        if len(parts) != 4 or mask is None:
            continue
        label, row, col_start, col_end = (int(value) for value in parts)
        if label != crack_label or not (0 <= row < height):
            continue
        mask[row, max(0, col_start) : min(width, col_end + 1)] = 255

    return mask


def adapt_crackforest(raw: Path) -> Iterator[SegSample]:
    """CrackForest: 118 road images with BSDS run-length ground truth.

    The ``.seg`` files are used rather than the ``.mat`` ones so the adapter has no
    scipy dependency.
    """
    image_dir = raw / "image"
    seg_dir = raw / "seg"
    if not image_dir.is_dir() or not seg_dir.is_dir():
        return
    for image_path in _iter_images(image_dir):
        seg_path = seg_dir / f"{image_path.stem}.seg"
        if not seg_path.exists():
            continue
        mask = _parse_bsds_seg(seg_path)
        if mask is None or not mask.any():
            continue
        yield SegSample(
            sample_id=f"crackforest_{image_path.stem}",
            image_path=image_path,
            mask=mask,
        )


def adapt_crack_segmentation_combined(raw: Path) -> Iterator[SegSample]:
    """khanhha/crack_segmentation: source code only.

    The repository does not vendor its corpus -- the images sit behind a Google
    Drive link -- so the checkout yields no image/mask pairs. It is kept in the
    catalogue for its ``test_imgs`` sanity images but contributes no training data.
    """
    return
    yield  # pragma: no cover - makes the function a generator


def adapt_crack_segmentation_kaggle(raw: Path) -> Iterator[SegSample]:
    """11,298 crack image/mask pairs aggregating six public corpora.

    The archive carries both a flat ``images``/``masks`` pair holding everything and
    a ``train``/``test`` division of the same files. The native split is used, since
    reading the flat directories as well would duplicate every sample and put the
    same image on both sides of the split.
    """
    root = raw / "crack_segmentation_dataset"
    if not root.is_dir():
        candidates = [p for p in raw.iterdir() if p.is_dir() and (p / "train").is_dir()]
        if not candidates:
            return
        root = candidates[0]

    for source_split, target_split in (("train", "train"), ("test", "test")):
        image_dir = root / source_split / "images"
        mask_dir = root / source_split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        for image_path in _iter_images(image_dir):
            mask_path = mask_dir / f"{image_path.stem}.jpg"
            if not mask_path.exists():
                matches = list(mask_dir.glob(f"{image_path.stem}.*"))
                if not matches:
                    continue
                mask_path = matches[0]
            resolved = target_split
            if target_split == "train":
                bucket = deterministic_split(image_path.stem, salt="cskaggle-val")
                resolved = "val" if bucket == "val" else "train"
            yield SegSample(
                sample_id=f"cskaggle_{image_path.stem}",
                image_path=image_path,
                mask=mask_path,
                split=resolved,
            )


def adapt_elpv(raw: Path) -> Iterator[ClsSample]:
    """ELPV: 2624 EL solar cell images labelled by defect probability.

    ``labels.csv`` is whitespace separated: ``<relative path> <probability> <type>``
    with probability in {0.0, 0.333, 0.667, 1.0}. Those four levels are kept as
    ordinal classes rather than thresholded to binary, because the 0.333 tier is
    exactly the ambiguous-defect case an inspection tool needs to flag for review.
    """
    data_dir = raw / "src" / "elpv_dataset" / "data"
    labels_csv = data_dir / "labels.csv"
    if not labels_csv.exists():
        return

    tiers = {0.0: "none", 0.3333: "possible", 0.6667: "probable", 1.0: "certain"}

    for line in labels_csv.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        relative, probability_text, cell_type = parts[0], parts[1], parts[2]
        image_path = data_dir / relative
        if not image_path.exists():
            continue
        probability = float(probability_text)
        nearest = min(tiers, key=lambda level: abs(level - probability))
        yield ClsSample(
            sample_id=f"elpv_{cell_type}_{image_path.stem}",
            image_path=image_path,
            class_name=tiers[nearest],
        )


def adapt_infrared_solar_modules(raw: Path) -> Iterator[ClsSample]:
    """Raptor Maps InfraredSolarModules: 20,000 IR module crops, 12 anomaly classes.

    This is the only openly licensed source found that carries the anomaly classes the
    solar pack actually declares -- hot spots, cracking, offline modules, soiling -- as
    named classes rather than as a defect probability. The classes are kept exactly as
    published rather than collapsed into the declared schema here, because a mapping is
    a decision about what counts as which defect and belongs where a reviewer can see
    it, not buried in a corpus adapter.

    Two properties of this set matter downstream and are not fixed here. The crops are
    24x40 pixels, so anything trained on them is a per-module classifier and not a
    localiser. And the distribution is severely imbalanced -- 10,000 No-Anomaly against
    249 Hot-Spot -- so a model trained without class weighting will score well by
    predicting "normal" and find nothing.
    """
    root = raw / "InfraredSolarModules"
    metadata_path = root / "module_metadata.json"
    if not metadata_path.exists():
        # The GitHub archive ships an inner zip; extract it once, in place.
        inner = next(raw.glob("*InfraredSolarModules.zip"), None)
        if inner is None:
            return
        _safe_extract(inner, raw)
    if not metadata_path.exists():
        return

    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, record in records.items():
        relative = str(record.get("image_filepath", ""))
        anomaly = str(record.get("anomaly_class", "")).strip()
        if not relative or not anomaly:
            continue
        image_path = root / relative
        if not image_path.exists():
            continue
        yield ClsSample(
            sample_id=f"irsm_{key}",
            image_path=image_path,
            class_name=_clean_class_name(anomaly),
        )


def adapt_sdnet2018(raw: Path) -> Iterator[ClsSample]:
    """SDNET2018: 56k concrete tiles under Decks / Pavements / Walls.

    Each surface folder splits into ``Cracked`` and ``Non-cracked``. Only the binary
    label is kept; the surface type is folded into the sample id so it stays
    traceable. (The original release used two-letter CD/UD-style folder names; the
    Kaggle mirror renamed them, so both spellings are matched.)
    """
    labels = {
        "cracked": "cracked",
        "non-cracked": "uncracked",
        "noncracked": "uncracked",
        "uncracked": "uncracked",
    }
    for surface_dir in sorted(p for p in raw.rglob("*") if p.is_dir()):
        name = surface_dir.name
        key = name.lower()
        if key in labels:
            class_name = labels[key]
            tag = f"{surface_dir.parent.name}_{name}"
        elif len(name) == 2 and name[0] in {"C", "U"} and name[1] in {"D", "P", "W"}:
            class_name = "cracked" if name[0] == "C" else "uncracked"
            tag = name
        else:
            continue
        for image_path in _iter_images(surface_dir):
            yield ClsSample(
                sample_id=f"sdnet_{tag}_{image_path.stem}",
                image_path=image_path,
                class_name=class_name,
            )


def adapt_surface_crack(raw: Path) -> Iterator[ClsSample]:
    """Surface crack detection: ``Positive`` / ``Negative`` folders of 227px tiles."""
    for folder, class_name in (("Positive", "cracked"), ("Negative", "uncracked")):
        for candidate in raw.rglob(folder):
            if not candidate.is_dir():
                continue
            for image_path in _iter_images(candidate):
                yield ClsSample(
                    sample_id=f"surfacecrack_{class_name}_{image_path.stem}",
                    image_path=image_path,
                    class_name=class_name,
                )


def _clean_class_name(name: str) -> str:
    """Normalise a class name from a YOLO data.yaml.

    Two things need removing. Inline list form (``names: ['a', 'b']``) leaves the
    surrounding quotes on each entry, and CODEBRIM's mirror bakes the class index
    into the label itself ("0 Efflorescence", "1 CorrosionStain"). The number is a
    leftover from however that project was imported, not part of the label, and
    keeping either makes every downstream metrics table read badly.
    """
    text = str(name).strip().strip("'\"").strip()
    cleaned = re.sub(r"^\d+\s+", "", text)
    return cleaned or text


def read_yolo_license(root: Path) -> str:
    """Recover the licence Roboflow records inside the export's data.yaml."""
    candidate = root / "data.yaml"
    if not candidate.exists():
        return ""
    for line in candidate.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("license:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return ""


def _read_yolo_class_names(root: Path) -> tuple[str, ...]:
    """Read class names out of a Roboflow ``data.yaml`` without a yaml dependency.

    The exports use one of two shapes -- ``names: ['a', 'b']`` or a block list --
    and both are simple enough to parse directly.
    """
    for candidate in (root / "data.yaml", root / "dataset.yaml"):
        if not candidate.exists():
            continue
        names: list[str] = []
        in_block = False
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                remainder = stripped[len("names:") :].strip()
                if remainder.startswith("["):
                    inner = remainder.strip("[]")
                    return tuple(
                        _clean_class_name(part) for part in inner.split(",") if part.strip()
                    )
                in_block = True
                continue
            if in_block:
                if stripped.startswith("-"):
                    names.append(stripped[1:].strip().strip("'\""))
                elif ":" in stripped and stripped.split(":", 1)[0].strip().isdigit():
                    names.append(stripped.split(":", 1)[1].strip().strip("'\""))
                else:
                    break
        if names:
            return tuple(_clean_class_name(n) for n in names)
    return ()


# Corrosion: the export carries the same label under two names, plus a negative class.
# A detector trained on the unfixed space emits findings called "non-corrosion", which
# is not a finding. Measured instance counts before the fix: Corrosion 805,
# Corrosion-detection 3, non-corrosion 2.
CORROSION_CLASS_MAP = {
    "Corrosion-detection": "Corrosion",   # the same label under the project's own name
    "non-corrosion": "",                  # a negative class is not a detection target
}

# Solar surface condition: two of the six declared classes have no instances at all, and
# one of them is named after its own index. Measured before the fix: Clear 6033,
# Snow-Covered 2010, Dusty 911, Physical-Damage 144, Bird-drop 0, "1" 0.
#
# "Clear" is kept deliberately. It is a healthy panel rather than a defect, but a
# detector trained without it has never been shown a sound panel and will call every
# module it sees something. What must not happen is a "Clear" box being reported to a
# client as a finding, and that is a consumer's job to enforce, not a reason to discard
# two thirds of the annotations.
SOLAR_PANEL_CLASS_MAP = {
    "Bird-drop": "",   # declared, zero instances: nothing was ever labelled with it
    "1": "",           # an export artefact, not a class
}


def adapt_roboflow_yolo(raw: Path, *, prefix: str,
                        class_map: dict[str, str] | None = None) -> Iterator[DetSample]:
    """Any Roboflow YOLO export: ``{train,valid,test}/{images,labels}`` + data.yaml.

    Some Universe projects ship a token validation split -- CODEBRIM's mirror has 90
    validation images against 2565 training ones, which is far too few to choose a
    checkpoint on across six classes. When the native split is that thin, part of the
    training set is deterministically reassigned to validation. The test split is
    never touched, so published comparisons stay valid.

    ``class_map`` renames or drops classes on the way in, and exists because Universe
    exports arrive with label spaces that are wrong rather than merely untidy. Two real
    cases from this project: a corrosion export carrying ``Corrosion`` and
    ``Corrosion-detection`` as separate classes when they are the same label under two
    names, and a solar export carrying a class literally named ``1`` alongside a
    ``Bird-drop`` class with no instances at all. Mapping a name to an empty string drops
    it. Doing this here, declared per dataset, keeps the decision visible next to the
    dataset it belongs to -- a model trained on an unfixed label space produces findings
    named ``non-corrosion``, which no downstream consumer can interpret.
    """
    class_names = _read_yolo_class_names(raw)
    if not class_names:
        return

    if class_map:
        # An empty name marks a dropped class; the detection writer skips those boxes.
        class_names = tuple(class_map.get(name, name) for name in class_names)

    def count_images(split_name: str) -> int:
        directory = raw / split_name / "images"
        return sum(1 for _ in _iter_images(directory)) if directory.is_dir() else 0

    train_count = count_images("train")
    val_count = count_images("valid") or count_images("val")
    # Below ~8% the validation set is too small to separate checkpoints reliably.
    rebalance = train_count > 0 and val_count < 0.08 * train_count

    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    for source_split, target_split in split_map.items():
        image_dir = raw / source_split / "images"
        label_dir = raw / source_split / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in _iter_images(image_dir):
            resolved = target_split
            if rebalance and target_split == "train":
                bucket = deterministic_split(image_path.stem, salt=f"{prefix}-val")
                resolved = "val" if bucket == "val" else "train"
            yield DetSample(
                sample_id=f"{prefix}_{source_split}_{image_path.stem}",
                image_path=image_path,
                label_path=label_dir / f"{image_path.stem}.txt",
                class_names=class_names,
                split=resolved,
            )


# The four damage types the CRDDC2022 benchmark scores, and the only four both RDD2022
# subsets have in common.
#
# The India annotations actually carry ten names: the benchmark four plus D01 (179),
# D11 (45), D43 (57), D44 (1062), D50 (28) and a single 'D0w0' that is a typo of D00.
# The China_Drone mirror carries the four plus 'Block crack' and 'Repair'.
#
# Keeping the union would give a model whose rarest classes appear in exactly one
# country's imagery, so it would learn "which country" as readily as "which damage".
# Restricting to the shared four costs about 1,372 India boxes and makes the corpus
# comparable to published RDD2022 numbers. The dropped names are listed rather than
# silently filtered so the loss is visible here.
RDD2022_BENCHMARK_CLASSES = ("D00", "D10", "D20", "D40")
RDD2022_DROPPED_CLASSES = ("D01", "D11", "D43", "D44", "D50", "D0w0", "Block crack", "Repair")
RDD2022_CLASS_MAP = {name: "" for name in RDD2022_DROPPED_CLASSES}


def adapt_rdd2022_india(raw: Path) -> Iterator[DetSample]:
    """RDD2022 India: Pascal VOC xml over ``India/train`` with an unlabelled test set.

    Only ``train`` carries annotations -- the 1,959 test images ship without xml because
    the benchmark scored them on a server -- so the native split is unusable and this
    derives train/val/test from the annotated images deterministically.
    """
    root = raw / "India" / "train"
    image_dir = root / "images"
    xml_dir = root / "annotations" / "xmls"
    if not image_dir.is_dir() or not xml_dir.is_dir():
        return

    for image_path in _iter_images(image_dir):
        annotation = xml_dir / f"{image_path.stem}.xml"
        if not annotation.is_file():
            continue
        yield DetSample(
            sample_id=f"rddindia_{image_path.stem}",
            image_path=image_path,
            label_path=annotation,
            class_names=RDD2022_BENCHMARK_CLASSES,
            split=deterministic_split(image_path.stem, salt="rdd2022-india"),
        )


def adapt_pvel_ad(raw: Path) -> Iterator[DetSample]:
    """PVEL-AD electroluminescence defects: Pascal VOC xml over ``trainval``.

    The archive holds 36,543 images but only 4,500 annotations. ``test/Annotations`` is
    present and empty -- the project's README says the test labels were released, this
    copy predates that -- so the 19,150 test images are unusable as supervision and are
    not emitted here. ``othertypes/good`` holds 11,353 anomaly-free cells, which are
    real negatives rather than unlabelled images, and are emitted with no boxes.

    Four of the twelve classes have almost no support: scratch (5 boxes), fragment (7),
    corner (9) and printing_error (32). They are dropped rather than trained on, because
    a detector that lists a class it has seen five times will still emit that class, and
    a named defect nobody can trust is worse than a refusal.
    """
    root = raw / "PVELAD" / "EL2021"
    image_dir = root / "trainval" / "JPEGImages"
    xml_dir = root / "trainval" / "Annotations"
    if not image_dir.is_dir() or not xml_dir.is_dir():
        return

    for image_path in _iter_images(image_dir):
        annotation = xml_dir / f"{image_path.stem}.xml"
        if not annotation.is_file():
            continue
        yield DetSample(
            sample_id=f"pvelad_{image_path.stem}",
            image_path=image_path,
            label_path=annotation,
            class_names=PVEL_AD_TRAINABLE_CLASSES,
            split=deterministic_split(image_path.stem, salt="pvel-ad"),
        )

    good_dir = root / "othertypes" / "good"
    if good_dir.is_dir():
        for image_path in _iter_images(good_dir):
            yield DetSample(
                sample_id=f"pveladgood_{image_path.stem}",
                image_path=image_path,
                label_path=None,
                class_names=PVEL_AD_TRAINABLE_CLASSES,
                split=deterministic_split(image_path.stem, salt="pvel-ad-good"),
            )


# Ordered by support in the trainval annotations, richest first.
PVEL_AD_TRAINABLE_CLASSES = (
    "finger",                  # 2958
    "crack",                   # 1260
    "black_core",              # 1028
    "thick_line",              #  981
    "horizontal_dislocation",  #  798
    "short_circuit",           #  492
    "vertical_dislocation",    #  137
    "star_crack",              #  135
)
PVEL_AD_UNTRAINABLE_CLASSES = ("printing_error", "corner", "fragment", "scratch")


def adapt_solar_pv_uav(raw: Path) -> Iterator[SegSample]:
    """Duke UAV PV segmentation: ``<name>.JPG`` beside ``<name>.png`` binary masks.

    Images are 4000x2250 and masks are three-channel 0/255, so the mask is reduced to a
    single channel by the segmentation writer. ``moving_labeled`` holds frames pulled
    from video in two flight modes; they are included but keep their mode in the sample
    id, because frames from one pass are near-duplicates and a reader needs to see that.
    """
    for source_dir, split in ((raw / "train_val_set", None), (raw / "test_set", "test")):
        if not source_dir.is_dir():
            continue
        for image_path in sorted(source_dir.glob("*.JPG")):
            mask_path = image_path.with_suffix(".png")
            if not mask_path.is_file():
                continue
            yield SegSample(
                sample_id=f"solarpvuav_{image_path.stem}",
                image_path=image_path,
                mask=mask_path,
                split=split or deterministic_split(image_path.stem, salt="solar-pv-uav"),
            )

    for mode in ("N_mode", "S_mode"):
        image_dir = raw / "moving_labeled" / "img" / mode
        label_dir = raw / "moving_labeled" / "labels" / mode
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for image_path in _iter_images(image_dir):
            mask_path = label_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                continue
            yield SegSample(
                sample_id=f"solarpvuav_{mode}_{image_path.stem}",
                image_path=image_path,
                mask=mask_path,
                # Frames from one video pass must not straddle a split, so the whole
                # mode goes to train rather than being diced frame by frame.
                split="train",
            )


RAIL_OBSTACLE_CLASSES = ("Branch", "IronRod", "Barrel", "Boulder", "Human", "Animal")


def adapt_uav_rsod_obstacles(raw: Path) -> Iterator[DetSample]:
    """UAV-RSOD obstacle detection: Pascal VOC xml beside each jpg in one flat folder.

    Images and annotations share a directory rather than the usual images/labels pair,
    and the split is the archive's own train/test with no validation set, so a validation
    split is carved deterministically out of train. The class list is read from the
    archive's class-names.txt rather than hardcoded, because a mismatch between the two
    would silently renumber every box.
    """
    root = raw / "images"
    names_file = root / "class-names.txt"
    class_names = RAIL_OBSTACLE_CLASSES
    if names_file.is_file():
        declared = tuple(
            line.strip() for line in names_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if declared:
            class_names = declared

    for source_split in ("train", "test"):
        split_dir = root / source_split
        if not split_dir.is_dir():
            continue
        for image_path in _iter_images(split_dir):
            annotation = image_path.with_suffix(".xml")
            if not annotation.is_file():
                continue
            if source_split == "test":
                split = "test"
            else:
                bucket = deterministic_split(image_path.stem, salt="uav-rsod-obstacles")
                split = "val" if bucket == "val" else "train"
            yield DetSample(
                sample_id=f"railobs_{source_split}_{image_path.stem}",
                image_path=image_path,
                label_path=annotation,
                class_names=class_names,
                split=split,
            )


def adapt_uav_rsod(raw: Path) -> Iterator[SegSample]:
    """UAV-RSOD segmentation: one image folder against two separate mask folders.

    The archive masks 'Rail Inside' and 'Rail Lines' as two independent binary sets over
    the same imagery. Only 'Rail Inside' is emitted.

    Emitting both looked reasonable and was wrong: the segmentation trainer is binary and
    has no notion of which aspect a mask belongs to, so it would receive the same image
    twice carrying two different definitions of foreground and learn the average of two
    contradictory tasks. Nothing would fail; the IoU would simply be poor for a reason
    nobody could see. 'Rail Lines' needs either a second task of its own or a multiclass
    trainer, and until one exists it is left in the archive rather than half-used.
    """
    image_dir = raw / "1 Images"
    mask_root = raw / "2 Annotations" / "2.2 Masking"
    if not image_dir.is_dir() or not mask_root.is_dir():
        return

    for aspect in ("Rail Inside",):
        mask_dir = mask_root / aspect
        if not mask_dir.is_dir():
            continue
        slug = aspect.lower().replace(" ", "")
        for image_path in _iter_images(image_dir):
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                continue
            yield SegSample(
                sample_id=f"railseg_{slug}_{image_path.stem}",
                image_path=image_path,
                mask=mask_path,
                # Salted on the stem alone so both aspects of one image share a split.
                split=deterministic_split(image_path.stem, salt="uav-rsod-seg"),
            )


# WeedsGalore semantic ids. The archive's masks carry 0/1/3/5 rather than a dense range,
# so remapping by position would relabel every pixel.
WEEDSGALORE_CLASSES = {0: "soil", 1: "maize", 3: "weed", 5: "weed"}


def adapt_weedsgalore(raw: Path) -> Iterator[SegSample]:
    """WeedsGalore multispectral crop/weed segmentation, split by the archive's own lists.

    Captures are named ``<date>_<id>`` and the archive ships splits/{train,val,test}.txt
    naming 156 of them. Those lists are used rather than a fresh hash, because the same
    field was flown on several dates and only the authors' split knows which captures
    overlap on the ground.

    There is no RGB file to read. Every capture ships as five separate single-band
    images -- B, G, R, RE and NIR -- so the RGB composite is built here and cached under
    the work root. That is also the limitation worth stating plainly: the red edge and
    near infrared bands are where crop/weed separation actually lives, and stacking them
    is a trainer change rather than an adapter one, so this corpus currently trains a
    vegetation model on the three bands that discriminate least.
    """
    split_dir = raw / "splits"
    assignments: dict[str, str] = {}
    for split_name, file_name in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        listing = split_dir / file_name
        if not listing.is_file():
            continue
        for line in listing.read_text(encoding="utf-8").splitlines():
            key = line.strip()
            if key:
                assignments[key] = split_name
    if not assignments:
        return

    for date_dir in sorted(p for p in raw.iterdir() if p.is_dir() and p.name[:2] == "20"):
        semantics_dir = date_dir / "semantics"
        images_dir = date_dir / "images"
        if not semantics_dir.is_dir() or not images_dir.is_dir():
            continue
        for mask_path in sorted(semantics_dir.glob("*.png")):
            key = mask_path.stem
            split = assignments.get(key)
            if split is None:
                # A capture the authors left out of every list. Dropping it is safer
                # than guessing which field it shares ground with.
                continue
            image_path = _weedsgalore_rgb(images_dir, key)
            if image_path is None:
                continue
            yield SegSample(
                sample_id=f"weeds_{key}",
                image_path=image_path,
                mask=mask_path,
                split=split,
            )


def _weedsgalore_rgb(images_dir: Path, key: str) -> Path | None:
    """Stack the R, G and B band files for one capture into a cached RGB png."""
    cache_dir = WORK_ROOT / "weedsgalore_rgb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    composite = cache_dir / f"{key}.png"
    if composite.is_file():
        return composite

    bands = []
    for suffix in ("R", "G", "B"):
        band_path = images_dir / f"{key}_{suffix}.png"
        if not band_path.is_file():
            return None
        bands.append(np.asarray(Image.open(band_path)))

    shapes = {band.shape[:2] for band in bands}
    if len(shapes) != 1:
        # Bands that disagree on size cannot be stacked without resampling, and
        # resampling one band against another silently shifts the spectral response.
        return None

    stacked = np.stack([_as_uint8(band) for band in bands], axis=-1)
    Image.fromarray(stacked).save(composite)
    return composite


def _as_uint8(band: np.ndarray) -> np.ndarray:
    """Reduce a band to single-channel uint8 without inventing dynamic range."""
    if band.ndim == 3:
        band = band[..., 0]
    if band.dtype == np.uint8:
        return band
    # 16-bit multispectral captures are scaled by the fixed bit depth rather than by
    # each image's own min and max: per-image normalisation would make brightness
    # incomparable between captures, which is exactly what a vegetation index needs.
    return (band.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# task assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One prepared training corpus, assembled from one or more raw datasets."""

    name: str
    kind: str  # segmentation | classification | detection
    datasets: tuple[str, ...]
    description: str


TASKS: dict[str, TaskSpec] = {
    "crack_seg": TaskSpec(
        name="crack_seg",
        kind="segmentation",
        datasets=("crack_segmentation_kaggle", "deepcrack", "crackforest"),
        description="Binary crack segmentation masks for SegFormer.",
    ),
    "crack_cls": TaskSpec(
        name="crack_cls",
        kind="classification",
        datasets=("sdnet2018", "surface_crack"),
        description="Crack / no-crack tiles for hard-negative mining and pretraining.",
    ),
    "solar_cls": TaskSpec(
        name="solar_cls",
        kind="classification",
        datasets=("elpv",),
        description="Solar cell defect severity tiers. CC BY-NC-SA: non-commercial only.",
    ),
    "solar_thermal_cls": TaskSpec(
        name="solar_thermal_cls",
        kind="classification",
        datasets=("infrared_solar_modules",),
        description=(
            "Per-module infrared anomaly classes -- hot spot, cracking, offline module, "
            "diode, soiling, shadowing and vegetation -- as published by Raptor Maps. "
            "MIT licensed. The crops are 24x40 px, so this trains a classifier over "
            "module polygons rather than a localiser."
        ),
    ),
    "solar_det": TaskSpec(
        name="solar_det",
        kind="detection",
        datasets=("solar_panel_defects",),
        description=(
            "Solar panel surface condition boxes for YOLO. Despite the source's title "
            "these are soiling and obstruction classes, not electrical faults, and two "
            "of its six classes carry no instances at all."
        ),
    ),
    "structural_det": TaskSpec(
        name="structural_det",
        kind="detection",
        datasets=("codebrim_structural",),
        description="Concrete bridge defect boxes: crack, spall, efflorescence, exposed bar, corrosion.",
    ),
    "corrosion_det": TaskSpec(
        name="corrosion_det",
        kind="detection",
        datasets=("corrosion_detection",),
        description="Metal corrosion and rust severity boxes for YOLO.",
    ),
    "pvel_ad_det": TaskSpec(
        name="pvel_ad_det",
        kind="detection",
        datasets=("pvel_ad",),
        description=(
            "Named and localised electroluminescence cell defects. Eight trainable "
            "classes; four long-tail classes are dropped rather than guessed at."
        ),
    ),
    "roads_det": TaskSpec(
        name="roads_det",
        kind="detection",
        datasets=("rdd2022_india", "rdd2022_china_drone"),
        description=(
            "Road surface damage restricted to the four CRDDC2022 benchmark classes "
            "that both the India and China_Drone subsets share."
        ),
    ),
    "rail_obstacle_det": TaskSpec(
        name="rail_obstacle_det",
        kind="detection",
        datasets=("uav_rsod_obstacles",),
        description="Obstacles in the rail corridor from Indian UAV imagery.",
    ),
    "rail_seg": TaskSpec(
        name="rail_seg",
        kind="segmentation",
        datasets=("uav_rsod",),
        description="Rail corridor extent from Indian UAV imagery.",
    ),
    "agriculture_seg": TaskSpec(
        name="agriculture_seg",
        kind="segmentation",
        datasets=("weedsgalore",),
        description="Maize crop and weed separation from multispectral UAV imagery.",
    ),
    "solar_module_seg": TaskSpec(
        name="solar_module_seg",
        kind="segmentation",
        datasets=("solar_pv_uav",),
        description="Binary PV module extent from UAV imagery, for inventory and thermal association.",
    ),
}

TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "crack": ("crack_seg", "crack_cls"),
    "solar": ("solar_cls", "solar_thermal_cls", "solar_det", "pvel_ad_det", "solar_module_seg"),
    "structural": ("structural_det",),
    "corrosion": ("corrosion_det",),
    "roads": ("roads_det",),
    "all": tuple(TASKS),
}

SegAdapter = Callable[[Path], Iterator[SegSample]]
ClsAdapter = Callable[[Path], Iterator[ClsSample]]
DetAdapter = Callable[[Path], Iterator[DetSample]]

ADAPTERS: dict[str, Callable[[Path], Iterator]] = {
    "deepcrack": adapt_deepcrack,
    "crackforest": adapt_crackforest,
    "crack_segmentation_combined": adapt_crack_segmentation_combined,
    "crack_segmentation_kaggle": adapt_crack_segmentation_kaggle,
    "elpv": adapt_elpv,
    "infrared_solar_modules": adapt_infrared_solar_modules,
    "sdnet2018": adapt_sdnet2018,
    "surface_crack": adapt_surface_crack,
    "solar_panel_defects": lambda raw: adapt_roboflow_yolo(
        raw, prefix="solarpv", class_map=SOLAR_PANEL_CLASS_MAP),
    "codebrim_structural": lambda raw: adapt_roboflow_yolo(raw, prefix="codebrim"),
    "corrosion_detection": lambda raw: adapt_roboflow_yolo(
        raw, prefix="corrosion", class_map=CORROSION_CLASS_MAP),
    "pvel_ad": adapt_pvel_ad,
    "rdd2022_india": adapt_rdd2022_india,
    "rdd2022_china_drone": lambda raw: adapt_roboflow_yolo(
        raw, prefix="rddchina", class_map=RDD2022_CLASS_MAP),
    "solar_pv_uav": adapt_solar_pv_uav,
    "uav_rsod": adapt_uav_rsod,
    "uav_rsod_obstacles": adapt_uav_rsod_obstacles,
    "weedsgalore": adapt_weedsgalore,
}


def resolve_tasks(names: Sequence[str]) -> list[TaskSpec]:
    selected: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key in TASK_GROUPS:
            selected.extend(TASK_GROUPS[key])
        elif key in TASKS:
            selected.append(key)
        else:
            raise KeyError(
                f"Unknown task {name!r}. Tasks: {', '.join(sorted(TASKS))}. "
                f"Groups: {', '.join(sorted(TASK_GROUPS))}."
            )
    seen: set[str] = set()
    ordered: list[TaskSpec] = []
    for key in selected:
        if key not in seen:
            seen.add(key)
            ordered.append(TASKS[key])
    return ordered


# Sample-id prefix each adapter stamps on its output, used to attribute already-
# prepared files back to the dataset they came from.
SOURCE_PREFIXES: dict[str, str] = {
    "cskaggle": "crack_segmentation_kaggle",
    "deepcrack": "deepcrack",
    "crackforest": "crackforest",
    "sdnet": "sdnet2018",
    "surfacecrack": "surface_crack",
    "elpv": "elpv",
    "irsm": "infrared_solar_modules",
    "codebrim": "codebrim_structural",
    "solarpv": "solar_panel_defects",
    "corrosion": "corrosion_detection",
    "pvelad": "pvel_ad",
    "pveladgood": "pvel_ad",
    "rddindia": "rdd2022_india",
    "rddchina": "rdd2022_china_drone",
    "solarpvuav": "solar_pv_uav",
}


def _scan_existing(spec: TaskSpec, task: PreparedTask) -> None:
    """Fill counts, sources, and classes for a task that is already on disk."""
    sources: set[str] = set()
    classes: set[str] = set()

    for split in ("train", "val", "test"):
        split_dir = task.root / split
        if not split_dir.is_dir():
            continue
        if spec.kind == "classification":
            files = [p for p in split_dir.rglob("*") if p.is_file() and p.suffix in IMAGE_SUFFIXES]
            classes.update(p.parent.name for p in files)
        else:
            image_dir = split_dir / "images"
            files = [p for p in image_dir.iterdir() if p.is_file()] if image_dir.is_dir() else []
        if files:
            task.counts[split] = len(files)
        for path in files:
            prefix = path.stem.split("_", 1)[0]
            dataset = SOURCE_PREFIXES.get(prefix)
            if dataset:
                sources.add(dataset)

    task.sources = [name for name in spec.datasets if name in sources]
    for name in task.sources:
        entry = DATASETS.get(name)
        if entry is not None:
            task.licenses[name] = entry.license
    if classes:
        task.class_names = sorted(classes)
    elif spec.kind == "detection":
        task.class_names = list(_read_yolo_class_names(task.root))


def prepare_task(spec: TaskSpec, *, force: bool = False) -> PreparedTask:
    """Build one prepared corpus from every raw dataset that is present on disk."""
    task = PreparedTask(task=spec.name, kind=spec.kind, root=PREPARED_ROOT / spec.name)

    if task.root.exists():
        if not force:
            existing = sum(1 for _ in task.root.rglob("*") if _.is_file())
            if existing:
                # Describe what is actually on disk rather than leaving the manifest
                # holding whatever an earlier, differently-scoped run recorded.
                _scan_existing(spec, task)
                task.warnings.append(
                    f"{task.root} already holds {existing} files; pass --force to rebuild."
                )
                return task
        shutil.rmtree(task.root)
    task.root.mkdir(parents=True, exist_ok=True)

    for dataset_name in spec.datasets:
        raw = DATA_ROOT / dataset_name
        if not raw.is_dir():
            task.warnings.append(f"{dataset_name}: not downloaded (run training.datasets.download).")
            continue
        adapter = ADAPTERS.get(dataset_name)
        if adapter is None:
            task.warnings.append(f"{dataset_name}: no adapter registered.")
            continue

        samples = list(adapter(raw))
        if not samples:
            task.warnings.append(f"{dataset_name}: yielded no usable samples.")
            continue

        if spec.kind == "segmentation":
            _write_segmentation(samples, task, salt=spec.name)
        elif spec.kind == "classification":
            _write_classification(samples, task, salt=spec.name)
        else:
            _write_detection(samples, task, salt=spec.name)

        task.sources.append(dataset_name)
        registry_entry = DATASETS.get(dataset_name)
        if registry_entry is not None:
            task.licenses[dataset_name] = registry_entry.license
        # A Roboflow export states its own licence; prefer that over the catalogue's
        # placeholder, since it is the authoritative record for what was downloaded.
        declared = read_yolo_license(raw)
        if declared:
            task.licenses[dataset_name] = declared

    return task


def write_manifest(tasks: Sequence[PreparedTask]) -> Path:
    """Record what was built, so a training run can state its data provenance."""
    manifest_path = PREPARED_ROOT / "manifest.json"
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("tasks", {})
        except json.JSONDecodeError:
            existing = {}

    for task in tasks:
        if not task.sources:
            continue
        existing[task.task] = {
            "kind": task.kind,
            "root": task.root.as_posix(),
            "counts": task.counts,
            "total": sum(task.counts.values()),
            "sources": task.sources,
            "licenses": task.licenses,
            "class_names": task.class_names,
            "split_fractions": {
                "val": VAL_FRACTION,
                "test": TEST_FRACTION,
                "note": "Datasets carrying a native split keep it; others hash by sample id.",
            },
            "warnings": task.warnings,
        }

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "tasks": existing}, indent=2), encoding="utf-8"
    )
    return manifest_path


def _print_catalogue() -> None:
    print(f"{'task':<16} {'kind':<15} {'state':<12} sources")
    print("-" * 78)
    for spec in TASKS.values():
        root = PREPARED_ROOT / spec.name
        if root.is_dir() and any(root.rglob("*")):
            state = "prepared"
        else:
            available = [d for d in spec.datasets if (DATA_ROOT / d).is_dir()]
            state = "ready" if available else "no raw data"
        print(f"{spec.name:<16} {spec.kind:<15} {state:<12} {', '.join(spec.datasets)}")
    print()
    print(f"Groups: {', '.join(sorted(TASK_GROUPS))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.datasets.prepare",
        description="Normalise downloaded datasets into trainer-ready layouts.",
    )
    parser.add_argument("tasks", nargs="*", help="Task or group names. Empty implies --list.")
    parser.add_argument("--list", action="store_true", help="Show the task catalogue and exit.")
    parser.add_argument("--force", action="store_true", help="Rebuild tasks that already exist.")
    args = parser.parse_args(argv)

    if args.list or not args.tasks:
        _print_catalogue()
        return 0

    try:
        specs = resolve_tasks(args.tasks)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    prepared: list[PreparedTask] = []
    for spec in specs:
        print(f"[{spec.name}] {spec.description}")
        task = prepare_task(spec, force=args.force)
        prepared.append(task)
        if task.counts:
            summary = "  ".join(f"{split}={count}" for split, count in sorted(task.counts.items()))
            print(f"  {summary}  total={sum(task.counts.values())}  from {', '.join(task.sources)}")
            if task.class_names:
                print(f"  classes: {', '.join(task.class_names)}")
        for warning in task.warnings:
            print(f"  warning: {warning}")

    manifest = write_manifest(prepared)
    print(f"\nManifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
