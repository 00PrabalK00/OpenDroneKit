"""Multi-file models: find every file a registry row publishes, and verify each one.

Grounding DINO is one graph plus a tokenizer and three configs; Florence-2 is four graphs
plus a tokenizer and four configs. The row's ``sha256`` covers the primary graph only
(``core.models.model_identity`` hashes that file), so a tokenizer swapped for another
model's -- which still loads and still returns boxes -- would go unnoticed. Every file in
the row's ``download.files`` therefore carries its own digest, and this module checks
each one at load time. A missing file or a digest mismatch refuses with the reason;
nothing runs on bytes the registry does not describe.

Files sit flat beside the primary graph, under the published asset names, exactly where
the first-run downloader puts them (``<weights root>/<key>/<asset name>``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import get_model_spec, load_registry, resolve_model_path

_DIGESTS: dict[tuple[str, int, float], str] = {}


class ModelFilesRefused(ValueError):
    """A multi-file model is missing a file, or a file is not the one registered."""


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    cached = _DIGESTS.get(key)
    if cached is not None:
        return cached
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    _DIGESTS[key] = digest
    return digest


def role_of(asset_name: str, model_key: str) -> str:
    """The file's role inside the model: the asset name without the key prefix.

    ``open_vocab_grounding_dino_tiny.onnx`` -> ``model``;
    ``vision_language_florence2_base_ft__vision_encoder_int8.onnx`` -> ``vision_encoder``;
    ``<key>__tokenizer.json`` -> ``tokenizer.json``. The precision suffix is dropped so
    the runner does not care whether the int8 or the fp32 graph was published.
    """
    name = Path(asset_name).name
    if name == f"{model_key}.onnx":
        return "model"
    prefix = f"{model_key}__"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(".onnx"):
        stem = name[: -len(".onnx")]
        for suffix in ("_int8", "_quantized", "_fp16", "_fp32"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        return stem
    return name


@dataclass(frozen=True)
class VerifiedModel:
    model_key: str
    kind: str
    primary: Path
    files: dict[str, Path]          # role -> path
    digests: dict[str, str]         # asset name -> sha256 (verified)
    model_sha256: str               # the primary graph's digest, as model_identity reports
    row: dict[str, Any]

    @property
    def model_used(self) -> str:
        return f"onnx:{self.primary.name}"


def verified_model(model_key: str, *, kind: str | None = None) -> VerifiedModel:
    """Resolve and verify every file of a registered multi-file model, or refuse."""
    spec = get_model_spec(model_key)
    if spec is None:
        raise ModelFilesRefused(f"{model_key!r} is not in the model registry.")
    if kind is not None and spec.kind != kind:
        raise ModelFilesRefused(f"{model_key} is a {spec.kind}, not a {kind}.")
    row = dict((load_registry().get("models") or {}).get(model_key) or {})
    primary = resolve_model_path(model_key)
    if primary is None or not primary.is_file():
        note = spec.status_note or "Its weights are not installed on this machine."
        raise ModelFilesRefused(f"{model_key} cannot run: {note}")
    entries = (row.get("download") or {}).get("files") or []
    if not entries:
        raise ModelFilesRefused(
            f"{model_key} lists no files with digests in the registry, so its tokenizer "
            "and configs cannot be verified; refusing.")
    files: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for entry in entries:
        name = Path(str(entry.get("name", ""))).name
        expected = str(entry.get("sha256", "")).strip().lower()
        if not name or not expected:
            raise ModelFilesRefused(f"{model_key}: registry file entry {entry!r} lacks a name or sha256.")
        path = primary.parent / name
        if not path.is_file():
            raise ModelFilesRefused(
                f"{model_key} cannot run: {name} is missing beside {primary.name}. "
                f"{spec.status_note}".strip())
        actual = sha256_file(path)
        if actual != expected:
            raise ModelFilesRefused(
                f"{model_key}: {name} on disk is not the registered file (recorded "
                f"{expected[:16]}..., found {actual[:16]}...). Refusing to run it.")
        files[role_of(name, model_key)] = path
        digests[name] = actual
    primary_digest = sha256_file(primary)
    if spec.sha256 and primary_digest != spec.sha256:
        raise ModelFilesRefused(
            f"{model_key}: {primary.name} is not the registered graph (recorded "
            f"{spec.sha256[:16]}..., found {primary_digest[:16]}...).")
    return VerifiedModel(model_key, spec.kind, primary, files, digests, primary_digest, row)
