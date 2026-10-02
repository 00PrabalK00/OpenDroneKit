"""First-run model-weights downloader that refuses anything it cannot verify.

Weights are not shipped in the installer (they are large, and some are not ours to
redistribute). This module fetches them on request, and only under three conditions:

1. A download location is *published* for the model: a ``download`` entry on its row in
   ``models/model_registry.json``, or in the JSON file named by ``ODK_MODEL_SOURCES``.
   Nothing here invents a URL. With no location the answer is "not available for
   download yet", stated as such.
2. The registry records the model's sha256. Without a recorded digest there is nothing
   to check the bytes against, so the download is refused before it starts.
3. The bytes that arrive hash to that digest. A mismatch is deleted and refused; it is
   never moved into the models folder.

A ``download`` entry is either a single file::

    "download": {"url": "https://.../crack_segmentation.onnx"}

which must hash to the row's ``sha256``; or, for ONNX graphs with an external-data
sidecar, several files, each with its own digest::

    "download": {"files": [
        {"name": "model.onnx",      "url": "https://...", "sha256": "..."},
        {"name": "model.onnx.data", "url": "https://...", "sha256": "..."}]}

after which the set is hashed exactly as ``core.models.model_identity`` hashes it and
must equal the row's ``sha256`` as well.

    python -m core.model_download --list
    python -m core.model_download --all
    python -m core.model_download crack_segmentation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

SOURCES_ENV = "ODK_MODEL_SOURCES"
ALLOWED_SCHEMES = ("https", "http", "file")
NOT_PUBLISHED = ("not available for download yet: no download URL has been published "
                 "for this model")

Progress = Callable[[str, int, int], None]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _registry() -> dict[str, Any]:
    from core.models import load_registry

    return load_registry()


def _external_sources() -> dict[str, Any]:
    path = os.environ.get(SOURCES_ENV, "").strip()
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    sources = payload.get("sources", payload) if isinstance(payload, dict) else {}
    return sources if isinstance(sources, dict) else {}


def download_spec(key: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """The published download location for a model, or None when there is none."""
    spec = _external_sources().get(key) or row.get("download")
    if isinstance(spec, str):
        spec = {"url": spec}
    if not isinstance(spec, dict):
        return None
    if spec.get("url") or spec.get("files"):
        return spec
    return None


def target_root() -> Path:
    """Where downloaded weights go: ODK_MODEL_WEIGHTS_DIR when set, else models/."""
    from core.models import extra_weights_root, models_root

    return extra_weights_root() or models_root()


def _installed(key: str) -> bool:
    from core.models import model_status

    return bool(model_status(key).get("exists"))


def availability(key: str) -> dict[str, Any]:
    """Whether a model can be downloaded, and if not, exactly why not."""
    models = _registry().get("models", {})
    row = models.get(key)
    if not isinstance(row, dict):
        return {"key": key, "downloadable": False, "installed": False,
                "reason": f"unknown model key {key!r}"}
    installed = _installed(key)
    expected = str(row.get("sha256") or "").strip().lower()
    spec = download_spec(key, row)
    base = {"key": key, "installed": installed, "sha256": expected,
            "status": row.get("status", "installed"), "path": row.get("path", "")}
    if spec is None:
        return {**base, "downloadable": False, "reason": NOT_PUBLISHED}
    if not expected:
        return {**base, "downloadable": False,
                "reason": "the registry records no sha256 for this model, so a download "
                          "could not be verified; refusing"}
    for url in _spec_urls(spec):
        scheme = urllib.parse.urlparse(url).scheme.lower()
        if scheme not in ALLOWED_SCHEMES:
            return {**base, "downloadable": False,
                    "reason": f"download URL scheme {scheme!r} is not allowed"}
    return {**base, "downloadable": True, "reason": "", "urls": _spec_urls(spec)}


def _spec_urls(spec: dict[str, Any]) -> list[str]:
    if spec.get("files"):
        return [str(f.get("url", "")) for f in spec["files"] if isinstance(f, dict)]
    return [str(spec.get("url", ""))]


def list_models() -> list[dict[str, Any]]:
    return [availability(key) for key in sorted(_registry().get("models", {}))]


def _fetch(url: str, dest: Path, progress: Progress | None, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenDroneKit-model-download"})
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as out:  # noqa: S310 - scheme checked
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            done += len(block)
            if progress:
                progress(url, done, total)


class VerificationError(RuntimeError):
    """The downloaded bytes are not the bytes the registry describes."""


def download_model(key: str, *, force: bool = False, progress: Progress | None = None,
                   timeout: float = 60.0) -> dict[str, Any]:
    """Download, verify and install one model. Never installs unverified bytes."""
    info = availability(key)
    if not info.get("downloadable"):
        return {"ok": False, "key": key, "error": info.get("reason", "not downloadable")}
    if info["installed"] and not force:
        return {"ok": True, "key": key, "skipped": True, "reason": "already installed"}

    row = _registry()["models"][key]
    spec = download_spec(key, row) or {}
    expected = info["sha256"]
    final = target_root() / str(row["path"])
    final.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="odk-dl-", dir=str(final.parent)) as tmp:
        staging = Path(tmp)
        try:
            if spec.get("files"):
                placed = _fetch_files(spec["files"], staging, final.name, progress, timeout)
                from core.semantic_engine import sha256_onnx_model

                main = staging / final.name
                got = sha256_onnx_model(main) if len(placed) > 1 else sha256_file(main)
                if got != expected:
                    raise VerificationError(
                        f"model digest {got} does not match the registry's {expected}")
            else:
                main = staging / final.name
                _fetch(str(spec["url"]), main, progress, timeout)
                got = sha256_file(main)
                if got != expected:
                    raise VerificationError(
                        f"sha256 {got} does not match the registry's {expected}")
                placed = [main]
        except VerificationError as exc:
            return {"ok": False, "key": key, "error": f"refused: {exc}; the file was discarded"}
        except (OSError, ValueError) as exc:
            return {"ok": False, "key": key, "error": f"download failed: {exc}"}
        written = []
        for part in placed:
            dest = final.parent / part.name
            os.replace(part, dest)
            written.append(str(dest))
    return {"ok": True, "key": key, "sha256": expected, "files": written}


def _fetch_files(files: list[Any], staging: Path, main_name: str,
                 progress: Progress | None, timeout: float) -> list[Path]:
    placed: list[Path] = []
    names = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("malformed download entry")
        name = Path(str(entry.get("name", ""))).name
        digest = str(entry.get("sha256", "")).strip().lower()
        if not name or not digest:
            raise VerificationError(f"file entry {entry!r} lacks a name or sha256")
        dest = staging / name
        _fetch(str(entry["url"]), dest, progress, timeout)
        got = sha256_file(dest)
        if got != digest:
            raise VerificationError(f"{name}: sha256 {got} does not match the published {digest}")
        placed.append(dest)
        names.add(name)
    if main_name not in names:
        raise VerificationError(f"the published files do not include {main_name}")
    return placed


def first_run_report() -> dict[str, Any]:
    """What a first launch should say about weights: what is missing and what can be fetched."""
    rows = list_models()
    missing = [r for r in rows if not r["installed"] and r.get("status") != "awaiting_weights"]
    return {
        "models": rows,
        "missing": [r["key"] for r in missing],
        "downloadable": [r["key"] for r in missing if r.get("downloadable")],
        "not_published": [r["key"] for r in missing if not r.get("downloadable")],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("keys", nargs="*", help="model keys to download")
    parser.add_argument("--all", action="store_true", help="download every missing model that is published")
    parser.add_argument("--list", action="store_true", help="list models and whether they can be downloaded")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pause", action="store_true", help="wait for Enter before exiting (Start-menu use)")
    args = parser.parse_args(argv)

    code = 0
    if args.list or (not args.keys and not args.all):
        for row in list_models():
            if row["installed"]:
                state, why = "installed", ""
            elif row.get("status") == "awaiting_weights":
                state, why = "no weights", "no weights exist for this model yet (registry: awaiting_weights)"
            elif row.get("downloadable"):
                state, why = "downloadable", ""
            else:
                state, why = "missing", row.get("reason", "")
            print(f"{row['key']:<40} {state:<13} {why}")
    keys = list(args.keys)
    if args.all:
        report = first_run_report()
        keys += report["missing"]
        if not report["missing"]:
            print("Every released model is already installed.")
    for key in keys:
        def show(url: str, done: int, total: int, _key: str = key) -> None:
            if total:
                print(f"\r{_key}: {done * 100 // total}%", end="", flush=True)
        result = download_model(key, force=args.force, progress=show)
        print()
        if result.get("ok"):
            print(f"{key}: {'already installed' if result.get('skipped') else 'downloaded and verified'}")
        else:
            print(f"{key}: {result.get('error')}")
            code = 1 if download_spec(key, _registry()["models"].get(key, {}) or {}) else code
    if args.pause:
        try:
            input("Press Enter to close.")
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
