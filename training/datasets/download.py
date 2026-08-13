"""Fetch training datasets from public HTTP, Kaggle, and Roboflow.

Downloads are resumable and content-hashed, because these archives are large enough
that a dropped connection halfway through is a normal event rather than an edge case.
Credentials are read from the environment or the user profile; nothing is stored in
the repository.

    python -m training.datasets.download --list
    python -m training.datasets.download public
    python -m training.datasets.download crack solar --data-root training/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import zipfile

# Allow `python training/datasets/download.py` as well as `-m`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.datasets.registry import DATASETS, GROUPS, DatasetSpec, resolve
else:
    from .registry import DATASETS, GROUPS, DatasetSpec, resolve


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
ARCHIVE_DIRNAME = "_archives"
CHUNK = 1 << 20


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def kaggle_token() -> str | None:
    """Kaggle API token from the environment or the standard user-profile locations."""
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if token:
        return token.strip()
    for candidate in (Path.home() / ".kaggle" / "access_token", Path.home() / ".kaggle" / "kaggle.json"):
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8").strip()
        if candidate.suffix == ".json":
            try:
                return str(json.loads(text).get("key") or "").strip() or None
            except json.JSONDecodeError:
                continue
        return text or None
    return None


def roboflow_key() -> str | None:
    """Roboflow private API key from the environment."""
    key = os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_KEY")
    return key.strip() if key else None


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_http(
    url: str,
    destination: Path,
    *,
    expected_sha256: str = "",
    progress: Callable[[str], None] | None = None,
    max_retries: int = 4,
) -> Path:
    """Download with HTTP range resume and exponential backoff.

    An existing complete file with a matching hash is left alone; a partial file is
    continued rather than restarted.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_sha256:
        if sha256_file(destination) == expected_sha256.lower():
            if progress:
                progress(f"  cached and verified: {destination.name}")
            return destination
        destination.unlink()

    for attempt in range(1, max_retries + 1):
        existing = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "OpenDroneKit/1.0"})
        if existing:
            request.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # A server that ignores Range replies 200 and would otherwise be
                # appended onto the partial file, corrupting it.
                resuming = response.status == 206 and existing > 0
                if existing and not resuming:
                    existing = 0
                total = int(response.headers.get("Content-Length") or 0) + existing
                mode = "ab" if resuming else "wb"
                downloaded = existing
                last_report = 0.0
                with open(destination, mode) as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        downloaded += len(block)
                        now = time.time()
                        if progress and now - last_report > 2.0:
                            last_report = now
                            share = f" ({100.0 * downloaded / total:.0f}%)" if total else ""
                            progress(f"  {_human(downloaded)}{share}")
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Download failed after {max_retries} attempts: {url} ({exc})") from exc
            delay = 2**attempt
            if progress:
                progress(f"  retry {attempt}/{max_retries - 1} in {delay}s after {type(exc).__name__}")
            time.sleep(delay)

    if expected_sha256:
        actual = sha256_file(destination)
        if actual != expected_sha256.lower():
            raise RuntimeError(f"Checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual}")
    return destination


def extract_archive(archive: Path, target: Path, progress: Callable[[str], None] | None = None) -> Path:
    """Extract a zip/tar into `target`, collapsing a single top-level wrapper folder.

    Member paths are validated before extraction so a crafted archive cannot write
    outside the target directory.
    """
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".extracted"
    if marker.exists():
        if progress:
            progress(f"  already extracted: {target.name}")
        return target

    staging = target.parent / f"{target.name}__staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    def is_safe(member_name: str) -> bool:
        resolved = (staging / member_name).resolve()
        return str(resolved).startswith(str(staging.resolve()))

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.namelist() if is_safe(m)]
            skipped = len(bundle.namelist()) - len(members)
            bundle.extractall(staging, members=members)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            members = [m for m in bundle.getmembers() if is_safe(m.name)]
            skipped = len(bundle.getmembers()) - len(members)
            bundle.extractall(staging, members=members)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive.name}")

    if skipped and progress:
        progress(f"  skipped {skipped} unsafe archive member(s)")

    entries = list(staging.iterdir())
    source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    for item in source.iterdir():
        destination = target / item.name
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True) if destination.is_dir() else destination.unlink()
        shutil.move(str(item), str(destination))
    shutil.rmtree(staging, ignore_errors=True)

    marker.write_text(f"{archive.name}\n{sha256_file(archive)}\n", encoding="utf-8")
    return target


def fetch_http(spec: DatasetSpec, data_root: Path, progress: Callable[[str], None]) -> dict[str, Any]:
    archive_dir = data_root / ARCHIVE_DIRNAME
    archive = archive_dir / (spec.archive_name or f"{spec.name}.zip")
    target = data_root / spec.name
    progress(f"  source: {spec.url}")
    download_http(spec.url, archive, progress=progress)
    extract_archive(archive, target, progress=progress)
    return {"path": str(target), "archive": str(archive), "sha256": sha256_file(archive)}


def fetch_kaggle(spec: DatasetSpec, data_root: Path, progress: Callable[[str], None]) -> dict[str, Any]:
    token = kaggle_token()
    if not token:
        raise RuntimeError(
            "No Kaggle credentials found. Set KAGGLE_API_TOKEN, or save the token to "
            f"{Path.home() / '.kaggle' / 'access_token'}."
        )
    os.environ.setdefault("KAGGLE_API_TOKEN", token)
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub>=0.4.1") from exc

    progress(f"  kaggle: {spec.slug}")
    downloaded = Path(kagglehub.dataset_download(spec.slug))
    target = data_root / spec.name
    if target.exists() and target.is_symlink():
        target.unlink()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # kagglehub keeps its own cache; link to it rather than duplicating gigabytes.
        try:
            target.symlink_to(downloaded, target_is_directory=True)
        except OSError:
            shutil.copytree(downloaded, target, dirs_exist_ok=True)
    return {"path": str(target), "cache": str(downloaded)}


def fetch_roboflow(spec: DatasetSpec, data_root: Path, progress: Callable[[str], None]) -> dict[str, Any]:
    key = roboflow_key()
    if not key:
        raise RuntimeError("No Roboflow API key found. Set ROBOFLOW_API_KEY in the environment.")
    if not (spec.workspace and spec.project):
        raise RuntimeError(
            f"Dataset {spec.name!r} has no Roboflow workspace/project pinned yet. "
            "Set them in training/datasets/registry.py before fetching."
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("roboflow is not installed. Run: pip install roboflow") from exc

    progress(f"  roboflow: {spec.workspace}/{spec.project} v{spec.version} as {spec.export_format}")
    client = Roboflow(api_key=key)
    project = client.workspace(spec.workspace).project(spec.project)
    target = data_root / spec.name
    target.mkdir(parents=True, exist_ok=True)
    dataset = project.version(spec.version).download(spec.export_format, location=str(target), overwrite=False)
    return {"path": str(getattr(dataset, "location", target))}


FETCHERS: dict[str, Callable[[DatasetSpec, Path, Callable[[str], None]], dict[str, Any]]] = {
    "http": fetch_http,
    "kaggle": fetch_kaggle,
    "roboflow": fetch_roboflow,
}


def fetch(specs: list[DatasetSpec], data_root: Path, *, quiet: bool = False) -> dict[str, Any]:
    """Fetch each dataset, continuing past failures so one missing key blocks nothing."""
    data_root.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    manifest: dict[str, Any] = {}
    for index, spec in enumerate(specs, start=1):
        header = f"[{index}/{len(specs)}] {spec.name} ({spec.kind}, ~{spec.approx_size_mb or '?'} MB)"
        progress(header)
        progress(f"  licence: {spec.license}")
        try:
            record = FETCHERS[spec.kind](spec, data_root, progress)
            record.update({"status": "ok", "task": spec.task, "target": spec.target, "license": spec.license})
            progress(f"  done -> {record['path']}")
        except Exception as exc:  # noqa: BLE001
            record = {"status": "failed", "reason": str(exc), "license": spec.license}
            progress(f"  FAILED: {exc}")
        manifest[spec.name] = record

    manifest_path = data_root / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(manifest)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    ok = sum(1 for r in manifest.values() if r.get("status") == "ok")
    progress(f"\n{ok}/{len(specs)} datasets ready. Manifest: {manifest_path}")
    return manifest


def print_catalogue() -> None:
    print(f"{'name':<28} {'kind':<9} {'task':<15} {'size':>8}  feeds")
    print("-" * 96)
    for spec in DATASETS.values():
        size = f"{spec.approx_size_mb} MB" if spec.approx_size_mb else "?"
        print(f"{spec.name:<28} {spec.kind:<9} {spec.task:<15} {size:>8}  {', '.join(spec.feeds) or '-'}")
    print("\nGroups:")
    for group, members in GROUPS.items():
        print(f"  {group:<12} {', '.join(members)}")
    print("\nCredentials detected:")
    print(f"  kaggle   : {'yes' if kaggle_token() else 'no  (set KAGGLE_API_TOKEN)'}")
    print(f"  roboflow : {'yes' if roboflow_key() else 'no  (set ROBOFLOW_API_KEY)'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OpenDroneKit training datasets.")
    parser.add_argument("names", nargs="*", help="Dataset or group names. Default: public")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Where datasets are stored.")
    parser.add_argument("--list", action="store_true", help="Show the catalogue and exit.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args(argv)

    if args.list:
        print_catalogue()
        return 0

    try:
        specs = resolve(args.names or ["public"])
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    manifest = fetch(specs, Path(args.data_root), quiet=args.quiet)
    return 0 if all(r.get("status") == "ok" for r in manifest.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
