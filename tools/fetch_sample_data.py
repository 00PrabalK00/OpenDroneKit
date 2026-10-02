"""Fetch the open sample datasets OpenDroneKit is tested and documented against.

    python tools/fetch_sample_data.py list
    python tools/fetch_sample_data.py fetch odm_aukerman --dest data/samples
    python tools/fetch_sample_data.py fetch odm_aukerman --dest data/samples --limit 8
    python tools/fetch_sample_data.py verify odm_aukerman --dest data/samples
    python tools/fetch_sample_data.py fetch odm_aukerman --dry-run      # print URLs only

What it promises:

* Files come from the original publisher (GitHub at a pinned commit, or Zenodo), never
  from an OpenDroneKit mirror. OpenDroneKit redistributes none of them.
* Every file is checked against the sha256 pinned in ``tools/sample_data/manifests``
  (the hashes of the copies the test suite runs on). A file that does not match is
  deleted and reported; nothing unverified is left looking like the dataset.
* Datasets behind an access request or a proprietary licence (``fetch: restricted``)
  are never downloaded: the script prints where to get them and stops.
* Subsets that were cut out of very large archives with range requests cannot be
  re-cut automatically; the script fetches what it can and says what is left
  (``manual``), and ``verify`` checks a copy you assembled yourself.

``--limit N`` fetches only the first N data files of the manifest (plus licence and
readme files), which is how the tutorials use an 8-frame Aukerman subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent / "sample_data"
CATALOG = HERE / "catalog.json"
MANIFESTS = HERE / "manifests"
USER_AGENT = "OpenDroneKit-sample-fetch/1.0 (+https://github.com/opendronekit)"
_ALWAYS = ("license", "licence", "readme", "copying")


class FetchRefused(RuntimeError):
    """The dataset cannot be fetched by this script; the message says how to get it."""


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    path: str          # relative, forward slashes, no leading "./"


def load_catalog(path: Path | None = None) -> dict[str, dict]:
    data = json.loads((path or CATALOG).read_text(encoding="utf-8"))
    return {row["name"]: row for row in data["datasets"]}


def read_manifest(name: str, manifests: Path | None = None) -> list[ManifestEntry]:
    """``<sha256> *./relative/path`` lines (sha256sum's binary-mode format)."""
    entries = []
    for line in ((manifests or MANIFESTS) / f"{name}.sha256").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rest = line.partition(" ")
        rest = rest.lstrip("*").strip()
        if rest.startswith("./"):
            rest = rest[2:]
        if len(digest) != 64 or not rest or ".." in Path(rest).parts:
            raise ValueError(f"Bad manifest line in {name}: {line!r}")
        entries.append(ManifestEntry(digest.lower(), rest))
    return entries


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upstream_url(dataset: dict, local_path: str) -> str:
    renamed = (dataset.get("rename") or {}).get(local_path)
    if renamed:
        return renamed if "://" in renamed else dataset["base_url"] + renamed
    remote = local_path
    for prefix, replacement in (dataset.get("strip_prefix") or {}).items():
        if remote.startswith(prefix):
            remote = replacement + remote[len(prefix):]
            break
    return dataset["base_url"] + urllib.request.quote(remote)


def select(entries: list[ManifestEntry], limit: int | None) -> list[ManifestEntry]:
    if not limit:
        return entries
    always = [e for e in entries if Path(e.path).name.lower().startswith(_ALWAYS)]
    data = [e for e in entries if e not in always]
    return always + data[:limit]


def _crlf_variant(path: Path, max_bytes: int = 4 << 20) -> bytes | None:
    """The file with CRLF line endings, if it is small text (no NUL bytes); else None."""
    if path.stat().st_size > max_bytes:
        return None
    data = path.read_bytes()
    if b"\0" in data:
        return None
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def download(url: str, target: Path, expected_sha256: str | None, *, timeout: float = 120) -> str:
    """Download to ``target`` via a temporary file; delete and raise on a hash mismatch."""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = target.with_name(target.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
    digest = h.hexdigest()
    if expected_sha256 and digest != expected_sha256:
        # Some reference copies were made with `git clone` on Windows, which rewrote the
        # line endings of text files (README, LICENSE, .gitignore) to CRLF. The bytes
        # served upstream are LF. Accept a text file only when its CRLF form matches the
        # pinned hash exactly, and store that form so `verify` agrees with the manifest.
        converted = _crlf_variant(tmp)
        if converted is not None and hashlib.sha256(converted).hexdigest() == expected_sha256:
            tmp.write_bytes(converted)
            tmp.replace(target)
            return expected_sha256
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {url}: got {digest}, pinned {expected_sha256}")
    tmp.replace(target)
    return digest


def verify(name: str, dest: Path, *, limit: int | None = None) -> dict[str, list[str]]:
    """Compare a local copy with the pinned manifest: ok / missing / mismatched."""
    entries = select(read_manifest(name), limit)
    result: dict[str, list[str]] = {"ok": [], "missing": [], "mismatched": []}
    for entry in entries:
        path = dest / entry.path
        if not path.is_file():
            result["missing"].append(entry.path)
        elif sha256_of(path) != entry.sha256:
            result["mismatched"].append(entry.path)
        else:
            result["ok"].append(entry.path)
    return result


def fetch(name: str, dest: Path, *, limit: int | None = None, dry_run: bool = False,
          log=print) -> dict[str, list[str]]:
    """Fetch dataset ``name`` into ``dest`` (the dataset's own folder)."""
    catalog = load_catalog()
    if name not in catalog:
        raise FetchRefused(f"Unknown dataset {name!r}. Known: {', '.join(sorted(catalog))}.")
    dataset = catalog[name]
    mode = dataset["fetch"]
    if mode == "restricted":
        raise FetchRefused(f"{name} is not fetched by OpenDroneKit ({dataset['licence']}). "
                           f"{dataset.get('manual', '')} Source: {dataset['source']}")
    if mode == "manual":
        raise FetchRefused(f"{name} cannot be fetched automatically. {dataset['manual']} "
                           f"Source: {dataset['source']}")
    entries = read_manifest(name)
    pinned = {e.path: e.sha256 for e in entries}
    fetched: dict[str, list[str]] = {"downloaded": [], "already_present": [], "failed": []}

    if mode == "files":
        for entry in select(entries, limit):
            target = dest / entry.path
            url = upstream_url(dataset, entry.path)
            if dry_run:
                log(f"{url} -> {target} (sha256 {entry.sha256[:12]}...)")
                continue
            if target.is_file() and sha256_of(target) == entry.sha256:
                fetched["already_present"].append(entry.path)
                continue
            try:
                download(url, target, entry.sha256)
                fetched["downloaded"].append(entry.path)
                log(f"ok   {entry.path}")
            except Exception as exc:  # noqa: BLE001 - one failure is reported, the rest continue
                fetched["failed"].append(f"{entry.path}: {exc}")
                log(f"FAIL {entry.path}: {exc}")
    elif mode == "archive":
        for archive in dataset["archives"]:
            save_as = archive["save_as"]
            target = dest / save_as
            if dry_run:
                log(f"{archive['url']} -> {target} (sha256 {str(pinned.get(save_as))[:12]}...)")
                continue
            if save_as not in pinned:
                raise FetchRefused(f"{save_as} has no pinned sha256 in the {name} manifest.")
            if not (target.is_file() and sha256_of(target) == pinned[save_as]):
                try:
                    download(archive["url"], target, pinned[save_as], timeout=600)
                    fetched["downloaded"].append(save_as)
                except Exception as exc:  # noqa: BLE001
                    fetched["failed"].append(f"{save_as}: {exc}")
                    continue
            else:
                fetched["already_present"].append(save_as)
            if archive.get("extract"):
                _extract(target, dest)
        if dataset.get("manual"):
            log(f"Not fetched automatically: {dataset['manual']}")
    else:
        raise FetchRefused(f"{name}: unknown fetch mode {mode!r} in the catalog.")
    return fetched


def _extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            resolved = (dest / member).resolve()
            if dest.resolve() not in resolved.parents and resolved != dest.resolve():
                raise ValueError(f"{archive.name} contains a path outside the folder: {member}")
        zf.extractall(dest)


def _print_list(catalog: dict[str, dict]) -> None:
    for name, row in catalog.items():
        size = f"{row['size_mb']} MB" if row.get("size_mb") else "-"
        print(f"{name:40s} {row['fetch']:10s} {row['licence']:28s} {size:>8s}  {row['title']}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List the datasets, their licence and how they are fetched.")
    for cmd in ("fetch", "verify"):
        p = sub.add_parser(cmd)
        p.add_argument("name")
        p.add_argument("--dest", type=Path, default=None,
                       help="Parent folder; the dataset goes in <dest>/<name> "
                            "(default: ./sample_data).")
        p.add_argument("--limit", type=int, default=None,
                       help="Only the first N data files of the manifest.")
        if cmd == "fetch":
            p.add_argument("--dry-run", action="store_true", help="Print URLs, download nothing.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    catalog = load_catalog()
    if args.command == "list":
        _print_list(catalog)
        return 0
    folder = (args.dest or Path("sample_data")) / args.name
    if args.command == "verify":
        if args.name not in catalog or catalog[args.name]["fetch"] == "restricted":
            print(f"{args.name}: no pinned manifest to verify against.")
            return 2
        result = verify(args.name, folder, limit=args.limit)
        print(f"{args.name}: {len(result['ok'])} ok, {len(result['missing'])} missing, "
              f"{len(result['mismatched'])} mismatched")
        for path in result["mismatched"][:20]:
            print(f"  MISMATCH {path}")
        return 0 if not result["missing"] and not result["mismatched"] else 1
    try:
        result = fetch(args.name, folder, limit=args.limit, dry_run=args.dry_run)
    except FetchRefused as exc:
        print(str(exc))
        return 3
    if not args.dry_run:
        print(f"{args.name}: {len(result['downloaded'])} downloaded, "
              f"{len(result['already_present'])} already present, {len(result['failed'])} failed "
              f"-> {folder}")
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
