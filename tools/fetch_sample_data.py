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
* Subsets that were cut out of very large archives (RDD2022 is 13 GB, the bridge and
  THED-PV archives several GB) are re-cut the same way: the script reads the archive's
  directory and then only the pinned members with HTTP range requests, so a 51 MB subset
  costs about 51 MB of traffic. Anything the catalog has no route to is listed as
  ``not_fetched`` with instructions, and ``verify`` checks a copy assembled by hand.

``--limit N`` fetches only the first N data files of the manifest (plus licence and
readme files), which is how the tutorials use an 8-frame Aukerman subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.parse
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


def sha256_of(path: Path, *, attempts: int = 3) -> str:
    for attempt in range(attempts):
        h = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            # A just-written file on a busy removable/exFAT volume can refuse an open for
            # a moment (EINVAL); a missing file is still reported as missing by the caller.
            if attempt == attempts - 1 or not path.exists():
                raise
            time.sleep(0.5)
    raise AssertionError("unreachable")


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


def _accept(data: bytes, target: Path, expected_sha256: str | None) -> str:
    """Write ``data`` to ``target`` if it matches the pin (or its CRLF text form does)."""
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        converted = _crlf_bytes(data)
        if converted is None or hashlib.sha256(converted).hexdigest() != expected_sha256:
            raise ValueError(f"sha256 mismatch: got {digest}, pinned {expected_sha256}")
        data, digest = converted, expected_sha256
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return digest


def _crlf_bytes(data: bytes, max_bytes: int = 4 << 20) -> bytes | None:
    if len(data) > max_bytes or b"\0" in data:
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


# -- reading single files out of a remote zip ------------------------------------------

class RangeFile:
    """A read-only, seekable view of a remote file, read with HTTP range requests.

    ``zipfile`` only needs ``seek``/``tell``/``read``, so this lets it read the central
    directory at the end of a multi-gigabyte archive and then only the members asked for,
    without downloading the archive. ``file://`` URLs are read from disk the same way.
    """

    BLOCK = 1 << 16

    def __init__(self, url: str, *, timeout: float = 120) -> None:
        self.timeout = timeout
        self.pos = 0
        self.requests = 0
        self._cache: dict[int, bytes] = {}
        self._local: Path | None = None
        if url.startswith("file:"):
            self._local = Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
            self.url, self.size = url, self._local.stat().st_size
            return
        self.source = url
        self._resolve()

    def _resolve(self) -> None:
        """Follow the publisher's redirect (to a signed storage URL that expires)."""
        probe = urllib.request.Request(self.source, headers={"User-Agent": USER_AGENT,
                                                             "Range": "bytes=0-0"})
        with urllib.request.urlopen(probe, timeout=self.timeout) as response:
            self.url = response.geturl()
            total = (response.headers.get("Content-Range") or "").rpartition("/")[2]
            if response.status != 206 or not total.isdigit():
                raise FetchRefused(f"{self.source} does not serve byte ranges; download the "
                                   "archive from the source page instead.")
            self.size = int(total)

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _get(self, start: int, stop: int) -> bytes:
        self.requests += 1
        if self._local is not None:
            with self._local.open("rb") as handle:
                handle.seek(start)
                return handle.read(stop - start)
        for attempt in range(3):
            request = urllib.request.Request(self.url, headers={
                "User-Agent": USER_AGENT, "Range": f"bytes={start}-{stop - 1}"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                break
            except urllib.error.HTTPError as exc:
                # A signed redirect target expires after a few minutes: resolve it again.
                if exc.code not in (403, 410) or attempt == 2:
                    raise
                self._resolve()
        if len(data) != stop - start:
            raise OSError(f"range {start}-{stop - 1} of {self.url} returned {len(data)} bytes")
        return data

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        stop = min(self.size, self.pos + n)
        if stop <= self.pos:
            return b""
        if stop - self.pos > 4 * self.BLOCK:          # a member's data: one request
            data = self._get(self.pos, stop)
        else:                                         # headers and directories: cached blocks
            parts, at = [], self.pos
            while at < stop:
                block = at // self.BLOCK
                if block not in self._cache:
                    self._cache[block] = self._get(block * self.BLOCK,
                                                   min(self.size, (block + 1) * self.BLOCK))
                chunk = self._cache[block][at - block * self.BLOCK: stop - block * self.BLOCK]
                parts.append(chunk)
                at += len(chunk)
            data = b"".join(parts)
        self.pos += len(data)
        return data


class _Window:
    """Bytes ``start .. start+size`` of another seekable file (a zip stored in a zip)."""

    def __init__(self, base, start: int, size: int) -> None:
        self.base, self.start, self.size, self.pos = base, start, size, 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        self.base.seek(self.start + self.pos)
        data = self.base.read(n)
        self.pos += len(data)
        return data


def open_remote_zip(url: str, nest: Iterable[str] = ()) -> zipfile.ZipFile:
    """Open a remote zip (and, through ``nest``, zips *stored* inside it) for reading."""
    handle = RangeFile(url)
    archive = zipfile.ZipFile(handle)
    base, offset = handle, 0
    for member in nest:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise FetchRefused(f"{member} is compressed inside {url}; it cannot be read in place.")
        base.seek(offset + info.header_offset)
        header = base.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"{member}: bad local header in {url}")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        offset += info.header_offset + 30 + name_len + extra_len
        archive = zipfile.ZipFile(_Window(handle, offset, info.compress_size))
    return archive


def _rule_for(dataset: dict, path: str) -> dict | None:
    for rule in dataset.get("members") or []:
        if path.startswith(rule["prefix"]):
            return rule
    return None


def _member(rule: dict, path: str) -> str:
    return rule.get("member_prefix", "") + path[len(rule["prefix"]):]


def _direct_url(dataset: dict, path: str) -> str | None:
    if (dataset.get("rename") or {}).get(path) or dataset.get("base_url"):
        return upstream_url(dataset, path)
    return None


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
    """Fetch dataset ``name`` into ``dest`` (the dataset's own folder).

    Returns lists of manifest paths: ``downloaded``, ``already_present``, ``failed`` (with
    the reason) and ``not_fetched`` (files the catalog gives no automatic route to).
    """
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
    if mode not in {"files", "archive"}:
        raise FetchRefused(f"{name}: unknown fetch mode {mode!r} in the catalog.")
    every = read_manifest(name)
    pinned = {e.path: e.sha256 for e in every}
    entries = select(every, limit)
    fetched: dict[str, list[str]] = {"downloaded": [], "already_present": [], "failed": [],
                                     "not_fetched": []}

    def present(path: str) -> bool:
        target = dest / path
        return target.is_file() and sha256_of(target) == pinned[path]

    # 1. Whole archives the publisher offers, optionally unpacked into a sub-folder.
    handled: set[str] = set()
    for archive in dataset.get("archives") or []:
        save_as = archive["save_as"]
        handled.add(save_as)
        target = dest / save_as
        if save_as not in pinned:
            raise FetchRefused(f"{save_as} has no pinned sha256 in the {name} manifest.")
        if dry_run:
            log(f"{archive['url']} -> {target} (sha256 {pinned[save_as][:12]}...)")
            continue
        if present(save_as):
            fetched["already_present"].append(save_as)
        else:
            try:
                download(archive["url"], target, pinned[save_as], timeout=600)
                fetched["downloaded"].append(save_as)
                log(f"ok   {save_as}")
            except Exception as exc:  # noqa: BLE001 - one failure is reported, the rest continue
                fetched["failed"].append(f"{save_as}: {exc}")
                log(f"FAIL {save_as}: {exc}")
                continue
        if archive.get("extract"):
            sub = str(archive.get("extract_to", "")).strip("/")
            for path in _extract(target, dest / sub if sub else dest, dest):
                if path not in pinned:
                    continue
                handled.add(path)
                if present(path):
                    fetched["downloaded"].append(path)
                else:
                    (dest / path).unlink()
                    fetched["failed"].append(f"{path}: sha256 mismatch after extraction")

    # 2. Every other pinned file: a direct URL, or one member read out of a remote zip.
    open_zips: dict[tuple[str, tuple[str, ...]], zipfile.ZipFile] = {}
    for entry in entries:
        if entry.path in handled:
            continue
        target = dest / entry.path
        rule = _rule_for(dataset, entry.path)
        url = None if rule else _direct_url(dataset, entry.path)
        if rule is None and url is None:
            if not dry_run and present(entry.path):
                fetched["already_present"].append(entry.path)
            else:
                fetched["not_fetched"].append(entry.path)
            continue
        if dry_run:
            where = (f"{rule['url']} :: " + " :: ".join(list(rule.get("nest", []))
                                                        + [_member(rule, entry.path)])
                     if rule else url)
            log(f"{where} -> {target} (sha256 {entry.sha256[:12]}...)")
            continue
        if present(entry.path):
            fetched["already_present"].append(entry.path)
            continue
        try:
            if rule:
                key = (rule["url"], tuple(rule.get("nest", [])))
                if key not in open_zips:
                    open_zips[key] = open_remote_zip(*key)
                _accept(open_zips[key].read(_member(rule, entry.path)), target, entry.sha256)
            else:
                download(url, target, entry.sha256)
            fetched["downloaded"].append(entry.path)
            log(f"ok   {entry.path}")
        except Exception as exc:  # noqa: BLE001
            fetched["failed"].append(f"{entry.path}: {exc}")
            log(f"FAIL {entry.path}: {exc}")
    if fetched["not_fetched"]:
        log(f"Not fetched automatically ({len(fetched['not_fetched'])} files). "
            f"{dataset.get('manual', '')}".strip())
    return fetched


def _extract(archive: Path, folder: Path, dataset_root: Path) -> list[str]:
    """Unpack ``archive`` into ``folder``; the files written, relative to the dataset."""
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX/")]
        for member in members:
            resolved = (folder / member).resolve()
            if folder.resolve() not in resolved.parents and resolved != folder.resolve():
                raise ValueError(f"{archive.name} contains a path outside the folder: {member}")
        zf.extractall(folder, members=members)
    return [(folder / m).relative_to(dataset_root).as_posix()
            for m in members if not m.endswith("/")]


def _print_list(catalog: dict[str, dict]) -> None:
    for name, row in catalog.items():
        size = f"{row['size_mb']} MB" if row.get("size_mb") else "-"
        print(f"{name:40s} {row['fetch']:10s} {row['licence']:28s} {size:>8s}  {row['title']}")
        if row.get("attribution"):
            print(f"{'':40s} credit: {row['attribution']}")


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
    if catalog[args.name].get("attribution"):
        print(f"Licence {catalog[args.name]['licence']}: credit {catalog[args.name]['attribution']}")
    if not args.dry_run:
        print(f"{args.name}: {len(result['downloaded'])} downloaded, "
              f"{len(result['already_present'])} already present, {len(result['failed'])} failed, "
              f"{len(result['not_fetched'])} not fetchable automatically -> {folder}")
    return 1 if result.get("failed") or result.get("not_fetched") else 0


if __name__ == "__main__":
    sys.exit(main())
