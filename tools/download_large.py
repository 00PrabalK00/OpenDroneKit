'''Checksum-verified parallel HTTP range downloader for large public archives.'''

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import re


CHUNK = 1024 * 1024


def byte_ranges(total: int, connections: int) -> list[tuple[int, int]]:
    if total <= 0 or connections <= 0:
        raise ValueError('total and connections must be positive')
    connections = min(connections, total)
    base, remainder = divmod(total, connections)
    ranges = []
    start = 0
    for index in range(connections):
        size = base + (1 if index < remainder else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _head(url: str) -> tuple[int, bool]:
    request = urllib.request.Request(
        url,
        method='HEAD',
        headers={'User-Agent': 'OpenDroneKit/1.0'},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get('Content-Length') or 0)
        ranges = str(response.headers.get('Accept-Ranges') or '').casefold() == 'bytes'
    if not total:
        raise RuntimeError('Server did not provide Content-Length.')
    return total, ranges


def _probe_range(url: str) -> int:
    '''Return total bytes when a one-byte GET proves range support.'''
    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'OpenDroneKit/1.0',
            'Range': 'bytes=0-0',
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = str(response.headers.get('Content-Range') or '')
        match = re.fullmatch(r'bytes 0-0/(\d+)', content_range)
        if response.status != 206 or match is None:
            raise RuntimeError(
                f'Server did not honor the range probe: HTTP {response.status}, '
                f'Content-Range {content_range!r}'
            )
        response.read(1)
    return int(match.group(1))


def _fetch_part(
    url: str,
    path: Path,
    start: int,
    end: int,
    *,
    retries: int = 5,
) -> Path:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return path
    if path.is_file() and path.stat().st_size > expected:
        path.unlink()
    for attempt in range(1, retries + 1):
        existing = path.stat().st_size if path.is_file() else 0
        request_start = start + existing
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'OpenDroneKit/1.0',
                'Range': f'bytes={request_start}-{end}',
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f'Server ignored range {request_start}-{end}: HTTP {response.status}'
                    )
                content_range = str(response.headers.get('Content-Range') or '')
                if not content_range.startswith(f'bytes {request_start}-{end}/'):
                    raise RuntimeError(f'Unexpected Content-Range: {content_range!r}')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('ab') as output:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        output.write(block)
            if path.stat().st_size == expected:
                return path
            raise RuntimeError(
                f'Part length mismatch: expected {expected}, got {path.stat().st_size}'
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('unreachable')


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def assemble_parts(
    parts: list[Path],
    destination: Path,
    *,
    expected_md5: str,
) -> Path:
    temporary = destination.with_name(destination.name + '.assembling')
    digest = hashlib.md5(usedforsecurity=False)
    with temporary.open('wb') as output:
        for part in parts:
            with part.open('rb') as source:
                while True:
                    block = source.read(CHUNK)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_md5.casefold():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'MD5 mismatch: expected {expected_md5}, got {actual}')
    os.replace(temporary, destination)
    return destination


def download_large(
    url: str,
    destination: str | Path,
    *,
    expected_md5: str,
    connections: int = 8,
    parts: int | None = None,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = md5_file(destination)
        if actual == expected_md5.casefold():
            print(f'cached and verified: {destination}', flush=True)
            return destination
        raise RuntimeError(
            f'Destination exists with the wrong MD5 ({actual}); remove it before retrying.'
        )
    total, supports_ranges = _head(url)
    if not supports_ranges:
        probed_total = _probe_range(url)
        if probed_total != total:
            total = probed_total
    part_count = int(parts or connections)
    ranges = byte_ranges(total, part_count)
    parts = [destination.with_name(f'{destination.name}.part{index:02d}') for index in range(len(ranges))]
    print(
        f'downloading {total:,} bytes in {len(parts)} independent ranged parts',
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(connections, len(parts))) as pool:
        futures = {
            pool.submit(_fetch_part, url, path, start, end): (index, path)
            for index, (path, (start, end)) in enumerate(zip(parts, ranges))
        }
        for future in as_completed(futures):
            index, path = futures[future]
            future.result()
            print(f'part {index + 1}/{len(parts)} complete ({path.stat().st_size:,} bytes)', flush=True)
    assemble_parts(parts, destination, expected_md5=expected_md5)
    for part in parts:
        part.unlink(missing_ok=True)
    print(f'MD5 verified: {expected_md5.casefold()}', flush=True)
    print(f'completed: {destination}', flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Parallel range download with MD5 verification.')
    parser.add_argument('url')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--md5', required=True)
    parser.add_argument('--connections', type=int, default=8)
    parser.add_argument(
        '--parts',
        type=int,
        help='Stable part count; set separately to resume with fewer active connections.',
    )
    args = parser.parse_args(argv)
    download_large(
        args.url,
        args.destination,
        expected_md5=args.md5,
        connections=args.connections,
        parts=args.parts,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
