'''Large downloads split and assemble bytes without gaps or overlap.'''

from __future__ import annotations

import hashlib

import pytest

from tools.download_large import (
    _probe_range,
    assemble_parts,
    byte_ranges,
    download_large,
    md5_file,
)


def test_byte_ranges_cover_every_byte_exactly_once():
    ranges = byte_ranges(103, 8)
    flattened = [value for start, end in ranges for value in range(start, end + 1)]
    assert flattened == list(range(103))
    assert max(end - start for start, end in ranges) <= 12


def test_parts_assemble_in_order_and_verify_md5(tmp_path):
    content = b'abcdefghijklmnopqrstuvwxyz' * 100
    parts = []
    for index, block in enumerate((content[:301], content[301:1700], content[1700:])):
        part = tmp_path / f'part{index}'
        part.write_bytes(block)
        parts.append(part)
    expected = hashlib.md5(content, usedforsecurity=False).hexdigest()
    output = assemble_parts(parts, tmp_path / 'archive.bin', expected_md5=expected)
    assert output.read_bytes() == content
    assert md5_file(output) == expected


def test_bad_checksum_does_not_publish_destination(tmp_path):
    part = tmp_path / 'part'
    part.write_bytes(b'corrupt')
    destination = tmp_path / 'archive.bin'
    with pytest.raises(RuntimeError, match='MD5 mismatch'):
        assemble_parts([part], destination, expected_md5='0' * 32)
    assert not destination.exists()


def test_range_probe_accepts_206_even_without_head_advertisement(monkeypatch):
    class Response:
        status = 206
        headers = {'Content-Range': 'bytes 0-0/9099481727'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return b'x'

    monkeypatch.setattr('urllib.request.urlopen', lambda request, timeout: Response())
    assert _probe_range('https://example.invalid/archive.zip') == 9099481727


def test_part_count_is_stable_when_worker_count_changes(tmp_path, monkeypatch):
    content = b'abcdefgh'
    expected = hashlib.md5(content, usedforsecurity=False).hexdigest()
    monkeypatch.setattr('tools.download_large._head', lambda url: (len(content), True))

    def fetch(url, path, start, end):
        path.write_bytes(content[start:end + 1])
        return path

    monkeypatch.setattr('tools.download_large._fetch_part', fetch)
    destination = tmp_path / 'archive.bin'
    download_large(
        'https://example.invalid/archive.bin',
        destination,
        expected_md5=expected,
        connections=2,
        parts=8,
    )
    assert destination.read_bytes() == content
