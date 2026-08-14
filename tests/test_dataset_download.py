'''Dataset archives are checksum-aware and extract only inside their target.'''

from __future__ import annotations

import hashlib
import io
import tarfile

from training.datasets.download import extract_archive, md5_file


def test_tar_extraction_filters_traversal_and_writes_verified_marker(tmp_path):
    archive = tmp_path / 'dataset.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        good = b'valid sample'
        good_info = tarfile.TarInfo('wrapper/images/sample.txt')
        good_info.size = len(good)
        bundle.addfile(good_info, io.BytesIO(good))
        bad = b'escape'
        bad_info = tarfile.TarInfo('../outside.txt')
        bad_info.size = len(bad)
        bundle.addfile(bad_info, io.BytesIO(bad))

    target = extract_archive(archive, tmp_path / 'dataset')
    assert (target / 'images' / 'sample.txt').read_bytes() == b'valid sample'
    assert not (tmp_path / 'outside.txt').exists()
    marker = target / '.extracted'
    lines = marker.read_text(encoding='utf-8').splitlines()
    assert lines[0] == archive.name
    assert lines[1] == hashlib.sha256(archive.read_bytes()).hexdigest()

    assert extract_archive(archive, target) == target


def test_md5_helper_matches_standard_digest(tmp_path):
    path = tmp_path / 'archive.bin'
    path.write_bytes(b'checksum fixture')
    assert md5_file(path) == hashlib.md5(b'checksum fixture', usedforsecurity=False).hexdigest()
