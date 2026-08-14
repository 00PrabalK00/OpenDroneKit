"""Object storage and the self-hosting compose file.

The containment tests matter most. Keys reach this layer from client-supplied
filenames, so a key containing `..` must never read or write outside the root. Those
tests assert the filesystem, not merely that an exception was raised.
"""

from __future__ import annotations

import os

import pytest

from services.api.storage import (
    LocalStorage,
    StorageError,
    build_storage,
    describe_storage,
)


@pytest.fixture
def store(tmp_path):
    return LocalStorage(root=tmp_path / "objects")


class TestLocalRoundTrip:
    def test_put_get_delete(self, store):
        payload = b"orthomosaic bytes"
        store.put("runs/1/ortho.tif", payload)
        assert store.exists("runs/1/ortho.tif")
        assert store.get("runs/1/ortho.tif") == payload
        assert store.delete("runs/1/ortho.tif") is True
        assert store.exists("runs/1/ortho.tif") is False

    def test_delete_of_a_missing_key_is_false_not_an_error(self, store):
        assert store.delete("never/written.bin") is False

    def test_get_of_a_missing_key_raises(self, store):
        with pytest.raises(StorageError):
            store.get("never/written.bin")

    def test_put_accepts_a_file_object(self, store, tmp_path):
        source = tmp_path / "source.bin"
        source.write_bytes(b"streamed")
        with source.open("rb") as handle:
            store.put("streamed.bin", handle)
        assert store.get("streamed.bin") == b"streamed"

    def test_list_filters_by_prefix(self, store):
        store.put("a/one.txt", b"1")
        store.put("a/two.txt", b"2")
        store.put("b/three.txt", b"3")
        assert store.list("a/") == ["a/one.txt", "a/two.txt"]
        assert len(store.list()) == 3


class TestKeyContainment:
    @pytest.mark.parametrize(
        "key",
        ["../escape.bin", "a/../../escape.bin", "..\\escape.bin", "a/b/../../../escape.bin"],
    )
    def test_a_parent_reference_is_refused(self, store, key):
        with pytest.raises(StorageError):
            store.put(key, b"should not be written")

    @pytest.mark.parametrize("key", ["", "   ", "/", "//", "."])
    def test_an_empty_or_meaningless_key_is_refused(self, store, key):
        with pytest.raises(StorageError):
            store.put(key, b"x")

    def test_nothing_is_written_outside_the_root(self, store, tmp_path):
        """Assert the filesystem, not just that an exception was raised."""
        for key in ["../escape.bin", "a/../../escape.bin"]:
            with pytest.raises(StorageError):
                store.put(key, b"payload")

        outside = [
            path for path in tmp_path.rglob("*")
            if path.is_file() and (tmp_path / "objects") not in path.parents
        ]
        assert outside == [], f"a write escaped the storage root: {outside}"

    def test_a_leading_slash_is_treated_as_relative(self, store):
        """An absolute-looking key must land inside the root, not at the filesystem root."""
        store.put("/runs/abs.bin", b"x")
        assert store.exists("runs/abs.bin")


class TestUrls:
    def test_local_url_is_a_file_uri_and_does_not_claim_to_be_signed(self, store):
        store.put("doc.txt", b"x")
        url = store.url("doc.txt", expires_s=60)
        assert url.startswith("file://")
        assert describe_storage()["signed_urls"] is False


class TestBackendSelection:
    def test_local_is_the_default(self, monkeypatch):
        monkeypatch.delenv("ODK_STORAGE_BACKEND", raising=False)
        assert build_storage().name == "local"

    def test_an_unknown_backend_raises_rather_than_falling_back(self, monkeypatch):
        """Silently storing data somewhere other than configured is worse than refusing."""
        monkeypatch.setenv("ODK_STORAGE_BACKEND", "dropbox")
        with pytest.raises(StorageError):
            build_storage()

    def test_s3_without_a_bucket_is_refused(self, monkeypatch):
        monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
        monkeypatch.delenv("ODK_S3_BUCKET", raising=False)
        with pytest.raises(StorageError):
            build_storage()

    def test_s3_backend_constructs_when_configured(self, monkeypatch):
        pytest.importorskip("boto3")
        monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("ODK_S3_BUCKET", "odk-test")
        monkeypatch.setenv("ODK_S3_ENDPOINT", "http://minio:9000")
        backend = build_storage()
        assert backend.name == "s3"
        assert backend.bucket == "odk-test"

    def test_describe_reports_the_live_backend(self, monkeypatch):
        monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("ODK_S3_BUCKET", "odk-test")
        described = describe_storage()
        assert described["backend"] == "s3"
        assert described["signed_urls"] is True


# Live S3 tests are skipped rather than mocked: a mocked boto3 proves the mock works,
# not the integration. Set ODK_TEST_S3_ENDPOINT against a real MinIO to run them.
LIVE_S3 = os.environ.get("ODK_TEST_S3_ENDPOINT", "")


@pytest.mark.skipif(not LIVE_S3, reason="Set ODK_TEST_S3_ENDPOINT to test against a live MinIO.")
class TestLiveS3:
    def test_round_trip(self, monkeypatch):
        pytest.importorskip("boto3")
        monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("ODK_S3_ENDPOINT", LIVE_S3)
        backend = build_storage()
        backend.put("test/round-trip.bin", b"live")
        assert backend.get("test/round-trip.bin") == b"live"
        assert backend.delete("test/round-trip.bin") is True
