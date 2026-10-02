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
# not the integration. Set ODK_TEST_S3_ENDPOINT (and ODK_S3_ACCESS_KEY /
# ODK_S3_SECRET_KEY, the keys the storage layer reads) against a real S3-compatible
# gateway -- the compose stack's SeaweedFS, AWS, Ceph -- to run them.
LIVE_S3 = os.environ.get("ODK_TEST_S3_ENDPOINT", "")


@pytest.fixture
def live_bucket(monkeypatch):
    """A bucket of this test's own on the live endpoint, removed afterwards.

    build_storage refuses S3 without ODK_S3_BUCKET (rightly), and this test used to set
    only the endpoint -- so against a real gateway it failed on configuration before it
    reached the network. The bucket is created with the same keys the storage layer
    uses, so a credential problem fails here, by name.
    """
    boto3 = pytest.importorskip("boto3")
    import uuid

    bucket = f"odk-test-{uuid.uuid4().hex[:12]}"
    client = boto3.client(
        "s3", endpoint_url=LIVE_S3, region_name=os.environ.get("ODK_S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("ODK_S3_ACCESS_KEY") or None,
        aws_secret_access_key=os.environ.get("ODK_S3_SECRET_KEY") or None,
    )
    client.create_bucket(Bucket=bucket)
    monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("ODK_S3_ENDPOINT", LIVE_S3)
    monkeypatch.setenv("ODK_S3_BUCKET", bucket)
    try:
        yield bucket
    finally:
        for item in client.list_objects_v2(Bucket=bucket).get("Contents", []):
            client.delete_object(Bucket=bucket, Key=item["Key"])
        client.delete_bucket(Bucket=bucket)


@pytest.mark.skipif(not LIVE_S3, reason="Set ODK_TEST_S3_ENDPOINT to test against a live S3 gateway.")
class TestLiveS3:
    def test_round_trip(self, live_bucket):
        backend = build_storage()
        assert backend.bucket == live_bucket
        backend.put("test/round-trip.bin", b"live")
        assert backend.get("test/round-trip.bin") == b"live"
        assert backend.exists("test/round-trip.bin") is True
        assert backend.delete("test/round-trip.bin") is True
        assert backend.exists("test/round-trip.bin") is False

    def test_streams_list_and_signed_urls(self, live_bucket):
        import io
        import urllib.request

        backend = build_storage()
        payload = bytes(range(256)) * 4096  # 1 MiB, uploaded as a stream
        assert backend.put("survey/a/images.bin", io.BytesIO(payload)) == f"s3://{live_bucket}/survey/a/images.bin"
        backend.put("survey/b/report.pdf", b"%PDF-1.4")
        assert backend.list("survey/") == ["survey/a/images.bin", "survey/b/report.pdf"]
        assert backend.list("survey/a/") == ["survey/a/images.bin"]
        # A genuinely signed URL: fetchable without credentials, and it is the object.
        with urllib.request.urlopen(backend.url("survey/a/images.bin", expires_s=120), timeout=30) as r:
            assert r.read() == payload

    def test_a_missing_object_raises_and_is_reported_absent(self, live_bucket):
        backend = build_storage()
        assert backend.exists("nothing/here.bin") is False
        with pytest.raises(Exception):
            backend.get("nothing/here.bin")

    def test_keys_that_escape_are_refused_before_the_network(self, live_bucket):
        backend = build_storage()
        with pytest.raises(StorageError):
            backend.put("../outside.bin", b"x")
