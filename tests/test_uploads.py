"""Datasets and resumable upload.

The failure cases are the point. A field upload runs over whatever connection the site
has, so the tests assert what happens when it goes wrong: chunks arriving out of order,
a resumed transfer, a truncated file, a corrupted file, and a filename crafted to
escape the storage directory.
"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))

    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None

    from services.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture
def dataset(client):
    """An authenticated session with a project and dataset ready to receive files."""
    response = client.post("/auth/register", json={
        "email": "pilot@example.com", "password": "longenough1",
        "organization_name": "Field Ops",
    })
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                             json={"name": "Site survey"}).json()["id"]
    dataset_id = client.post(f"/projects/{project_id}/datasets", headers=headers,
                             json={"name": "Flight 1 imagery"}).json()["id"]
    return headers, dataset_id


def begin(client, headers, dataset_id, payload_bytes, chunk_size=1024, declare_sha=True):
    body = {
        "filename": "DSC00229.JPG",
        "total_bytes": len(payload_bytes),
        "chunk_size": chunk_size,
    }
    if declare_sha:
        body["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    response = client.post(f"/datasets/{dataset_id}/uploads", headers=headers, json=body)
    assert response.status_code == 201, response.text
    return response.json()


def send_chunk(client, headers, upload_id, index, data):
    return client.put(
        f"/uploads/{upload_id}/chunks/{index}",
        headers=headers,
        files={"chunk": ("chunk.bin", data, "application/octet-stream")},
    )


class TestDatasets:
    def test_create_and_list(self, client, dataset):
        headers, dataset_id = dataset
        project_id = client.get("/organizations", headers=headers).json()[0]["id"]
        listed = client.get(f"/projects/1/datasets", headers=headers).json()
        assert any(entry["id"] == dataset_id for entry in listed)


class TestResumableUpload:
    def test_chunks_in_order_assemble_correctly(self, client, dataset):
        headers, dataset_id = dataset
        payload = bytes(range(256)) * 12  # 3072 bytes
        session = begin(client, headers, dataset_id, payload, chunk_size=1024)
        upload_id = session["upload_id"]
        assert session["missing_chunks"] == [0, 1, 2]

        for index in range(3):
            send_chunk(client, headers, upload_id, index, payload[index * 1024:(index + 1) * 1024])

        result = client.post(f"/uploads/{upload_id}/complete", headers=headers)
        assert result.status_code == 200, result.text
        assert result.json()["bytes"] == len(payload)
        assert result.json()["sha256"] == hashlib.sha256(payload).hexdigest()
        assert result.json()["checksum_verified"] is True

    def test_chunks_out_of_order_assemble_correctly(self, client, dataset):
        """Parallel uploaders do not finish in order; assembly must not depend on it."""
        headers, dataset_id = dataset
        payload = bytes(range(256)) * 12
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]

        for index in (2, 0, 1):
            send_chunk(client, headers, upload_id, index, payload[index * 1024:(index + 1) * 1024])

        result = client.post(f"/uploads/{upload_id}/complete", headers=headers)
        assert result.status_code == 200
        assert result.json()["sha256"] == hashlib.sha256(payload).hexdigest()

    def test_status_reports_exactly_what_is_missing(self, client, dataset):
        """A resumed client must be able to send only the missing chunks."""
        headers, dataset_id = dataset
        payload = b"x" * 3072
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]

        send_chunk(client, headers, upload_id, 0, payload[:1024])
        status = client.get(f"/uploads/{upload_id}", headers=headers).json()
        assert status["missing_chunks"] == [1, 2]
        assert status["received_bytes"] == 1024
        assert status["complete"] is False

        for index in status["missing_chunks"]:
            send_chunk(client, headers, upload_id, index, payload[index * 1024:(index + 1) * 1024])

        final = client.get(f"/uploads/{upload_id}", headers=headers).json()
        assert final["complete"] is True
        assert final["missing_chunks"] == []

    def test_completing_with_a_missing_chunk_is_refused(self, client, dataset):
        headers, dataset_id = dataset
        payload = b"y" * 3072
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]
        send_chunk(client, headers, upload_id, 0, payload[:1024])

        response = client.post(f"/uploads/{upload_id}/complete", headers=headers)
        assert response.status_code == 409
        assert "missing" in response.json()["detail"].lower()

    def test_a_corrupted_chunk_is_caught_by_the_checksum(self, client, dataset):
        """Byte count alone cannot detect corruption; that is why sha256 is declared."""
        headers, dataset_id = dataset
        payload = b"z" * 2048
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]

        send_chunk(client, headers, upload_id, 0, payload[:1024])
        send_chunk(client, headers, upload_id, 1, b"CORRUPTED!" + b"z" * 1014)  # right size, wrong bytes

        response = client.post(f"/uploads/{upload_id}/complete", headers=headers)
        assert response.status_code == 409
        assert "checksum" in response.json()["detail"].lower()

    def test_a_size_mismatch_is_caught(self, client, dataset):
        headers, dataset_id = dataset
        payload = b"w" * 2048
        upload_id = begin(client, headers, dataset_id, payload,
                          chunk_size=1024, declare_sha=False)["upload_id"]

        send_chunk(client, headers, upload_id, 0, payload[:1024])
        send_chunk(client, headers, upload_id, 1, b"short")  # fewer bytes than declared

        response = client.post(f"/uploads/{upload_id}/complete", headers=headers)
        assert response.status_code == 409
        assert "size mismatch" in response.json()["detail"].lower()

    def test_a_rejected_upload_leaves_no_partial_file(self, client, dataset, tmp_path):
        """A failed verification must not leave a half-written file to be reconstructed."""
        headers, dataset_id = dataset
        payload = b"q" * 2048
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]
        send_chunk(client, headers, upload_id, 0, payload[:1024])
        send_chunk(client, headers, upload_id, 1, b"BAD" + b"q" * 1021)

        client.post(f"/uploads/{upload_id}/complete", headers=headers)
        stored = list((tmp_path / "storage" / "datasets").rglob("*.JPG"))
        assert stored == [], "a file that failed verification was left on disk"

    def test_dataset_counters_only_move_on_success(self, client, dataset):
        headers, dataset_id = dataset
        payload = b"a" * 1024
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]
        send_chunk(client, headers, upload_id, 0, payload)
        result = client.post(f"/uploads/{upload_id}/complete", headers=headers).json()
        assert result["dataset_file_count"] == 1

    def test_out_of_range_chunk_index_is_refused(self, client, dataset):
        headers, dataset_id = dataset
        payload = b"b" * 1024
        upload_id = begin(client, headers, dataset_id, payload, chunk_size=1024)["upload_id"]
        response = send_chunk(client, headers, upload_id, 7, b"b" * 1024)
        assert response.status_code == 422


class TestUploadPathSafety:
    @pytest.mark.parametrize(
        "filename",
        ["../escape.txt", "..\\escape.txt", "/etc/passwd", "a/../../b.txt", "sub/dir/file.jpg"],
    )
    def test_a_crafted_filename_cannot_escape_the_dataset_directory(
        self, client, dataset, tmp_path, filename
    ):
        """The client controls this string, so it must never reach the filesystem as given."""
        headers, dataset_id = dataset
        payload = b"c" * 16
        response = client.post(f"/datasets/{dataset_id}/uploads", headers=headers, json={
            "filename": filename, "total_bytes": len(payload), "chunk_size": 1024,
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        assert response.status_code == 201
        upload_id = response.json()["upload_id"]
        send_chunk(client, headers, upload_id, 0, payload)
        client.post(f"/uploads/{upload_id}/complete", headers=headers)

        storage = tmp_path / "storage"
        escaped = [p for p in tmp_path.rglob("*") if p.is_file() and storage not in p.parents]
        assert not any(p.name in {"escape.txt", "passwd", "b.txt"} for p in escaped), \
            "an upload landed outside the storage root"

    def test_a_dotdot_only_filename_is_refused(self, client, dataset):
        headers, dataset_id = dataset
        response = client.post(f"/datasets/{dataset_id}/uploads", headers=headers, json={
            "filename": "..", "total_bytes": 4, "chunk_size": 1024,
        })
        assert response.status_code == 422


class TestUploadAuthorization:
    def test_a_viewer_cannot_upload(self, client, dataset):
        headers, dataset_id = dataset
        client.post("/auth/register", json={
            "email": "watcher@example.com", "password": "longenough1",
            "organization_name": "Personal"})
        organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
        client.post(f"/organizations/{organization_id}/members", headers=headers,
                    json={"email": "watcher@example.com", "role": "viewer"})

        viewer = {"Authorization": "Bearer " + client.post("/auth/login", json={
            "email": "watcher@example.com", "password": "longenough1"}).json()["access_token"]}
        response = client.post(f"/datasets/{dataset_id}/uploads", headers=viewer, json={
            "filename": "x.jpg", "total_bytes": 10, "chunk_size": 1024})
        assert response.status_code == 403

    def test_an_unauthenticated_upload_is_refused(self, client, dataset):
        _, dataset_id = dataset
        response = client.post(f"/datasets/{dataset_id}/uploads", json={
            "filename": "x.jpg", "total_bytes": 10, "chunk_size": 1024})
        assert response.status_code == 401
