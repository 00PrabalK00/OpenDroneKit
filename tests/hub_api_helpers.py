"""Helpers for tests that drive the Hub API through its real HTTP surface.

Nothing here stubs the API: requests go through FastAPI's validation, the real routers
and an isolated SQLite database. Uploads use the resumable chunk protocol a client
would, so a dataset in a test holds exactly the bytes a user would have sent.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE_IMAGE = REPO / "tests" / "fixtures" / "concrete_cracked_public_domain.jpg"
TERMINAL = {"done", "failed", "cancelled"}


def reset_engine() -> None:
    import services.api.db as db_module

    db_module._engine = None
    db_module._SessionLocal = None


def configure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
    reset_engine()


def register(client, email: str = "owner@example.com", organization: str = "Survey Co"):
    response = client.post("/auth/register", json={
        "email": email, "password": "longenough1", "organization_name": organization,
    })
    assert response.status_code == 201, response.text
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    return headers, organization_id


def project_with_dataset(client, headers, organization_id, name: str = "Site"):
    project_id = client.post(f"/organizations/{organization_id}/projects", headers=headers,
                             json={"name": name, "client": "Municipal Works"}).json()["id"]
    dataset_id = client.post(f"/projects/{project_id}/datasets", headers=headers,
                             json={"name": "Imagery"}).json()["id"]
    return project_id, dataset_id


def upload(client, headers, dataset_id: int, path: Path = FIXTURE_IMAGE,
           chunk_size: int = 128 * 1024) -> dict:
    data = Path(path).read_bytes()
    begin = client.post(f"/datasets/{dataset_id}/uploads", headers=headers, json={
        "filename": Path(path).name, "total_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "chunk_size": chunk_size,
    })
    assert begin.status_code == 201, begin.text
    upload_id = begin.json()["upload_id"]
    for index in range(max(1, -(-len(data) // chunk_size))):
        part = data[index * chunk_size:(index + 1) * chunk_size]
        sent = client.put(f"/uploads/{upload_id}/chunks/{index}", headers=headers,
                          files={"chunk": ("chunk", part, "application/octet-stream")})
        assert sent.status_code == 200, sent.text
    done = client.post(f"/uploads/{upload_id}/complete", headers=headers)
    assert done.status_code == 200, done.text
    return done.json()


def wait(client, headers, path: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    body = client.get(path, headers=headers).json()
    while body.get("status") not in TERMINAL and time.monotonic() < deadline:
        time.sleep(0.1)
        body = client.get(path, headers=headers).json()
    return body


def weights_installed(model_key: str) -> bool:
    from core.models import model_status

    return bool(model_status(model_key).get("exists"))
