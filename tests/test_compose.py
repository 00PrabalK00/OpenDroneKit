"""The self-hosting deployment definition.

These tests read the compose file as data. They cannot prove the stack boots -- that
needs a Docker daemon -- but they do catch the mistakes that make a first run fail
confusingly: a service starting before its database accepts connections, a missing
healthcheck, or a credential committed to the repository.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DOCKER_DIR = Path(__file__).resolve().parents[1] / "infrastructure" / "docker"
COMPOSE_PATH = DOCKER_DIR / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    if not COMPOSE_PATH.exists():
        pytest.skip("docker-compose.yml not present")
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


class TestServices:
    def test_the_expected_services_are_defined(self, compose):
        services = compose["services"]
        for name in ("postgis", "seaweedfs", "api", "worker"):
            assert name in services, f"{name} service missing"

    def test_the_database_is_postgis_not_plain_postgres(self, compose):
        """Plain postgres would accept the schema and then fail on spatial queries."""
        image = compose["services"]["postgis"]["image"]
        assert "postgis" in image.lower()

    def test_every_long_running_service_has_a_healthcheck(self, compose):
        for name, service in compose["services"].items():
            if name.endswith("-init"):
                continue  # one-shot containers exit; a healthcheck is meaningless
            assert "healthcheck" in service, f"{name} has no healthcheck"

    def test_the_api_waits_for_a_healthy_database(self, compose):
        """postgis reports 'up' well before it accepts connections on first boot."""
        depends = compose["services"]["api"]["depends_on"]
        assert depends["postgis"]["condition"] == "service_healthy"
        # Not merely the object store up: the bucket made, with the API's own keys.
        assert depends["seaweedfs-init"]["condition"] == "service_completed_successfully"

    def test_the_api_serves_the_asgi_app(self, compose):
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        dockerfile = (DOCKER_DIR / "Dockerfile.api").read_text(encoding="utf-8")
        assert "services.api.main:app" in dockerfile or "services.api.main:app" in text

    def test_the_bucket_is_created_before_first_use(self, compose):
        """Without this the first upload fails against a bucket that does not exist."""
        assert "seaweedfs-init" in compose["services"]
        init = compose["services"]["seaweedfs-init"]
        assert init["depends_on"]["seaweedfs"]["condition"] == "service_healthy"


class TestObjectStore:
    """MinIO stopped publishing images (minio/minio and quay.io/minio/minio both refuse
    pulls as of 2026-09-29), so a stack naming them could not start at all."""

    def test_no_service_uses_an_image_that_can_no_longer_be_pulled(self, compose):
        for name, service in compose["services"].items():
            image = str(service.get("image", ""))
            assert "minio" not in image.lower(), f"{name} still uses {image}"

    def test_the_object_store_is_seaweedfs_with_its_s3_gateway(self, compose):
        service = compose["services"]["seaweedfs"]
        assert service["image"].startswith("chrislusf/seaweedfs")
        script = " ".join(service["command"])
        assert "server" in script and "-s3" in script and "-s3.port=8333" in script
        # The gateway's identity is the API's keys, from the environment, via a config
        # file -- without one SeaweedFS rejects every signed request (AccessDenied).
        assert "-s3.config=" in script
        assert "$$ODK_S3_ACCESS_KEY" in script and "$$ODK_S3_SECRET_KEY" in script
        env = service["environment"]
        assert "${ODK_S3_ACCESS_KEY:?" in env["ODK_S3_ACCESS_KEY"]

    def test_the_api_and_worker_point_at_the_gateway_that_exists(self, compose):
        for name in ("api", "worker"):
            env = compose["services"][name]["environment"]
            assert env["ODK_S3_ENDPOINT"] == "http://seaweedfs:8333", name
            # The variable names the storage layer reads did not change.
            for variable in ("ODK_S3_BUCKET", "ODK_S3_ACCESS_KEY", "ODK_S3_SECRET_KEY"):
                assert variable in env, (name, variable)


class TestPersistence:
    def test_named_volumes_exist_for_stateful_services(self, compose):
        volumes = compose.get("volumes", {})
        for name in ("postgis_data", "seaweedfs_data"):
            assert name in volumes, f"{name} volume missing; data would not survive a restart"

    def test_the_database_persists_to_a_volume(self, compose):
        mounts = compose["services"]["postgis"]["volumes"]
        assert any("postgis_data" in str(mount) for mount in mounts)


class TestSecrets:
    def test_no_literal_secret_is_committed(self, compose):
        """Every credential must come from the environment, never from this file.

        Only YAML mappings are examined. A shell command inside an entrypoint legally
        contains `$${VAR}` -- compose's escape for a variable the shell expands -- and
        that is an environment reference, not a committed secret.
        """
        secret_names = ("PASSWORD", "SECRET_KEY", "ACCESS_KEY")

        def walk(node, path="") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{path}.{key}" if path else str(key))
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]")
            else:
                leaf = path.rsplit(".", 1)[-1].upper()
                if any(name in leaf for name in secret_names):
                    value = str(node).strip()
                    assert not value or "${" in value, \
                        f"a credential appears to be hard-coded at {path}: {value!r}"

        walk(compose)

    def test_required_secrets_fail_fast_when_unset(self, compose):
        """`:?` makes compose refuse to start rather than booting insecurely."""
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        for variable in ("POSTGRES_PASSWORD", "ODK_SECRET_KEY", "ODK_S3_ACCESS_KEY"):
            assert re.search(rf"\$\{{{variable}:\?", text), \
                f"{variable} should use ${{VAR:?message}} so an unset value stops the stack"

    def test_env_example_exists_and_holds_no_real_secret(self):
        example = DOCKER_DIR / ".env.example"
        assert example.exists(), ".env.example is needed for a first run"
        text = example.read_text(encoding="utf-8")
        assert "ODK_SECRET_KEY=" in text
        # The signing key must be left blank rather than shipped with a default that
        # a deployment might keep.
        for line in text.splitlines():
            if line.startswith("ODK_SECRET_KEY="):
                assert line.strip() == "ODK_SECRET_KEY="

    def test_a_real_env_file_is_not_committed(self):
        assert not (DOCKER_DIR / ".env").exists(), "infrastructure/docker/.env must not be committed"


class TestImages:
    def test_dockerfiles_exist(self):
        assert (DOCKER_DIR / "Dockerfile.api").exists()
        assert (DOCKER_DIR / "Dockerfile.worker").exists()

    def test_the_api_runs_as_a_non_root_user(self):
        """A web-facing process should not be able to rewrite its own code."""
        text = (DOCKER_DIR / "Dockerfile.api").read_text(encoding="utf-8")
        assert re.search(r"^USER\s+\w+", text, re.M), "Dockerfile.api never drops root"

    def test_the_worker_runs_as_a_non_root_user(self):
        text = (DOCKER_DIR / "Dockerfile.worker").read_text(encoding="utf-8")
        assert re.search(r"^USER\s+\w+", text, re.M), "Dockerfile.worker never drops root"

    def test_geospatial_system_libraries_are_installed(self):
        """rasterio and pyproj need GDAL and PROJ present in the image."""
        text = (DOCKER_DIR / "Dockerfile.api").read_text(encoding="utf-8")
        assert "gdal" in text.lower()
        assert "proj" in text.lower()


# Booting the object store for real needs a Docker daemon and the seaweedfs image, so it
# runs only when asked. Skipped is not passed: this is what proves the compose service
# starts, accepts the API's keys, and holds the bucket the API writes to.
LIVE_COMPOSE = os.environ.get("ODK_TEST_DOCKER_COMPOSE", "") == "1"


@pytest.mark.skipif(not LIVE_COMPOSE, reason="Set ODK_TEST_DOCKER_COMPOSE=1 to boot the object store with docker compose.")
def test_the_compose_object_store_boots_and_the_storage_layer_round_trips(tmp_path, monkeypatch):
    import secrets
    import socket
    import subprocess
    import urllib.request

    pytest.importorskip("boto3")
    from services.api.storage import build_storage

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    access, secret = "odk" + secrets.token_hex(6), secrets.token_hex(16)
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"POSTGRES_PASSWORD=unused\nODK_SECRET_KEY=unused\nODK_S3_ACCESS_KEY={access}\n"
        f"ODK_S3_SECRET_KEY={secret}\nODK_S3_BUCKET=odk-compose-test\nODK_S3_PORT={port}\n",
        encoding="utf-8")
    project = "odkcompose" + secrets.token_hex(3)
    base = ["docker", "compose", "-p", project, "-f", str(COMPOSE_PATH), "--env-file", str(env_file)]
    # Compose prefers the shell's variables over --env-file, so a developer's exported
    # ODK_S3_* would silently become the gateway's keys. Only the file speaks here.
    compose_env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("ODK_", "POSTGRES_"))}

    def compose(*args):
        return subprocess.run(base + list(args), capture_output=True, text=True, timeout=300,
                              env=compose_env)

    try:
        up = compose("up", "-d", "--wait", "seaweedfs")
        assert up.returncode == 0, up.stderr
        init = compose("run", "--rm", "seaweedfs-init")
        assert init.returncode == 0 and "bucket ready" in init.stdout, (init.stdout, init.stderr)
        # A second run finds the bucket already there, and that is success too.
        again = compose("run", "--rm", "seaweedfs-init")
        assert again.returncode == 0 and "bucket ready" in again.stdout, (again.stdout, again.stderr)

        monkeypatch.setenv("ODK_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("ODK_S3_ENDPOINT", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("ODK_S3_BUCKET", "odk-compose-test")
        monkeypatch.setenv("ODK_S3_ACCESS_KEY", access)
        monkeypatch.setenv("ODK_S3_SECRET_KEY", secret)
        storage = build_storage()
        storage.put("survey/one.bin", b"compose")
        assert storage.get("survey/one.bin") == b"compose"
        assert storage.list("survey/") == ["survey/one.bin"]
        # A signed URL really works against the gateway.
        assert urllib.request.urlopen(storage.url("survey/one.bin"), timeout=30).read() == b"compose"
        assert storage.delete("survey/one.bin") is True
        assert storage.exists("survey/one.bin") is False

        # Any other key is refused: the gateway has exactly one identity.
        monkeypatch.setenv("ODK_S3_SECRET_KEY", "not-the-secret")
        with pytest.raises(Exception):
            build_storage().put("survey/two.bin", b"x")
    finally:
        compose("down", "-v")
