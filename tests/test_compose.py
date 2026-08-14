"""The self-hosting deployment definition.

These tests read the compose file as data. They cannot prove the stack boots -- that
needs a Docker daemon -- but they do catch the mistakes that make a first run fail
confusingly: a service starting before its database accepts connections, a missing
healthcheck, or a credential committed to the repository.
"""

from __future__ import annotations

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
        for name in ("postgis", "minio", "api", "worker"):
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
        assert depends["minio"]["condition"] == "service_healthy"

    def test_the_api_serves_the_asgi_app(self, compose):
        text = COMPOSE_PATH.read_text(encoding="utf-8")
        dockerfile = (DOCKER_DIR / "Dockerfile.api").read_text(encoding="utf-8")
        assert "services.api.main:app" in dockerfile or "services.api.main:app" in text

    def test_the_bucket_is_created_before_first_use(self, compose):
        """Without this the first upload fails against a bucket that does not exist."""
        assert "minio-init" in compose["services"]


class TestPersistence:
    def test_named_volumes_exist_for_stateful_services(self, compose):
        volumes = compose.get("volumes", {})
        for name in ("postgis_data", "minio_data"):
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
