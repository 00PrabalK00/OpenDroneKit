"""CI must fail when a test skips, not pass quietly.

This project's tests skip when their dependency is absent, which is right on a laptop --
a developer without Docker should not be blocked -- and a lie in CI, where the whole
point is that the dependency IS there. A spatial suite that skips because the database
never started reports green while checking nothing.

fl.sitl is the concrete case. The flight tests pass against real ArduPilot in a container
and skip under a plain pytest, so the registry row reads not_started on any laptop no
matter how many times the container goes green. CI is the only place that gap can close,
and only if CI notices the difference between passing and skipping.

So these tests check the workflow asserts its own tests ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


class TestTheWorkflowExists:
    def test_it_is_valid_yaml(self, workflow) -> None:
        assert isinstance(workflow, dict)

    def test_it_runs_on_push_and_pull_request(self, workflow) -> None:
        # `on` parses as the boolean True in YAML 1.1, which is a classic trap.
        triggers = workflow.get("on") or workflow.get(True)
        assert "push" in triggers and "pull_request" in triggers

    def test_the_three_jobs_are_present(self, workflow) -> None:
        assert set(workflow["jobs"]) == {"tests", "spatial", "sitl"}


class TestSkipsCannotPassAsSuccess:
    """The point of the file."""

    def test_the_spatial_job_asserts_its_tests_ran(self, raw) -> None:
        assert "spatial tests skipped in CI" in raw, (
            "the spatial job would be green when PostGIS never came up"
        )

    def test_the_sitl_job_asserts_its_tests_ran(self, raw) -> None:
        assert "SITL tests did not run" in raw, (
            "a skipped SITL suite is exactly what makes fl.sitl read not_started; a green "
            "tick over a skip would hide that permanently"
        )

    def test_the_sitl_job_requires_a_pass_count(self, raw) -> None:
        assert 'grep -E "[0-9]+ passed" sitl.log' in raw


class TestTheJobsHaveWhatTheyNeed:
    def test_the_spatial_job_runs_a_real_postgis(self, workflow) -> None:
        services = workflow["jobs"]["spatial"].get("services", {})
        assert "postgis" in services
        assert "postgis" in services["postgis"]["image"]

    def test_the_spatial_service_is_health_checked(self, workflow) -> None:
        # Starting the container is not the same as it being ready to accept a connection.
        options = workflow["jobs"]["spatial"]["services"]["postgis"].get("options", "")
        assert "health-cmd" in options

    def test_the_test_job_has_node_for_the_ui_check(self, workflow) -> None:
        # The workspace UI is verified by executing all fourteen workspaces under Node.
        steps = workflow["jobs"]["tests"]["steps"]
        assert any("setup-node" in str(step.get("uses", "")) for step in steps)

    def test_the_sitl_job_allows_time_to_build_ardupilot(self, workflow) -> None:
        # Building from source takes 15-25 minutes; a short timeout would fail the job
        # for being slow rather than for being wrong.
        assert workflow["jobs"]["sitl"]["timeout-minutes"] >= 60

    def test_every_job_has_a_timeout(self, workflow) -> None:
        for name, job in workflow["jobs"].items():
            assert job.get("timeout-minutes"), f"{name} could hang forever"


class TestStatusCannotSilentlyRegress:
    def test_ci_runs_the_strict_status_check(self, raw) -> None:
        """Status is computed from passing tests, so a broken test downgrades a row.

        Without --strict that downgrade lands in a generated document nobody reads on
        the day it happens.
        """
        assert "feature_status.py --strict" in raw
