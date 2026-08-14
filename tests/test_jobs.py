"""Job lifecycle.

A job that dies without reporting used to read "running" forever, which showed the
UI a phantom job with a Cancel button that could never do anything.
"""

from __future__ import annotations

import time

import pytest

from app.jobs import JobManager


def _wait_for(manager: JobManager, job_id: str, statuses: set[str], timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job and job["status"] in statuses:
            return job
        time.sleep(0.02)
    return manager.get(job_id)


def test_successful_job_reports_done_and_its_result():
    manager = JobManager()

    def work(progress, should_cancel):
        progress(50, "half")
        return 42

    job_id = manager.submit("ok", work)
    job = _wait_for(manager, job_id, {"done", "failed"})
    assert job["status"] == "done"
    assert job["result"] == 42
    assert job["percent"] == 100


def test_cooperative_cancel_stops_the_job():
    manager = JobManager()

    def work(progress, should_cancel):
        for index in range(500):
            progress(index % 100, f"step {index}")
            time.sleep(0.01)
        return "never"

    job_id = manager.submit("slow", work)
    _wait_for(manager, job_id, {"running"})
    assert manager.cancel(job_id) is True

    job = _wait_for(manager, job_id, {"cancelled", "failed", "done"})
    assert job["status"] == "cancelled"


def test_ordinary_exception_is_recorded_as_failed():
    manager = JobManager()

    def work(progress, should_cancel):
        raise ValueError("bad input")

    job_id = manager.submit("boom", work)
    job = _wait_for(manager, job_id, {"failed", "done"})
    assert job["status"] == "failed"
    assert "ValueError" in job["error"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_base_exception_does_not_strand_the_job_as_running():
    """SystemExit bypasses `except Exception` and used to leave status stuck.

    The manager records the failure and then re-raises, so the worker thread really
    does die; pytest notices that, which is the behaviour under test.
    """
    manager = JobManager()

    def work(progress, should_cancel):
        progress(5, "about to die")
        raise SystemExit("abrupt")

    job_id = manager.submit("hard-exit", work)
    job = _wait_for(manager, job_id, {"failed", "cancelled", "done"})
    assert job["status"] == "failed", "a job that died must never keep reading 'running'"
    assert "SystemExit" in job["error"]


def test_cancelling_a_finished_job_is_refused():
    manager = JobManager()
    job_id = manager.submit("quick", lambda progress, should_cancel: 1)
    _wait_for(manager, job_id, {"done", "failed"})
    assert manager.cancel(job_id) is False


def test_active_lists_only_unfinished_jobs():
    manager = JobManager()
    job_id = manager.submit("quick", lambda progress, should_cancel: 1)
    _wait_for(manager, job_id, {"done", "failed"})
    assert all(job["id"] != job_id for job in manager.active())


def test_progress_messages_are_logged():
    manager = JobManager()

    def work(progress, should_cancel):
        progress(10, "first")
        progress(90, "second")
        return None

    job_id = manager.submit("logged", work)
    job = _wait_for(manager, job_id, {"done", "failed"})
    assert any("first" in line for line in job["log"])
    assert any("second" in line for line in job["log"])
