"""A job that finishes has to be observable as finished.

The reconstruction ran, wrote an orthomosaic, a DSM, a mesh and a camera track, and the
application went on saying it was working -- for five hours, until the process was killed.
Nothing was wrong with the reconstruction. `Api.job_status` answers `ok(job={...})`, both
pollers read `.state` off the envelope instead of the record, and `undefined` matches
none of the terminal branches, so the loop that was supposed to stop never could.

That is the same defect shape as `create_project`'s nested id, and it is invisible to
every test that asserts a call returns ok: the envelope is correct, the reading of it is
not. So this tests the round trip an operator actually depends on -- submit, poll,
observe an end -- rather than the shape of one reply.
"""

from __future__ import annotations

from pathlib import Path
import re
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def api(tmp_path, monkeypatch):
    # The store resolves its path relative to the working directory, so the test has to
    # move rather than configure. Without this it writes to the developer's real project
    # database and blocks on the running desktop app's write lock.
    monkeypatch.chdir(tmp_path)
    from app.api import Api
    from app.session import AppSession

    session = AppSession()
    session.create_project("Job polling", str(tmp_path / "project"))
    return Api(session)


def poll_to_completion(api, job_id: str, timeout: float = 20.0) -> dict:
    """Poll the way a caller must, and require that the polling can end."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = api.job_status(job_id).get("job") or {}
        state = str(record.get("state") or record.get("status") or "")
        if state in {"done", "failed", "cancelled"}:
            return record
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} never reported a terminal state within {timeout}s -- "
        "this is the bug that polled a finished reconstruction forever"
    )


def test_a_finished_job_reports_that_it_finished(api):
    job_id = api._jobs.submit("test", lambda progress, should_cancel, **_: {"value": 42})
    record = poll_to_completion(api, job_id)
    assert record["status"] == "done"
    assert record["result"]["value"] == 42


def test_a_failed_job_reports_the_reason(api):
    def explode(progress, should_cancel, **_):
        raise RuntimeError("no images in the folder")

    record = poll_to_completion(api, api._jobs.submit("test", explode))
    assert record["status"] == "failed"
    # The reason has to survive to the caller: "failed" with no why is not a report.
    assert "no images in the folder" in str(record.get("error", ""))


def test_the_status_record_is_nested_under_job(api):
    """Name the shape, so a future change to it fails here and not in the field."""
    envelope = api.job_status(api._jobs.submit("test", lambda progress, should_cancel, **_: {}))
    assert envelope["ok"] is True
    assert "job" in envelope, "callers unwrap .job; moving it silently breaks every poller"
    assert "status" not in envelope, "a status at the top level would make both readings look right"


def test_the_cockpit_unwraps_the_job_record():
    """The UI poller is JavaScript and cannot be reached from pytest -- so read it.

    A browser test would not have caught this either: the loop kept running and kept
    toasting, which looks alive. Only the absence of an end is the symptom, and that takes
    longer to notice than anyone waits.
    """
    source = (REPO_ROOT / "app" / "web" / "js" / "workspace" / "shell.js").read_text(encoding="utf-8")
    watcher = source[source.index("job_status"):][:600]
    assert re.search(r"\.job\s*\|\|", watcher), (
        "shell.js reads job state off the ok() envelope again -- terminal states "
        "will be unreachable and jobs will appear to run forever"
    )
