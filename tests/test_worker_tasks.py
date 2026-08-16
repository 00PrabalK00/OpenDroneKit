"""Cancellation has to cross the gap between the process that asks and the one that runs.

core/processing_runs.py cancels through a module-level dict of threading.Events keyed by
run id. That is correct while the process calling stop_processing_run is the process
running the pipeline, and silently wrong the moment a worker is on another machine: the
API sets an Event in its own memory, marks the run cancelled, and the worker keeps
reconstructing for another forty minutes. The operator sees "cancelled" and the cluster
keeps burning.

So the request travels through Redis and the worker checks it at each progress callback
-- a stage boundary, where the run's recorded state and the files on disk agree.

The direction of the failure mode matters and is tested below: an unreachable broker
must answer "no cancel pending", never "cancel". Treating a network blip as a
cancellation would throw away an hour of reconstruction that was going fine.
"""

from __future__ import annotations

import pytest

pytest.importorskip("redis")

from services.worker.celery_app import broker_reachable  # noqa: E402
from services.worker.tasks import (  # noqa: E402
    CANCEL_PREFIX,
    cancel_key,
    cancel_requested,
    clear_cancel,
    execute_run,
    register,
    request_cancel,
)


def _live() -> bool:
    return bool(broker_reachable(timeout_s=0.5).get("reachable"))


live = pytest.mark.skipif(not _live(), reason="no Redis broker reachable")


class TestKeys:
    def test_the_key_is_namespaced_by_run(self) -> None:
        assert cancel_key("abc").startswith(CANCEL_PREFIX)
        assert cancel_key("abc") != cancel_key("abd")


class TestUnreachableBrokerFailsSafe:
    def test_no_broker_means_no_cancel_pending(self, monkeypatch) -> None:
        """The safe direction: a blip must not look like an operator pressing stop."""
        import services.worker.celery_app as module

        monkeypatch.setenv("ODK_BROKER_URL", "redis://127.0.0.1:6391/0")
        assert cancel_requested("whatever") is False

    def test_a_failed_request_is_reported_not_raised(self, monkeypatch) -> None:
        monkeypatch.setenv("ODK_BROKER_URL", "redis://127.0.0.1:6391/0")
        assert request_cancel("whatever") is False


@live
class TestCancelCrossesProcesses:
    def test_a_request_is_visible_to_a_separate_reader(self) -> None:
        """The whole point: the asker and the runner are not the same process."""
        run_id = "cross-process-run"
        clear_cancel(run_id)
        assert cancel_requested(run_id) is False
        assert request_cancel(run_id) is True
        assert cancel_requested(run_id) is True
        clear_cancel(run_id)
        assert cancel_requested(run_id) is False

    def test_cancelling_one_run_does_not_cancel_another(self) -> None:
        clear_cancel("run-a"); clear_cancel("run-b")
        request_cancel("run-a")
        assert cancel_requested("run-a") is True
        assert cancel_requested("run-b") is False
        clear_cancel("run-a")

    def test_the_key_is_cleared_after_the_run_finishes(self, tmp_path, monkeypatch) -> None:
        """A stale cancel key would kill the next run that reused the id."""
        run_id = "finishing-run"
        request_cancel(run_id)

        import services.worker.tasks as module

        monkeypatch.setattr(module, "run_pipeline", lambda *a, **k: None, raising=False)
        called = {}

        def fake_pipeline(project_root, rid, stages, progress_callback=None):
            called["ran"] = True
            if progress_callback:
                progress_callback(50.0, "half")
            return type("R", (), {"status": "completed"})()

        monkeypatch.setattr("core.processing_runs.run_pipeline", fake_pipeline)
        monkeypatch.setattr("core.processing_runs.stop_processing_run", lambda *a, **k: None)
        monkeypatch.setattr("core.processing_runs.get_processing_status",
                            lambda *a, **k: type("S", (), {"progress_percent": 100.0})())

        execute_run(str(tmp_path), run_id)
        assert called.get("ran")
        assert cancel_requested(run_id) is False, "the cancel key outlived its run"


class TestTheWorkerAsksTheRunToStop:
    def test_a_pending_cancel_reaches_stop_processing_run(self, tmp_path, monkeypatch) -> None:
        stopped = []
        monkeypatch.setattr("services.worker.tasks.cancel_requested", lambda rid: True)
        monkeypatch.setattr("services.worker.tasks.clear_cancel", lambda rid: None)
        monkeypatch.setattr("core.processing_runs.stop_processing_run",
                            lambda root, rid: stopped.append(rid))
        monkeypatch.setattr("core.processing_runs.get_processing_status",
                            lambda *a, **k: type("S", (), {"progress_percent": 12.0})())

        def fake_pipeline(project_root, rid, stages, progress_callback=None):
            progress_callback(10.0, "stage one")
            return type("R", (), {"status": "cancelled"})()

        monkeypatch.setattr("core.processing_runs.run_pipeline", fake_pipeline)
        result = execute_run(str(tmp_path), "r1")
        assert stopped == ["r1"], "the cancel never reached the running pipeline"
        assert result["status"] == "cancelled"

    def test_no_cancel_lets_the_run_finish(self, tmp_path, monkeypatch) -> None:
        stopped = []
        monkeypatch.setattr("services.worker.tasks.cancel_requested", lambda rid: False)
        monkeypatch.setattr("services.worker.tasks.clear_cancel", lambda rid: None)
        monkeypatch.setattr("core.processing_runs.stop_processing_run",
                            lambda root, rid: stopped.append(rid))
        monkeypatch.setattr("core.processing_runs.get_processing_status",
                            lambda *a, **k: type("S", (), {"progress_percent": 100.0})())
        monkeypatch.setattr("core.processing_runs.run_pipeline",
                            lambda p, r, s, progress_callback=None: (
                                progress_callback(100.0, "done"),
                                type("R", (), {"status": "completed"})())[1])
        result = execute_run(str(tmp_path), "r2")
        assert stopped == []
        assert result["status"] == "completed"

    def test_a_failure_is_re_raised_so_the_broker_can_retry(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("services.worker.tasks.cancel_requested", lambda rid: False)
        monkeypatch.setattr("services.worker.tasks.clear_cancel", lambda rid: None)

        def boom(*a, **k):
            raise RuntimeError("colmap died")

        monkeypatch.setattr("core.processing_runs.run_pipeline", boom)
        with pytest.raises(RuntimeError, match="colmap died"):
            execute_run(str(tmp_path), "r3")


class TestRegistration:
    def test_the_task_registers_onto_an_app(self) -> None:
        celery = pytest.importorskip("celery")
        app = celery.Celery("t", broker="memory://", backend="cache+memory://")
        task = register(app)
        assert task.name == "odk.processing.run"
        assert "odk.processing.run" in app.tasks
