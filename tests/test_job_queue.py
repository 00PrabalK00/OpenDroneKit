"""Concurrency is bounded, priority is respected, and a retry admits what it did.

Reconstruction is memory-bound before it is CPU-bound. Eight simultaneous jobs on a
four-core box do not finish sooner than four -- they finish later, or one gets
OOM-killed halfway through and loses hours of work that was nearly done. So the worker
count is a promise, not a hint, and the first test here is that it holds under load.

The retry tests are the ones that matter most. A retry asserts the failure was
transient, and that assertion is usually wrong: a deterministic failure retried three
times is the same failure three times, an hour later. So retries are opt-in, every
attempt keeps its own error, and a job out of attempts reports `failed` rather than
sitting in a state that looks like work still to come.
"""

from __future__ import annotations

import threading
import time

import pytest

from core.job_queue import JobCancelled, JobQueue


@pytest.fixture
def queue():
    q = JobQueue(workers=2)
    q.start()
    yield q
    q.shutdown()


class TestConcurrencyIsBounded:
    def test_never_more_jobs_run_at_once_than_workers(self) -> None:
        """The promise the whole module exists to keep."""
        peak = 0
        live = 0
        lock = threading.Lock()

        def work():
            nonlocal peak, live
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lock:
                live -= 1

        q = JobQueue(workers=3)
        q.start()
        try:
            for _ in range(20):
                q.submit("load", work)
            assert q.join(timeout=30.0)
        finally:
            q.shutdown()
        assert peak <= 3, f"{peak} jobs ran at once against a 3-worker limit"

    def test_a_queue_needs_at_least_one_worker(self) -> None:
        with pytest.raises(ValueError):
            JobQueue(workers=0)

    def test_every_submitted_job_eventually_runs(self, queue) -> None:
        ids = [queue.submit("t", lambda: 7) for _ in range(15)]
        assert queue.join(timeout=30.0)
        assert all(queue.status(i)["status"] == "done" for i in ids)


class TestPriority:
    def test_higher_priority_work_runs_first(self) -> None:
        order: list[str] = []
        lock = threading.Lock()

        def record(tag: str):
            def run():
                with lock:
                    order.append(tag)
                time.sleep(0.01)
            return run

        # One worker, so ordering is the queue's decision rather than a race.
        q = JobQueue(workers=1)
        try:
            blocker = threading.Event()
            q.submit("blocker", blocker.wait, priority=0)
            for index in range(5):
                q.submit(f"low-{index}", record(f"low-{index}"), priority=200)
            q.submit("urgent", record("urgent"), priority=1)
            q.start()
            blocker.set()
            assert q.join(timeout=30.0)
        finally:
            q.shutdown()
        assert order[0] == "urgent", f"priority ignored: {order}"

    def test_equal_priority_runs_first_in_first_out(self) -> None:
        order: list[int] = []

        def record(index: int):
            return lambda: order.append(index)

        q = JobQueue(workers=1)
        try:
            for index in range(6):
                q.submit(f"job-{index}", record(index), priority=50)
            q.start()
            assert q.join(timeout=30.0)
        finally:
            q.shutdown()
        assert order == sorted(order), f"equal priority did not run in order: {order}"

    def test_low_priority_work_still_completes(self, queue) -> None:
        # Strict priority can starve; it must not deadlock.
        low = queue.submit("low", lambda: "done", priority=900)
        for _ in range(5):
            queue.submit("high", lambda: None, priority=1)
        assert queue.join(timeout=30.0)
        assert queue.status(low)["status"] == "done"


class TestRetries:
    def test_a_job_without_retries_fails_once(self, queue) -> None:
        calls = []

        def always_fails():
            calls.append(1)
            raise RuntimeError("nope")

        job = queue.submit("once", always_fails)
        assert queue.join(timeout=30.0)
        assert len(calls) == 1, "a job that did not ask for retries was retried"
        assert queue.status(job)["status"] == "failed"

    def test_retries_stop_at_the_limit(self, queue) -> None:
        calls = []

        def always_fails():
            calls.append(1)
            raise RuntimeError("still nope")

        job = queue.submit("thrice", always_fails, max_attempts=3)
        assert queue.join(timeout=30.0)
        assert len(calls) == 3
        assert queue.status(job)["attempts_made"] == 3

    def test_a_transient_failure_succeeds_on_a_later_attempt(self, queue) -> None:
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise OSError("disk busy")
            return "recovered"

        job = queue.submit("flaky", flaky, max_attempts=5)
        assert queue.join(timeout=30.0)
        assert queue.status(job)["status"] == "done"
        assert queue.result(job) == "recovered"

    def test_every_attempt_keeps_its_own_error(self, queue) -> None:
        """Three different failures is a different problem from the same one thrice."""
        errors = iter(["OSError: disk busy", "ValueError: bad header", "RuntimeError: gone"])

        def varied():
            raise RuntimeError(next(errors))

        job = queue.submit("varied", varied, max_attempts=3)
        assert queue.join(timeout=30.0)
        history = queue.status(job)["attempt_history"]
        assert len(history) == 3
        assert len({entry["error"] for entry in history}) == 3, (
            "attempt errors were collapsed; the summary hides which failure you have"
        )

    def test_an_exhausted_job_is_failed_not_queued(self, queue) -> None:
        """Work that will never run again must not look like work still to come."""
        job = queue.submit("doomed", lambda: (_ for _ in ()).throw(RuntimeError("x")),
                           max_attempts=2)
        assert queue.join(timeout=30.0)
        assert queue.status(job)["status"] == "failed"

    def test_max_attempts_below_one_is_refused(self, queue) -> None:
        with pytest.raises(ValueError):
            queue.submit("bad", lambda: None, max_attempts=0)


class TestCancellation:
    def test_a_queued_job_can_be_cancelled_before_it_starts(self) -> None:
        q = JobQueue(workers=1)
        try:
            blocker = threading.Event()
            q.submit("blocker", blocker.wait, priority=0)
            victim = q.submit("victim", lambda: "should not run", priority=10)
            q.start()
            assert q.cancel(victim)
            blocker.set()
            assert q.join(timeout=30.0)
            assert q.status(victim)["status"] == "cancelled"
            assert q.result(victim) is None
        finally:
            q.shutdown()

    def test_a_running_job_cancels_cooperatively(self, queue) -> None:
        started = threading.Event()

        def cooperative(should_cancel):
            started.set()
            for _ in range(500):
                if should_cancel():
                    raise JobCancelled()
                time.sleep(0.005)
            return "finished"

        job = queue.submit("coop", cooperative)
        assert started.wait(timeout=5.0)
        assert queue.cancel(job)
        assert queue.join(timeout=30.0)
        assert queue.status(job)["status"] == "cancelled"

    def test_a_finished_job_cannot_be_cancelled_retroactively(self, queue) -> None:
        # Reporting a completed job as cancelled would discard a real result.
        job = queue.submit("quick", lambda: 1)
        assert queue.join(timeout=30.0)
        assert queue.cancel(job) is False
        assert queue.status(job)["status"] == "done"

    def test_cancelling_an_unknown_job_is_false_not_an_error(self, queue) -> None:
        assert queue.cancel("nosuchjob") is False


class TestReporting:
    def test_the_report_counts_what_is_queued_and_running(self, queue) -> None:
        for _ in range(4):
            queue.submit("t", lambda: time.sleep(0.01))
        report = queue.report()
        assert report["workers"] == 2
        assert set(report["by_status"]) <= {"queued", "running", "done", "failed", "cancelled"}
        assert queue.join(timeout=30.0)

    def test_waiting_time_is_reported_so_starvation_is_visible(self) -> None:
        """Strict priority can starve. The only symptom is a job that never starts."""
        q = JobQueue(workers=1)
        try:
            blocker = threading.Event()
            q.submit("blocker", blocker.wait, priority=0)
            q.submit("starved", lambda: None, priority=900)
            q.start()
            time.sleep(0.1)
            assert q.report()["longest_wait_s"] > 0.0
            blocker.set()
            assert q.join(timeout=30.0)
        finally:
            q.shutdown()

    def test_retried_jobs_are_counted(self, queue) -> None:
        queue.submit("flaky", lambda: (_ for _ in ()).throw(OSError("x")), max_attempts=2)
        assert queue.join(timeout=30.0)
        assert queue.report()["retried_jobs"] == 1

    def test_an_unknown_job_id_raises(self, queue) -> None:
        with pytest.raises(KeyError):
            queue.status("nope")


class TestTheApiSurface:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("queue", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_a_session_without_a_queue_says_so_rather_than_pretending(self, api) -> None:
        """The desktop shell runs jobs directly. Reporting an empty queue would imply
        there is one and that it is idle, which is a different thing from absent."""
        result = api.processing_queue_report()
        assert not result["ok"]
        assert "no processing queue" in result["error"].lower()

    def test_an_attached_queue_reports_through_the_api(self, api) -> None:
        queue = JobQueue(workers=2)
        queue.start()
        try:
            api._session.processing_queue = queue
            queue.submit("t", lambda: 1)
            assert queue.join(timeout=30.0)
            result = api.processing_queue_report()
            assert result["ok"], result.get("error")
            assert result["workers"] == 2
        finally:
            queue.shutdown()
