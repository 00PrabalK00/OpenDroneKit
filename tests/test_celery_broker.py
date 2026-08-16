"""The broker is configured for hour-long, memory-hungry jobs.

Celery's defaults are tuned for short tasks. Reconstruction is neither short nor cheap,
and three of those defaults are actively wrong for it:

  * Early acknowledgement loses a job when a worker is OOM-killed mid-run -- which is
    the failure to expect here, since photogrammetry is memory-bound before it is
    CPU-bound.
  * Prefetching hoards tasks into worker buffers, so the queue reads as busy while
    workers sit idle holding jobs they have not started, and the depth number lies.
  * A short visibility timeout hands a slow job to a second worker while the first is
    still running it, so the same reconstruction runs twice and one result is discarded.

These tests assert the configuration rather than the framework: Celery works, the
question is whether it has been told the right things about this workload.

Tests needing a live broker skip when none is reachable. That is deliberate -- a broker
test that passes against a mock proves the mock agrees with itself.
"""

from __future__ import annotations

import pytest

pytest.importorskip("celery")
pytest.importorskip("redis")

from services.worker.celery_app import (  # noqa: E402
    broker_reachable,
    broker_url,
    build_app,
    queue_depth,
)


@pytest.fixture(scope="module")
def app():
    return build_app("odk-test")


class TestConfiguredForLongJobs:
    def test_acknowledgement_is_late(self, app) -> None:
        """A worker killed mid-job must not take the job with it."""
        assert app.conf.task_acks_late is True

    def test_a_lost_worker_returns_its_task(self, app) -> None:
        # The OS killing a worker is not the task failing; it is the task not finishing.
        assert app.conf.task_reject_on_worker_lost is True

    def test_workers_do_not_hoard_tasks(self, app) -> None:
        """Prefetch above 1 makes queue depth a lie for hour-long work."""
        assert app.conf.worker_prefetch_multiplier == 1

    def test_a_started_task_is_distinguishable_from_a_queued_one(self, app) -> None:
        assert app.conf.task_track_started is True

    def test_the_visibility_timeout_outlasts_a_real_job(self, app) -> None:
        """Too short and the same reconstruction runs twice on two workers."""
        timeout = app.conf.broker_transport_options["visibility_timeout"]
        assert timeout >= 60 * 60, f"visibility timeout of {timeout}s is shorter than one job"

    def test_results_outlive_a_slow_poll(self, app) -> None:
        assert app.conf.result_expires >= 60 * 60

    def test_priority_is_available(self, app) -> None:
        options = app.conf.broker_transport_options
        assert options["queue_order_strategy"] == "priority"
        assert len(options["priority_steps"]) > 1

    def test_only_json_is_accepted(self, app) -> None:
        # Pickle would let a queue entry execute arbitrary code on a worker.
        assert app.conf.accept_content == ["json"]
        assert "pickle" not in app.conf.accept_content


class TestBrokerReporting:
    def test_an_unreachable_broker_is_reported_not_raised(self, monkeypatch) -> None:
        """Celery does not connect on import, so a bad broker looks like a task that
        never runs. The check turns that into an answer."""
        import services.worker.celery_app as module

        monkeypatch.setenv("ODK_BROKER_URL", "redis://127.0.0.1:6390/0")
        report = module.broker_reachable(timeout_s=0.25)
        assert report["reachable"] is False
        assert report["error"]
        assert "6390" in report["broker"]

    def test_the_report_names_the_broker_it_tried(self) -> None:
        assert broker_reachable(timeout_s=0.25)["broker"] == broker_url()


def _live() -> bool:
    return bool(broker_reachable(timeout_s=0.5).get("reachable"))


live_broker = pytest.mark.skipif(not _live(), reason="no Redis broker reachable")


@live_broker
class TestAgainstARealBroker:
    """Run against Redis itself. A mock would only agree with itself."""

    def test_the_broker_answers(self) -> None:
        assert broker_reachable()["reachable"] is True

    def test_queue_depth_is_readable(self) -> None:
        assert queue_depth() >= 0

    def test_a_task_is_queued_rather_than_run_inline(self, app) -> None:
        """The point of a broker: submitting does not execute.

        With no worker running, the task must sit in Redis. If it ran inline the
        submitting process would block on an hour-long reconstruction, which is the
        behaviour the queue exists to remove.
        """
        import redis

        @app.task(name="odk.test.noop")
        def noop(value):
            return value

        client = redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0)
        queue = "odk-test-queue"
        client.delete(queue)
        before = client.llen(queue)
        noop.apply_async(args=[1], queue=queue)
        after = client.llen(queue)
        client.delete(queue)
        assert after == before + 1, "the task did not reach the broker"

    def test_the_queue_survives_the_submitting_process(self, app) -> None:
        """The property core/job_queue.py cannot offer.

        An in-process heap dies with its process. A restarted API pod must find its
        backlog still there, or jobs vanish without failing and nothing reports it.
        """
        import redis

        @app.task(name="odk.test.durable")
        def durable(value):
            return value

        client = redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0)
        queue = "odk-durable-queue"
        client.delete(queue)
        durable.apply_async(args=[7], queue=queue)

        # A completely separate client, standing in for a restarted process.
        fresh = redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0)
        depth = fresh.llen(queue)
        fresh.delete(queue)
        assert depth == 1, "the queued task did not outlive its submitter"
