"""Celery over Redis: processing that survives the process that submitted it.

core/job_queue.py bounds concurrency inside one process, which is the right answer for
a desktop shell and the wrong one for a deployment. Two things it cannot do:

  * Survive a restart. Everything queued lives in a Python heap, so an API pod that
    restarts loses the backlog silently -- the jobs do not fail, they simply cease to
    exist, and nothing reports their absence.
  * Use more than one machine. Reconstruction is memory-bound, so the way to run four
    large jobs is four hosts, not four threads.

Redis holds the queue; Celery runs the workers. Both are open source and neither needs a
managed service -- `docker compose up redis` is the whole infrastructure requirement.

Settings worth knowing, because each encodes a decision rather than a default:

``task_acks_late``
    A task is acknowledged when it FINISHES, not when it is picked up. A worker
    OOM-killed mid-reconstruction -- the failure this system should expect, since
    photogrammetry is memory-bound before it is CPU-bound -- would otherwise take its
    job with it. Late acknowledgement puts the job back on the queue instead.

``worker_prefetch_multiplier = 1``
    Celery's default hoards tasks into each worker's local buffer. For short tasks that
    is a throughput win; for hour-long reconstructions it means a queue that looks busy
    while three workers sit idle holding jobs they have not started. One at a time, so
    the queue's depth is the truth.

``task_reject_on_worker_lost``
    A worker killed by the OS did not fail the task -- it never finished it. Rejecting
    puts the work back rather than recording a failure nobody caused.

``task_track_started``
    Without it a running job is indistinguishable from a queued one, which is the same
    reporting gap core/job_queue.py exists to avoid.

No result is silently discarded: ``result_expires`` is long enough that a client polling
a slow job still finds its answer.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_BROKER = "redis://localhost:6379/0"


def broker_url() -> str:
    return os.environ.get("ODK_BROKER_URL", DEFAULT_BROKER)


def result_backend() -> str:
    # Same Redis by default. Split them with ODK_RESULT_BACKEND when result traffic
    # would compete with the queue for memory.
    return os.environ.get("ODK_RESULT_BACKEND", broker_url())


def build_app(name: str = "opendronekit") -> Any:
    """Construct the Celery application.

    A function rather than a module-level singleton so tests, and any deployment that
    needs a second broker, can build one without importing a connection as a side
    effect of importing this module.
    """
    from celery import Celery

    app = Celery(name, broker=broker_url(), backend=result_backend())
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Long enough that a client polling an hour-long reconstruction still finds its
        # result, short enough that Redis does not accumulate them forever.
        result_expires=60 * 60 * 24,
        # Priority support. Redis needs the queue split into per-priority lists, and
        # Celery does that when told how many steps there are.
        broker_transport_options={
            "priority_steps": list(range(10)),
            "queue_order_strategy": "priority",
            # A task not acknowledged within this window returns to the queue. It must
            # exceed the longest job or a slow reconstruction is handed to a second
            # worker while the first is still running it.
            "visibility_timeout": 60 * 60 * 6,
        },
    )
    return app


def broker_reachable(timeout_s: float = 2.0) -> dict[str, Any]:
    """Whether the broker is actually there, reported rather than assumed.

    Celery does not connect on import, so a misconfigured broker surfaces as a task that
    never runs. An explicit check turns that into an answer at startup.
    """
    url = broker_url()
    try:
        import redis
    except ImportError:
        return {"reachable": False, "broker": url,
                "error": "the redis package is not installed"}
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=timeout_s)
        client.ping()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"reachable": False, "broker": url, "error": f"{type(exc).__name__}: {exc}"}
    return {"reachable": True, "broker": url}


def queue_depth(queue: str = "celery") -> int:
    """How many tasks are waiting. Raises if the broker cannot be reached.

    Depth is meaningful only because prefetch is 1: with Celery's default the number
    would undercount everything already sitting in worker buffers.
    """
    import redis

    client = redis.Redis.from_url(broker_url(), socket_connect_timeout=2.0)
    return int(client.llen(queue))
