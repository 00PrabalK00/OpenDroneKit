"""Standalone inference worker for API AI jobs (``ODK_AI_WORKER=external``).

    python -m services.worker.ai_worker            # poll forever
    python -m services.worker.ai_worker --once     # drain the queue, then exit

Needs the same ODK_DATABASE_URL and ODK_STORAGE_PATH as the API. Any number of these
can run: each job is claimed with a conditional update, so none is run twice.
"""

from __future__ import annotations

import argparse
import time


def drain(max_jobs: int | None = None) -> list[tuple[int, str]]:
    """Run pending jobs until none are left (or max_jobs ran). Returns (id, status)."""
    from services.api.ai_jobs import claim_next, run_ai_job

    done: list[tuple[int, str]] = []
    while max_jobs is None or len(done) < max_jobs:
        job_id = claim_next()
        if job_id is None:
            break
        done.append((job_id, run_ai_job(job_id, already_claimed=True)))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m services.worker.ai_worker")
    parser.add_argument("--once", action="store_true", help="drain pending jobs and exit")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    from services.api.db import init_db

    init_db()
    while True:
        for job_id, status in drain():
            print(f"ai job {job_id}: {status}", flush=True)
        if args.once:
            return 0
        time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
