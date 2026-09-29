"""Cancel has to stop a reconstruction, including the native COLMAP process.

Api.run_reconstruction received should_cancel and never passed it on, and COLMAP steps
ran under subprocess.run with a two-hour timeout, so Cancel on a dense run did nothing
until the run finished. The pipeline was worse: its progress wrapper swallowed every
exception from the callback, including the job runner's cancel signal.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from core.errors import OperationCancelled


def test_a_running_native_step_is_killed_on_cancel():
    from core.reconstruction_colmap import ColmapReconstructor

    engine = ColmapReconstructor(profile="fast")
    flag = threading.Event()
    engine._should_cancel = flag.is_set
    threading.Timer(0.5, flag.set).start()

    started = time.monotonic()
    with pytest.raises(OperationCancelled):
        engine._run_command([sys.executable, "-c", "import time; time.sleep(60)"])
    assert time.monotonic() - started < 10, "cancel must not wait for the step to finish"


def test_a_native_step_that_finishes_returns_its_output():
    from core.reconstruction_colmap import ColmapReconstructor

    engine = ColmapReconstructor(profile="fast")
    done = engine._run_command([sys.executable, "-c", "import sys; print('ok'); print('warn', file=sys.stderr)"])
    assert done.returncode == 0
    assert done.stdout.strip() == "ok" and done.stderr.strip() == "warn"


def test_the_pipeline_does_not_swallow_the_job_runners_cancel(tmp_path):
    """The job runner cancels by raising from progress(); the pipeline caught it."""
    from app.jobs import JobManager
    from core.pipeline import PipelineConfig, StructuralFaultPipeline

    images = tmp_path / "images"
    images.mkdir()
    import cv2
    import numpy as np

    for index in range(3):
        cv2.imwrite(str(images / f"{index}.png"), np.full((64, 64, 3), 128, np.uint8))

    config = PipelineConfig(reconstruction_engine="custom", run_coverage_validation=False)
    pipeline = StructuralFaultPipeline(config)
    manager = JobManager()
    gate = threading.Event()

    def work(progress, should_cancel, **_):
        def gated(percent, message):
            gate.set()
            time.sleep(0.2)
            progress(percent, message)
        return pipeline.run(images, tmp_path / "out", progress_callback=gated, should_cancel=should_cancel)

    job_id = manager.submit("pipeline", work)
    assert gate.wait(10)
    assert manager.cancel(job_id)
    deadline = time.time() + 30
    while time.time() < deadline and manager.get(job_id)["status"] == "running":
        time.sleep(0.05)
    assert manager.get(job_id)["status"] == "cancelled"
