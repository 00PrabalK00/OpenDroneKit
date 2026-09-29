"""A second reconstruction into the folder a running one is writing to is refused.

Audit: pressing Process while New Job was still running started a second COLMAP run into
the same reconstruction folder; it died with "PermissionError: [WinError 32]" on a file
the first run held open, after the toast had said it started. The cockpit now refuses the
second one up front and says which job holds the folder.
"""

from __future__ import annotations

import time

from cockpit_harness import Cockpit, make_api, write_geotagged_jpeg


def _dataset(api, tmp_path):
    folder = tmp_path / "images"
    for i in range(4):
        write_geotagged_jpeg(folder / f"IMG_{i:03d}.jpg", -81.75 + i * 1e-4, 41.30)
    assert api.set_active_dataset(str(folder))["ok"]
    return folder


def _settle(api, job_id):
    api.cancel_job(job_id)
    for _ in range(600):
        if api.job_status(job_id)["job"]["status"] not in {"pending", "running"}:
            return
        time.sleep(0.1)


def test_a_second_run_into_the_same_folder_is_refused_until_the_first_ends(tmp_path) -> None:
    api = make_api(tmp_path)
    _dataset(api, tmp_path)
    first = api.run_reconstruction({"engine": "auto"})
    assert first["ok"], first
    second = api.run_reconstruction({"engine": "auto"})
    try:
        assert second["ok"] is False
        assert "already writing to" in second["error"] and first["job_id"] in second["error"]
    finally:
        _settle(api, first["job_id"])
    # Once the first has ended, the folder is free again.
    third = api.run_reconstruction({"engine": "auto"})
    assert third["ok"], third
    _settle(api, third["job_id"])


def test_the_cockpit_shows_the_refusal_on_the_second_press(tmp_path) -> None:
    api = make_api(tmp_path)
    _dataset(api, tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Processing");
            await ODK.press("New Job");
            await ODK.answer();
            await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Job queued")), 10000);
            await ODK.press("Process");
            await ODK.answer();
            const refusal = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("already writing to")), 10000);
            return { refusal };
        """)
    started = [c for c in cockpit.called("run_reconstruction") if c["ok"]]
    try:
        assert result["refusal"] and result["refusal"].startswith("Process:")
        assert len(started) == 1
    finally:
        for job in api.list_jobs()["jobs"]:
            _settle(api, job["id"])


def test_size_job_reports_the_verdict_the_application_computed(tmp_path) -> None:
    """Size Job read keys size_job never returns, so every dataset "may not fit"."""
    api = make_api(tmp_path)
    _dataset(api, tmp_path)
    expected = api.size_reconstruction_job(4)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Processing");
            await ODK.press("Size Job");
            return { toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("images")), 10000) };
        """)
    verdict = "Fits" if expected["can_run_as_one_job"] else "Does not fit as one job"
    assert result["toast"].startswith(f"{verdict}: {expected['note']}"), (result, expected)
    assert "may not fit" not in result["toast"] or not expected["can_run_as_one_job"]
