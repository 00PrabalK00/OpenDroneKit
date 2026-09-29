"""A second reconstruction into the folder a running one is writing to is refused.

Audit: pressing Process while New Job was still running started a second COLMAP run into
the same reconstruction folder; it died with "PermissionError: [WinError 32]" on a file
the first run held open, after the toast had said it started. The cockpit now refuses the
second one up front and says which job holds the folder.
"""

from __future__ import annotations

import pytest

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
    verdict = ("Fits" if expected["can_run_as_one_job"]
               else "Cannot run here now" if not expected["fits_one_image"]
               else "Does not fit as one job")
    assert result["toast"].startswith(f"{verdict}: {expected['note']}"), (result, expected)
    assert "may not fit" not in result["toast"] or not expected["can_run_as_one_job"]


def test_size_job_does_not_contradict_itself_on_a_machine_short_of_memory(tmp_path, monkeypatch) -> None:
    """Verifier: with 573 MB free the toast read "Does not fit as one job: 8 images should
    run as a single job ... Run in chunks of about 8 images". The free-memory probe is the
    machine boundary, so it is the one thing set here; the rest is the real path."""
    import core.job_sizing as sizing

    monkeypatch.setattr(sizing, "available_memory_mb", lambda: 573.0)
    api = make_api(tmp_path)
    _dataset(api, tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Processing");
            await ODK.press("Size Job");
            return { toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("images")), 10000) };
        """)
    toast = result["toast"]
    assert toast.startswith("Fits: 4 images should run as a single job"), toast
    assert "Memory is tight" in toast
    assert "Does not fit" not in toast and "chunks of about" not in toast


@pytest.mark.parametrize("free_mb", [36.0, 30.0])
def test_size_job_refuses_when_not_even_one_image_fits(tmp_path, monkeypatch, free_mb) -> None:
    """Verifier: at 36 MB free the toast offered "chunks of about 1 fit" though one image
    is estimated at 45 MB. Chunking cannot help there, and the toast must say so."""
    import core.job_sizing as sizing

    monkeypatch.setattr(sizing, "available_memory_mb", lambda: free_mb)
    api = make_api(tmp_path)
    _dataset(api, tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Processing");
            await ODK.press("Size Job");
            return { toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("images") || t.includes("image(s)")), 10000) };
        """)
    toast = result["toast"]
    assert toast.startswith("Cannot run here now: This machine cannot run this job now"), toast
    assert f"only {free_mb:.0f} MB is free" in toast
    assert "splitting the job cannot help" in toast
    assert "chunks of about" not in toast and "Fits" not in toast
