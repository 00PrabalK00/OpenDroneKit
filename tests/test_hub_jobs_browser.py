"""The browser Hub uploads imagery and runs, watches, cancels and downloads jobs.

The Hub API adapter had job helpers that no control used, so processing and AI jobs
were reachable only through curl or the SDK. This drives hub.html in a real headless
Chromium against the real API under uvicorn: sign up, create a project and dataset,
upload a photo through the chunked, checksummed protocol, run a detection job, an AI
job and a report job, and download what they wrote.
"""

from __future__ import annotations

import json

import pytest

from hub_api_helpers import FIXTURE_IMAGE
from test_hub_login_browser import _fill, api, browser, hub_page  # noqa: F401 - fixtures

TEXT = "document.querySelector({!r}).textContent"


def _job_line(page, kind: str, state: str = "done"):
    return page.wait_for(
        "(() => { const rows = Array.from(document.querySelectorAll('#job-list .record'))"
        f"  .map(r => r.textContent).filter(t => t.includes({json.dumps(': ' + kind + ' — ')})); "
        f"  const hit = rows.find(t => t.includes({json.dumps(' — ' + state + ' ')})); "
        "  const failed = rows.find(t => t.includes(' — failed ')); "
        "  return hit || (failed ? 'FAILED: ' + failed : ''); })()",
        timeout=120,
    )


def _signed_in_on_a_project(page, api_url: str) -> None:
    _fill(page, api_url, "jobs@example.com", "longenough1", "Jobs Co")
    page.js("document.querySelector('#hub-sign-up').click(), true")
    page.wait_for("document.querySelector('#hub-connection-state').textContent === 'connected'")
    page.js("""(() => {
        const form = document.querySelector('#project-form');
        form.querySelector('[name=name]').value = 'Bridge 7';
        form.requestSubmit();
        return true;
    })()""")
    page.wait_for("!!document.querySelector('#project-list [data-project-id]')")
    page.js("document.querySelector('#project-list [data-project-id]').click(), true")
    page.js("document.querySelector('.hub-nav [data-panel=jobs]').click(), true")
    page.wait_for("document.querySelector('#jobs-project-note').textContent.includes('Bridge 7')")
    page.wait_for("document.querySelector('#job-list').textContent === 'No jobs in this project yet.'")


def test_upload_run_and_download_jobs_from_the_hub(browser, api):
    page = browser
    _signed_in_on_a_project(page, api)

    page.js("document.querySelector('#new-dataset-name').value = 'Deck underside', "
            "document.querySelector('#create-dataset').click(), true")
    page.wait_for("document.querySelector('#job-dataset').selectedOptions[0].textContent"
                  ".startsWith('Deck underside')")

    root = page.send("DOM.getDocument")["root"]["nodeId"]
    node = page.send("DOM.querySelector", nodeId=root, selector="#dataset-files")["nodeId"]
    page.send("DOM.setFileInputFiles", nodeId=node, files=[str(FIXTURE_IMAGE)])
    page.js("document.querySelector('#dataset-form').requestSubmit(), true")
    uploaded = page.wait_for(
        "(t => t.includes('checksum') || t.includes('failed') ? t : '')"
        "(document.querySelector('#upload-progress').textContent)")
    assert f"{FIXTURE_IMAGE.stat().st_size} bytes, checksum verified" in uploaded, uploaded
    assert page.js("document.querySelector('#job-dataset').selectedOptions[0].textContent") \
        == "Deck underside (1 files)"

    # Detection with whatever the server can actually run; the heuristic is labelled so.
    page.js("(s => { s.value = 'detection'; s.dispatchEvent(new Event('change')); })"
            "(document.querySelector('#job-kind')), true")
    assert page.js("document.querySelector('[data-job-kind=detection]').hidden") is False
    assert page.js("document.querySelector('#job-detection-model').value")
    page.js("document.querySelector('#job-form').requestSubmit(), true")
    detection = _job_line(page, "detection")
    assert not detection.startswith("FAILED"), detection
    assert "finding(s)" in detection and "recorded unreviewed" in detection

    # An AI segmentation job, then download the mask it wrote through the Hub.
    page.js("(s => { s.value = 'segmentation'; s.dispatchEvent(new Event('change')); })"
            "(document.querySelector('#ai-task')), true")
    page.js("document.querySelector('#ai-job-form').requestSubmit(), true")
    ai = _job_line(page, "ai:segmentation")
    assert not ai.startswith("FAILED"), ai
    ref = page.js("""(() => {
        const rows = Array.from(document.querySelectorAll('#job-list .record'))
          .filter(r => r.textContent.includes(': ai:segmentation — done'));
        const button = rows[0].querySelector('[data-job-artifact][data-ref$="_mask.png"]');
        return button ? button.dataset.ref : '';
    })()""")
    assert ref.endswith("_mask.png"), ref
    size = page.js(f"""(async () => {{
        const hub = new ODKHubApi.HubApi(document.querySelector('#hub-api-url').value,
                                         document.querySelector('#hub-token').value);
        const button = document.querySelector('[data-ref={json.dumps(ref)}]');
        const blob = await hub.downloadArtifact(Number(button.dataset.jobArtifact), {json.dumps(ref)});
        const head = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
        return head[1] === 0x50 && head[2] === 0x4E && head[3] === 0x47 ? blob.size : -1;
    }})()""")
    assert size > 0
    page.js(f"document.querySelector('[data-ref={json.dumps(ref)}]').click(), true")
    page.wait_for("document.querySelector('#hub-status').textContent.startsWith('Download ')"
                  " && document.querySelector('#hub-status').textContent.endsWith('complete.')")

    # A report over the unreviewed findings the detection job just recorded.
    page.js("(s => { s.value = 'report'; s.dispatchEvent(new Event('change')); })"
            "(document.querySelector('#job-kind')), true")
    page.js("document.querySelector('#job-include-unreviewed').checked = true, "
            "document.querySelector('#job-form').requestSubmit(), true")
    report = _job_line(page, "report")
    assert not report.startswith("FAILED"), report
    assert page.js("!!document.querySelector('#job-list [data-ref$=\".pdf\"]')")


def test_the_hub_shows_why_a_job_cannot_run(browser, api):
    page = browser
    _signed_in_on_a_project(page, api)

    # A reconstruction without a dataset is refused in the page, before any request.
    page.js("document.querySelector('#job-dataset').value = '', "
            "document.querySelector('#job-form').requestSubmit(), true")
    assert "needs a dataset" in page.wait_for(
        "(t => t.startsWith('Job:') ? t : '')(document.querySelector('#hub-status').textContent)")

    # A measurement with no finished reconstruction to measure on says so.
    page.js("(s => { s.value = 'measurement'; s.dispatchEvent(new Event('change')); })"
            "(document.querySelector('#job-kind')), true")
    page.wait_for(TEXT.format("#job-source") + " === 'No finished reconstruction'")

    # Every model the registry lists appears, and one that cannot run here is disabled
    # with the server's reason rather than hidden.
    models = page.js("""Array.from(document.querySelectorAll('#ai-model option'))
        .map(o => ({key: o.value, disabled: o.disabled, reason: o.title}))""")
    assert any(m["key"] == "heuristic" and not m["disabled"] for m in models), models
    for model in models:
        if model["disabled"]:
            assert model["reason"], model
