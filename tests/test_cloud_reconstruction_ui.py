"""The Run on control: a reconstruction runs on the named server, or the screen says why not.

The desktop Api could already take execution_mode="cloud", but no screen sent it, so an
operator could only reach the server from the CLI. Both windows now have a Run on choice.
These open the real pages in headless Chromium against the real Api (tests/live_ui.py
replaces only the pywebview transport) and, for the round trip, the real services/api
app on a local socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_cloud_reconstruction import _survey_copy, server  # noqa: F401  (fixture)


def _desktop_api(root: Path, images: Path):
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    api = Api(AppSession(ProjectStore(root / "odk.db")))
    assert api.create_project("cloud ui", str(root / "project"))["ok"]
    assert api.import_dataset(str(images))["ok"]
    return api


def _images(root: Path) -> Path:
    images = root / "images"
    images.mkdir()
    import cv2
    import numpy as np

    cv2.imwrite(str(images / "a.png"), np.zeros((64, 64, 3), np.uint8))
    return images


CLASSIC_STATUS = ("T.until(() => { const s = document.querySelector('#status-message').textContent;"
                  " return /%s/.test(s) && s; }, 60000, 'status %s')")


def test_the_classic_window_refuses_a_server_run_without_url_or_token(tmp_path, monkeypatch):
    from live_ui import LiveUI

    monkeypatch.delenv("ODK_CLOUD_TOKEN", raising=False)
    api = _desktop_api(tmp_path, _images(tmp_path))
    ui = LiveUI(api, "home", page="index.html", ready="!!window.odk")
    try:
        hidden = ui.js("document.querySelector('#recon-cloud-fields').hidden")
        ui.js("const s = document.querySelector('#recon-run-on'); s.value = 'cloud';"
              " s.dispatchEvent(new Event('change')); true")
        shown = ui.js("!document.querySelector('#recon-cloud-fields').hidden")
        ui.js("document.querySelector('#btn-recon').click(); true")
        no_url = ui.js(CLASSIC_STATUS % ("server project URL", "url"))
        ui.js("document.querySelector('#recon-cloud-endpoint').value = 'http://127.0.0.1:9/projects/1';"
              " document.querySelector('#btn-recon').click(); true")
        no_token = ui.js(CLASSIC_STATUS % ("API token", "token"))
        token_type = ui.js("document.querySelector('#recon-cloud-token').type")
    finally:
        ui.close()
    assert hidden is True and shown is True, "the server fields follow the Run on choice"
    assert "needs the server project URL" in no_url
    assert "needs an API token for http://127.0.0.1:9/projects/1" in no_token
    assert token_type == "password"
    assert not api.list_jobs().get("jobs"), "a refused request must not start a job"


def test_the_cockpit_process_button_refuses_a_server_run_without_a_token(tmp_path, monkeypatch):
    from live_ui import LiveUI

    monkeypatch.delenv("ODK_CLOUD_TOKEN", raising=False)
    api = _desktop_api(tmp_path, _images(tmp_path))
    ui = LiveUI(api, "processing")
    try:
        ui.js("(async () => {"
              " const field = (label) => [...document.querySelectorAll('.field')]"
              "   .find((f) => f.querySelector('label').textContent === label);"
              " await T.until(() => field('Run on'), 30000, 'Run on field');"
              " const set = (label, v) => { const i = field(label).querySelector('input, select');"
              "   i.value = v; i.dispatchEvent(new Event('change')); };"
              " set('Run on', 'OpenDroneKit server');"
              " set('Server project URL', 'http://127.0.0.1:9/projects/1');"
              " return true; })()")
        # Process is refused until the shell has read the open project; wait for the
        # header to name it, as a user would see it before clicking.
        ui.js("T.until(() => document.body.innerText.includes('cloud ui'), 60000, 'project loaded')")
        ui.js("T.tool('Process')")
        ui.js("T.ok('Confirm')")
        refusal = ui.js("T.toast('needs an API token', 60000)")
    finally:
        ui.close()
    assert "http://127.0.0.1:9/projects/1" in refusal
    assert not api.list_jobs().get("jobs")


def test_the_classic_window_reconstructs_on_the_server(server, tmp_path):  # noqa: F811
    if not server["serves_artifacts"]:
        pytest.skip("This services/api has no artifact download route.")
    pytest.importorskip("rasterio")
    from live_ui import LiveUI

    images = _survey_copy(tmp_path)
    api = _desktop_api(tmp_path, images)
    ui = LiveUI(api, "home", page="index.html", ready="!!window.odk")
    try:
        ui.js("document.querySelector('#recon-profile').value = 'fast_preview';"
              " const s = document.querySelector('#recon-run-on'); s.value = 'cloud';"
              " s.dispatchEvent(new Event('change'));"
              f" document.querySelector('#recon-cloud-endpoint').value = {json.dumps(server['endpoint'])};"
              f" document.querySelector('#recon-cloud-token').value = {json.dumps(server['token'])};"
              " document.querySelector('#btn-recon').click(); true")
        body = ui.js("(async () => { const w = (ms) => new Promise((r) => setTimeout(r, ms));"
                     " for (let i = 0; i < 4500; i++) {"
                     "   const t = document.querySelector('#modal-title');"
                     "   if (t && t.textContent === 'Reconstruction Complete')"
                     "     return document.querySelector('#modal-body').textContent;"
                     "   await w(200); }"
                     " return document.querySelector('#status-message').textContent; })()")
    finally:
        ui.close()
    assert "Ran on" in body and "OpenDroneKit server" in body, body
    assert "EPSG:32617" in body, body
    jobs = api.list_jobs()["jobs"]
    assert jobs and jobs[0]["result"]["execution_mode_used"] == "cloud"
    names = [layer.get("name", "") for layer in api.list_layers()["layers"]]
    assert any("rtho" in name for name in names), names
