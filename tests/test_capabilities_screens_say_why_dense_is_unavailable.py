"""Every screen that reports dense stereo shows the probed answer and its reason.

The capability used to read "yes" whenever a COLMAP binary existed. It is now decided by
running patch-match once (tests/test_dense_stereo_probe.py), and the reason -- on the
RTX 5060 dev box, a COLMAP compiled with a newer CUDA than the driver supports -- is the
thing an operator needs in order to fix it. These tests open the real pages in headless
Chromium against the real Api, with this machine's real COLMAP probed from scratch, and
read what each capability screen shows:

- the cockpit's Tools > Environment Capabilities menu item,
- the cockpit's Developers > Infrastructure panel and Processing > Engine tab,
- the classic window's Environment Capabilities dialog.

Only the pywebview transport is replaced (tests/live_ui.py).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    """A real Api whose dense probe runs now, against this machine's COLMAP."""
    import os

    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    root = tmp_path_factory.mktemp("caps_ui")
    previous = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
    # A fresh probe cache: the answer on screen must come from a run in this test,
    # not from whatever an earlier session left in the user's profile.
    os.environ["APPDATA"] = str(root / "appdata")
    os.environ["XDG_CONFIG_HOME"] = str(root / "appdata")
    try:
        api = Api(AppSession(ProjectStore(root / "odk.db")))
        assert api.create_project("capabilities ui", str(root / "project"))["ok"]
        caps = api.capabilities()["capabilities"]
        yield api, caps
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _reason(caps) -> str:
    reason = caps["dense_stereo_reason"]
    assert reason, "the capability must always say why, available or not"
    return reason


def test_the_probe_names_the_cause_and_the_fix_when_the_driver_cannot_load_the_kernels(probed):
    """The dev box's failure: CUDA 13.2 kernels under a driver that supports CUDA 13.1."""
    _, caps = probed
    reason = _reason(caps)
    if caps["dense_stereo"]:
        pytest.skip("Dense stereo works on this machine; there is no failure to explain.")
    if "unsupported toolchain" not in reason:
        pytest.skip(f"This machine fails dense stereo for another reason: {reason}")
    assert "Cause:" in reason and "Fix:" in reason
    assert "ODK_COLMAP" in reason and "INSTALLATION.md" in reason


def test_the_cockpit_menu_lists_each_capability_with_the_dense_reason(probed):
    from live_ui import LiveUI

    api, caps = probed
    ui = LiveUI(api, "home")
    try:
        ui.js("window.odk.onMenu({ action: 'tools.capabilities' }); true")
        text = ui.js("T.until(() => { const a = document.querySelector('.modal textarea.modal-reveal');"
                     " return a && a.value; }, 120000, 'capabilities box')")
    finally:
        ui.close()
    lines = text.splitlines()
    assert f"dense_stereo: {str(caps['dense_stereo']).lower()}" in lines
    assert f"dense_stereo_reason: {_reason(caps)}" in lines, "one key per line, reason readable"
    assert not any(line.startswith("capabilities: {") for line in lines), \
        "the envelope was printed as one line of JSON"


def test_the_developers_infrastructure_panel_shows_the_reason(probed):
    from live_ui import LiveUI

    api, caps = probed
    ui = LiveUI(api, "developers")
    try:
        rows = json.loads(ui.js(
            "T.until(() => { const t = T.table('Component');"
            " return t && t.some((r) => r[0] === 'Dense stereo') && JSON.stringify(t); },"
            " 120000, 'infrastructure table')"))
    finally:
        ui.close()
    dense = next(r for r in rows if r[0] == "Dense stereo")
    assert dense[1] == ("available" if caps["dense_stereo"] else "not available")
    assert dense[2] == _reason(caps)


def test_the_processing_engine_tab_shows_the_reason(probed):
    from live_ui import LiveUI

    api, caps = probed
    ui = LiveUI(api, "processing")
    try:
        text = ui.js(
            "(async () => { const tab = await T.until(() => [...document.querySelectorAll('*')]"
            ".find((e) => e.children.length === 0 && e.textContent.trim() === 'Engine'), 30000, 'Engine tab');"
            " tab.click();"
            " return T.until(() => { const p = T.props().join('\\n');"
            " return p.includes('Dense stereo detail') && p; }, 120000, 'engine properties'); })()")
    finally:
        ui.close()
    assert "GPU dense stereo" in text
    assert _reason(caps) in text


def test_the_classic_dialog_shows_the_flag_the_binary_and_the_reason(probed):
    from live_ui import LiveUI

    api, caps = probed
    ui = LiveUI(api, "home", page="index.html", ready="!!window.odk")
    try:
        ui.js("window.odk.onMenu({ action: 'tools.capabilities' }); true")
        text = ui.js("(async () => { const w = (ms) => new Promise((r) => setTimeout(r, ms));"
                     " for (let i = 0; i < 300; i++) {"
                     "   const t = document.querySelector('#modal-title');"
                     "   if (t && t.textContent === 'Environment Capabilities')"
                     "     return document.querySelector('#modal-body').textContent;"
                     "   await w(200); }"
                     " return ''; })()")
    finally:
        ui.close()
    assert text, "the Environment Capabilities dialog never opened"
    assert "Dense MVS" in text
    assert _reason(caps) in text
    assert (caps["colmap_binary"] or "not found") in text
