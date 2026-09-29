"""A project with nothing in it is refused a report, with the reason, on every path.

The audit found report_readiness() answering {'ok': True, 'missing': []} for a fresh
project: "Nothing missing" in the dialog, and Generate wrote a document of empty sections
that reads as a survey which found nothing. The desktop Api was taught to refuse; the
engine's own workspace (core.report_engine.generate_report, which a processing run's
report stage calls) still declared the same empty project ready. The refusal now lives
in the engine's readiness check, so both agree.

These drive the real Api and the shipped pages (headless browser, real Api behind the
bridge) the way a user meets it: open the Generate dialog on a new project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from core.report_engine import NOTHING_TO_REPORT
from desktop_window import run_page

REASON = "nothing to be about"


def _empty_project(base: Path) -> tuple[Api, Path]:
    api = Api(AppSession(ProjectStore(base / "state.db")))
    root = base / "project"
    assert api.create_project("Empty site", str(root))["ok"]
    return api, root


class TestThroughTheApi:
    @pytest.mark.parametrize("report_type", ["standard", "engineering", "executive",
                                             "defect_only", "dataset_quality",
                                             "mission_summary"])
    def test_every_type_is_not_ready_and_says_why(self, tmp_path, report_type) -> None:
        api, _ = _empty_project(tmp_path)
        readiness = api.report_readiness("", report_type, ["html"])
        assert readiness["ok"] is True, readiness  # the call worked; the answer is no
        assert readiness["ready"] is False
        assert readiness["missing"][0] == NOTHING_TO_REPORT

    def test_generate_refuses_and_writes_nothing(self, tmp_path) -> None:
        api, root = _empty_project(tmp_path)
        refused = api.generate_report("", "Empty", "standard")
        assert refused["ok"] is False and REASON in refused["error"]
        assert not list(root.glob("reports/**/report.html"))

    def test_one_saved_mission_is_enough_to_be_ready(self, tmp_path) -> None:
        """The refusal is about emptiness, not a blanket no."""
        api, _ = _empty_project(tmp_path)
        api.set_aoi([[-81.7504, 41.3038], [-81.7494, 41.3038], [-81.7494, 41.3046],
                     [-81.7504, 41.3046]])
        assert api.plan_mission({"mode": "grid", "altitude_m": 60})["ok"]
        assert api.save_mission("Deck grid")["ok"]
        readiness = api.report_readiness("", "standard", ["html"])
        assert readiness["ready"] is True, readiness
        assert NOTHING_TO_REPORT not in readiness["missing"]


WINDOW_DRIVER = r"""
const $ = (s) => document.querySelector(s);
const out = {};
$('#btn-report').click();
await __waitFor(() => __modalOpen('Generate Report')
  && /Ready|Warning|Missing|Note/.test($('#rp-readiness').textContent));
out.readiness = $('#rp-readiness').textContent;
__clickAction('Generate');
await __waitFor(() => /Report not generated/.test($('#status-message').textContent));
out.generate = $('#status-message').textContent;
__finish(out);
"""

COCKPIT_DRIVER = r"""
const qa = (s) => Array.from(document.querySelectorAll(s));
const seen = [];
new MutationObserver(() => qa('.toast').forEach((t) => {
  if (!seen.includes(t.textContent)) seen.push(t.textContent);
})).observe(document.body, { childList: true, subtree: true });
const out = {};
const lines = () => qa('.console .line').map((l) => l.textContent).join(' | ');
await __waitFor(() => /missing|ready/.test(lines()));
out.readiness = lines();
qa('.tbtn').find((b) => b.textContent === 'Generate').click();
await __waitFor(() => seen.some((t) => t.startsWith('Generate')));
out.generate = seen.filter((t) => t.startsWith('Generate')).join(' | ');
__finish(out);
"""


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    base = tmp_path_factory.mktemp("empty_window")
    api, root = _empty_project(base)
    result = run_page(api, WINDOW_DRIVER, tmp_path_factory.mktemp("profile_ew"))
    assert "error" not in result, result
    result["root"] = root
    return result


@pytest.fixture(scope="module")
def cockpit(tmp_path_factory):
    base = tmp_path_factory.mktemp("empty_cockpit")
    api, root = _empty_project(base)
    result = run_page(
        api, COCKPIT_DRIVER, tmp_path_factory.mktemp("profile_ec"), page_name="workspace.html",
        before_js="localStorage.setItem('odk.workspace.last', 'reports');",
        ready_js="Array.from(document.querySelectorAll('.tbtn')).some("
                 "(b) => b.textContent === 'Generate')",
    )
    assert "error" not in result, result
    result["root"] = root
    return result


class TestTheDefaultWindow:
    def test_the_dialog_says_not_ready_and_why(self, window) -> None:
        text = window["readiness"]
        assert "Nothing missing" not in text
        assert "Not ready" in text and REASON in text

    def test_generate_refuses_with_the_reason(self, window) -> None:
        assert REASON in window["generate"]
        assert not list(Path(window["root"]).glob("reports/**/report.html"))


class TestTheCockpit:
    def test_the_readiness_panel_is_blocked_with_the_reason(self, cockpit) -> None:
        text = cockpit["readiness"]
        assert REASON in text
        assert "Generate will refuse" in text

    def test_generate_refuses_with_the_reason(self, cockpit) -> None:
        assert REASON in cockpit["generate"], cockpit["generate"]
        assert not list(Path(cockpit["root"]).glob("reports/**/report.html"))
