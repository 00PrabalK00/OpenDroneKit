"""The cockpit keeps asking which project is open until the application answers.

Found while verifying the reports area under machine load: the cockpit read get_state
once at startup. When that one call failed, the status bar said "connected -- 1
project(s)" and "no project open" for the whole session, and every project action
answered "Open a project first." although a project was open. The tests worked around
it by clicking again; a user had no such loop.

The shipped workspace.html runs in headless Chromium with the real Api behind the
bridge; only get_state is made to fail for its first calls, the way a busy or locked
store makes it fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore
from desktop_window import run_page


class FirstReadsFail:
    """The real Api, whose first `failures` get_state calls raise."""

    def __init__(self, api: Api, failures: int) -> None:
        self._api = api
        self._left = failures
        self.state_reads = 0

    def get_state(self) -> dict[str, Any]:
        self.state_reads += 1
        if self._left > 0:
            self._left -= 1
            raise RuntimeError("database is locked")
        return self._api.get_state()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)


DRIVER = r"""
const statusText = () => (document.querySelector('.status-project') || {}).textContent || '';
await __waitFor(() => statusText().includes('Read later'), 30000);
const out = { status: statusText() };
// A project action is not refused as "no project" now that the project is known.
Array.from(document.querySelectorAll('.tbtn')).find((b) => b.textContent === 'Inspection Report').click();
await __waitFor(() => document.querySelector('.toast') || document.querySelector('textarea.modal-reveal'), 30000);
out.said = Array.from(document.querySelectorAll('.toast')).map((t) => t.textContent).join(' | ');
__finish(out);
"""


def _project(tmp_path: Path) -> Api:
    api = Api(AppSession(ProjectStore(tmp_path / "state.db")))
    assert api.create_project("Read later", str(tmp_path / "project"))["ok"]
    return api


def test_a_failed_first_read_of_the_open_project_is_retried(tmp_path) -> None:
    api = FirstReadsFail(_project(tmp_path), failures=3)
    shown = run_page(
        api, DRIVER, tmp_path / "profile", page_name="workspace.html",
        before_js="localStorage.setItem('odk.workspace.last', 'reports');",
        ready_js="Array.from(document.querySelectorAll('.tbtn')).some("
                 "(b) => b.textContent === 'Inspection Report')",
    )
    assert "error" not in shown, shown
    assert "Read later" in shown["status"]
    assert "Open a project first" not in shown["said"], shown
    assert api.state_reads > 3
