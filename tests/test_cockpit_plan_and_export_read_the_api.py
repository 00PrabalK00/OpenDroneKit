"""The cockpit's Plan and Export buttons must read what the Api actually returns.

Plan read result.waypoint_count, which plan_mission has never returned, so every plan
reported "Planned ? waypoints." Export read result.files / result.exported, while
export_mission answers {written, directory}, so a successful export fell through to
"Nothing to export yet" or jumped to Export Products.

This runs the real actions.js in node against the real Api. Only the transport is
substituted: pywebview's bridge becomes an HTTP call into an Api instance in this
process, so the responses are the Api's own, not a hand-written shape.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

ACTIONS_JS = Path(__file__).resolve().parents[1] / "app" / "web" / "js" / "workspace" / "actions.js"
AOI = [[-81.7535, 41.3035], [-81.7521, 41.3035], [-81.7521, 41.3044], [-81.7535, 41.3044]]


@pytest.fixture
def bridge(tmp_path):
    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("cockpit", root_dir=str(tmp_path / "project"))
    api = Api(session)
    assert api.set_aoi(AOI)["ok"]
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            method = self.path.rsplit("/", 1)[1]
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"[]")
            with lock:
                result = getattr(api, method)(*args)
            body = json.dumps(result, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield api, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def run_actions(url: str, options: dict) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    script = f"""
globalThis.window = {{ pywebview: {{ api: new Proxy({{}}, {{ get: (_, m) => async (...args) =>
  (await fetch({json.dumps(url)} + m, {{ method: "POST", body: JSON.stringify(args) }})).json() }}) }} }};
const {{ ACTIONS }} = await import({json.dumps(ACTIONS_JS.as_uri())});
const ctx = {{ missionOptions: () => ({json.dumps(options)}), prompt: async () => null, choose: async () => null }};
const plan = await ACTIONS.Plan.run(ctx);
const exported = await ACTIONS.Export.run(ctx);
console.log(JSON.stringify({{ plan, exported }}));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_plan_reports_the_real_waypoint_count(bridge) -> None:
    api, url = bridge
    out = run_actions(url, {"template": "grid", "altitude_m": 50, "speed_m_s": 3})
    planned = len(api._session.mission_plan.waypoints)
    assert "?" not in out["plan"]["message"]
    assert f"Planned {planned} waypoints" in out["plan"]["message"]


def test_export_reports_the_files_it_wrote(bridge) -> None:
    api, url = bridge
    out = run_actions(url, {"template": "orbit", "altitude_m": 50})
    exported = out["exported"]
    assert exported["message"].startswith("Exported 6 mission file(s)"), exported["message"]
    assert len(exported["files"]) == 6
    assert all(Path(f).exists() for f in exported["files"])
