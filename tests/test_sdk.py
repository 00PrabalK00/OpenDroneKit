"""Developer SDK verification against real modules, mission logic and local HTTP."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from core.drone import MockDroneClient
from sdk import (
    DroneSession,
    MissionPlanRequest,
    OpenDroneKitClient,
    PluginKind,
    PluginRegistry,
    plan_mission,
)


class TestPluginSystem:
    def test_all_documented_plugin_points_load_a_real_manifest_module(self, tmp_path, monkeypatch):
        module = tmp_path / "district_plugin.py"
        module.write_text(
            "def create_provider(url='http://tiles.local'):\n"
            "    return {'kind': 'wmts', 'url': url}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        manifest = tmp_path / "plugins.json"
        manifest.write_text(
            json.dumps({
                "api_version": "1",
                "plugins": [{
                    "kind": "map_provider",
                    "name": "district-wmts",
                    "factory": "district_plugin:create_provider",
                }],
            }),
            encoding="utf-8",
        )

        registry = PluginRegistry()
        loaded = registry.load_manifest(manifest)

        assert {kind.value for kind in PluginKind} == {
            "drone", "camera", "payload", "mission_type", "engine", "model",
            "exporter", "report_template", "map_provider",
        }
        assert loaded[0].name == "district-wmts"
        assert registry.create("map_provider", "district-wmts", url="http://localhost/wmts") == {
            "kind": "wmts", "url": "http://localhost/wmts"
        }

    def test_duplicate_plugin_names_are_refused(self, tmp_path, monkeypatch):
        module = tmp_path / "duplicate_plugin.py"
        module.write_text("def make():\n    return object()\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        manifest = tmp_path / "plugins.json"
        manifest.write_text(json.dumps({
            "api_version": "1",
            "plugins": [{"kind": "camera", "name": "same", "factory": "duplicate_plugin:make"}],
        }), encoding="utf-8")
        registry = PluginRegistry()
        registry.load_manifest(manifest)
        try:
            registry.load_manifest(manifest)
        except ValueError as exc:
            assert "already registered" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Duplicate plugin registration was accepted")


class _ApiHandler(BaseHTTPRequestHandler):
    polls = 0

    def log_message(self, *_):
        return

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.headers["Authorization"] == "Bearer local-token"
        if self.path == "/organizations/4/projects":
            self._json(200, [{"id": 7, "name": "Quarry", "client": "Client"}])
        elif self.path == "/organizations/4/assets":
            self._json(200, [{"id": 2, "name": "Pile A", "asset_type": "stockpile"}])
        elif self.path == "/jobs/9":
            type(self).polls += 1
            status = "completed" if type(self).polls >= 2 else "running"
            self._json(200, {"id": 9, "status": status, "percent": 100 if status == "completed" else 50})
        elif self.path == "/jobs/9/log":
            self._json(200, {"job_id": 9, "status": "completed", "log": "real local response"})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        assert self.headers["Authorization"] == "Bearer local-token"
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/projects/7/jobs":
            assert payload["kind"] == "reconstruction"
            assert payload["dataset_id"] == 12
            self._json(202, {"id": 9, "status": "queued", "percent": 0})
        elif self.path == "/jobs/9/cancel":
            self._json(200, {"id": 9, "status": "running", "message": "Cancelling..."})
        else:
            self._json(404, {"detail": "not found"})


class TestDeveloperLibraries:
    def test_mission_and_drone_facades_use_the_real_core_implementations(self):
        request = MissionPlanRequest(
            mission_name="SDK survey",
            polygon_lonlat=[
                [77.5900, 12.9700], [77.5910, 12.9700],
                [77.5910, 12.9710], [77.5900, 12.9710],
            ],
            altitude_m=60,
            mode="grid",
        )
        plan = plan_mission(request)
        assert len(plan.waypoints) > 0
        assert plan.path_distance_m > 0

        adapter = MockDroneClient()
        with DroneSession(adapter, "mock://sdk") as session:
            telemetry = session.telemetry_dict()
            assert telemetry["connected"] is True
        assert adapter.is_connected() is False

    def test_job_client_sends_real_http_and_waits_for_terminal_state(self):
        _ApiHandler.polls = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ApiHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenDroneKitClient(
                f"http://127.0.0.1:{server.server_port}", token="local-token"
            )
            assert client.list_projects(4)[0]["name"] == "Quarry"
            assert client.list_assets(4)[0]["asset_type"] == "stockpile"
            submitted = client.submit_job(7, kind="reconstruction", dataset_id=12)
            assert submitted.id == 9
            finished = client.wait_for_job(9, timeout_s=2, poll_interval_s=0.01)
            assert finished.status == "completed"
            assert client.job_log(9)["log"] == "real local response"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
