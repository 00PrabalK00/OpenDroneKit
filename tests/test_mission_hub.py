"""Secure mission preview and real compiled-plan simulation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading

from browser_evidence import dump_dom_command
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("email_validator")

from fastapi.testclient import TestClient  # noqa: E402


AOI = [[77.5900, 12.9700], [77.5910, 12.9700], [77.5910, 12.9710], [77.5900, 12.9710]]
WEB = Path(__file__).resolve().parents[1] / "app" / "web"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODK_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("ODK_SECRET_KEY", "test-secret-long-enough-for-hmac-sha256!")
    import services.api.db as db_module
    db_module._engine = None
    db_module._SessionLocal = None
    from services.api.main import app
    with TestClient(app) as test_client:
        yield test_client
    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture
def compiled_mission(client):
    registered = client.post("/auth/register", json={
        "email": "mission@example.com", "password": "longenough1",
        "organization_name": "Compiled Mission Co",
    }).json()
    headers = {"Authorization": f"Bearer {registered['access_token']}"}
    organization_id = client.get("/organizations", headers=headers).json()[0]["id"]
    project_id = client.post(
        f"/organizations/{organization_id}/projects", headers=headers,
        json={"name": "Bengaluru Survey"},
    ).json()["id"]
    response = client.post(f"/projects/{project_id}/missions", headers=headers, json={
        "name": "Grid A", "template": "grid", "aoi": AOI, "altitude_m": 60,
        "speed_m_s": 8, "gimbal_tilt_deg": -90, "aircraft_model": "DJI Mavic 3E",
        "battery_start_pct": 96, "battery_usable_minutes": 31,
    })
    assert response.status_code == 201, response.text
    return headers, project_id, response.json()["id"]


class TestMissionPreviewSharing:
    def test_secure_share_contains_complete_preview_from_compiled_plan(
        self, client, compiled_mission,
    ):
        headers, project_id, mission_id = compiled_mission
        plan = client.get(f"/missions/{mission_id}/plan", headers=headers).json()["plan"]
        created = client.post(f"/projects/{project_id}/shares", headers=headers, json={
            "password": "client-view", "include_missions": True,
            "include_defects": False,
        }).json()
        response = client.get(
            f"/public/shares/{created['url_token']}",
            headers={"X-Share-Password": "client-view"},
        )
        assert response.status_code == 200
        preview = response.json()["missions"][0]
        assert preview["mission_id"] == mission_id
        assert np.allclose(preview["path"]["coordinates"], plan["waypoints"])
        assert preview["area"]["status"] == "measured_from_georeferenced_aoi"
        assert preview["area"]["area_m2"] > 10_000
        assert preview["altitude"] == {
            "minimum_m": 60.0, "maximum_m": 60.0,
            "reference": "compiled_waypoint_altitude",
        }
        assert preview["duration_min"] == pytest.approx(plan["estimated_time_min"])
        assert preview["drone"]["model"] == "DJI Mavic 3E"
        assert preview["safety_areas"]["geofence"] == plan["safety_constraints"]["geofence"]


class TestCompiledMissionSimulation:
    def test_playback_uses_compiled_commands_gimbal_captures_and_battery(
        self, client, compiled_mission,
    ):
        headers, _, mission_id = compiled_mission
        plan = client.get(f"/missions/{mission_id}/plan", headers=headers).json()["plan"]
        simulation = client.get(f"/missions/{mission_id}/simulation", headers=headers).json()
        assert simulation["source"] == "persisted_compiled_plan"
        assert np.allclose(
            [row["position"] for row in simulation["timeline"]], plan["waypoints"]
        )
        assert all(row["gimbal_pitch_deg"] == -90 for row in simulation["timeline"])
        assert np.allclose(simulation["capture_points"], plan["waypoints"])
        assert simulation["battery"]["basis"] == "operator_declared_usable_minutes"
        assert simulation["timeline"][-1]["battery_pct"] < 96

    def test_missing_terrain_is_explicit_and_has_no_fake_surface(
        self, client, compiled_mission,
    ):
        headers, _, mission_id = compiled_mission
        simulation = client.get(f"/missions/{mission_id}/simulation", headers=headers).json()
        assert simulation["terrain"]["status"] == "unavailable"
        assert simulation["terrain"]["samples"] == []
        assert "does not draw a flat surface" in simulation["terrain"]["reason"]


class TestMissionHubBrowser:
    def test_browser_plays_the_real_api_payload(self, client, compiled_mission, tmp_path):
        headers, _, mission_id = compiled_mission
        payload = client.get(f"/missions/{mission_id}/simulation", headers=headers).json()
        script = (WEB / "js" / "hub-missions.js").read_bytes()
        encoded = json.dumps(payload).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return

            def do_GET(self):
                if self.path == "/hub-missions.js":
                    body, kind = script, "text/javascript"
                elif self.path == "/mission.json":
                    body, kind = encoded, "application/json"
                elif self.path == "/":
                    body = b'''<!doctype html><body data-result="pending"><canvas id="mission" width="720" height="420"></canvas><script src="/hub-missions.js"></script><script>window.onload=async()=>{try{const payload=await (await fetch('/mission.json')).json();const player=new ODKHubMissions.MissionSimulator(document.getElementById('mission'));const sim=player.load(payload);const frame=player.setFrame(sim.timeline.length-1);document.body.dataset.result=`frames:${sim.timeline.length};capture:${sim.capture_points.length};terrain:${sim.terrain.status};battery:${frame.battery_pct===null?'none':'estimated'}`;}catch(e){document.body.dataset.result='error:'+e.message;}}</script></body>'''
                    kind = "text/html"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            completed = subprocess.run(dump_dom_command(
                "mission playback evidence",
                tmp_path / "edge-profile",
                f"http://127.0.0.1:{server.server_port}/",
                virtual_time_ms=5000,
                webgl=False,
            ), capture_output=True, text=True, timeout=40, check=True)
            expected = len(payload["timeline"])
            assert (
                f'data-result="frames:{expected};capture:{expected};terrain:unavailable;'
                'battery:estimated"' in completed.stdout
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class TestMissionHubWiring:
    def test_hub_exposes_secure_share_and_playback_controls(self):
        html = (WEB / "hub.html").read_text(encoding="utf-8")
        script = (WEB / "js" / "hub.js").read_text(encoding="utf-8")
        assert 'id="mission-share-form"' in html
        assert 'id="mission-simulation-canvas"' in html
        assert "getMissionSimulation" in (
            WEB / "js" / "hub-api.js"
        ).read_text(encoding="utf-8")
        assert "createMissionShare" in script and "loadMissionSimulation" in script
