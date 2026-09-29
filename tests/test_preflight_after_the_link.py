"""Preflight after the link changes, driven through the Api against a real MAVLink peer.

Found driving preflight the way an operator does -- connect, plan, confirm, upload, lose
the link, disconnect:

* a mission the vehicle had accepted read "Drone not connected -- upload pending" once
  the link went silent;
* confirmations ("camera checked by eye") survived Disconnect and were shown as
  confirmed for whatever aircraft came next;
* the cockpit checklist refreshes every 5 s and wrote an audit row each time;
* a drone profile declaring "medium" obstacle avoidance passed as "medium OA enabled."
  though nothing read it from the aircraft.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("MAVLINK20", "1")
pytest.importorskip("pymavlink")

from app.api import Api  # noqa: E402
from app.session import AppSession  # noqa: E402
from app.store import ProjectStore  # noqa: E402
from fake_mavlink_vehicle import FakeVehicle, free_udp_port  # noqa: E402

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


@pytest.fixture
def flown(tmp_path):
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    api.create_project("preflight link", str(tmp_path / "project"))
    port = free_udp_port()
    vehicle = FakeVehicle(port).start()
    assert api.connect_vehicle(f"udpin:127.0.0.1:{port}", "mavlink")["ok"]
    time.sleep(2.4)
    api.set_aoi(AOI)
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    yield api, vehicle
    api.disconnect_vehicle()
    vehicle.stop()


def checks(report):
    return {c["id"]: c for c in report["checks"]}


def test_an_uploaded_mission_is_not_called_pending_when_the_link_goes_quiet(flown):
    api, vehicle = flown
    assert api.upload_mission()["ok"]
    assert checks(api.run_preflight())["mission_uploaded"]["severity"] == "pass"
    vehicle.go_silent()
    time.sleep(4.5)
    uploaded = checks(api.run_preflight())["mission_uploaded"]
    assert "pending" not in uploaded["message"]
    assert "Uploaded earlier" in uploaded["message"] and uploaded["severity"] != "pass"


def test_confirmations_do_not_outlive_the_connection(flown):
    api, _ = flown
    confirmed = checks(api.run_preflight(["camera", "gimbal", "weather"]))
    assert confirmed["camera"]["confirmed"] and confirmed["gimbal"]["confirmed"]
    api.disconnect_vehicle()
    after = api.run_preflight()
    assert not checks(after)["camera"]["confirmed"], "a confirmation outlived its aircraft"
    assert {"camera", "gimbal", "weather"} <= set(after["pending_confirmation"])


def test_the_checklist_refresh_does_not_flood_the_audit_log(flown):
    api, _ = flown

    def preflight_rows():
        return [e for e in api.audit_log(500)["events"] if e["event_type"] == "preflight_run"]

    before = len(preflight_rows())
    for _ in range(6):                      # thirty seconds of the cockpit checklist
        api.run_preflight()
    assert len(preflight_rows()) - before <= 1
    api.run_preflight(["weather"])          # an operator's confirmation is recorded
    assert len(preflight_rows()) - before == 2


def test_a_declared_obstacle_avoidance_is_not_a_measured_one():
    from core.preflight import check_obstacle_avoidance
    from core.settings import DroneProfile

    result = check_obstacle_avoidance(DroneProfile(obstacle_avoidance="medium"))
    assert result.severity != "pass" and result.requires_manual
    assert "declares medium" in result.message
