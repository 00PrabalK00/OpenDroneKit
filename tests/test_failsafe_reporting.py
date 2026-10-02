"""How the app reports the vehicle's failsafes, fence, GPS and lost link.

These pin the reporting logic in isolation. Whether ArduPilot actually sends what is
parsed here, and does what it says, is proven against real ArduCopter SITL in
tests/sitl/test_copter_failsafes.py.
"""

from __future__ import annotations

import json
import time

from app.session import vehicle_alerts
from core.drone import DroneTelemetry
from core.flight import FlightManager, FlightSession, FlightState
from core.flight_log import FlightLog, write_gpx
from core.preflight import check_battery_status


def _telemetry(**raw) -> DroneTelemetry:
    return DroneTelemetry(connected=True, gps_fix=3, satellites=10, battery_pct=80.0,
                          battery_v=12.1, flight_mode="AUTO", raw=dict(raw))


def test_nothing_reported_means_no_failsafe_fields():
    payload = DroneTelemetry(connected=True, gps_fix=3).to_dict()
    for key in ("failsafe", "battery_failsafe", "fence_breached", "gps_stale", "vehicle_messages"):
        assert key not in payload


def test_vehicle_failsafe_state_comes_from_its_heartbeat():
    assert _telemetry(system_status=5).to_dict()["failsafe"] is True
    assert _telemetry(system_status=4).to_dict()["failsafe"] is False


def test_battery_failsafe_is_the_vehicles_charge_state_not_the_percentage():
    payload = _telemetry(battery_charge_state=2).to_dict()
    assert payload["battery_failsafe"] is True and payload["battery_state"] == "low"
    assert payload["battery_pct"] == 80.0
    assert _telemetry(battery_charge_state=1).to_dict()["battery_failsafe"] is False


def test_fence_breach_is_reported_with_its_type():
    payload = _telemetry(fence_status={"breach_status": 1, "breach_count": 2,
                                       "breach_type": 3}).to_dict()
    assert payload["fence_breached"] is True
    assert payload["fence_breach_count"] == 2
    assert payload["fence_breach_type"] == "boundary"


def test_a_gps_report_that_stopped_is_not_a_3d_fix():
    tel = _telemetry(gps_received_at=time.time() - 10.0)
    tel.timestamp = time.time()
    payload = tel.to_dict()
    assert payload["gps_fix"] == 0 and payload["gps_stale"] is True


def test_alerts_quote_the_vehicle():
    tel = _telemetry(battery_charge_state=2, system_status=5,
                     vehicle_messages=[{"time": time.time(), "severity": 4,
                                        "text": "Battery 1 is low 10.60V used 120 mAh"}])
    alerts = {a["kind"]: a["message"] for a in vehicle_alerts(tel.to_dict())}
    assert "Battery 1 is low" in alerts["battery_failsafe"]
    assert "failsafe" in alerts


def test_preflight_blocks_on_the_vehicles_battery_failsafe():
    check = check_battery_status(_telemetry(battery_charge_state=3))
    assert check.severity == "blocking" and "failsafe" in check.message


class _SilentAfter:
    def __init__(self, tel: DroneTelemetry) -> None:
        self.tel = tel

    def is_connected(self) -> bool:
        return True

    def get_telemetry(self) -> DroneTelemetry:
        return self.tel


def test_the_flight_session_enters_and_leaves_link_lost(tmp_path):
    manager = FlightManager()
    tel = _telemetry()
    tel.timestamp = time.time() - 10.0
    manager._drone = _SilentAfter(tel)
    manager._session = FlightSession(id="s", project_id="p", mission_id="m", drone_id="d",
                                     log_dir=str(tmp_path), state=FlightState("s", status="flying"))
    manager._apply_telemetry(tel)
    assert manager._session.state.status == "link_lost"
    assert "Telemetry lost" in manager._session.state.warnings

    tel.timestamp = time.time()
    tel.flight_mode = "RTL"
    manager._apply_telemetry(tel)
    state = manager._session.state
    assert state.status == "flying"
    assert state.link_gaps[-1]["duration_s"] >= 9.0 and state.link_gaps[-1]["mode_after"] == "RTL"
    events = [json.loads(line)["event"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == ["link_lost", "link_restored"]


def test_the_flight_log_marks_a_gap_and_does_not_draw_across_it(tmp_path):
    log = FlightLog()
    start = 1_700_000_000.0
    for i, t in enumerate((0.0, 1.0, 2.0, 12.0, 13.0)):
        log.record({"timestamp": start + t, "latitude": 1.0 + i * 1e-4, "longitude": 2.0,
                    "gps_fix": 3, "flight_mode": "AUTO" if t < 5 else "RTL"})
    summary = log.summary()
    assert len(summary["link_gaps"]) == 1
    assert summary["link_gaps"][0]["duration_s"] == 10.0
    assert summary["link_gaps"][0]["mode_after"] == "RTL"
    gpx = write_gpx(log, tmp_path / "f.gpx").read_text(encoding="utf-8")
    assert gpx.count("<trkseg") == 2
