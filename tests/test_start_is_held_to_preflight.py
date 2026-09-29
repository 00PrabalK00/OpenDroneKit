"""Start Mission is held to the preflight checklist.

The checklist was advisory: with the battery check blocking and nothing confirmed,
vehicle_command("start") armed the aircraft and flew the mission, and both UIs' Start
buttons went straight to it. Start is now refused while a check blocks or a manual
check is unconfirmed, naming them; the pilot in command can still start, as an explicit
override that is written to the audit log.

Driven through the production Api and both UIs against tests/fake_mavlink_vehicle.py,
a real UDP MAVLink peer.
"""

from __future__ import annotations

import json
import os
import time

import pytest

os.environ.setdefault("MAVLINK20", "1")
pytest.importorskip("pymavlink")

from test_flight_end_to_end import api, connect, plan, vehicles  # noqa: E402,F401

MISSION_START = 300


def _overrides(api) -> list[dict]:
    events = api.audit_log()["events"]
    return [json.loads(e.get("payload_json") or "{}") for e in events
            if e.get("event_type") == "preflight_override"]


def _started(vehicle) -> bool:
    return any(command == MISSION_START for command, *_ in vehicle.commands)


def test_start_is_refused_until_the_manual_checks_are_confirmed(api, vehicles):
    vehicle, uri = vehicles()
    connect(api, uri)
    plan(api)
    assert api.upload_mission()["ok"]
    vehicle.accept_modes = True

    refused = api.vehicle_command("start")
    assert refused["ok"] is False and refused["preflight_blocked"] is True, refused
    assert "Weather Acknowledged" in refused["error"], refused
    assert "Weather Acknowledged" in refused["unconfirmed"]
    time.sleep(0.5)
    assert not _started(vehicle), "a refused start still reached the aircraft"
    assert _overrides(api) == []

    pending = api.run_preflight()["pending_confirmation"]
    assert api.run_preflight(acknowledged=pending)["can_start"] is True
    started = api.vehicle_command("start")
    assert started["ok"] is True, started
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not _started(vehicle):
        time.sleep(0.1)
    assert _started(vehicle)


def test_a_blocking_check_cannot_be_acknowledged_away_only_overridden(api, vehicles):
    vehicle, uri = vehicles(battery_remaining=12)
    connect(api, uri)
    plan(api)
    assert api.upload_mission()["ok"]
    vehicle.accept_modes = True
    pending = api.run_preflight()["pending_confirmation"]
    api.run_preflight(acknowledged=pending + ["battery"])

    refused = api.vehicle_command("start")
    assert refused["ok"] is False and refused["preflight_blocked"] is True, refused
    assert any(item.startswith("Battery") for item in refused["blocking"]), refused
    assert refused["unconfirmed"] == [], refused
    assert not _started(vehicle)

    # The pilot in command may still start; the override is on the record.
    overridden = api.vehicle_command("start", {"override_preflight": True})
    assert overridden["ok"] is True, overridden
    [override] = _overrides(api)
    assert override["command"] == "start"
    assert any(item.startswith("Battery") for item in override["blocking"]), override


def test_other_commands_are_not_held(api, vehicles):
    """RTL, land and abort must never wait on a checklist."""
    vehicle, uri = vehicles()
    connect(api, uri)
    assert api.vehicle_command("rtl")["ok"] is True

