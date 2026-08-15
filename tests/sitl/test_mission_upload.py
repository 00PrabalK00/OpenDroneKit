"""Mission protocol round-trip against an actual ArduPilot mission store."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.sitl

_POSITION_COMMANDS = {16, 19, 21, 22, 82}


def test_planner_mission_round_trips_through_ardupilot(sitl_client, sitl_mission):
    expected = list(sitl_mission.items)
    upload = sitl_client.upload_mission(expected)
    assert upload.success, upload.message

    actual = sitl_client.download_mission(timeout_s=30.0)
    assert len(actual) == len(expected)
    assert [item["seq"] for item in actual] == [item["seq"] for item in expected]
    assert [item["command"] for item in actual] == [item["command"] for item in expected]

    for sent, received in zip(expected, actual, strict=True):
        if int(sent["command"]) not in _POSITION_COMMANDS:
            continue
        assert received["lat"] == pytest.approx(sent["lat"], abs=2e-7)
        assert received["lon"] == pytest.approx(sent["lon"], abs=2e-7)
        assert received["alt"] == pytest.approx(sent["alt"], abs=0.05)
