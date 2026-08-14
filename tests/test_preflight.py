"""Preflight checks.

The whole point of a preflight is to be pessimistic. A check that cannot determine
something must say so, because a checklist showing fourteen ticks reads as fourteen
things verified, and a pilot who has been told the compass is fine will not go and look
at it. So the tests below concentrate on the difference between passed, failed, and not
established -- particularly for the sensors an autopilot only reports if asked, where
silence is easily mistaken for health.

They also pin the boundary: nothing here refuses to fly. Blocking checks report that
they block, and the operator remains the one who decides.
"""

from __future__ import annotations

import pytest

from core.drone import DroneTelemetry, MockDroneClient
from core.preflight import (
    SENSOR_BITS,
    SEVERITY_BLOCK,
    SEVERITY_PASS,
    SEVERITY_WARN,
    can_start_mission,
    check_battery_status,
    check_compass,
    check_drone_connection,
    check_gimbal,
    check_gps_status,
    check_home_position,
    check_imu,
    confirm_manual_check,
    run_preflight,
)

ALL_SENSORS = SENSOR_BITS["gyro"] | SENSOR_BITS["accelerometer"] | SENSOR_BITS["compass"]


def telemetry_with(health: int | None = ALL_SENSORS, present: int | None = None,
                   **overrides) -> DroneTelemetry:
    """Telemetry carrying the sensor bitmasks a real SYS_STATUS message would."""
    telemetry = DroneTelemetry(connected=True, gps_fix=3, satellites=14, hdop=0.8,
                               home_set=True, battery_pct=85.0, **overrides)
    if health is not None:
        telemetry.raw["sensors_present"] = ALL_SENSORS if present is None else present
        telemetry.raw["sensors_enabled"] = ALL_SENSORS
        telemetry.raw["sensors_health"] = health
    return telemetry


class TestCompass:
    def test_a_healthy_compass_passes(self):
        result = check_compass(telemetry_with())
        assert result.severity == SEVERITY_PASS

    def test_an_unhealthy_compass_blocks(self):
        """A bad compass arms happily and flies away; this is the check that matters."""
        health = ALL_SENSORS & ~SENSOR_BITS["compass"]
        result = check_compass(telemetry_with(health=health))

        assert result.severity == SEVERITY_BLOCK
        assert "Calibrate" in result.fix_action

    def test_an_unreported_compass_warns_rather_than_passing(self):
        """Silence is not health. A tick here would stop the pilot looking."""
        result = check_compass(DroneTelemetry(connected=True))

        assert result.severity == SEVERITY_WARN
        assert "not a healthy sensor" in result.fix_action

    def test_an_absent_compass_warns_and_asks_for_confirmation(self):
        result = check_compass(telemetry_with(health=0, present=0))
        assert result.severity == SEVERITY_WARN
        assert "without one" in result.fix_action


class TestImu:
    def test_a_healthy_imu_passes(self):
        assert check_imu(telemetry_with()).severity == SEVERITY_PASS

    def test_an_unhealthy_gyro_blocks_and_is_named(self):
        health = ALL_SENSORS & ~SENSOR_BITS["gyro"]
        result = check_imu(telemetry_with(health=health))

        assert result.severity == SEVERITY_BLOCK
        assert "gyroscope" in result.message

    def test_an_unhealthy_accelerometer_blocks_and_is_named(self):
        health = ALL_SENSORS & ~SENSOR_BITS["accelerometer"]
        result = check_imu(telemetry_with(health=health))

        assert result.severity == SEVERITY_BLOCK
        assert "accelerometer" in result.message

    def test_both_faulty_names_both(self):
        health = SENSOR_BITS["compass"]
        result = check_imu(telemetry_with(health=health))
        assert "gyroscope" in result.message and "accelerometer" in result.message

    def test_an_unreported_imu_warns(self):
        assert check_imu(DroneTelemetry(connected=True)).severity == SEVERITY_WARN


class TestGimbal:
    def test_a_ready_gimbal_passes(self):
        assert check_gimbal(True).severity == SEVERITY_PASS

    def test_a_failed_gimbal_blocks(self):
        result = check_gimbal(False)
        assert result.severity == SEVERITY_BLOCK
        assert "power-cycle" in result.fix_action.lower()

    def test_an_unreported_gimbal_needs_the_operator_to_look(self):
        """A stuck gimbal points the survey somewhere else and nothing downstream notices."""
        result = check_gimbal(None)
        assert result.severity == SEVERITY_WARN
        assert result.requires_manual is True
        assert result.ok is False, "an unconfirmed manual check must not read as passed"


class TestGps:
    def test_an_rtk_fix_passes(self):
        result = check_gps_status(DroneTelemetry(gps_fix=5, satellites=20, hdop=0.5))
        assert result.severity == SEVERITY_PASS
        assert "RTK" in result.message

    def test_a_marginal_fix_warns_rather_than_blocking(self):
        result = check_gps_status(DroneTelemetry(gps_fix=3, satellites=6, hdop=2.5))
        assert result.severity == SEVERITY_WARN

    def test_no_fix_blocks(self):
        result = check_gps_status(DroneTelemetry(gps_fix=0, satellites=0))
        assert result.severity == SEVERITY_BLOCK

    def test_an_unset_home_position_blocks(self):
        assert check_home_position(DroneTelemetry(home_set=False)).severity == SEVERITY_BLOCK


class TestBattery:
    def test_a_full_pack_passes(self):
        result = check_battery_status(DroneTelemetry(battery_pct=95.0))
        assert result.severity == SEVERITY_PASS

    def test_a_critical_pack_blocks(self):
        result = check_battery_status(DroneTelemetry(battery_pct=5.0),
                                      warn_pct=30.0, critical_pct=15.0)
        assert result.severity == SEVERITY_BLOCK


class TestConnection:
    def test_a_disconnected_vehicle_blocks(self):
        result = check_drone_connection(MockDroneClient())
        assert result.severity == SEVERITY_BLOCK

    def test_a_connected_vehicle_passes(self):
        client = MockDroneClient()
        client.connect("mock://vehicle")
        assert check_drone_connection(client).severity == SEVERITY_PASS


class TestWholeReport:
    def _profile(self):
        from core.settings import DroneProfile

        return DroneProfile()

    def test_a_report_covers_every_specified_subsystem(self):
        """Connection, battery, GPS, compass, IMU, home, storage, camera, gimbal, geofence."""
        client = MockDroneClient()
        client.connect("mock://vehicle")

        report = run_preflight("p", "m", "d", client, self._profile())
        ids = {check.id for check in report.checks}

        for required in ("drone_connection", "gps_status", "home_position",
                         "battery", "compass", "imu", "camera",
                         "gimbal", "storage", "geofence"):
            assert required in ids, f"{required} is not checked before arming"

    def test_a_report_with_blocking_issues_cannot_start(self):
        report = run_preflight("p", "m", "d", MockDroneClient(), self._profile())
        assert report.can_start is False
        assert report.blocking_issues

    def test_confirming_a_manual_check_records_who_said_so(self):
        client = MockDroneClient()
        client.connect("mock://vehicle")
        report = run_preflight("p", "m", "d", client, self._profile())

        confirmed = confirm_manual_check(report, "gimbal", "Checked by eye, level and free.")
        assert confirmed.confirmed is True
        assert "level and free" in confirmed.operator_note

    def test_confirming_an_unknown_check_is_refused(self):
        from core.errors import AppError

        report = run_preflight("p", "m", "d", MockDroneClient(), self._profile())
        with pytest.raises(AppError):
            confirm_manual_check(report, "no_such_check")

    def test_the_report_counts_what_blocks_and_what_merely_warns(self):
        report = run_preflight("p", "m", "d", MockDroneClient(), self._profile())
        payload = report.to_dict()

        assert payload["blocking_count"] == len(report.blocking_issues)
        assert payload["can_start"] is False

    def test_can_start_mission_agrees_with_the_report(self):
        report = run_preflight("p", "m", "d", MockDroneClient(), self._profile())
        assert can_start_mission(report) == report.can_start

    def test_sensor_checks_are_skipped_when_there_is_no_vehicle(self):
        """Reporting a healthy compass on a vehicle nobody is talking to would be a lie."""
        report = run_preflight("p", "m", "d", MockDroneClient(), self._profile())
        ids = {check.id for check in report.checks}
        assert "compass" not in ids and "imu" not in ids
