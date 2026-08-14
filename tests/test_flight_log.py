"""Flight logging and export.

The moment a flight log matters is after the flight, when someone asks where the
aircraft was or what altitude something was surveyed at. So the tests check that the
record survives export intact, and that the two things which would quietly falsify it
do not happen: a sample with no GPS fix must not appear in a track as a position, and a
gap in the recording must not be smoothed over with a line nobody observed.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET

import pytest

from core.flight_log import (
    MIN_USABLE_FIX,
    FlightLog,
    FlightSample,
    export,
    export_all,
    write_csv,
    write_gpx,
    write_json,
    write_kml,
)

BASE_TIME = 1_800_000_000.0
BASE_LAT, BASE_LON = 41.3042, -81.7505


def sample(offset_s: float, *, fix: int = 3, lat_step: float = 0.0,
           battery: float = 90.0, alt: float = 60.0) -> FlightSample:
    return FlightSample(
        timestamp=BASE_TIME + offset_s,
        latitude=BASE_LAT + lat_step, longitude=BASE_LON,
        altitude_rel_m=alt, altitude_abs_m=alt + 280.0,
        heading_deg=90.0, speed_mps=8.0,
        battery_pct=battery, battery_v=22.5,
        gps_fix=fix, satellites=14 if fix >= 3 else 0, hdop=0.8 if fix >= 3 else 99.9,
        flight_mode="AUTO", armed=True, waypoint_index=1, waypoint_total=10,
        link_quality_pct=98.0,
    )


@pytest.fixture
def flight() -> FlightLog:
    log = FlightLog(aircraft="M300-1", pilot="P. Khare", mission_name="Bridge 14")
    for i in range(10):
        log.samples.append(sample(i * 2.0, lat_step=i * 0.0001, battery=90.0 - i))
    return log


class TestRecording:
    def test_telemetry_is_recorded_as_a_sample(self):
        log = FlightLog()
        log.record({"timestamp": BASE_TIME, "latitude": BASE_LAT, "longitude": BASE_LON,
                    "gps_fix": 3, "battery_pct": 88.0})
        assert len(log.samples) == 1
        assert log.samples[0].battery_pct == pytest.approx(88.0)

    def test_an_object_with_attributes_records_the_same_way(self):
        """The bridge hands over a dataclass, not a dict."""
        from core.drone import DroneTelemetry

        log = FlightLog()
        log.record(DroneTelemetry(latitude=BASE_LAT, longitude=BASE_LON, gps_fix=3))
        assert log.samples[0].latitude == pytest.approx(BASE_LAT)

    def test_duration_and_distance_come_from_the_samples(self, flight):
        assert flight.duration_s == pytest.approx(18.0)
        assert flight.distance_m() > 0


class TestPositionValidity:
    def test_a_sample_with_no_fix_carries_no_position(self):
        assert sample(0.0, fix=0).has_position is False
        assert sample(0.0, fix=1).has_position is False
        assert sample(0.0, fix=MIN_USABLE_FIX).has_position is True

    def test_a_zero_coordinate_with_a_claimed_fix_is_still_rejected(self):
        """0,0 is a receiver reporting its uninitialised state, not a place at sea."""
        null_island = FlightSample(timestamp=BASE_TIME, latitude=0.0, longitude=0.0,
                                   gps_fix=3)
        assert null_island.has_position is False

    def test_unfixed_samples_are_counted_in_the_summary(self, flight):
        flight.samples.append(sample(20.0, fix=0))
        summary = flight.summary()
        assert summary["samples_without_fix"] == 1
        assert summary["positioned_samples"] == 10

    def test_distance_ignores_samples_without_a_fix(self, flight):
        """Otherwise a dropout teleports the aircraft to null island and back."""
        before = flight.distance_m()
        flight.samples.insert(5, sample(9.0, fix=0))
        assert flight.distance_m() == pytest.approx(before)


class TestCsv:
    def test_every_sample_reaches_the_csv(self, flight, tmp_path):
        flight.samples.append(sample(20.0, fix=0))
        path = write_csv(flight, tmp_path / "flight.csv")

        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 11, "the CSV is the full record, including unfixed samples"

    def test_the_csv_carries_the_columns_an_operator_reads(self, flight, tmp_path):
        path = write_csv(flight, tmp_path / "flight.csv")
        header = path.read_text(encoding="utf-8").splitlines()[0]
        for column in ("timestamp_utc", "latitude", "altitude_rel_m", "battery_pct",
                       "gps_fix", "flight_mode"):
            assert column in header

    def test_elapsed_time_is_relative_to_the_first_sample(self, flight, tmp_path):
        path = write_csv(flight, tmp_path / "flight.csv")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert float(rows[0]["elapsed_s"]) == pytest.approx(0.0)
        assert float(rows[-1]["elapsed_s"]) == pytest.approx(18.0)


class TestJson:
    def test_the_json_keeps_the_summary_alongside_the_samples(self, flight, tmp_path):
        path = write_json(flight, tmp_path / "flight.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["summary"]["sample_count"] == 10
        assert len(payload["samples"]) == 10

    def test_the_summary_states_that_nothing_is_interpolated(self, flight, tmp_path):
        path = write_json(flight, tmp_path / "flight.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "nothing is interpolated" in payload["summary"]["note"]


class TestGpx:
    def test_the_gpx_track_holds_one_point_per_fixed_sample(self, flight, tmp_path):
        path = write_gpx(flight, tmp_path / "flight.gpx")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        points = [e for e in root.iter() if e.tag.endswith("trkpt")]
        assert len(points) == 10

    def test_unfixed_samples_are_omitted_from_the_track(self, flight, tmp_path):
        """A track is a claim about position; a sample without a fix is not one."""
        flight.samples.append(sample(20.0, fix=0))
        path = write_gpx(flight, tmp_path / "flight.gpx")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        points = [e for e in root.iter() if e.tag.endswith("trkpt")]

        assert len(points) == 10, "an unfixed sample must not appear as a position"
        for point in points:
            assert abs(float(point.get("lat"))) > 1e-6

    def test_gpx_uses_lat_lon_attributes_in_that_order(self, flight, tmp_path):
        path = write_gpx(flight, tmp_path / "flight.gpx")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        first = next(e for e in root.iter() if e.tag.endswith("trkpt"))

        assert float(first.get("lat")) == pytest.approx(BASE_LAT, abs=0.01)
        assert float(first.get("lon")) == pytest.approx(BASE_LON, abs=0.01)


class TestKml:
    def test_the_kml_track_is_written_in_lon_lat_alt_order(self, flight, tmp_path):
        """KML reverses GPX's ordering, and getting it wrong relocates the flight."""
        path = write_kml(flight, tmp_path / "flight.kml")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        coordinates = next(e for e in root.iter() if e.tag.endswith("coordinates")).text
        first = coordinates.split()[0].split(",")

        assert float(first[0]) == pytest.approx(BASE_LON, abs=0.01)
        assert float(first[1]) == pytest.approx(BASE_LAT, abs=0.01)
        assert float(first[2]) > 100.0

    def test_the_track_is_drawn_at_its_real_altitude(self, flight, tmp_path):
        """Draping it on the ground would discard the altitude that was flown."""
        path = write_kml(flight, tmp_path / "flight.kml")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        mode = next(e for e in root.iter() if e.tag.endswith("altitudeMode"))
        assert mode.text == "absolute"

    def test_unfixed_samples_are_omitted_from_the_kml_too(self, flight, tmp_path):
        flight.samples.append(sample(20.0, fix=0))
        path = write_kml(flight, tmp_path / "flight.kml")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        coordinates = next(e for e in root.iter() if e.tag.endswith("coordinates")).text
        assert len(coordinates.split()) == 10


class TestExportDispatch:
    def test_the_format_is_chosen_by_name(self, flight, tmp_path):
        for fmt in ("csv", "json", "gpx", "kml"):
            assert export(flight, tmp_path / f"f.{fmt}", fmt).exists()

    def test_an_unsupported_format_is_refused_and_lists_the_real_ones(self, flight, tmp_path):
        with pytest.raises(ValueError, match="Use one of"):
            export(flight, tmp_path / "f.xls", "xls")

    def test_export_all_writes_every_format(self, flight, tmp_path):
        from pathlib import Path

        written = export_all(flight, tmp_path / "logs", stem="bridge14")
        assert set(written) == {"csv", "json", "gpx", "kml"}
        assert all(Path(p).exists() for p in written.values())


class TestEmptyAndDegenerate:
    def test_an_empty_log_exports_without_crashing(self, tmp_path):
        empty = FlightLog(mission_name="Never flew")
        for fmt in ("csv", "json", "gpx", "kml"):
            assert export(empty, tmp_path / f"e.{fmt}", fmt).exists()

    def test_an_empty_log_reports_zero_rather_than_guessing(self, tmp_path):
        summary = FlightLog().summary()
        assert summary["sample_count"] == 0
        assert summary["duration_s"] == 0.0
        assert summary["max_altitude_rel_m"] is None

    def test_a_single_sample_has_no_duration_and_no_distance(self):
        log = FlightLog()
        log.samples.append(sample(0.0))
        assert log.duration_s == 0.0
        assert log.distance_m() == 0.0

    def test_a_log_with_no_fixes_produces_an_empty_track_not_a_false_one(self, tmp_path):
        log = FlightLog(mission_name="No signal")
        for i in range(5):
            log.samples.append(sample(i * 2.0, fix=0))

        path = write_gpx(log, tmp_path / "nofix.gpx")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        assert [e for e in root.iter() if e.tag.endswith("trkpt")] == []
