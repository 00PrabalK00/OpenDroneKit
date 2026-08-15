"""RTK/PPK input checking: camera events, base observations, and the clock between them.

The failure mode being tested for is a survey that processes cleanly and is out by a
metre. It happens when the base station was not recording for part of the flight, when
the aircraft never held a fixed solution, or when GPS time is converted to UTC with the
wrong leap-second offset. None of those raise anything by themselves; each produces
coordinates that look exactly like good ones.

Fixtures are written in the real file formats -- a DJI ``.MRK`` event list and a RINEX
observation header -- because parsing is half of what is being checked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.rtk import (
    DEFAULT_LEAP_SECONDS,
    RtkError,
    align_events_to_base,
    gps_to_utc,
    positioning_report,
    read_base_station,
    read_camera_events,
)

# 15 August 2026 is GPS week 2382. Second-of-week 259200 is midday on the Thursday.
WEEK = 2382
NOON = 259200.0


def mrk_line(sequence, seconds, week=WEEK, flag=50, std=0.012):
    return (
        f"{sequence}\t{seconds:.6f}\t{week}\t"
        f"12.5,N\t-8.0,E\t3.5,V\t"
        f"18.5204{sequence:02d},Lat\t73.8567{sequence:02d},Lon\t560.412,Ellh\t"
        f"{std},Ns\t{std},Es\t{std * 2},Vs\t{flag},Q"
    )


def write_mrk(path, count=5, first_seconds=NOON, step=2.0, flag=50):
    lines = [mrk_line(i + 1, first_seconds + i * step, flag=flag) for i in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def rinex_header(first: datetime, last: datetime | None = None, *, marker="BASE01",
                 approx=True, end_of_header=True, interval=1.0):
    def epoch(value: datetime, label: str) -> str:
        body = (f"  {value.year}    {value.month:2d}    {value.day:2d}    "
                f"{value.hour:2d}    {value.minute:2d}   {value.second:10.7f}     GPS")
        return f"{body:<60}{label}"

    lines = [
        f"{'     3.04           OBSERVATION DATA    M':<60}RINEX VERSION / TYPE",
        f"{marker:<60}MARKER NAME",
        f"{'1234                 TRIMBLE R12         5.20':<60}REC # / TYPE / VERS",
        f"{'5678                 TRM115000.00':<60}ANT # / TYPE",
    ]
    if approx:
        lines.append(f"{'  1116000.0000   5843000.0000   2011000.0000':<60}APPROX POSITION XYZ")
    lines.append(epoch(first, "TIME OF FIRST OBS"))
    if last is not None:
        lines.append(epoch(last, "TIME OF LAST OBS"))
    if interval is not None:
        lines.append(f"{f'    {interval:.3f}':<60}INTERVAL")
    if end_of_header:
        lines.append(f"{'':<60}END OF HEADER")
    return "\n".join(lines) + "\n"


def write_rinex(path, **kwargs):
    first = kwargs.pop("first", gps_to_utc(WEEK, NOON) - timedelta(minutes=5))
    last = kwargs.pop("last", gps_to_utc(WEEK, NOON) + timedelta(minutes=30))
    path.write_text(rinex_header(first, last, **kwargs), encoding="utf-8")
    return path


class TestGpsTime:
    def test_the_gps_epoch_itself_converts_back_with_the_leap_offset(self):
        assert gps_to_utc(0, DEFAULT_LEAP_SECONDS) == datetime(1980, 1, 6, tzinfo=timezone.utc)

    def test_leap_seconds_are_subtracted_not_ignored(self):
        """GPS time does not observe leap seconds; ignoring them shifts every event."""
        with_leap = gps_to_utc(WEEK, NOON, leap_seconds=18)
        without = gps_to_utc(WEEK, NOON, leap_seconds=0)
        assert (without - with_leap).total_seconds() == 18

    def test_a_week_is_seven_days(self):
        assert (gps_to_utc(WEEK + 1, NOON) - gps_to_utc(WEEK, NOON)).days == 7

    def test_a_nonsense_timestamp_is_refused(self):
        with pytest.raises(RtkError, match="non-negative week"):
            gps_to_utc(-1, 0.0)


class TestCameraEvents:
    def test_every_shutter_event_is_read(self, tmp_path):
        events = read_camera_events(write_mrk(tmp_path / "flight.MRK", count=7))
        assert len(events) == 7
        assert [e.sequence for e in events] == list(range(1, 8))

    def test_corrections_are_converted_from_millimetres(self, tmp_path):
        """A correction read as metres would be a thousand times too large."""
        event = read_camera_events(write_mrk(tmp_path / "flight.MRK"))[0]

        assert event.north_correction_m == pytest.approx(0.0125)
        assert event.east_correction_m == pytest.approx(-0.008)
        assert event.vertical_correction_m == pytest.approx(0.0035)

    def test_the_solution_flag_is_carried_through(self, tmp_path):
        fixed = read_camera_events(write_mrk(tmp_path / "fixed.MRK", flag=50))[0]
        floating = read_camera_events(write_mrk(tmp_path / "float.MRK", flag=1))[0]

        assert fixed.solution == "fixed"
        assert floating.solution == "float"

    def test_an_unrecognised_flag_is_reported_not_assumed_good(self, tmp_path):
        event = read_camera_events(write_mrk(tmp_path / "odd.MRK", flag=7))[0]
        assert event.solution == "unknown(7)"

    def test_an_empty_event_file_is_refused(self, tmp_path):
        path = tmp_path / "empty.MRK"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(RtkError, match="no camera events"):
            read_camera_events(path)

    def test_a_missing_event_file_is_refused(self, tmp_path):
        with pytest.raises(RtkError, match="not found"):
            read_camera_events(tmp_path / "absent.MRK")


class TestBaseStation:
    def test_the_station_position_and_window_are_read(self, tmp_path):
        base = read_base_station(write_rinex(tmp_path / "base.26o"))

        assert base.marker_name == "BASE01"
        assert base.approx_xyz_m[0] == pytest.approx(1116000.0)
        assert base.duration_s == pytest.approx(35 * 60)
        assert base.interval_s == pytest.approx(1.0)

    def test_a_file_without_a_base_position_is_refused(self, tmp_path):
        """An unknown base offsets the whole survey identically and invisibly."""
        path = tmp_path / "nopos.26o"
        path.write_text(rinex_header(gps_to_utc(WEEK, NOON), approx=False), encoding="utf-8")

        with pytest.raises(RtkError, match="where the base stood is unknown"):
            read_base_station(path)

    def test_a_truncated_header_is_refused(self, tmp_path):
        path = tmp_path / "truncated.26o"
        path.write_text(rinex_header(gps_to_utc(WEEK, NOON), end_of_header=False),
                        encoding="utf-8")

        with pytest.raises(RtkError, match="END OF HEADER"):
            read_base_station(path)


class TestAlignment:
    def test_a_flight_inside_the_session_is_fully_covered(self, tmp_path):
        events = read_camera_events(write_mrk(tmp_path / "flight.MRK", count=10))
        base = read_base_station(write_rinex(tmp_path / "base.26o"))

        alignment = align_events_to_base(events, base)
        assert alignment["coverage_fraction"] == 1.0
        assert alignment["uncovered_before"] == []

    def test_events_before_the_base_started_are_named(self, tmp_path):
        """The base was switched on late; those frames cannot be corrected."""
        events = read_camera_events(write_mrk(tmp_path / "flight.MRK", count=6))
        base = read_base_station(write_rinex(
            tmp_path / "base.26o",
            first=gps_to_utc(WEEK, NOON + 5),
            last=gps_to_utc(WEEK, NOON + 600)))

        alignment = align_events_to_base(events, base)
        assert alignment["uncovered_before"] == [1, 2, 3]
        assert alignment["coverage_fraction"] == pytest.approx(0.5)

    def test_events_after_the_base_stopped_are_named(self, tmp_path):
        events = read_camera_events(write_mrk(tmp_path / "flight.MRK", count=6))
        base = read_base_station(write_rinex(
            tmp_path / "base.26o",
            first=gps_to_utc(WEEK, NOON - 60),
            last=gps_to_utc(WEEK, NOON + 5)))

        alignment = align_events_to_base(events, base)
        assert alignment["uncovered_after"] == [4, 5, 6]

    def test_a_session_with_no_declared_end_cannot_demonstrate_coverage(self, tmp_path):
        events = read_camera_events(write_mrk(tmp_path / "flight.MRK"))
        path = tmp_path / "base.26o"
        path.write_text(rinex_header(gps_to_utc(WEEK, NOON - 60)), encoding="utf-8")
        base = read_base_station(path)

        with pytest.raises(RtkError, match="unverified"):
            align_events_to_base(events, base)

    def test_the_leap_second_assumption_changes_coverage_and_is_recorded(self, tmp_path):
        """A wrong offset shifts every event by a second; the report says which was used."""
        events = read_camera_events(write_mrk(tmp_path / "flight.MRK"))
        base = read_base_station(write_rinex(tmp_path / "base.26o"))

        assert align_events_to_base(events, base, leap_seconds=17)["leap_seconds"] == 17


class TestPositioningReport:
    def test_a_clean_fixed_flight_is_usable_and_says_what_it_means(self, tmp_path):
        report = positioning_report(write_mrk(tmp_path / "flight.MRK", count=8),
                                    write_rinex(tmp_path / "base.26o"))

        assert report["usable_for_ppk"] is True
        assert report["fixed_fraction"] == 1.0
        # 12 mm north and 12 mm east combine to 17 mm horizontally, not 12.
        assert "1.7 cm" in report["accuracy_statement"]
        assert "not the accuracy of the final deliverable" in report["accuracy_statement"]

    def test_it_never_claims_to_have_computed_a_solution(self, tmp_path):
        report = positioning_report(write_mrk(tmp_path / "flight.MRK"),
                                    write_rinex(tmp_path / "base.26o"))

        assert any("not a PPK solution" in caveat for caveat in report["caveats"])

    def test_a_float_flight_is_not_described_as_rtk(self, tmp_path):
        """The client asked for RTK; the aircraft recorded float. Say float."""
        report = positioning_report(write_mrk(tmp_path / "flight.MRK", flag=1),
                                    write_rinex(tmp_path / "base.26o"))

        assert report["fixed_fraction"] == 0.0
        assert "does not support an RTK or PPK accuracy claim" in report["accuracy_statement"]

    def test_a_partly_fixed_flight_refuses_a_single_accuracy_figure(self, tmp_path):
        path = tmp_path / "mixed.MRK"
        path.write_text("\n".join([
            mrk_line(1, NOON, flag=50), mrk_line(2, NOON + 2, flag=50),
            mrk_line(3, NOON + 4, flag=1), mrk_line(4, NOON + 6, flag=1),
        ]) + "\n", encoding="utf-8")
        report = positioning_report(path, write_rinex(tmp_path / "base.26o"))

        assert report["fixed_fraction"] == pytest.approx(0.5)
        assert "quote the fixed and non-fixed portions separately" in report["accuracy_statement"]

    def test_partial_base_coverage_blocks_the_survey_with_the_reason(self, tmp_path):
        report = positioning_report(
            write_mrk(tmp_path / "flight.MRK", count=6),
            write_rinex(tmp_path / "base.26o",
                        first=gps_to_utc(WEEK, NOON + 5),
                        last=gps_to_utc(WEEK, NOON + 600)))

        assert report["usable_for_ppk"] is False
        assert any("nothing in the output marking which is which" in reason
                   for reason in report["blocking"])

    def test_the_leap_second_assumption_is_always_stated(self, tmp_path):
        report = positioning_report(write_mrk(tmp_path / "flight.MRK"),
                                    write_rinex(tmp_path / "base.26o"))

        assert any("leap seconds" in caveat for caveat in report["caveats"])


class TestThroughTheApi:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("ppk", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_good_inputs_are_reported_as_usable(self, api, tmp_path):
        result = api.check_ppk_inputs(str(write_mrk(tmp_path / "flight.MRK")),
                                      str(write_rinex(tmp_path / "base.26o")))

        assert result["ok"] is True
        assert result["usable_for_ppk"] is True
        assert result["caveats"]

    def test_partial_coverage_is_surfaced_before_processing(self, api, tmp_path):
        result = api.check_ppk_inputs(
            str(write_mrk(tmp_path / "flight.MRK", count=6)),
            str(write_rinex(tmp_path / "base.26o",
                            first=gps_to_utc(WEEK, NOON + 5),
                            last=gps_to_utc(WEEK, NOON + 600))))

        assert result["usable_for_ppk"] is False
        assert result["blocking"]

    def test_an_unreadable_base_file_is_refused_with_the_reason(self, api, tmp_path):
        path = tmp_path / "base.26o"
        path.write_text(rinex_header(gps_to_utc(WEEK, NOON), approx=False), encoding="utf-8")
        result = api.check_ppk_inputs(str(write_mrk(tmp_path / "flight.MRK")), str(path))

        assert result["ok"] is False
        assert "where the base stood is unknown" in result["error"]
