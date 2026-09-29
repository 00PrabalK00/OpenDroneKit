"""The camera event file as a DJI RTK aircraft actually writes it.

The reader was written against an invented layout (tagged ``Ns``/``Es``/``Vs`` standard
deviations, an unbracketed week). The published Phantom 4 RTK layout puts the GPS week
in brackets and writes the three standard deviations untagged after the height, so a
real file was refused on its first line -- and a file with untagged deviations that did
get through reported them as 0.0, a perfect fix nobody recorded. The solution flag
table also called DJI's 16 (single) and 34 (float) "fixed".
"""

from __future__ import annotations

import pytest

from core.rtk import RtkError, read_camera_events
from test_rtk import NOON, WEEK, write_mrk, write_rinex


def dji_line(sequence: int, seconds: float, week: int = WEEK, flag: int = 50) -> str:
    """One line in the published Phantom 4 RTK layout, tab separated."""
    return "\t".join([
        str(sequence), f"{seconds:.6f}", f"[{week}]",
        "    -2,N", "    26,E", "   183,V",
        "22.64574539,Lat", "113.98244453,Lon", "-3.006,Ellh",
        "0.012595, 0.011226, 0.028003", f"{flag},Q",
    ])


def write_dji(path, count: int = 4, flag: int = 50):
    path.write_text("\n".join(dji_line(i + 1, NOON + 2 * i, flag=flag) for i in range(count)) + "\n",
                    encoding="utf-8")
    return path


class TestDjiLayout:
    def test_a_bracketed_week_and_untagged_deviations_are_read(self, tmp_path):
        event = read_camera_events(write_dji(tmp_path / "DJI_0001_Timestamp.MRK"))[0]
        assert event.gps_week == WEEK
        assert event.gps_seconds == pytest.approx(NOON)
        assert event.std_north_m == pytest.approx(0.012595)
        assert event.std_east_m == pytest.approx(0.011226)
        assert event.std_vertical_m == pytest.approx(0.028003)
        assert event.north_correction_m == pytest.approx(-0.002)
        assert event.east_correction_m == pytest.approx(0.026)
        assert event.vertical_correction_m == pytest.approx(0.183)
        assert event.latitude_deg == pytest.approx(22.64574539)
        assert event.longitude_deg == pytest.approx(113.98244453)
        assert event.height_m == pytest.approx(-3.006)
        assert event.solution == "fixed"

    @pytest.mark.parametrize("flag,solution", [(50, "fixed"), (34, "float"),
                                               (16, "single"), (0, "none"), (7, "unknown(7)")])
    def test_the_solution_flag_means_what_dji_says(self, tmp_path, flag, solution):
        event = read_camera_events(write_dji(tmp_path / "f.MRK", flag=flag))[0]
        assert event.solution == solution

    def test_a_line_without_deviations_is_refused_not_read_as_zero(self, tmp_path):
        path = tmp_path / "bare.MRK"
        path.write_text("\t".join(["1", f"{NOON}", f"[{WEEK}]", "22.6,Lat", "113.9,Lon",
                                   "-3.0,Ellh", "50,Q"]) + "\n", encoding="utf-8")
        with pytest.raises(RtkError, match="no standard deviations"):
            read_camera_events(path)

    def test_the_tagged_layout_still_reads(self, tmp_path):
        event = read_camera_events(write_mrk(tmp_path / "tagged.MRK"))[0]
        assert event.std_north_m == pytest.approx(0.012)


class TestThroughTheApi:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("ppk", root_dir=str(tmp_path / "project"))
        return Api(session)

    def test_a_dji_file_is_checked_and_its_deviations_reported(self, api, tmp_path):
        result = api.check_ppk_inputs(str(write_dji(tmp_path / "DJI_0001_Timestamp.MRK")),
                                      str(write_rinex(tmp_path / "base.26o")))
        assert result["ok"] is True, result
        assert result["usable_for_ppk"] is True
        # hypot(0.012595, 0.011226). It used to be 0.0.
        assert result["median_horizontal_std_m"] == pytest.approx(0.0169, abs=1e-4)

    def test_a_float_dji_flight_makes_no_rtk_claim(self, api, tmp_path):
        result = api.check_ppk_inputs(str(write_dji(tmp_path / "f.MRK", flag=34)),
                                      str(write_rinex(tmp_path / "base.26o")))
        assert result["fixed_fraction"] == 0.0
        assert "does not support an RTK or PPK accuracy claim" in result["accuracy_statement"]


def test_check_ppk_in_the_cockpit_reports_what_the_check_found(tmp_path):
    """The button read keys the check never returns and said "check out" regardless."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore
    from cockpit_actions_in_a_browser import run_actions
    from core.rtk import gps_to_utc

    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("ppk", root_dir=str(tmp_path / "project"))
    api = Api(session)
    events = write_dji(tmp_path / "DJI_0001_Timestamp.MRK", count=6)
    good = write_rinex(tmp_path / "good.26o")
    short = write_rinex(tmp_path / "short.26o", first=gps_to_utc(WEEK, NOON + 5),
                        last=gps_to_utc(WEEK, NOON + 600))
    results = run_actions(api, tmp_path, [
        {"name": "good", "action": "Check PPK"},
        {"name": "short", "action": "Check PPK"},
    ], picks=[str(events), str(good), str(events), str(short)])

    assert "[ok] PPK inputs check out: 6 of 6 events inside the base window" in results["good"]
    assert "Every camera event was recorded with a fixed solution" in results["good"]
    assert "PPK inputs are not usable (3 of 6 events" in results["short"], results["short"]
    assert "outside the base station's observation window" in results["short"]
