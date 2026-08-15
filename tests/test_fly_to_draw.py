"""Defining a survey boundary by flying to its corners.

The value of this is that the operator is standing next to a boundary the basemap gets
wrong. The danger is that every failure mode produces a boundary that looks fine: a
corner marked without a fix has coordinates, a double press produces an edge, and a
crossed-over outline still returns an area. So the tests below are mostly refusals, each
checking that the reason reaches the operator while they are still on site and can
re-mark.
"""

from __future__ import annotations

import pytest

from mission.fly_to_draw import (
    BoundaryMark,
    BoundaryRefused,
    boundary_from_marks,
    mark_from_telemetry,
)

# A square roughly 90 m on a side near Bengaluru.
CORNERS = [(77.5940, 12.9700), (77.5948, 12.9700), (77.5948, 12.9708), (77.5940, 12.9708)]


def fixed(lon, lat, **overrides):
    payload = {"connected": True, "longitude": lon, "latitude": lat, "altitude_m": 30.0,
               "fix_type": 3, "satellites": 14, "hdop": 0.8}
    payload.update(overrides)
    return payload


def marks(corners=CORNERS, **overrides):
    return [mark_from_telemetry(fixed(lon, lat, **overrides)) for lon, lat in corners]


class TestMarking:
    def test_a_good_fix_records_the_corner_with_its_quality(self):
        mark = mark_from_telemetry(fixed(77.5940, 12.9700))

        assert mark.longitude == pytest.approx(77.5940)
        assert mark.fix_type == 3
        assert mark.satellites == 14
        assert mark.marked_utc.endswith("+00:00")

    def test_a_corner_without_a_3d_fix_is_refused(self):
        """It would look exactly like a good corner in the finished boundary."""
        with pytest.raises(BoundaryRefused, match="not a 3D fix"):
            mark_from_telemetry(fixed(77.5940, 12.9700, fix_type=1))

    def test_marking_with_no_vehicle_connected_is_refused(self):
        with pytest.raises(BoundaryRefused, match="No vehicle is connected"):
            mark_from_telemetry({"connected": False, "reason": "Link lost."})

    def test_telemetry_without_a_position_is_refused(self):
        with pytest.raises(BoundaryRefused, match="no coordinates"):
            mark_from_telemetry({"connected": True, "fix_type": 3})

    def test_a_position_off_the_planet_is_refused(self):
        with pytest.raises(BoundaryRefused, match="not on Earth"):
            mark_from_telemetry(fixed(255.0, 12.97))

    def test_alternative_telemetry_field_names_are_accepted(self):
        """Drivers differ; a corner should not be lost to a field name."""
        mark = mark_from_telemetry({
            "connected": True, "lon": 77.594, "lat": 12.97, "alt_m": 25.0,
            "gps_fix_type": 4, "satellites_visible": 18, "eph": 0.6,
        })
        assert mark.fix_type == 4
        assert mark.satellites == 18


class TestBuildingTheBoundary:
    def test_four_flown_corners_become_an_area(self):
        boundary = boundary_from_marks(marks())

        assert boundary["corner_count"] == 4
        assert boundary["source"] == "flown_marks"
        # ~0.087 deg east by 0.0008 north at this latitude: roughly 0.7 ha.
        assert 0.5 < boundary["area_hectares"] < 1.0

    def test_the_perimeter_closes_the_ring(self):
        boundary = boundary_from_marks(marks())
        # Four sides of a rough square, not three.
        assert boundary["perimeter_m"] > 300.0

    def test_the_weakest_corner_is_reported_not_the_average(self):
        rough = marks()
        rough[2] = BoundaryMark(longitude=77.5948, latitude=12.9708, fix_type=3,
                                satellites=5, hdop=2.6)
        boundary = boundary_from_marks(rough)

        assert "5 satellites" in " ".join(boundary["limits"])
        assert any("HDOP above 2" in limit for limit in boundary["limits"])

    def test_it_never_claims_to_be_a_surveyed_boundary(self):
        boundary = boundary_from_marks(marks())
        assert any("not a surveyed boundary" in limit for limit in boundary["limits"])


class TestRefusals:
    def test_two_corners_do_not_make_an_area(self):
        with pytest.raises(BoundaryRefused, match="no inside to survey"):
            boundary_from_marks(marks(CORNERS[:2]))

    def test_a_double_press_is_refused_rather_than_becoming_an_edge(self):
        doubled = list(CORNERS[:2]) + [(77.59480001, 12.97000001)] + [CORNERS[3]]
        with pytest.raises(BoundaryRefused, match="double press"):
            boundary_from_marks(marks(doubled))

    def test_a_crossed_outline_is_refused_even_though_it_has_an_area(self):
        """A bow-tie returns a number; the number means nothing."""
        bowtie = [CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]]
        with pytest.raises(BoundaryRefused, match="no inside"):
            boundary_from_marks(marks(bowtie))


class TestThroughTheApi:
    @pytest.fixture
    def api(self, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
        session.create_project("fly to draw", root_dir=str(tmp_path / "project"))
        return Api(session)

    def fly(self, api, monkeypatch, corners=CORNERS, **overrides):
        sequence = iter(corners)

        def telemetry():
            lon, lat = next(sequence)
            return fixed(lon, lat, **overrides)

        monkeypatch.setattr(api._session, "telemetry", telemetry)
        return [api.mark_boundary_corner() for _ in corners]

    def test_flying_the_corners_produces_the_aoi(self, api, monkeypatch):
        responses = self.fly(api, monkeypatch)
        assert all(r["ok"] for r in responses)

        result = api.boundary_from_marks()
        assert result["ok"] is True
        assert result["corner_count"] == 4
        assert len(api._session.aoi_polygon) == 4

    def test_the_boundary_can_be_reviewed_without_replacing_the_aoi(self, api, monkeypatch):
        self.fly(api, monkeypatch)
        result = api.boundary_from_marks(apply_as_aoi=False)

        assert result["ok"] is True
        assert api._session.aoi_polygon == []

    def test_marking_without_a_fix_fails_with_the_reason(self, api, monkeypatch):
        monkeypatch.setattr(api._session, "telemetry",
                            lambda: fixed(77.594, 12.97, fix_type=0))
        result = api.mark_boundary_corner()

        assert result["ok"] is False
        assert "not a 3D fix" in result["error"]

    def test_too_few_corners_is_refused_at_the_api(self, api, monkeypatch):
        self.fly(api, monkeypatch, corners=CORNERS[:2])
        result = api.boundary_from_marks()

        assert result["ok"] is False
        assert "at least three" in result["error"]

    def test_the_marks_can_be_cleared_and_re_flown(self, api, monkeypatch):
        self.fly(api, monkeypatch)
        assert api.clear_boundary_marks()["corner_count"] == 0

        self.fly(api, monkeypatch)
        assert api.boundary_from_marks()["corner_count"] == 4
