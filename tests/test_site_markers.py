"""The things on a site that are not part of the flight.

A mission plan says where the aircraft goes and nothing about where the pilot stands,
where the van parks, or which corner of the yard has the overhead line. Those live in
somebody's head on the day and in nobody's head six months later when a different crew
reflies it.

The tests worth having are about a hazard that cannot be missed: a fixed kind list so it
cannot be typed in a way that filters miss, a radius that counts toward clearance, and
markers that inform the crew without ever silently constraining the aircraft.
"""

from __future__ import annotations

import pytest

from core.site_markers import (
    MARKER_KINDS,
    MarkerRefused,
    MarkerStore,
    SiteMarker,
    hazards_near,
    validate,
)

SITE = [-81.752, 41.304]


def marker(name: str = "crane", kind: str = "hazard", **kwargs) -> SiteMarker:
    base = {"name": name, "kind": kind, "points": [list(SITE)]}
    base.update(kwargs)
    return SiteMarker(**base)  # type: ignore[arg-type]


class TestAMarkerMustMeanSomething:
    def test_a_marker_needs_a_name(self) -> None:
        with pytest.raises(MarkerRefused, match="dot on a map"):
            validate(marker(name="  "))

    def test_an_invented_kind_is_refused_and_the_real_ones_named(self) -> None:
        """Free text is how a hazard gets typed as "danger" and missed by every filter."""
        with pytest.raises(MarkerRefused) as exc:
            validate(marker(kind="danger"))
        assert "hazard" in str(exc.value)

    @pytest.mark.parametrize("kind", sorted(MARKER_KINDS))
    def test_every_documented_kind_is_accepted(self, kind) -> None:
        validate(marker(kind=kind))

    def test_a_position_off_the_earth_is_refused(self) -> None:
        with pytest.raises(MarkerRefused, match="not on Earth"):
            validate(marker(points=[[400.0, 41.0]]))

    def test_two_points_is_neither_a_place_nor_an_area(self) -> None:
        with pytest.raises(MarkerRefused, match="neither"):
            validate(marker(points=[SITE, [-81.751, 41.305]]))

    def test_three_points_make_an_area(self) -> None:
        area = marker(kind="restricted",
                      points=[SITE, [-81.751, 41.304], [-81.751, 41.305]])
        validate(area)
        assert area.is_area is True

    def test_a_radius_on_an_area_is_refused(self) -> None:
        """It already encloses ground; a radius as well is two answers to one question."""
        with pytest.raises(MarkerRefused, match="already encloses"):
            validate(marker(points=[SITE, [-81.751, 41.304], [-81.751, 41.305]],
                            radius_m=20.0))

    def test_a_negative_radius_is_refused(self) -> None:
        with pytest.raises(MarkerRefused):
            validate(marker(radius_m=-5.0))


class TestStoringThem:
    def test_a_marker_round_trips(self, tmp_path) -> None:
        store = MarkerStore(tmp_path)
        store.add(marker("overhead line", "hazard", radius_m=25.0,
                         note="11 kV, runs east-west"))
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].radius_m == pytest.approx(25.0)
        assert "11 kV" in loaded[0].note

    def test_two_markers_cannot_share_a_name(self, tmp_path) -> None:
        """Names are how a marker is referred to on the radio, so two hazards called "the
        crane" is a briefing problem rather than a storage one."""
        store = MarkerStore(tmp_path)
        store.add(marker("the crane"))
        with pytest.raises(MarkerRefused, match="already"):
            store.add(marker("The Crane"))

    def test_an_invalid_marker_never_reaches_disk(self, tmp_path) -> None:
        store = MarkerStore(tmp_path)
        with pytest.raises(MarkerRefused):
            store.add(marker(kind="danger"))
        assert store.load() == []

    def test_removing_an_unknown_id_says_so(self, tmp_path) -> None:
        with pytest.raises(MarkerRefused):
            MarkerStore(tmp_path).remove("nope")

    def test_a_corrupt_file_reads_as_none(self, tmp_path) -> None:
        store = MarkerStore(tmp_path)
        store.path.write_text("{ not json", encoding="utf-8")
        assert store.load() == []

    def test_geojson_carries_points_and_areas(self, tmp_path) -> None:
        """So the briefing can be handed to somebody who does not run this software."""
        store = MarkerStore(tmp_path)
        store.add(marker("crane", "hazard"))
        store.add(marker("no-go", "restricted",
                         points=[SITE, [-81.751, 41.304], [-81.751, 41.305]]))
        kinds = {f["geometry"]["type"] for f in store.to_geojson()["features"]}
        assert kinds == {"Point", "Polygon"}

    def test_an_area_polygon_is_closed(self, tmp_path) -> None:
        store = MarkerStore(tmp_path)
        store.add(marker("no-go", "restricted",
                         points=[SITE, [-81.751, 41.304], [-81.751, 41.305]]))
        ring = store.to_geojson()["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]


class TestHazardsNearTheFlight:
    def test_a_hazard_on_the_line_is_reported(self) -> None:
        hazard = marker("crane", "hazard")
        found = hazards_near([hazard], [SITE], clearance_m=30.0)
        assert len(found) == 1
        assert found[0]["name"] == "crane"

    def test_a_hazard_far_away_is_not(self) -> None:
        hazard = marker("crane", "hazard", points=[[-81.90, 41.40]])
        assert hazards_near([hazard], [SITE], clearance_m=30.0) == []

    def test_the_hazards_own_radius_counts(self) -> None:
        """A 40 m crane 60 m from the line is closer than the coordinates suggest, and
        that difference is the whole reason a hazard is more than a dot."""
        far = [-81.7513, 41.304]  # roughly 58 m east
        without = hazards_near([marker("crane", "hazard", points=[far])], [SITE], 30.0)
        with_radius = hazards_near(
            [marker("crane", "hazard", points=[far], radius_m=40.0)], [SITE], 30.0)
        assert without == []
        assert len(with_radius) == 1

    def test_only_hazards_are_checked(self) -> None:
        """A parking spot next to the flight line is not a warning; reporting it teaches
        people to ignore the warnings that matter."""
        assert hazards_near([marker("van", "access")], [SITE], 30.0) == []

    def test_the_closest_comes_first(self) -> None:
        near = marker("near", "hazard", points=[SITE])
        mid = marker("mid", "hazard", points=[[-81.7515, 41.304]])
        found = hazards_near([mid, near], [SITE], clearance_m=1000.0)
        assert [f["name"] for f in found] == ["near", "mid"]

    def test_it_reports_rather_than_constrains(self) -> None:
        """Markers are information. A planner that silently rerouted around a note
        somebody typed would be worse than one that says nothing, because the crew would
        stop reading the notes."""
        import inspect

        from core import site_markers

        source = inspect.getsource(site_markers)
        assert "does NOT stop a flight" in source
