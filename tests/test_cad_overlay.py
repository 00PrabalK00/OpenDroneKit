"""Putting the drawing on top of what was actually built.

An overlay answers "does the site match the design". That makes a badly placed overlay
worse than none at all: a plan two metres east of where it belongs reads as a construction
error, and someone goes and measures a wall that is exactly where it should be.

So most of these tests are about refusing to place something in a position that is not
known to be right, and about telling the operator when geometry was not understood rather
than quietly drawing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.cad_overlay import (
    CadDrawing,
    OverlayRefused,
    alignment_report,
    place_raster,
    read_dxf,
    reproject,
)

# Group code / value pairs on alternating lines. A line, a closed polyline, a circle,
# and a block reference this reader does not flatten.
DXF = """0
SECTION
2
ENTITIES
0
LINE
10
0.0
20
0.0
11
10.0
21
0.0
0
LWPOLYLINE
70
1
10
0.0
20
0.0
10
10.0
20
0.0
10
10.0
20
5.0
0
CIRCLE
10
5.0
20
2.5
40
1.0
0
INSERT
2
SOMEBLOCK
0
ENDSEC
0
EOF
"""


@pytest.fixture
def drawing_file(tmp_path) -> Path:
    path = tmp_path / "site.dxf"
    path.write_text(DXF, encoding="utf-8")
    return path


@pytest.fixture
def drawing(drawing_file) -> CadDrawing:
    return read_dxf(drawing_file)


class TestReadingADrawing:
    def test_it_reads_the_entities_a_site_plan_uses(self, drawing) -> None:
        assert drawing.entity_counts == {"LINE": 1, "LWPOLYLINE": 1, "CIRCLE": 1}

    def test_a_closed_polyline_is_closed(self, drawing) -> None:
        polyline = drawing.polylines[1]
        assert polyline[0] == polyline[-1]

    def test_a_circle_becomes_a_polyline(self, drawing) -> None:
        assert len(drawing.polylines[2]) > 8

    def test_bounds_cover_every_vertex(self, drawing) -> None:
        assert drawing.bounds() == pytest.approx((0.0, 0.0, 10.0, 5.0))

    def test_geometry_it_cannot_flatten_is_counted_not_dropped(self, drawing) -> None:
        """A plan whose walls live inside BLOCK references would otherwise overlay as an
        empty drawing, which on top of an orthomosaic reads as "the design matches
        nothing" rather than "this reader did not understand the file"."""
        assert drawing.skipped_entities == {"INSERT": 1}

    def test_a_drawing_with_nothing_readable_says_what_it_found(self, tmp_path) -> None:
        path = tmp_path / "blocks.dxf"
        path.write_text("0\nSECTION\n0\nINSERT\n2\nWALLS\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
        with pytest.raises(OverlayRefused, match="INSERT"):
            read_dxf(path)

    def test_a_missing_file_is_refused(self, tmp_path) -> None:
        with pytest.raises(OverlayRefused):
            read_dxf(tmp_path / "nope.dxf")

    def test_a_malformed_tag_does_not_lose_the_walls(self, tmp_path) -> None:
        """Real drawings carry vendor extensions; one unreadable pair should not discard
        the geometry around it."""
        path = tmp_path / "messy.dxf"
        path.write_text(
            "not-a-code\ngarbage\n0\nLINE\n10\n0.0\n20\n0.0\n11\n5.0\n21\n5.0\n0\nEOF\n",
            encoding="utf-8")
        assert read_dxf(path).entity_counts["LINE"] == 1


class TestPuttingItInTheProjectsCoordinates:
    def test_reprojection_moves_the_geometry(self, drawing) -> None:
        moved = reproject(drawing, 32617, 4326)
        assert moved.bounds() != drawing.bounds()
        assert -180.0 <= moved.bounds()[0] <= 180.0

    def test_the_source_crs_is_recorded(self, drawing) -> None:
        assert reproject(drawing, 32617, 4326).source_epsg == 32617

    def test_the_same_crs_is_left_alone(self, drawing) -> None:
        same = reproject(drawing, 32617, 32617)
        assert same.bounds() == pytest.approx(drawing.bounds())

    def test_entity_counts_survive_reprojection(self, drawing) -> None:
        """The counts are how an operator knows what was read; losing them in transit
        would hide an unflattened block."""
        moved = reproject(drawing, 32617, 4326)
        assert moved.entity_counts == drawing.entity_counts
        assert moved.skipped_entities == drawing.skipped_entities


class TestPlacingAPlainImage:
    @pytest.fixture
    def image(self, tmp_path) -> Path:
        path = tmp_path / "plan.png"
        path.write_bytes(b"not really a png")
        return path

    def test_a_sane_box_is_accepted(self, image) -> None:
        placed = place_raster(image, 0.0, 0.0, 10.0, 10.0, epsg=32617)
        assert placed.bounds() == (0.0, 0.0, 10.0, 10.0)

    def test_swapped_east_and_west_is_refused(self, image) -> None:
        """It would render mirrored, and an operator comparing a mirrored plan against
        the orthomosaic would be reading a placement mistake as a build error."""
        with pytest.raises(OverlayRefused, match="mirror"):
            place_raster(image, 10.0, 0.0, 0.0, 10.0, epsg=32617)

    def test_swapped_north_and_south_is_refused(self, image) -> None:
        with pytest.raises(OverlayRefused, match="flip"):
            place_raster(image, 0.0, 10.0, 10.0, 0.0, epsg=32617)

    def test_a_zero_width_box_is_refused(self, image) -> None:
        with pytest.raises(OverlayRefused):
            place_raster(image, 5.0, 0.0, 5.0, 10.0, epsg=32617)

    @pytest.mark.parametrize("box", [
        (-200.0, 0.0, -190.0, 10.0),
        (0.0, -100.0, 10.0, -95.0),
        (170.0, 0.0, 200.0, 10.0),
    ])
    def test_impossible_lat_lon_is_refused(self, image, box) -> None:
        with pytest.raises(OverlayRefused):
            place_raster(image, *box, epsg=4326)

    def test_a_missing_image_is_refused(self, tmp_path) -> None:
        with pytest.raises(OverlayRefused):
            place_raster(tmp_path / "gone.png", 0.0, 0.0, 1.0, 1.0)


class TestSayingWhenItLandedInTheWrongPlace:
    def test_an_aligned_drawing_reports_no_warning(self, drawing) -> None:
        report = alignment_report(drawing, [0.0, 0.0, 10.0, 5.0])
        assert report["overlaps"] is True
        assert report["centre_offset_m"] == pytest.approx(0.0, abs=1e-9)
        assert "warning" not in report

    def test_a_drawing_nowhere_near_the_survey_is_called_out(self, drawing) -> None:
        """The common failure: a plan in local site coordinates, or a DXF whose origin is
        a corner of the sheet. Without this the operator concludes the building moved."""
        report = alignment_report(drawing, [5000.0, 5000.0, 5010.0, 5005.0])
        assert report["overlaps"] is False
        assert report["centre_offset_m"] > 1000.0
        assert "CRS" in report["warning"]

    def test_a_small_offset_still_overlaps(self, drawing) -> None:
        """Two metres out is the case the overlay EXISTS to show, so it must not be
        reported as a placement failure."""
        # Drawing centre (5, 2.5); target centre (7, 2.5). Two metres, which is exactly
        # the size of discrepancy an overlay is put on a site to reveal.
        report = alignment_report(drawing, [2.0, 0.0, 12.0, 5.0])
        assert report["overlaps"] is True
        assert report["centre_offset_m"] == pytest.approx(2.0, abs=1e-6)

    def test_an_empty_drawing_cannot_be_aligned(self) -> None:
        with pytest.raises(OverlayRefused):
            alignment_report(CadDrawing(), [0.0, 0.0, 1.0, 1.0])
