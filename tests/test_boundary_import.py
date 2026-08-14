"""Importing an area of interest from the files surveyors actually hand over.

The failure that matters here is not a crash but a plausible wrong answer: swapped
axes put the survey somewhere real and wrong, and nothing downstream would object. So
these tests check the landing position, not merely that a list of numbers came back.
"""

from __future__ import annotations

import zipfile

import pytest

from mission.boundary_import import (
    BoundaryImportError,
    describe_boundary,
    read_boundary,
    read_csv,
    read_geojson,
    read_gpx,
    read_kml,
    read_kmz,
)

# A small block in Cleveland, Ohio: negative longitude, positive latitude, so a swap
# would be both obvious and impossible in valid WGS84.
CORNERS = [(-81.7510, 41.3035), (-81.7490, 41.3035),
           (-81.7490, 41.3050), (-81.7510, 41.3050)]

KML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><Polygon>
<outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs>
</Polygon></Placemark></Document></kml>"""


def kml_text(corners=CORNERS) -> str:
    coords = " ".join(f"{lon},{lat},0" for lon, lat in corners)
    return KML_TEMPLATE.format(coords=coords)


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestKML:
    def test_a_kml_polygon_is_read_in_lon_lat_order(self, tmp_path):
        """KML writes lon,lat; reading it as lat,lon would move the site to Somalia."""
        points = read_kml(write(tmp_path, "aoi.kml", kml_text()))
        assert len(points) == 4
        assert points[0][0] == pytest.approx(-81.7510)
        assert points[0][1] == pytest.approx(41.3035)

    def test_the_repeated_closing_point_is_not_kept_as_a_corner(self, tmp_path):
        closed = [*CORNERS, CORNERS[0]]
        assert len(read_kml(write(tmp_path, "closed.kml", kml_text(closed)))) == 4

    def test_a_linestring_boundary_is_accepted(self, tmp_path):
        coords = " ".join(f"{lon},{lat}" for lon, lat in CORNERS)
        text = ('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                f"<Placemark><LineString><coordinates>{coords}</coordinates>"
                "</LineString></Placemark></kml>")
        assert len(read_kml(write(tmp_path, "line.kml", text))) == 4

    def test_malformed_kml_is_refused_with_a_reason(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="not valid KML"):
            read_kml(write(tmp_path, "bad.kml", "<kml><unclosed>"))

    def test_kml_without_coordinates_is_refused(self, tmp_path):
        text = '<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document/></kml>'
        with pytest.raises(BoundaryImportError, match="No <coordinates>"):
            read_kml(write(tmp_path, "empty.kml", text))


class TestKMZ:
    def test_a_kmz_is_unzipped_and_read(self, tmp_path):
        path = tmp_path / "aoi.kmz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("doc.kml", kml_text())
        points = read_kmz(path)
        assert len(points) == 4
        assert points[0][1] == pytest.approx(41.3035)

    def test_a_kmz_with_an_oddly_named_kml_still_works(self, tmp_path):
        path = tmp_path / "odd.kmz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("files/boundary.kml", kml_text())
        assert len(read_kmz(path)) == 4

    def test_a_kmz_containing_no_kml_is_refused(self, tmp_path):
        path = tmp_path / "nokml.kmz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("readme.txt", "nothing here")
        with pytest.raises(BoundaryImportError, match="no .kml file"):
            read_kmz(path)

    def test_a_file_that_is_not_a_zip_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="not a readable KMZ"):
            read_kmz(write(tmp_path, "fake.kmz", "definitely not a zip"))


class TestGeoJSON:
    def test_a_polygon_feature_is_read(self, tmp_path):
        ring = [[lon, lat] for lon, lat in CORNERS]
        text = ('{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
                '"geometry":{"type":"Polygon","coordinates":[' + str(ring).replace("'", "") + "]}}]}")
        points = read_geojson(write(tmp_path, "aoi.geojson", text))
        assert len(points) == 4
        assert points[0][0] == pytest.approx(-81.7510)

    def test_a_bare_geometry_without_a_feature_wrapper_is_read(self, tmp_path):
        ring = str([[lon, lat] for lon, lat in CORNERS]).replace("'", "")
        text = '{"type":"Polygon","coordinates":[' + ring + "]}"
        assert len(read_geojson(write(tmp_path, "bare.geojson", text))) == 4

    def test_invalid_json_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="not valid JSON"):
            read_geojson(write(tmp_path, "bad.geojson", "{not json"))

    def test_geojson_without_geometry_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="No polygon or line"):
            read_geojson(write(tmp_path, "points.geojson",
                               '{"type":"FeatureCollection","features":[]}'))


class TestGPX:
    def test_a_walked_track_is_read_in_lat_lon_attribute_order(self, tmp_path):
        """GPX is the reverse of KML: lat and lon are attributes, and order matters."""
        points = "".join(f'<trkpt lat="{lat}" lon="{lon}"/>' for lon, lat in CORNERS)
        text = ('<?xml version="1.0"?><gpx version="1.1"><trk><trkseg>'
                f"{points}</trkseg></trk></gpx>")
        result = read_gpx(write(tmp_path, "walk.gpx", text))
        assert result[0][0] == pytest.approx(-81.7510)  # longitude first out
        assert result[0][1] == pytest.approx(41.3035)

    def test_standalone_waypoints_are_used_when_there_is_no_track(self, tmp_path):
        points = "".join(f'<wpt lat="{lat}" lon="{lon}"/>' for lon, lat in CORNERS)
        text = f'<?xml version="1.0"?><gpx version="1.1">{points}</gpx>'
        assert len(read_gpx(write(tmp_path, "marks.gpx", text))) == 4

    def test_an_empty_gpx_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="No track, route or waypoint"):
            read_gpx(write(tmp_path, "empty.gpx", '<?xml version="1.0"?><gpx/>'))


class TestCSV:
    def test_a_headed_csv_is_read_by_column_name(self, tmp_path):
        rows = "\n".join(f"{lat},{lon}" for lon, lat in CORNERS)
        points = read_csv(write(tmp_path, "corners.csv", f"latitude,longitude\n{rows}"))
        assert points[0][0] == pytest.approx(-81.7510)
        assert points[0][1] == pytest.approx(41.3035)

    def test_column_order_follows_the_header_not_the_position(self, tmp_path):
        """A lon-first file must not be read as lat-first just because it came first."""
        rows = "\n".join(f"{lon},{lat}" for lon, lat in CORNERS)
        points = read_csv(write(tmp_path, "lonfirst.csv", f"lon,lat\n{rows}"))
        assert points[0][0] == pytest.approx(-81.7510)
        assert points[0][1] == pytest.approx(41.3035)

    def test_a_csv_without_a_header_is_refused_rather_than_guessed(self, tmp_path):
        """28,77 is Delhi one way round and the Somali coast the other."""
        with pytest.raises(BoundaryImportError, match="no latitude/longitude header"):
            read_csv(write(tmp_path, "bare.csv", "28.6,77.2\n28.7,77.3\n28.8,77.4"))

    def test_extra_columns_are_ignored(self, tmp_path):
        text = "name,latitude,longitude,notes\n" + "\n".join(
            f"corner{i},{lat},{lon},fenced" for i, (lon, lat) in enumerate(CORNERS))
        assert len(read_csv(write(tmp_path, "rich.csv", text))) == 4

    def test_a_header_with_no_numeric_rows_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="no rows with numeric"):
            read_csv(write(tmp_path, "words.csv", "lat,lon\nnorth,west\nsouth,east"))

    def test_an_empty_file_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="is empty"):
            read_csv(write(tmp_path, "empty.csv", ""))


class TestValidation:
    def test_an_out_of_range_coordinate_is_refused(self, tmp_path):
        """Latitude 141 is the signature of swapped axes, not a real place."""
        text = "lat,lon\n141.3,-81.7\n141.4,-81.8\n141.5,-81.9"
        with pytest.raises(BoundaryImportError, match="out-of-range"):
            read_csv(write(tmp_path, "swapped.csv", text))

    def test_a_boundary_with_too_few_corners_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="at least 3"):
            read_csv(write(tmp_path, "two.csv", "lat,lon\n41.30,-81.75\n41.31,-81.76"))

    def test_duplicate_consecutive_points_do_not_count_as_corners(self, tmp_path):
        text = "lat,lon\n41.30,-81.75\n41.30,-81.75\n41.31,-81.76"
        with pytest.raises(BoundaryImportError, match="at least 3"):
            read_csv(write(tmp_path, "dupe.csv", text))


class TestDispatch:
    def test_the_format_is_chosen_by_extension(self, tmp_path):
        assert len(read_boundary(write(tmp_path, "aoi.kml", kml_text()))) == 4

    def test_an_unsupported_extension_is_refused_and_lists_what_works(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="Supported:"):
            read_boundary(write(tmp_path, "aoi.dxf", "irrelevant"))

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(BoundaryImportError, match="does not exist"):
            read_boundary(tmp_path / "nope.kml")


class TestDescribe:
    def test_the_summary_locates_and_sizes_the_boundary(self, tmp_path):
        points = read_kml(write(tmp_path, "aoi.kml", kml_text()))
        summary = describe_boundary(points)

        assert summary["point_count"] == 4
        # Cleveland is UTM zone 17 north.
        assert summary["utm_epsg"] == 32617
        assert summary["centroid"][0] == pytest.approx(-81.75, abs=0.01)
        # ~167 m by ~167 m, so a couple of hectares.
        assert 2.0 < summary["area_hectares"] < 4.0

    def test_an_imported_boundary_can_be_planned_over(self, tmp_path):
        """The point of importing is to plan, so prove the handover works."""
        from mission.planner import MissionPlanner

        points = read_kml(write(tmp_path, "aoi.kml", kml_text()))
        plan = MissionPlanner().generate(mode="grid", polygon_lonlat=points, altitude_m=60.0)
        assert len(plan.waypoints) > 0
