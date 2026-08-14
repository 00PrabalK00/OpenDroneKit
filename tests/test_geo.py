"""The geospatial foundation: similarity solve, CRS selection, raster round trip.

`solve_similarity_umeyama` is what converts arbitrary structure-from-motion units
into metres in a real coordinate system. If it is wrong, every downstream area,
length, and volume is wrong by the same factor while still looking plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import geo


class TestUmeyamaSimilarity:
    def test_recovers_a_known_transform(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(40, 3)) * 10.0

        angle = np.deg2rad(37.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        scale = 2.75
        translation = np.array([120.0, -45.0, 8.0])
        target = scale * (rotation @ source.T).T + translation

        found_scale, found_rotation, found_translation = geo.solve_similarity_umeyama(source, target)

        assert found_scale == pytest.approx(scale, rel=1e-9)
        assert found_rotation == pytest.approx(rotation, abs=1e-9)
        assert found_translation == pytest.approx(translation, abs=1e-7)

    def test_recovered_transform_maps_points_back(self):
        rng = np.random.default_rng(11)
        source = rng.normal(size=(25, 3))
        angle = np.deg2rad(-15.0)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        target = 0.4 * (rotation @ source.T).T + np.array([1.0, 2.0, 3.0])

        scale, found_rotation, translation = geo.solve_similarity_umeyama(source, target)
        mapped = scale * (found_rotation @ source.T).T + translation
        assert mapped == pytest.approx(target, abs=1e-9)

    def test_rotation_is_proper(self):
        """A reflection would mirror the survey; det must be +1, never -1."""
        rng = np.random.default_rng(3)
        source = rng.normal(size=(30, 3))
        target = rng.normal(size=(30, 3))
        _, rotation, _ = geo.solve_similarity_umeyama(source, target)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


class TestUtmZoneSelection:
    @pytest.mark.parametrize(
        "lat,lon,expected",
        [
            (41.3042, -81.7505, 32617),   # Aukerman, Ohio -> UTM 17N
            (51.5074, -0.1278, 32630),    # London -> 30N
            (-33.8688, 151.2093, 32756),  # Sydney -> 56S
            (0.0, 0.0, 32631),            # null island -> 31N
        ],
    )
    def test_zone_matches_known_locations(self, lat, lon, expected):
        assert geo.auto_utm_epsg(lat, lon) == expected

    def test_southern_hemisphere_uses_the_327xx_band(self):
        assert str(geo.auto_utm_epsg(-20.0, 30.0)).startswith("327")
        assert str(geo.auto_utm_epsg(20.0, 30.0)).startswith("326")

    def test_norway_exception(self):
        """Zone 32 is widened over southern Norway; 59N 6E falls in it, not zone 31."""
        assert geo.auto_utm_epsg(59.0, 6.0) == 32632


class TestCoordinateTransforms:
    def test_wgs84_to_projected_round_trips(self):
        lon, lat = -81.7505, 41.3042
        epsg = geo.auto_utm_epsg(lat, lon)
        easting, northing = geo.wgs84_to_projected(lon, lat, epsg)
        back_lon, back_lat = geo.projected_to_wgs84(easting, northing, epsg)
        assert back_lon == pytest.approx(lon, abs=1e-9)
        assert back_lat == pytest.approx(lat, abs=1e-9)

    def test_projected_coordinates_are_metres(self):
        """One degree of latitude is about 111 km; the projection must reflect that."""
        epsg = geo.auto_utm_epsg(41.0, -81.75)
        _, north_a = geo.wgs84_to_projected(-81.75, 41.00, epsg)
        _, north_b = geo.wgs84_to_projected(-81.75, 41.01, epsg)
        assert abs(north_b - north_a) == pytest.approx(1110.0, rel=0.02)

    def test_haversine_matches_a_known_distance(self):
        """London to Paris is about 343 km great-circle.

        Note the signature is (lat, lon) while GeoJSON and most of this codebase
        order coordinates (lon, lat); passing them the wrong way round here returns
        403 km rather than failing, so the ordering is asserted explicitly.
        """
        metres = geo.haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
        assert metres == pytest.approx(343_000, rel=0.02)

    def test_haversine_is_symmetric(self):
        forward = geo.haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
        backward = geo.haversine_m(48.8566, 2.3522, 51.5074, -0.1278)
        assert forward == pytest.approx(backward)

    def test_haversine_of_a_point_with_itself_is_zero(self):
        assert geo.haversine_m(41.3042, -81.7505, 41.3042, -81.7505) == pytest.approx(0.0)


class TestGeoTiffRoundTrip:
    def test_written_raster_reads_back_with_its_georeferencing(self, tmp_path):
        array = np.arange(64, dtype=np.float32).reshape(8, 8)
        target = tmp_path / "raster.tif"
        geo.write_geotiff(
            target, array, epsg=32617, west=500000.0, north=4570000.0,
            pixel_size=0.5, cog=False,
        )
        data, meta = geo.read_geotiff(target)

        assert meta["epsg"] == 32617
        assert meta["width"] == 8 and meta["height"] == 8
        assert data[0] == pytest.approx(array)
        assert meta["transform"][0] == pytest.approx(0.5)
        # North-up rasters carry a negative y scale.
        assert meta["transform"][4] == pytest.approx(-0.5)

    def test_nodata_is_preserved(self, tmp_path):
        array = np.full((4, 4), 5.0, dtype=np.float32)
        target = tmp_path / "nodata.tif"
        # Offset from the origin so the transform is not the identity, which GDAL
        # may discard entirely rather than writing a geotransform.
        geo.write_geotiff(
            target, array, epsg=32617, west=500000.0, north=4570000.0,
            pixel_size=1.0, nodata=-9999.0, cog=False,
        )
        _, meta = geo.read_geotiff(target)
        assert meta["nodata"] == pytest.approx(-9999.0)


class TestVectorOutput:
    def test_polygon_ring_is_closed(self):
        feature = geo.polygon_feature([[0, 0], [1, 0], [1, 1]], {"id": "a"})
        ring = feature["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "GeoJSON polygons must close their ring"

    def test_geojson_declares_its_crs_and_features(self, tmp_path):
        import json

        target = tmp_path / "out.geojson"
        geo.write_geojson(
            target,
            [geo.point_feature(-81.75, 41.30, {"id": "p1"})],
            epsg=4326,
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 1

    def test_polygon_area_is_in_square_metres(self):
        """A 100 m square at the equator is 10,000 m2."""
        step = 100.0 / 111_320.0  # metres -> degrees of latitude
        ring = [[0.0, 0.0], [step, 0.0], [step, step], [0.0, step]]
        area = geo.polygon_area_m2(ring)
        assert area == pytest.approx(10_000.0, rel=0.02)
