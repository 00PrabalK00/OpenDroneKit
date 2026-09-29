"""The Api behaviour the cockpit fixes rest on, checked without a browser.

  * measure_geometry: what the map's Distance, Area and Measure tools report;
  * set_aoi: refuses a world-sized area (the audit's 2.7 billion ha AOI hung plan_mission
    at 6.4 GB), and clears on null rather than raising a TypeError;
  * create_project: a taken name is a sentence, not "IntegrityError: UNIQUE constraint";
  * switching project takes the last one's AOI, no-fly zones and plan with it;
  * the same file or folder registered twice is one layer and one dataset;
  * rest_endpoints: the Developers route list is the REST service's own;
  * rest_request: loopback GETs only.
"""

from __future__ import annotations

import pytest

from cockpit_harness import make_api, write_geotagged_jpeg
from core import geo

W, E = -81.7520, -81.7513
S, N = 41.3040, 41.3044


@pytest.fixture
def api(tmp_path):
    return make_api(tmp_path)


class TestMeasureGeometry:
    def test_distance_is_the_geodesic_path_length(self, api) -> None:
        path = [[W, S], [E, S], [E, N]]
        result = api.measure_geometry(path, "distance")
        expected = geo.haversine_m(S, W, S, E) + geo.haversine_m(S, E, N, E)
        assert result["ok"] and result["length_m"] == pytest.approx(expected, abs=1e-3)
        assert len(result["segments_m"]) == 2

    def test_area_and_perimeter(self, api) -> None:
        ring = [[W, S], [E, S], [E, N], [W, N]]
        result = api.measure_geometry(ring, "area")
        assert result["area_m2"] == pytest.approx(geo.polygon_area_m2(ring), rel=1e-6)
        assert result["perimeter_m"] > 0 and result["vertices"] == 4

    def test_too_few_points_and_bad_coordinates_are_refused(self, api) -> None:
        assert not api.measure_geometry([[W, S]], "distance")["ok"]
        assert not api.measure_geometry([[W, S], [E, S]], "area")["ok"]
        assert not api.measure_geometry([[W, 95.0], [E, S]], "distance")["ok"]
        assert not api.measure_geometry([[W, S], [E, S]], "volume")["ok"]


class TestAreaOfInterest:
    def test_a_world_sized_area_is_refused_before_planning(self, api) -> None:
        huge = [[-120.0, -40.0], [120.0, -40.0], [120.0, 60.0], [-120.0, 60.0]]
        result = api.set_aoi(huge)
        assert not result["ok"] and "Zoom in" in result["error"]
        assert api._session.aoi_polygon == []

    def test_null_clears(self, api) -> None:
        api.set_aoi([[W, S], [E, S], [E, N]])
        assert api.set_aoi(None)["ok"]
        assert api._session.aoi_polygon == []


class TestProjects:
    def test_a_taken_name_is_explained(self, api, tmp_path) -> None:
        result = api.create_project("harness site", str(tmp_path / "again"))
        assert not result["ok"]
        assert "already exists" in result["error"] and "IntegrityError" not in result["error"]

    def test_a_new_project_does_not_inherit_the_last_ones_site(self, api, tmp_path) -> None:
        api.set_aoi([[W, S], [E, S], [E, N]])
        api.set_no_fly_zones([[[W, S], [E, S], [E, N]]])
        api.plan_mission({})
        assert api.create_project("Elsewhere", str(tmp_path / "elsewhere"))["ok"]
        state = api.get_state()
        assert state["aoi"] == [] and state["no_fly"] == [] and not state["has_mission"]


class TestNoDuplicates:
    def test_the_same_layer_file_is_one_layer(self, api, tmp_path) -> None:
        layer = tmp_path / "zone.geojson"
        geo.write_geojson(layer, [geo.point_feature(W, S, {})])
        first = api.add_layer_from_file(str(layer))["layer"]
        second = api.add_layer_from_file(str(layer))["layer"]
        assert first["id"] == second["id"]
        assert len(api.list_layers()["layers"]) == 1

    def test_reimporting_a_folder_is_one_dataset(self, api, tmp_path) -> None:
        folder = tmp_path / "flight"
        write_geotagged_jpeg(folder / "A.JPG", W, S)
        write_geotagged_jpeg(folder / "B.JPG", E, N)
        api.import_dataset(str(folder))
        api.import_dataset(str(folder))
        assert len(api.list_datasets()["datasets"]) == 1
        photo_layers = [l for l in api.list_layers()["layers"] if "photo points" in l["name"]]
        assert len(photo_layers) == 1
        # A folder whose contents changed is genuinely a different record.
        write_geotagged_jpeg(folder / "C.JPG", W, N)
        api.import_dataset(str(folder))
        assert len(api.list_datasets()["datasets"]) == 2


class TestRestService:
    def test_the_route_list_is_the_services_own(self, api) -> None:
        pytest.importorskip("fastapi")
        pytest.importorskip("email_validator")
        from services.api.main import app

        def flatten(routes):
            # Newer FastAPI keeps included routers as one wrapper entry each.
            for route in routes:
                if hasattr(route, "original_router"):
                    yield from flatten(route.original_router.routes)
                else:
                    yield route

        served = {
            (method, route.path)
            for route in flatten(app.routes)
            for method in (getattr(route, "methods", None) or {"WEBSOCKET"})
            if method not in {"HEAD", "OPTIONS"}
            and not route.path.startswith(("/docs", "/redoc", "/openapi"))
        }
        listed = {(e["method"], e["path"]) for e in api.rest_endpoints()["endpoints"]}
        assert listed == served
        # The hand-written list this replaced.
        assert ("GET", "/findings") not in listed and ("GET", "/processing/jobs") not in listed

    def test_requests_stay_on_this_machine(self, api) -> None:
        assert not api.rest_request("/health", "http://example.com")["ok"]
        assert not api.rest_request("health")["ok"]

    def test_a_closed_port_is_reported_as_not_running(self, api) -> None:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        result = api.rest_request("/health", f"http://127.0.0.1:{port}")
        assert not result["ok"] and "not answering" in result["error"]


class TestCaptureMatching:
    def test_match_needs_a_plan_and_images(self, api) -> None:
        assert "No mission is planned" in api.match_captures()["error"]
        api.set_aoi([[W, S], [E, S], [E, N], [W, N]])
        api.plan_mission({})
        assert "Select the flight" in api.match_captures()["error"]
        assert "Run Match Captures first" in api.request_reflight()["error"]
