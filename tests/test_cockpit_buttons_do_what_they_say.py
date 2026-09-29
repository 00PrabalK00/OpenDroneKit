"""Buttons that were labelled one thing and called another, driven end to end.

Audit, before this change:

  Projects   "New Folder"        called add_layer_from_file
             "Archive"           revoked share links ("Revoked 0 link(s)")
  Verification "Load Flight"     called check_interrupted_flight
             "Match Captures"    counted saved mission versions
             "Request Reflight"  logged maintenance against aircraft 1 (and failed)
  Developers "Copy cURL"         built a POST to the first registered webhook
             "Send Request"      counted the fields capabilities() returned
             Endpoints panel     nine hand-written routes, most of which do not exist
  Processing "Set CRS"           TypeError: missing 1 required positional argument
  Digital Twin every render      TypeError in build_asset_inventory (no instances)
  Home       Import              "Imported ? images"; a re-import duplicated the dataset

Each test drives the real page against the real Api with real files on disk.
"""

from __future__ import annotations

import pytest

from cockpit_harness import Cockpit, make_api, write_geotagged_jpeg

W, E = -81.7520, -81.7513
S, N = 41.3040, 41.3044


@pytest.fixture
def flown(tmp_path):
    """A project with a planned mission and a folder of geotagged images from the site.

    The images sit on a few of the plan's own capture points and one well outside it,
    so the match has matched, missed and unplanned captures to report."""
    api = make_api(tmp_path)
    assert api.set_aoi([[W, S], [E, S], [E, N], [W, N]])["ok"]
    assert api.plan_mission({"altitude_m": 40})["ok"]
    from core.capture_matching import planned_captures_from_plan

    planned = planned_captures_from_plan(api._session.mission_plan_dict)
    folder = tmp_path / "flight"
    for i, capture in enumerate(planned[:5]):
        write_geotagged_jpeg(folder / f"IMG_{i:04d}.JPG", capture.longitude, capture.latitude)
    write_geotagged_jpeg(folder / "IMG_9999.JPG", -81.7400, 41.3100)
    return api, folder, planned


def test_import_then_load_flight_match_and_request_reflight(flown, tmp_path) -> None:
    api, folder, planned = flown
    with Cockpit(api, tmp_path / "profile", picks={"folder": str(folder)}) as cockpit:
        result = cockpit.run("""
            await ODK.press("Import");
            const first = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Imported")));
            await ODK.press("Import");
            await ODK.sleep(1500);
            await ODK.nav("Verification");
            await ODK.press("Load Flight");
            const rows = ODK.$$(".modal-backdrop .modal .modal-row").length;
            await ODK.pick("flight");
            const loaded = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Loaded")));
            await ODK.press("Match Captures");
            await ODK.answer("5");
            const matched = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("planned captures matched")), 20000);
            await ODK.sleep(1500);
            const props = ODK.panel("Capture Properties");
            const missing = ODK.panel("Missing Captures");
            await ODK.press("Request Reflight");
            await ODK.answer("battery swap gap");
            const reflight = await ODK.waitFor(() => ODK.toasts().find((t) => t.includes("to refly")), 20000);
            return { first, rows, loaded, matched, props, missing, reflight };
        """)
    assert result["first"] == "Imported 6 image(s), 6 geotagged."
    assert result["rows"] == 1, "importing the same folder twice listed it twice"
    assert not cockpit.called("check_interrupted_flight")
    report = api.capture_report()
    assert report["ok"] and report["matched"] == 5
    assert len(report["missed"]) == len(planned) - 5
    assert report["unplanned"], "the off-plan image was not reported"
    assert result["matched"].startswith(f"5 of {len(planned)} planned captures matched")
    assert "Matched 5" in result["props"]
    assert str(planned[5].index) in result["missing"]
    # The reflight is the missed points, on disk and on the map; no maintenance record.
    assert not cockpit.called("log_maintenance")
    layer = next(l for l in api.list_layers()["layers"] if l["name"] == "Reflight points")
    assert layer["metadata"]["feature_count"] == len(planned) - 5
    assert f"{len(planned) - 5} point(s) to refly" in result["reflight"]


def test_set_crs_checks_the_images_and_sets_the_reconstruction_crs(flown, tmp_path) -> None:
    api, folder, _ = flown
    api.import_dataset(str(folder))
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Processing");
            await ODK.press("Set CRS");
            await ODK.answer("32617");
            return { said: await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Output CRS"))) };
        """)
    checked = cockpit.called("check_spatial_reference")
    assert checked and checked[0]["ok"]
    paths, _gcps, epsg = checked[0]["args"]
    assert len(paths) == 6 and epsg == 32617
    assert result["said"].startswith("Output CRS EPSG:32617")
    assert "6/6 images geotagged" in result["said"]


def test_developer_screen_lists_real_routes_and_requests_them(tmp_path) -> None:
    api = make_api(tmp_path)
    routes = api.rest_endpoints()["endpoints"]
    target = next(r for r in routes if r["method"] == "GET" and "{" not in r["path"]
                  and r["path"] != "/health")
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Developers");
            await ODK.waitFor(() => ODK.panel("Endpoints") && !ODK.panel("Endpoints").startsWith("Reading"));
            const panel = ODK.panel("Endpoints");
            // Open the group and pick the route, as an operator would.
            const group = ODK.$$(".panel .tree-node").find((n) => n.innerText.includes({target["router"]!r}));
            group.onclick({{ offsetX: 0 }});   // the twisty: expand the router's routes
            await ODK.sleep(200);
            const row = ODK.$$(".panel .tree-node").find((n) => n.innerText.trim().endsWith({(target["method"] + " " + target["path"])!r}));
            row.onclick({{ offsetX: 999 }});
            await ODK.sleep(300);
            await ODK.press("Copy cURL");
            const dialog = await ODK.waitFor(() => ODK.$(".modal-backdrop .modal textarea"));
            const curl = dialog.value;
            ODK.$(".modal-backdrop .modal .modal-actions .tbtn.primary").click();
            await ODK.sleep(200);
            await ODK.press("Send Request");
            const sent = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Send Request:") || t.startsWith({target["path"]!r})), 20000);
            return {{ panel, curl, sent }};
        """)
    assert "GET /projects " not in result["panel"] + " "
    assert f'curl -X GET "http://127.0.0.1:8000{target["path"]}"' in result["curl"]
    assert "webhook" not in result["curl"].lower()
    sent = cockpit.called("rest_request")
    assert sent and sent[0]["args"][0] == target["path"]
    # Whatever the service answered -- or that nothing is listening -- is what is shown.
    assert "HTTP " in result["sent"] or "not answering" in result["sent"]


def test_renamed_buttons_and_the_twin_render_without_errors(tmp_path) -> None:
    api = make_api(tmp_path)
    layer = tmp_path / "zone.geojson"
    layer.write_text('{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
                     '"geometry":{"type":"Point","coordinates":[-81.75,41.30]}}]}', encoding="utf-8")
    with Cockpit(api, tmp_path / "profile", picks={"file": str(layer)}) as cockpit:
        result = cockpit.run("""
            await ODK.nav("Projects");
            const toolbar = ODK.$(".toolbar").innerText.split("\\n");
            await ODK.press("Add Layer");
            const added = await ODK.waitFor(() => ODK.toasts().includes("Layer added."));
            await ODK.nav("Digital Twin");
            await ODK.sleep(1500);
            return { toolbar, added: Boolean(added), twin: ODK.panel("Scene Hierarchy") };
        """)
    assert "Add Layer" in result["toolbar"] and "New Folder" not in result["toolbar"]
    assert "Revoke Links" in result["toolbar"] and "Archive" not in result["toolbar"]
    assert result["added"] and "zone" in result["twin"]
    assert not cockpit.called("build_asset_inventory")
    assert not [c for c in cockpit.calls if "TypeError" in str(c["error"])]
