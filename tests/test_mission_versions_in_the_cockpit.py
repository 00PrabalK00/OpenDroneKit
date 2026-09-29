"""The cockpit's choose() pickers for saved missions work once there is history.

The pickers rendered, and every one past the first saved version was broken:

* mission_version_history raised "TypeError: object of type 'int' has no len()" as soon
  as a mission had two versions (save_mission stores the waypoint COUNT); tryCall hid
  it, and Compare Versions said "Two saved versions are needed" over three;
* Compare Versions read diff.changes / diff.fields, which the Api never sends, so any
  pair of versions was "identical";
* History labelled rows from h.summary, also never sent, so every row read "v1";
* Repeat Survey offered "same-camera" and "same-terrain", which mission.repeat refuses
  as unknown modes, never offered a saved survey, and a saved version given to the Api
  was refused with "no waypoints" because the row carries a count, not coordinates;
* Add Marker never asked for a radius, so a hazard made in the cockpit was a point.

Driven through workspace.html in headless Chromium against the real Api.
"""

from __future__ import annotations

from cockpit_harness import Cockpit, make_api
from mission.versioning import diff_versions, version_history

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]


def _saved(api, name, **options):
    assert api.plan_mission(options)["ok"]
    assert api.save_mission(name)["ok"]


def test_history_of_saved_versions_counts_waypoints_without_raising(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    _saved(api, "North", altitude_m=55)
    _saved(api, "North", altitude_m=80)
    _saved(api, "Roof", altitude_m=40)
    history = api.mission_version_history("")
    assert history["ok"], history
    rows = {(h["mission_name"], h["version_num"]): h for h in history["history"]}
    # Each version is compared with the one before it OF THE SAME MISSION.
    assert rows[("Roof", 1)]["changes"] == ["First saved version."]
    assert any(c.startswith("Altitude: 55 m -> 80 m") for c in rows[("North", 2)]["changes"])
    assert rows[("North", 2)]["waypoints"] < rows[("North", 1)]["waypoints"]

    diff = api.diff_mission_versions(1, 2, "North")["diff"]
    assert not diff["identical"] and diff["waypoints_before"] > diff["waypoints_after"] > 0

    # The pure function takes a count or a list.
    a = {"version_num": 1, "plan_summary": {"waypoints": 12, "altitude_m": 50.0}}
    b = {"version_num": 2, "plan_summary": {"waypoints": [[0, 0], [1, 1]], "altitude_m": 50.0}}
    assert diff_versions(a, b).waypoint_delta == -10
    assert version_history([a, b])[1]["changes"] == ["Waypoints: 12 -> 2 (10 fewer)"]


def test_a_saved_version_can_be_repeated(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    _saved(api, "North", altitude_m=55)
    repeat = api.repeat_mission(1, "exact", "", "", "North")
    assert repeat["ok"], repeat
    saved = api.list_mission_versions("North")["versions"][0]
    assert repeat["repeat"]["capture_count"] > 0
    assert repeat["repeat"]["camera"] == "mavic2pro"
    assert repeat["repeat"]["altitude_m"] == 55.0
    assert saved["version_num"] == 1


def test_compare_history_repeat_and_marker_in_the_cockpit(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    _saved(api, "North", altitude_m=55)
    _saved(api, "North", altitude_m=80)
    _saved(api, "North", altitude_m=80, front_overlap_pct=85)
    _saved(api, "Roof", altitude_m=40)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            const out = {};
            const toast = (prefix, n = 1) => ODK.waitFor(() => {
              const hits = ODK.toasts().filter((t) => t.includes(prefix));
              return hits.length >= n ? hits[hits.length - 1] : null;
            }, 20000);
            await ODK.nav("Mission Planning");

            await ODK.press("Compare Versions");
            out.missionRow = await ODK.pick("North");
            out.fromRows = ODK.$$(".modal-backdrop .modal .modal-row").map((r) => r.textContent);
            await ODK.pick("v1");
            await ODK.pick("v2");
            out.compare = await toast("North v1 to v2");

            await ODK.press("History");
            await ODK.pick("North");
            out.historyRows = ODK.$$(".modal-backdrop .modal .modal-row").map((r) => r.textContent);
            await ODK.pick("v1");
            out.restored = await toast("Restored North v1");
            out.errors = ODK.toasts().filter((t) => /TypeError|undefined|NaN/.test(t));
            return out;
        """, virtual_time_ms=120000, timeout_s=480)
        later = cockpit.run("""
            const out = {};
            const toast = (prefix, n = 1) => ODK.waitFor(() => {
              const hits = ODK.toasts().filter((t) => t.includes(prefix));
              return hits.length >= n ? hits[hits.length - 1] : null;
            }, 20000);
            await ODK.nav("Mission Planning");
            await ODK.press("Repeat Survey");
            out.repeatRows = ODK.$$(".modal-backdrop .modal .modal-row").map((r) => r.textContent);
            await ODK.pick("Roof v1");
            out.modeRows = ODK.$$(".modal-backdrop .modal .modal-row").map((r) => r.textContent);
            await ODK.pick("exact");
            out.repeat = await toast("Repeat of Roof v1");

            await ODK.press("Add Marker");
            await ODK.pick("hazard —");
            await ODK.answer("Mast");
            // No map click: Esc, then type the position.
            await ODK.key("Escape");
            await ODK.answer("-81.7500"); await ODK.answer("41.3045");
            await ODK.answer("15");
            out.marker = await toast('Added hazard "Mast"');
            out.errors = ODK.toasts().filter((t) => /TypeError|undefined|NaN|Unknown repeat mode/.test(t));
            return out;
        """, virtual_time_ms=120000, timeout_s=480)
    seen.update({k: v for k, v in later.items() if k != "errors"})
    seen["errors"] = seen["errors"] + later["errors"]
    assert seen["errors"] == [], seen
    assert "North" in seen["missionRow"], seen
    assert any("v1" in r and "grid" in r and "First saved version." in r for r in seen["fromRows"]), seen
    assert "Altitude: 55 m -> 80 m" in seen["compare"], seen
    assert "identical" not in seen["compare"], seen
    # Newest first, each saying what changed; not three rows reading "v1".
    assert seen["historyRows"][0].startswith("v3"), seen
    assert any("Front overlap: 75 % -> 85 %" in r for r in seen["historyRows"]), seen
    assert "as v4" in seen["restored"], seen
    assert [v["version_num"] for v in api.list_mission_versions("North")["versions"]].count(4) == 1

    assert any(r.startswith("Roof v1") for r in seen["repeatRows"]), seen
    assert not any("same-camera" in r or "same-terrain" in r for r in seen["modeRows"]), seen
    assert "capture point(s) at 40 m with mavic2pro" in seen["repeat"], seen
    assert "comparable" in seen["repeat"], seen

    assert "(radius 15 m)" in seen["marker"], seen
    [marker] = api.list_site_markers()["markers"]
    assert marker["name"] == "Mast" and marker["radius_m"] == 15.0, marker
    failed = [c for c in cockpit.calls if not c["ok"] and c["method"] in (
        "mission_version_history", "diff_mission_versions", "restore_mission_version",
        "repeat_mission", "add_site_marker")]
    assert failed == [], failed
