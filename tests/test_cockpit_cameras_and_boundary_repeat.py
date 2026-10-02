"""Two cockpit pickers that answered with figures that were not real.

* Cameras passed the camera's display NAME to describe_camera. The camera database is
  keyed by profile key, so every camera fell through to the generic placeholder and the
  toast printed the placeholder's GSD (1.98 cm at 60 m) under the real camera's name,
  with the placeholder warning dropped. The Mavic 2 Pro's real figure is 1.41 cm.
* Repeat Survey > modified boundary cleared the waypoints and never re-planned, so it
  reported "Re-planned ..." over 0 capture points -- even over the original's own area,
  which it also called a partial overlap.

Driven through workspace.html in headless Chromium against the real Api.
"""

from __future__ import annotations

from cockpit_harness import Cockpit, make_api
from mission.cameras import describe, resolve
from mission.repeat import REPEAT_MODIFIED_BOUNDARY, repeat_mission

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]
SHIFTED = [[-81.7500, 41.3030], [-81.7460, 41.3030], [-81.7460, 41.3060], [-81.7500, 41.3060]]


def test_a_camera_named_by_its_display_name_is_still_that_camera():
    profile, known = resolve("DJI Mavic 2 Pro (Hasselblad L1D-20c)")
    assert known and profile.key == "mavic2pro"
    assert describe("nonsense-camera")["known"] is False


def test_a_modified_boundary_repeat_is_planned_over_the_new_area():
    original = {"template": "grid", "camera": "mavic2pro", "altitude_m": 60.0,
                "front_overlap_pct": 75.0, "side_overlap_pct": 65.0,
                "waypoints": [[-81.75, 41.304, 60.0]] * 5, "polygon": AOI}
    same = repeat_mission(original, mode=REPEAT_MODIFIED_BOUNDARY, boundary=AOI)
    assert len(same.waypoints) > 0
    assert same.comparability == "comparable"
    moved = repeat_mission(original, mode=REPEAT_MODIFIED_BOUNDARY, boundary=SHIFTED)
    assert len(moved.waypoints) > 0
    assert moved.comparability == "partial_overlap"
    # Every new capture point lies over the new area, not the old one.
    lons = [w[0] for w in moved.waypoints]
    assert min(lons) > -81.7505 and max(lons) < -81.7455
    assert any("50%" in c or "inside the original boundary" in c for c in moved.changes)


def test_cameras_and_boundary_repeat_in_the_cockpit(tmp_path):
    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    assert api.save_mission("Quarry")["ok"]
    truth = describe("mavic2pro", altitude_m=60.0)["gsd_cm"]
    placeholder = describe("not-a-camera", altitude_m=60.0)["gsd_cm"]
    assert abs(truth - placeholder) > 0.1

    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run("""
            const out = {};
            const toast = (prefix, n = 1) => ODK.waitFor(() => {
              const hits = ODK.toasts().filter((t) => t.includes(prefix));
              return hits.length >= n ? hits[hits.length - 1] : null;
            }, 20000);
            await ODK.nav("Mission Planning");

            await ODK.press("Cameras");
            await ODK.pick("DJI Mavic 2 Pro");
            await ODK.answer("60");
            out.camera = await toast("at 60 m: GSD");

            await ODK.press("Repeat Survey");
            await ODK.pick("Quarry v1");
            await ODK.pick("modified boundary");
            out.same = await toast("Repeat of Quarry v1");
            out.errors = ODK.toasts().filter((t) => /TypeError|undefined|NaN/.test(t));
            return out;
        """, virtual_time_ms=120000, timeout_s=480)
        assert api.set_aoi(SHIFTED)["ok"]
        moved = cockpit.run("""
            const toast = (prefix, n) => ODK.waitFor(() => {
              const hits = ODK.toasts().filter((t) => t.includes(prefix));
              return hits.length >= n ? hits[hits.length - 1] : null;
            }, 20000);
            await ODK.nav("Mission Planning");
            await ODK.press("Repeat Survey");
            await ODK.pick("Quarry v1");
            await ODK.pick("modified boundary");
            return (await toast("Repeat of Quarry v1", 1)) || ODK.toasts().join(" || ");
        """, virtual_time_ms=120000, timeout_s=480)

    assert seen["errors"] == [], seen
    assert seen["camera"].startswith("DJI Mavic 2 Pro"), seen
    assert f"GSD {truth:.2f} cm" in seen["camera"], seen
    assert f"GSD {placeholder:.2f} cm" not in seen["camera"], seen

    assert "modified_boundary" in seen["same"], seen
    assert " 0 capture point(s)" not in seen["same"], seen
    assert "comparable." in seen["same"] and "partial_overlap" not in seen["same"], seen
    assert "same area as the original" in seen["same"], seen

    assert moved and "partial_overlap" in moved, moved
    assert " 0 capture point(s)" not in moved, moved
    assert "inside the original boundary" in moved, moved
