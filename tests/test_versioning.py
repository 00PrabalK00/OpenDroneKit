"""Mission version diff and restore.

The question a diff has to answer is "is this the plan that was signed off", usually on
site and under time pressure. So the tests check that a real change is described the way
an operator would say it, that a derived field changing does not bury the decision that
caused it, and above all that restoring an old version does not delete the record of
what was flown in between.
"""

from __future__ import annotations

import pytest

from mission.versioning import diff_versions, restore_version, version_history


def version(num: int, **summary) -> dict:
    base = {
        "altitude_m": 60.0, "front_overlap_pct": 75.0, "side_overlap_pct": 65.0,
        "estimated_gsd_cm": 1.4, "estimated_time_min": 6.1, "path_distance_m": 1478.0,
        "camera": "mavic2pro", "template": "grid",
        "waypoints": [[0, 0, 60]] * 144,
        "polygon": [[-81.75, 41.30], [-81.74, 41.30], [-81.74, 41.31]],
        "safety_constraints": {"no_fly_polygons": []},
    }
    base.update(summary)
    return {
        "id": num, "version_num": num, "mission_name": "Bridge deck",
        "template": base["template"], "created_at": f"2026-08-14T0{num}:00:00Z",
        "note": "", "flight_recipe": {"world_poses": []}, "plan_summary": base,
    }


class TestDiff:
    def test_identical_versions_report_no_differences(self):
        result = diff_versions(version(1), version(2))
        assert result.is_empty is True
        assert result.summary() == ["No differences."]

    def test_an_altitude_change_is_described_in_operator_terms(self):
        result = diff_versions(version(1), version(2, altitude_m=75.0))
        assert any("Altitude: 60 m -> 75 m" in line for line in result.summary())

    def test_a_changed_waypoint_count_is_quantified_both_ways(self):
        fewer = diff_versions(version(1), version(2, waypoints=[[0, 0, 60]] * 96))
        assert fewer.waypoint_delta == -48
        assert any("48 fewer" in line for line in fewer.summary())

        more = diff_versions(version(1), version(2, waypoints=[[0, 0, 60]] * 200))
        assert any("56 more" in line for line in more.summary())

    def test_a_redrawn_boundary_is_one_change_not_one_per_vertex(self):
        """A redrawn area is a single decision, however many vertices moved."""
        result = diff_versions(
            version(1), version(2, polygon=[[-81.76, 41.29], [-81.73, 41.29],
                                            [-81.73, 41.32], [-81.76, 41.32]]))
        assert result.geofence_changed is True
        assert sum(1 for line in result.summary() if "redrawn" in line) == 1

    def test_a_new_no_fly_zone_is_flagged_as_affecting_the_route(self):
        result = diff_versions(
            version(1),
            version(2, safety_constraints={"no_fly_polygons": [[[0, 0], [1, 0], [1, 1]]]}))
        assert result.no_fly_changed is True
        assert any("No-fly" in line for line in result.summary())

    def test_several_changes_are_all_reported(self):
        result = diff_versions(
            version(1), version(2, altitude_m=80.0, front_overlap_pct=85.0,
                                camera="phantom4rtk"))
        text = " ".join(result.summary())
        assert "Altitude" in text and "Front overlap" in text and "Camera" in text

    def test_floating_point_noise_is_not_reported_as_a_change(self):
        """Recomputing a plan must not look like someone edited it."""
        result = diff_versions(version(1), version(2, estimated_time_min=6.1 + 1e-12))
        assert result.is_empty is True

    def test_the_version_numbers_are_carried_on_the_diff(self):
        result = diff_versions(version(3), version(7, altitude_m=70.0))
        assert (result.from_version, result.to_version) == (3, 7)

    def test_a_boolean_reads_as_on_or_off_rather_than_true_or_false(self):
        result = diff_versions(version(1, terrain_follow_enabled=False),
                               version(2, terrain_follow_enabled=True))
        assert any("off -> on" in line for line in result.summary())

    def test_a_summary_arriving_as_json_text_is_still_compared(self):
        """Versions come back from SQLite with the summary still encoded."""
        import json

        older = version(1)
        newer = version(2, altitude_m=90.0)
        older["plan_summary"] = json.dumps(older["plan_summary"])
        newer["plan_summary"] = json.dumps(newer["plan_summary"])

        result = diff_versions(older, newer)
        assert any("Altitude" in line for line in result.summary())


class TestHistory:
    def test_the_first_version_has_nothing_to_compare_against(self):
        history = version_history([version(1)])
        assert history[0]["changes"] == ["First saved version."]

    def test_each_version_is_described_against_its_predecessor(self):
        history = version_history([
            version(1),
            version(2, altitude_m=70.0),
            version(3, altitude_m=70.0, camera="phantom4rtk"),
        ])
        assert any("Altitude" in c for c in history[1]["changes"])
        assert any("Camera" in c for c in history[2]["changes"])

    def test_history_is_ordered_even_when_the_input_is_not(self):
        history = version_history([version(3), version(1), version(2)])
        assert [h["version_num"] for h in history] == [1, 2, 3]


class FakeStore:
    """Records saves so restore can be checked without a database."""

    def __init__(self):
        self.saved: list[dict] = []

    def save_mission_version(self, **kwargs):
        entry = dict(kwargs)
        entry["version_num"] = len(self.saved) + 10
        self.saved.append(entry)
        return entry


class TestRestore:
    def test_restoring_writes_a_new_version_rather_than_overwriting(self):
        """History is append-only: what was flown in between must survive."""
        store = FakeStore()
        result = restore_version(store, project_id=1, version=version(3, altitude_m=55.0))

        assert len(store.saved) == 1
        assert result["version_num"] == 10, "restore must create a newer version"

    def test_the_restored_content_matches_the_version_it_came_from(self):
        store = FakeStore()
        original = version(3, altitude_m=55.0, camera="phantom4rtk")
        restore_version(store, project_id=1, version=original)

        saved = store.saved[0]
        assert saved["plan_summary"]["altitude_m"] == 55.0
        assert saved["plan_summary"]["camera"] == "phantom4rtk"

    def test_the_restore_says_where_it_came_from(self):
        """An unexplained version in the audit trail is a puzzle for someone later."""
        store = FakeStore()
        restore_version(store, project_id=1, version=version(3))
        assert "version 3" in store.saved[0]["note"]

    def test_a_custom_note_is_kept(self):
        store = FakeStore()
        restore_version(store, project_id=1, version=version(3),
                        note="Reverting: client rejected the 75 m plan.")
        assert "client rejected" in store.saved[0]["note"]

    def test_the_restored_version_points_back_at_its_source(self):
        store = FakeStore()
        restore_version(store, project_id=1, version=version(4))
        assert store.saved[0]["parent_version_id"] == 4

    def test_a_recipe_stored_as_json_text_is_restored_as_a_recipe(self):
        import json

        store = FakeStore()
        source = version(2)
        source["flight_recipe"] = json.dumps({"world_poses": [{"lon": 1.0, "lat": 2.0}]})
        restore_version(store, project_id=1, version=source)

        assert isinstance(store.saved[0]["flight_recipe"], dict)
        assert store.saved[0]["flight_recipe"]["world_poses"][0]["lon"] == 1.0


class TestAgainstRealStorage:
    def test_save_restore_and_diff_work_against_the_real_store(self, tmp_path):
        """The fake store proves the logic; this proves the integration."""
        from app.store import ProjectStore

        store = ProjectStore(tmp_path / "projects.db")
        project = store.create_project("Bridge 14")
        project_id = project["id"] if isinstance(project, dict) else project

        first = store.save_mission_version(
            project_id=project_id, mission_name="deck", template="grid",
            flight_recipe={"world_poses": []},
            plan_summary={"altitude_m": 60.0, "waypoints": [[0, 0, 60]] * 10},
        )
        store.save_mission_version(
            project_id=project_id, mission_name="deck", template="grid",
            flight_recipe={"world_poses": []},
            plan_summary={"altitude_m": 80.0, "waypoints": [[0, 0, 80]] * 6},
        )

        versions = store.list_mission_versions(project_id, "deck")
        assert len(versions) >= 2

        ordered = sorted(versions, key=lambda v: v["version_num"])
        changes = diff_versions(ordered[0], ordered[1]).summary()
        assert any("Altitude" in c for c in changes)

        restored = restore_version(store, project_id, ordered[0])
        assert restored["version_num"] > ordered[-1]["version_num"]
        # The intervening version is still there.
        assert len(store.list_mission_versions(project_id, "deck")) >= 3
