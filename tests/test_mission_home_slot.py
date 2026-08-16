"""MAVLink reserves mission sequence 0 for home, and a plan must allow for it.

Found by SITL, and findable no other way. The planner emitted NAV_TAKEOFF (22) as the
first mission item, the bridge sent it at sequence 0, and ArduPilot -- which stores its
own home there and overwrites whatever arrives -- read it back as NAV_WAYPOINT (16).

Nothing failed. The upload reported success and the item count matched. The aircraft
would simply not have taken off: it would have proceeded toward the first waypoint at
whatever altitude it already had, which on the ground is none.

Every mock-based test of the upload path passed throughout, because a mock stores what
it is given. Only a real autopilot has an opinion about sequence 0.
"""

from __future__ import annotations

from core.mission_planner_bridge import _without_home_slot, _with_home_slot

TAKEOFF = 22
WAYPOINT = 16
CAMERA_TRIGGER = 206  # non-positional: params carry values, not degrees


def _plan() -> list[dict]:
    return [
        {"seq": 0, "command": TAKEOFF, "lat": -35.363, "lon": 149.165, "alt": 10.0},
        {"seq": 1, "command": WAYPOINT, "lat": -35.364, "lon": 149.166, "alt": 10.0},
        {"seq": 2, "command": CAMERA_TRIGGER, "lat": 1.0, "lon": 0.0, "alt": 0.0},
    ]


class TestHomeSlotIsReserved:
    def test_the_first_real_command_no_longer_sits_at_sequence_zero(self) -> None:
        sent = _with_home_slot(_plan())
        assert sent[0]["command"] == WAYPOINT, "sequence 0 must be the home placeholder"
        assert sent[1]["command"] == TAKEOFF, (
            "the takeoff moved out of the home slot; this is the bug SITL caught"
        )

    def test_sequences_are_contiguous_from_zero(self) -> None:
        sent = _with_home_slot(_plan())
        assert [item["seq"] for item in sent] == [0, 1, 2, 3]

    def test_nothing_from_the_plan_is_dropped(self) -> None:
        plan = _plan()
        sent = _with_home_slot(plan)
        assert len(sent) == len(plan) + 1
        assert [item["command"] for item in sent[1:]] == [i["command"] for i in plan]

    def test_only_one_item_is_marked_current(self) -> None:
        sent = _with_home_slot(_plan())
        assert sum(int(item.get("current", 0)) for item in sent) == 1

    def test_home_borrows_the_first_positional_fix_not_null_island(self) -> None:
        """A vehicle without home set should see somewhere plausible, not 0,0.

        The autopilot overwrites this either way, so the value is not flown -- but a
        placeholder at Null Island is the kind of coordinate that reaches a log or a
        map and gets believed.
        """
        sent = _with_home_slot(_plan())
        assert sent[0]["lat"] == -35.363
        assert sent[0]["lon"] == 149.165

    def test_home_is_taken_from_a_positional_item_not_a_camera_trigger(self) -> None:
        # Non-positional items carry plain values in lat/lon; command 206 above has
        # lat 1.0, which is a real place off the coast of Africa and not a home.
        plan = [
            {"seq": 0, "command": CAMERA_TRIGGER, "lat": 1.0, "lon": 0.0, "alt": 0.0},
            {"seq": 1, "command": WAYPOINT, "lat": -35.364, "lon": 149.166, "alt": 10.0},
        ]
        sent = _with_home_slot(plan)
        assert sent[0]["lat"] == -35.364

    def test_an_empty_plan_stays_empty(self) -> None:
        assert _with_home_slot([]) == []


class TestDownloadMirrorsUpload:
    def test_a_round_trip_returns_the_plan_that_was_uploaded(self) -> None:
        plan = _plan()
        # What the vehicle stores is the uploaded list behind its own home item.
        stored = _with_home_slot(plan)
        assert _without_home_slot(stored) == [
            dict(item, seq=index, current=0) for index, item in enumerate(plan)
        ]

    def test_the_caller_never_sees_the_home_item(self) -> None:
        stored = _with_home_slot(_plan())
        returned = _without_home_slot(stored)
        assert len(returned) == len(_plan())
        assert returned[0]["command"] == TAKEOFF

    def test_an_empty_mission_downloads_as_empty(self) -> None:
        assert _without_home_slot([]) == []

    def test_a_mission_of_home_alone_downloads_as_empty(self) -> None:
        # A vehicle with no mission still reports its home item.
        assert _without_home_slot([{"seq": 0, "command": WAYPOINT}]) == []
