"""Crash recovery.

If the laptop dies mid-mission, the aircraft does not. The failure mode worth designing
against is an application that reopens, sees an incomplete mission, and quietly
re-uploads or restarts it -- sending an airborne aircraft back to the start of its
route. So every test here checks that recovery produces information and choices, never
an action, and that the software says plainly what it cannot know.
"""

from __future__ import annotations

import json

import pytest

from core.drone import DroneTelemetry
from core.flight_state import (
    PHASE_ABORTED,
    PHASE_COMPLETED,
    PHASE_FLYING,
    PHASE_PAUSED,
    PHASE_PLANNED,
    PHASE_UPLOADED,
    FlightState,
    clear_state,
    load_state,
    record_transition,
    recover,
    save_state,
    state_path,
)


def flying_telemetry() -> DroneTelemetry:
    return DroneTelemetry(
        connected=True, armed=True, flight_mode="AUTO",
        latitude=41.3042, longitude=-81.7505, altitude_rel_m=62.0,
        battery_pct=64.0, waypoint_index=37, waypoint_total=144,
    )


class TestPersistence:
    def test_a_transition_is_written_and_read_back(self, tmp_path):
        record_transition(tmp_path, PHASE_UPLOADED, mission_name="Bridge 14")
        state = load_state(tmp_path)

        assert state.phase == PHASE_UPLOADED
        assert state.mission_name == "Bridge 14"

    def test_telemetry_is_captured_with_the_transition(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())
        state = load_state(tmp_path)

        assert state.last_latitude == pytest.approx(41.3042)
        assert state.waypoint_index == 37
        assert state.waypoint_total == 144
        assert state.last_battery_pct == pytest.approx(64.0)

    def test_later_transitions_keep_earlier_fields(self, tmp_path):
        record_transition(tmp_path, PHASE_PLANNED, mission_name="Bridge 14")
        record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())

        state = load_state(tmp_path)
        assert state.mission_name == "Bridge 14"
        assert state.phase == PHASE_FLYING

    def test_no_state_reads_as_none_not_as_an_empty_flight(self, tmp_path):
        assert load_state(tmp_path) is None

    def test_the_state_file_is_written_atomically(self, tmp_path):
        """A half-written file would either fail to parse or describe a flight that
        never happened."""
        record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())

        # No temporary files left behind, and the result is valid JSON.
        assert list(tmp_path.glob("*.tmp")) == []
        json.loads(state_path(tmp_path).read_text(encoding="utf-8"))

    def test_state_can_be_cleared(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING)
        clear_state(tmp_path)
        assert load_state(tmp_path) is None

    def test_clearing_nothing_is_not_an_error(self, tmp_path):
        clear_state(tmp_path)


class TestCleanShutdown:
    @pytest.mark.parametrize("phase", [PHASE_COMPLETED, PHASE_ABORTED])
    def test_a_finished_mission_is_marked_clean(self, tmp_path, phase):
        state = record_transition(tmp_path, phase)
        assert state.clean_shutdown is True
        assert state.possibly_airborne is False

    def test_an_interrupted_flight_is_not_marked_clean(self, tmp_path):
        state = record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())
        assert state.clean_shutdown is False
        assert state.possibly_airborne is True

    def test_a_paused_mission_may_still_be_airborne(self, tmp_path):
        """Paused means loitering, which is still flying."""
        state = record_transition(tmp_path, PHASE_PAUSED)
        assert state.possibly_airborne is True

    def test_an_uploaded_but_unflown_mission_is_not_airborne(self, tmp_path):
        state = record_transition(tmp_path, PHASE_UPLOADED)
        assert state.possibly_airborne is False


class TestRecovery:
    def test_nothing_recorded_means_nothing_to_recover(self, tmp_path):
        result = recover(tmp_path)
        assert result["recovered"] is False
        assert result["requires_operator"] is False

    def test_a_clean_finish_needs_no_operator_decision(self, tmp_path):
        record_transition(tmp_path, PHASE_COMPLETED, mission_name="Bridge 14")
        result = recover(tmp_path)

        assert result["requires_operator"] is False
        assert result["possibly_airborne"] is False

    def test_an_interrupted_flight_requires_the_operator(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, mission_name="Bridge 14",
                          telemetry=flying_telemetry())
        result = recover(tmp_path)

        assert result["requires_operator"] is True
        assert result["possibly_airborne"] is True

    def test_recovery_never_resumes_anything_by_itself(self):
        """The defect this module exists to prevent."""
        import inspect

        from core import flight_state

        source = inspect.getsource(flight_state.recover)
        for forbidden in ("upload_mission", "start_mission", "set_flight_mode", "arm("):
            assert forbidden not in source, (
                f"recover() must not call {forbidden}: re-uploading to an airborne "
                "aircraft would send it back to the start of its route"
            )

    def test_the_summary_says_the_aircraft_may_still_be_flying(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())
        summary = recover(tmp_path)["summary"]

        assert "may still be airborne" in summary
        assert "cannot tell" in summary

    def test_the_summary_reports_where_and_how_far_it_had_got(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, mission_name="Bridge 14",
                          telemetry=flying_telemetry())
        summary = recover(tmp_path)["summary"]

        assert "waypoint 37 of 144" in summary
        assert "41.3042" in summary
        assert "64%" in summary

    def test_the_note_explains_why_nothing_was_resumed(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())
        assert "back to the start of its route" in recover(tmp_path)["note"]

    def test_the_operator_is_offered_choices_not_a_default(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, telemetry=flying_telemetry())
        options = recover(tmp_path)["options"]

        assert "reconnect_and_observe" in options
        assert "resume_remaining" in options
        assert "discard_state" in options

    def test_a_missing_position_does_not_fabricate_one(self, tmp_path):
        record_transition(tmp_path, PHASE_FLYING, mission_name="Bridge 14")
        summary = recover(tmp_path)["summary"]

        assert "Last seen at" not in summary
        assert "may still be airborne" in summary


class TestCorruptState:
    def test_a_corrupt_file_is_reported_as_unknown_not_as_no_flight(self, tmp_path):
        """Losing the record is different from there having been nothing to record."""
        state_path(tmp_path).write_text("{ not json", encoding="utf-8")
        result = recover(tmp_path)

        assert result["corrupt"] is True
        assert result["requires_operator"] is True
        assert "unknown" in result["summary"]

    def test_a_corrupt_file_still_tells_the_operator_to_check_the_aircraft(self, tmp_path):
        state_path(tmp_path).write_text("garbage", encoding="utf-8")
        assert "before doing anything else" in recover(tmp_path)["summary"]

    def test_an_unknown_field_in_the_file_does_not_break_loading(self, tmp_path):
        """A state file written by a later version must not lose the whole record."""
        save_state(tmp_path, FlightState(phase=PHASE_FLYING, mission_name="Bridge 14"))
        payload = json.loads(state_path(tmp_path).read_text(encoding="utf-8"))
        payload["some_future_field"] = 42
        state_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")

        state = load_state(tmp_path)
        assert state is not None
        assert state.mission_name == "Bridge 14"
