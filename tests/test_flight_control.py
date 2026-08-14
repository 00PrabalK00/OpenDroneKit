"""Who has control, and taking it back.

Every test here is shaped by one asymmetry. Telling a pilot the aircraft is flying
itself when they actually have control is a moment's confusion. Telling them they have
control when the aircraft is still flying its mission is how people get hurt. So an
unrecognised mode counts as autonomous, a mode change is not believed until the vehicle
confirms it, and a failed hand-back says which mode the aircraft is still in rather than
reporting a generic error.
"""

from __future__ import annotations

import pytest

from core.drone import CommandResult, DroneTelemetry
from core.flight_control import (
    CONTROL_ASSISTED,
    CONTROL_AUTONOMOUS,
    CONTROL_MANUAL,
    CONTROL_UNKNOWN,
    HANDBACK_MODES,
    classify_mode,
    control_state,
    take_manual_control,
)


class FakeVehicle:
    """A vehicle that accepts only the modes it is told to accept."""

    def __init__(self, mode="AUTO", armed=True, accepts=(), reject_message="Denied."):
        self.mode = mode
        self.armed = armed
        self.accepts = {m.upper() for m in accepts}
        self.reject_message = reject_message
        self.requested: list[str] = []

    def set_flight_mode(self, mode: str) -> CommandResult:
        self.requested.append(mode.upper())
        if mode.upper() in self.accepts:
            self.mode = mode.upper()
            return CommandResult(True, "set_flight_mode", f"Mode is {mode.upper()}.")
        return CommandResult(False, "set_flight_mode", self.reject_message)

    def get_telemetry(self) -> DroneTelemetry:
        return DroneTelemetry(connected=True, armed=self.armed, flight_mode=self.mode)


class TestClassification:
    @pytest.mark.parametrize("mode", ["AUTO", "MISSION", "RTL", "GUIDED", "AUTO.MISSION"])
    def test_autonomous_modes_are_recognised(self, mode):
        assert classify_mode(mode) == CONTROL_AUTONOMOUS

    @pytest.mark.parametrize("mode", ["LOITER", "POSHOLD", "ALT_HOLD", "BRAKE"])
    def test_assisted_modes_are_recognised(self, mode):
        assert classify_mode(mode) == CONTROL_ASSISTED

    @pytest.mark.parametrize("mode", ["STABILIZE", "MANUAL", "ACRO"])
    def test_manual_modes_are_recognised(self, mode):
        assert classify_mode(mode) == CONTROL_MANUAL

    def test_case_and_spacing_do_not_matter(self):
        assert classify_mode("  auto  ") == CONTROL_AUTONOMOUS

    def test_an_unrecognised_mode_is_unknown_not_manual(self):
        """Guessing wrong in this direction is the dangerous one."""
        assert classify_mode("SOME_VENDOR_MODE") == CONTROL_UNKNOWN

    def test_an_empty_mode_is_unknown(self):
        assert classify_mode("") == CONTROL_UNKNOWN


class TestControlState:
    def test_an_autonomous_flight_says_the_sticks_will_not_take_control(self):
        """The fact that surprises people, stated where they will read it."""
        state = control_state(DroneTelemetry(armed=True, flight_mode="AUTO"))

        assert state.control == CONTROL_AUTONOMOUS
        assert state.pilot_has_control is False
        assert "sticks will not take control" in state.description

    def test_an_assisted_mode_reports_the_pilot_has_control(self):
        state = control_state(DroneTelemetry(armed=True, flight_mode="LOITER"))
        assert state.pilot_has_control is True

    def test_a_manual_mode_says_you_are_flying_it(self):
        state = control_state(DroneTelemetry(armed=True, flight_mode="STABILIZE"))
        assert state.control == CONTROL_MANUAL
        assert "you are flying it" in state.description

    def test_an_unknown_mode_does_not_credit_the_pilot_with_control(self):
        state = control_state(DroneTelemetry(armed=True, flight_mode="WEIRD"))

        assert state.pilot_has_control is False
        assert "Assume it is still flying itself" in state.description

    def test_a_disarmed_aircraft_says_nothing_is_flying(self):
        state = control_state(DroneTelemetry(armed=False, flight_mode="AUTO"))
        assert "Nothing is flying" in state.description

    def test_a_dict_of_telemetry_works_the_same_way(self):
        state = control_state({"armed": True, "flight_mode": "LOITER"})
        assert state.control == CONTROL_ASSISTED


class TestTakeManualControl:
    def test_control_is_handed_back_and_confirmed(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=["LOITER"])
        result = take_manual_control(vehicle)

        assert result["ok"] is True
        assert result["mode"] == "LOITER"
        assert vehicle.mode == "LOITER"

    def test_loiter_is_tried_first_because_it_holds_position(self):
        """A pilot taking over mid-mission may not have their hands on the sticks."""
        vehicle = FakeVehicle(mode="AUTO", accepts=HANDBACK_MODES)
        take_manual_control(vehicle)
        assert vehicle.requested[0] == "LOITER"

    def test_a_refused_mode_is_followed_by_the_next_candidate(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=["STABILIZE"])
        result = take_manual_control(vehicle)

        assert result["ok"] is True
        assert result["mode"] == "STABILIZE"
        assert len(vehicle.requested) > 1

    def test_a_preferred_mode_is_tried_before_the_defaults(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=HANDBACK_MODES)
        take_manual_control(vehicle, preferred="ALT_HOLD")
        assert vehicle.requested[0] == "ALT_HOLD"

    def test_when_nothing_is_accepted_the_failure_names_the_current_mode(self):
        """A generic error would leave the pilot unsure whether the mission stopped."""
        vehicle = FakeVehicle(mode="AUTO", accepts=[])
        result = take_manual_control(vehicle)

        assert result["ok"] is False
        assert "still in AUTO" in result["error"]
        assert "transmitter" in result["error"]

    def test_a_failure_does_not_claim_the_mission_stopped(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=[])
        result = take_manual_control(vehicle)
        assert "do not assume the mission has stopped" in result["error"]

    def test_every_attempt_is_reported_so_the_reason_is_visible(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=["STABILIZE"],
                              reject_message="Needs position estimate.")
        result = take_manual_control(vehicle)

        refused = [a for a in result["attempts"] if not a["ok"]]
        assert refused
        assert "position estimate" in refused[0]["message"]

    def test_a_success_says_the_mission_is_interrupted_not_cancelled(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=["LOITER"])
        assert "interrupted, not cancelled" in take_manual_control(vehicle)["note"]

    def test_a_driver_that_cannot_change_mode_says_to_use_the_transmitter(self):
        class NoModeControl:
            def get_telemetry(self):
                return DroneTelemetry(armed=True, flight_mode="AUTO")

        result = take_manual_control(NoModeControl())
        assert result["ok"] is False
        assert "transmitter" in result["error"]

    def test_the_resulting_state_is_returned_for_display(self):
        vehicle = FakeVehicle(mode="AUTO", accepts=["LOITER"])
        result = take_manual_control(vehicle)
        assert result["state"]["pilot_has_control"] is True


class TestCommandedIsNotConfirmed:
    def test_a_mode_change_is_believed_only_when_the_vehicle_agrees(self):
        """A driver that reports success without changing mode must not be trusted."""

        class LyingVehicle(FakeVehicle):
            def set_flight_mode(self, mode: str) -> CommandResult:
                self.requested.append(mode.upper())
                # Claims success, never changes mode.
                return CommandResult(True, "set_flight_mode", "Sent.")

        vehicle = LyingVehicle(mode="AUTO")
        result = take_manual_control(vehicle)

        # The hand-back reports what the driver told it, but the state it returns comes
        # from the vehicle, so the disagreement is visible rather than hidden.
        assert result["state"]["mode"] == "AUTO"
        assert result["state"]["pilot_has_control"] is False

    def test_the_bridge_verifies_mode_changes_by_default(self):
        """The real driver waits for the heartbeat instead of trusting transmission."""
        import inspect

        from core.mission_planner_bridge import MissionPlannerDroneClient

        signature = inspect.signature(MissionPlannerDroneClient.set_flight_mode)
        assert signature.parameters["verify"].default is True

    def test_the_bridge_resolves_mode_numbers_to_names(self):
        """custom_mode is a number; "3" is not a flight mode a pilot recognises."""
        from core.mission_planner_bridge import MissionPlannerDroneClient

        client = MissionPlannerDroneClient()
        # With no connection there is no mapping, and an empty name is the honest
        # answer rather than a stringified integer.
        assert client._mode_name(3) == ""
