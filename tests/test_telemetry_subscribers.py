"""Live telemetry reaches its subscribers, and one bad subscriber cannot stop the rest.

The registry carried this as implemented with the note "subscribe() added to the bridge;
no UI consumer test". The publish path runs on the MAVLink listener thread, and the
failure it must survive is mundane: a UI callback raises -- a widget disposed while the
aircraft is still flying, a formatting error on an unexpected value -- and if that
exception propagates it takes the listener thread with it.

Losing the listener does not look like a crash. The window stays open and the numbers
stop moving, which reads as a quiet aircraft rather than a dead telemetry feed. That is
the worst possible presentation of the failure, so it is the one pinned here.
"""

from __future__ import annotations

import pytest

from core.mission_planner_bridge import MissionPlannerDroneClient


@pytest.fixture
def client() -> MissionPlannerDroneClient:
    # No connection: _publish and the subscriber registry are pure local state, and
    # exercising them without a vehicle is the point -- this is about delivery, not
    # about MAVLink.
    return MissionPlannerDroneClient()


class TestDelivery:
    def test_a_subscriber_receives_published_events(self, client) -> None:
        seen = []
        client.subscribe(seen.append)
        client._publish({"type": "telemetry", "battery": 0.82})
        assert seen == [{"type": "telemetry", "battery": 0.82}]

    def test_every_subscriber_receives_the_same_event(self, client) -> None:
        first, second = [], []
        client.subscribe(first.append)
        client.subscribe(second.append)
        client._publish({"type": "telemetry"})
        assert first == second == [{"type": "telemetry"}]

    def test_subscribing_twice_does_not_double_deliver(self, client) -> None:
        # The shell re-subscribes on reconnect; duplicate delivery would double-count
        # anything a consumer accumulates.
        seen = []
        client.subscribe(seen.append)
        client.subscribe(seen.append)
        client._publish({"type": "telemetry"})
        assert len(seen) == 1

    def test_unsubscribing_stops_delivery(self, client) -> None:
        seen = []
        client.subscribe(seen.append)
        client.unsubscribe(seen.append)
        client._publish({"type": "telemetry"})
        assert seen == []

    def test_unsubscribing_something_never_registered_is_harmless(self, client) -> None:
        client.unsubscribe(lambda event: None)  # must not raise


class TestOneBadSubscriberCannotStopTheFeed:
    """The failure this design exists to survive."""

    def test_a_raising_subscriber_does_not_propagate(self, client) -> None:
        def explode(event):
            raise RuntimeError("the widget was disposed mid-flight")

        client.subscribe(explode)
        client._publish({"type": "telemetry"})  # must not raise

    def test_a_raising_subscriber_does_not_starve_the_others(self, client) -> None:
        def explode(event):
            raise RuntimeError("boom")

        seen = []
        client.subscribe(explode)
        client.subscribe(seen.append)
        client._publish({"type": "telemetry", "alt": 30.0})
        assert seen == [{"type": "telemetry", "alt": 30.0}], (
            "a broken subscriber swallowed the event for everyone else; on a real "
            "flight the numbers stop moving and that reads as a quiet aircraft"
        )

    def test_the_feed_keeps_working_after_a_subscriber_has_raised(self, client) -> None:
        calls = []

        def sometimes(event):
            calls.append(event)
            raise ValueError("still broken")

        client.subscribe(sometimes)
        client._publish({"seq": 1})
        client._publish({"seq": 2})
        assert len(calls) == 2, "delivery stopped after the first failure"


class TestTelemetrySnapshot:
    def test_get_telemetry_returns_a_snapshot_without_a_vehicle(self, client) -> None:
        # An unconnected client must answer with defaults rather than raising: the UI
        # asks for telemetry before the aircraft is up.
        telemetry = client.get_telemetry()
        assert telemetry is not None
        assert hasattr(telemetry, "waypoint_index")

    def test_the_waypoint_index_starts_at_the_plan_s_first_item(self, client) -> None:
        """Plan numbering, not the vehicle's.

        ArduPilot counts from its reserved home item at sequence 0, so its indices run
        one ahead of the plan the operator drew. The bridge reports plan numbering
        everywhere, and a default that did not match would put the progress display one
        waypoint out before the aircraft had moved.
        """
        assert client.get_telemetry().waypoint_index == 0
