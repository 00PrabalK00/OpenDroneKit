"""The flight screen showed an aircraft that was not there.

Airframe "M350 RTK", link strong, mode AUTO, armed yes. Battery 78%, alt 60.2 m AGL,
22 satellites, HDOP 0.6, link 98%, flight time 06:12. A wind gust of 9.2 m/s at 06:02.
An event log reading armed, mission started, waypoint 31 reached, capture 62 triggered.
All of it written into the source, all of it rendered with nothing connected.

**The preflight checklist was the worst control in the cockpit.** Nine items, eight
ticked -- autopilot healthy, GPS fix RTK fixed, home position set, battery 96%, mission
uploaded, geofence uploaded -- checked against nothing. A checklist exists to be
believed. A pre-ticked one is worse than no checklist, because an operator who would have
checked now has a reason not to. It also displayed "Storage 0 GB free" with a tick.

Every item on it turns out to be derivable from telemetry the vehicle already reports, so
it is now an actual preflight check rather than a picture of one.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js"


@pytest.fixture(scope="module")
def source() -> str:
    return WORKSPACES.read_text(encoding="utf-8")


def strip_comments(js: str) -> str:
    """Remove /* */ blocks and // lines.

    The line-prefix version used elsewhere in this suite only catches lines beginning
    with `*`, and the block comments in workspaces.js indent their continuation lines
    with plain text. So "Storage 0 GB free" survived in a comment *about* having removed
    it, and the guard passed on the prose describing the defect. That is the second time
    a guard in this suite has been satisfied by its own explanation.
    """
    out = []
    i = 0
    while i < len(js):
        if js.startswith("/*", i):
            end = js.find("*/", i + 2)
            i = len(js) if end == -1 else end + 2
            continue
        if js.startswith("//", i):
            end = js.find("\n", i)
            i = len(js) if end == -1 else end
            continue
        out.append(js[i])
        i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def flight(source) -> str:
    """The flight workspace, comments stripped."""
    block = source.split("const flight = {")[1].split("\nconst verification")[0]
    return strip_comments(block)


FABRICATED = [
    ("M350 RTK", "an airframe that is not connected"),
    ("78%", "a battery reading from no aircraft"),
    ("60.2", "an altitude from no aircraft"),
    ("512.6", "an altitude from no aircraft"),
    ("24.1 V", "a pack voltage from no aircraft"),
    ("06:12", "a flight time for a flight that did not happen"),
    ("9.2 m/s", "a wind reading from no anemometer"),
    ("waypoint 31", "an event from a flight that did not happen"),
    ("capture 62", "an event from a flight that did not happen"),
    ("Battery 96%", "a preflight check that passed against nothing"),
    ("Storage 0 GB free", "a check that passed while reporting no free storage"),
]


@pytest.mark.parametrize("needle,why", FABRICATED, ids=[n for n, _ in FABRICATED])
def test_the_flight_screen_invents_nothing(flight, needle, why) -> None:
    assert needle not in flight, f"{needle} is still rendered -- {why}"


class TestThePreflightChecklistActuallyChecks:
    def test_it_reads_the_aircraft(self, flight) -> None:
        block = flight.split('title: "Preflight"')[1][:2000]
        assert "live({" in block
        assert '"telemetry"' in block

    def test_an_unknown_check_is_not_a_passed_check(self, flight) -> None:
        """The rule the whole project runs on, at the point it matters most.

        Weather cannot be derived from telemetry. Dropping the item would quietly
        shorten the checklist; ticking it would be a lie. It is shown as unknown.
        """
        block = flight.split('title: "Preflight"')[1][:2200]
        assert "null" in block, "there is no third state between pass and fail"
        assert "unknown" in block

    def test_it_still_asks_about_weather(self, flight) -> None:
        block = flight.split('title: "Preflight"')[1][:2200]
        assert "Weather" in block, "an item that cannot be automated was dropped instead"

    def test_no_check_is_hardcoded_to_pass(self, flight) -> None:
        block = flight.split('title: "Preflight"')[1][:2200]
        # Every item is a [label, expression] pair; a literal `true` would be a tick
        # that no telemetry can clear.
        assert ", true]" not in block


class TestASimulatedAircraftSaysSo:
    def test_the_aircraft_panel_marks_simulation(self, flight) -> None:
        """SITL and the mock driver report telemetry that is shaped exactly like a real
        vehicle's. On the screen that arms the aircraft, the difference has to be
        visible."""
        block = flight.split('title: "Aircraft"')[1][:1400]
        assert "is_simulated" in block
        assert "SIMULATED" in block


class TestTheFieldsExist:
    """Every telemetry field the flight screen renders must be one DroneTelemetry has.

    This class of error has now happened three times in this cockpit -- `gsd` to the
    planner, `capture_count` on the estimates, `severity` / `message` / `created_at` on
    the notifications, and `entries` on the audit log, which really returns `events`. A
    plausible field name renders as blank or `undefined`. It never raises, so nothing
    catches it but a test like this one.
    """

    def test_telemetry_fields_are_declared(self, flight) -> None:
        drone = (ROOT / "core" / "drone.py").read_text(encoding="utf-8")
        declared = drone.split("class DroneTelemetry")[1].split("@property")[0]
        properties = set(re.findall(r"def (\w+)", drone.split("class DroneTelemetry")[1][:2500]))

        # session.telemetry() adds these to the dataclass's own fields before returning:
        # which driver produced the reading, and whether the vehicle is real. They are
        # part of the payload the panel receives even though DroneTelemetry has no such
        # attributes.
        session = (ROOT / "app" / "session.py").read_text(encoding="utf-8")
        added = set(re.findall(r'payload\["(\w+)"\]\s*=', session))
        assert {"driver", "is_simulated"} <= added, (
            "session.telemetry() no longer adds the fields the aircraft panel reads"
        )

        used = set(re.findall(r"\btm\.(\w+)", flight))
        missing = sorted(
            f for f in used
            if f"{f}:" not in declared and f not in properties and f not in added
        )
        assert not missing, f"nothing produces these telemetry fields: {missing}"

    def test_the_audit_log_is_read_by_its_real_key(self, flight) -> None:
        api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
        assert "return ok(events=" in api.split("def audit_log")[1][:300]
        block = flight.split('"audit_log"')[1][:900]
        assert "a.events" in block, "the event log reads a key audit_log does not return"

    def test_notification_fields_are_the_real_ones(self, flight) -> None:
        notif = (ROOT / "core" / "notifications.py").read_text(encoding="utf-8")
        shape = notif.split("def to_dict")[1][:600]
        block = flight.split('"list_notifications"')[1][:900]
        for field in set(re.findall(r"\bitem\.(\w+)", block)):
            assert f'"{field}"' in shape, f"Notification.to_dict() has no {field}"
