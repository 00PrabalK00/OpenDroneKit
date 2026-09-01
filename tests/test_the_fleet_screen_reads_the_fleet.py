"""The Fleet screen showed a fleet that did not exist, because nothing could list one.

Four aircraft (M350-01 "flying", M350-02, M300-01 "service", Mavic-3E-01), two named
pilots with flight hours, three batteries with charge, cycle counts, health and cell
temperatures, and two maintenance records. None of it existed.

This one was not only a wiring mistake. `fleet_status()` returns *counts* -- four
aircraft, three batteries, two overdue -- which is what a dashboard tile needs and not
what this screen needs. There was no call that could name an aircraft. The honest path
did not exist, so the screen was drawn from constants, which is the same shape as the
canvas overlays before `canvas()` could carry an element.

So `list_fleet()` is a new capability rather than a rewiring, and these tests exercise it
against a real database.

Two things it deliberately does not do. It does not report a battery's charge or
temperature: those come off the pack over a link this build does not have, and inventing
a temperature for a lithium pack is not a neutral placeholder. And `hours_to_service` is
None when no interval is configured, which is not the same as "not due" and must never
render as a comfortable number.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import re

import pytest

pytest.importorskip("sqlalchemy", reason="the service database layer needs SQLAlchemy")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def database():
    path = Path(tempfile.mkdtemp()) / "fleet_screen.db"
    previous = os.environ.get("ODK_DATABASE_URL")
    os.environ["ODK_DATABASE_URL"] = f"sqlite:///{path.as_posix()}"

    import services.api.db as db_module

    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001
    db_module.init_db()
    yield db_module

    if previous is None:
        os.environ.pop("ODK_DATABASE_URL", None)
    else:
        os.environ["ODK_DATABASE_URL"] = previous
    db_module._engine = None          # noqa: SLF001
    db_module._SessionLocal = None    # noqa: SLF001


@pytest.fixture(scope="module")
def org(database):
    from services.api.models import Organization

    with database.get_session_factory()() as session:
        record = Organization(name="Fleet org", slug="fleet-org")
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


@pytest.fixture(scope="module")
def stocked(org):
    """One aircraft with a service interval, one pack past its cycle limit, one pilot."""
    from app import desktop_ops
    from services.api.models import Aircraft
    from services.api.db import get_session_factory

    craft = desktop_ops.add_aircraft(org, "ODK-01", "Matrice 350", "SN-AAA")
    desktop_ops.add_battery(org, "B-01", capacity_mah=5880, cycle_limit=200)
    desktop_ops.add_battery(org, "B-02", capacity_mah=5880, cycle_limit=100)
    desktop_ops.add_pilot(org, "Test Pilot", "LIC-1")

    with get_session_factory()() as session:
        record = session.get(Aircraft, craft["id"])
        record.flight_hours = 96.0
        record.service_interval_hours = 100.0
        record.hours_at_last_service = 0.0
        session.commit()

    # Push one pack past its rated cycles without retiring it: in service and out of
    # spec is the state worth surfacing, and the one an invented table never shows.
    from services.api.models import Battery
    from sqlalchemy import select

    with get_session_factory()() as session:
        pack = session.scalars(select(Battery).where(Battery.serial_number == "B-02")).one()
        pack.cycle_count = 140
        session.commit()

    return org


class TestTheFleetCanBeListedAtAll:
    def test_it_names_the_aircraft(self, stocked) -> None:
        from app import desktop_ops

        fleet = desktop_ops.list_fleet(stocked)
        assert [a["name"] for a in fleet["aircraft"]] == ["ODK-01"]
        assert fleet["aircraft"][0]["serial_number"] == "SN-AAA"

    def test_it_names_the_batteries_and_the_pilots(self, stocked) -> None:
        from app import desktop_ops

        fleet = desktop_ops.list_fleet(stocked)
        assert {b["serial_number"] for b in fleet["batteries"]} == {"B-01", "B-02"}
        assert [p["display_name"] for p in fleet["pilots"]] == ["Test Pilot"]

    def test_fleet_status_still_only_counts(self, stocked) -> None:
        """The two calls answer different questions and both are wanted."""
        from app import desktop_ops

        assert desktop_ops.fleet_status(stocked)["aircraft"] == 1


class TestTheNumbersThatDecideWhetherItFlies:
    def test_hours_to_service_counts_down(self, stocked) -> None:
        from app import desktop_ops

        craft = desktop_ops.list_fleet(stocked)["aircraft"][0]
        # 100 h interval, 96 h flown since the last service.
        assert craft["hours_to_service"] == pytest.approx(4.0)

    def test_no_interval_is_not_the_same_as_not_due(self, stocked) -> None:
        """The distinction the UI has to keep.

        An aircraft with no service interval configured has an unknown service state.
        Reporting a number there would say "not due" about an airframe nobody has set a
        schedule for.

        The column is NOT NULL, so "no schedule" is stored as 0 rather than NULL -- which
        is exactly why this must be handled deliberately: 0 is a number, and arithmetic
        on it produces a plausible, wrong answer instead of an error.
        """
        from app import desktop_ops
        from services.api.db import get_session_factory
        from services.api.models import Aircraft
        from sqlalchemy import select

        with get_session_factory()() as session:
            record = session.scalars(select(Aircraft).where(Aircraft.name == "ODK-01")).one()
            kept = record.service_interval_hours
            record.service_interval_hours = 0.0
            session.commit()
        try:
            craft = desktop_ops.list_fleet(stocked)["aircraft"][0]
            assert craft["hours_to_service"] is None
        finally:
            with get_session_factory()() as session:
                record = session.scalars(select(Aircraft).where(Aircraft.name == "ODK-01")).one()
                record.service_interval_hours = kept
                session.commit()

    def test_a_pack_in_service_past_its_cycle_limit_is_flagged(self, stocked) -> None:
        """Retired is a decision someone made. Past its limit and still in service is
        the state that puts the pack in an aircraft when it should be on a shelf."""
        from app import desktop_ops

        packs = {b["serial_number"]: b for b in desktop_ops.list_fleet(stocked)["batteries"]}
        assert packs["B-02"]["over_cycle_limit"] is True
        assert packs["B-02"]["retired"] is False
        assert packs["B-01"]["over_cycle_limit"] is False


class TestTheScreenRendersOnlyWhatIsRecorded:
    @pytest.fixture(scope="class")
    def screen(self) -> str:
        source = (REPO_ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js").read_text(
            encoding="utf-8")
        block = source.split("const fleet = {")[1].split("\nconst reports")[0]
        out, i = [], 0
        while i < len(block):
            if block.startswith("/*", i):
                end = block.find("*/", i + 2)
                i = len(block) if end == -1 else end + 2
                continue
            out.append(block[i])
            i += 1
        return "".join(out)

    @pytest.mark.parametrize("needle", [
        "M350-01", "M300-01", "Mavic-3E-01", "A. Sharma", "R. Iyer",
        "1ZNBJ9", "09.01.0034", "142.6", "B-05", "B-07", "31 ", "propeller set replaced",
    ])
    def test_the_invented_fleet_is_gone(self, screen, needle) -> None:
        assert needle not in screen

    def test_charge_and_temperature_are_not_shown(self, screen) -> None:
        """Nothing in this build reads either from a pack."""
        assert "Charge" not in screen
        assert "Temp" not in screen

    def test_every_field_it_renders_is_one_list_fleet_returns(self, screen) -> None:
        ops = (REPO_ROOT / "app" / "desktop_ops.py").read_text(encoding="utf-8")
        shape = ops.split("def list_fleet")[1].split("\ndef ")[0]
        for prefix in ("a", "b", "p", "m"):
            for field in set(re.findall(rf"\b{prefix}\.(\w+)\b", screen)):
                if field in {"id", "map", "filter", "find", "length", "slice"}:
                    continue
                assert f'"{field}"' in shape, f"list_fleet() does not return {field}"
