"""What the application says it can plan must be what it can plan.

app/api.py carried a hand-written list of fifteen mission templates. The planner accepted
twenty-two. So wind turbine, dome, box, closed loop, multi-facade and smart adaptive were
all fully implemented, all reachable by name, and absent from the application's own
inventory of what it could do.

Nobody chooses a mission type they cannot see, which makes an unadvertised template
indistinguishable from one that was never built. This is the third time in one session
that working code turned out to be unreachable through the surface a user touches --
after pylon/thermal/multispectral, and after the cockpit panels.

So the list is derived from the alias table rather than written twice, and these tests
hold the two together.
"""

from __future__ import annotations

import pytest

from app.api import Api
from app.session import AppSession
from mission.planner import TEMPLATE_ALIASES, _normalize_template, available_templates

# Present before this fix, and required to stay. A template that silently disappears
# from the menu is the same failure in the other direction.
PREVIOUSLY_ADVERTISED = {
    "grid", "double_grid", "corridor", "facade", "tower_mapping", "solar_inspection",
    "orbit", "panorama", "bubble_360", "waypoints", "linear_inspection", "lateral_capture",
    "roof_inspection", "magnetic_mapping", "linked_mission",
}

# Implemented and unadvertised until this fix.
WERE_INVISIBLE = {
    "wind_turbine", "dome_inspection", "box_inspection",
    "closed_loop", "multi_facade", "smart_adaptive",
}


@pytest.fixture(scope="module")
def advertised() -> list[str]:
    return Api(AppSession()).mission_templates()["templates"]


def test_nothing_that_used_to_be_offered_has_gone(advertised) -> None:
    missing = PREVIOUSLY_ADVERTISED - set(advertised)
    assert not missing, f"templates dropped from the menu: {sorted(missing)}"


def test_the_ones_that_were_invisible_are_offered(advertised) -> None:
    missing = WERE_INVISIBLE - set(advertised)
    assert not missing, f"still implemented and unreachable from the menu: {sorted(missing)}"


def test_the_menu_is_exactly_what_the_planner_accepts(advertised) -> None:
    """The invariant, rather than a snapshot of today's count.

    Derived from the alias table, so adding an alias adds the template to the menu and
    there is no second list to keep in step.
    """
    assert sorted(advertised) == available_templates()
    assert set(advertised) == set(TEMPLATE_ALIASES.values())


@pytest.mark.parametrize("template", sorted(set(TEMPLATE_ALIASES.values())))
def test_every_advertised_template_survives_normalisation(template) -> None:
    """A canonical name must resolve to itself.

    _normalize_template falls back to "grid" for anything it does not know, so a template
    offered in the menu but absent from the table would quietly plan a lawnmower grid
    instead -- the worst outcome available, because it produces a plausible mission that
    is not the one that was asked for.
    """
    assert _normalize_template(template) == template


def test_an_unknown_template_still_falls_back_to_grid() -> None:
    """Deriving the menu must not change what happens to a name nobody recognises."""
    assert _normalize_template("not_a_real_template") == "grid"
    assert _normalize_template("") == "grid"


def test_the_api_does_not_carry_its_own_copy_of_the_list() -> None:
    """The hand-written list is how the two drifted apart in the first place."""
    # Read the file rather than inspect.getsource: @guard wraps the method, so getsource
    # returns the decorator's wrapper and the assertion passes on the wrong text.
    from pathlib import Path

    api_source = (Path(__file__).resolve().parents[1] / "app" / "api.py").read_text(encoding="utf-8")
    body = api_source.split("def mission_templates")[1].split("\n    @guard")[0]
    assert "available_templates" in body
    assert '"tower_mapping"' not in body, "the API is listing templates by hand again"
