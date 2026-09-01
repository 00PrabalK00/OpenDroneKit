"""A panel called Layers, in a mapping application, with no layer control in it.

`layerTree()` rendered six fixed rows -- Basemap, Terrain (SRTM), Orthomosaic, DSM,
Mission, Geofence -- on every project regardless of what it held. A project with no
orthomosaic still showed "Orthomosaic", so the panel whose job is to say which products
exist said the same thing for all of them. Three workspaces then appended their own rows:
Obstacles, No-fly regions, Defects, Thermal, Change, Volumes, Slope map. Labels for layers
that were never registered.

The worse half is the one that took longest to notice. **There was no toggle.**
`set_layer_visible` and `set_layer_opacity` have been on the Api the whole time, both
implemented, and nothing anywhere in the interface ever called either one.

That is this project's recurring shape in its plainest form: the feature exists, it works,
its tests pass, and there is nothing a person can press that arrives at it. It is the same
finding as the special mission types, the report templates, and the mission form -- and it
is why the rule this codebase now runs on is that a feature is not done when its function
exists, but when something a person can press reaches it.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_JS = ROOT / "app" / "web" / "js" / "workspace"


def strip_comments(js: str) -> str:
    out, i = [], 0
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
def workspaces() -> str:
    return strip_comments((WORKSPACE_JS / "workspaces.js").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def layer_tree(workspaces) -> str:
    return workspaces.split("const layerTree")[1].split("\n});")[0]


class TestTheToggleReachesTheApplication:
    def test_visibility_calls_the_api(self, layer_tree) -> None:
        assert "set_layer_visible" in layer_tree, (
            "the Layers panel still cannot change a layer's visibility"
        )

    def test_the_control_reverts_when_the_call_fails(self, layer_tree) -> None:
        """A checkbox showing a state the application did not accept is precisely how
        the rest of this cockpit went wrong."""
        assert "box.checked = !box.checked" in layer_tree

    def test_the_api_really_has_the_method(self) -> None:
        api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
        assert "def set_layer_visible" in api

    def test_tryCall_is_imported_where_it_is_used(self, workspaces) -> None:
        """It was not. The panel would have thrown a ReferenceError on first render."""
        assert 'from "./api.js"' in workspaces
        assert "tryCall" in workspaces.split('from "./api.js"')[0]


class TestItListsTheProjectsOwnLayers:
    def test_it_asks_for_them(self, layer_tree) -> None:
        assert 'calls: ["list_layers"]' in layer_tree

    def test_no_layer_name_is_written_into_the_source(self, workspaces) -> None:
        invented = [
            "Basemap — Satellite", "Terrain (SRTM)", "No-fly regions",
            "Slope map", "l-ortho", "l-dsm", "l-fence", "l-nofly", "l-vol",
        ]
        for name in invented:
            assert name not in workspaces, f"{name} is still a hardcoded layer row"

    def test_a_layer_without_a_crs_says_so(self, layer_tree) -> None:
        """register_raster_layer() adds rasters with no CRS and flags them, precisely so
        the UI can explain why they cannot be placed rather than dropping them."""
        assert "crs_epsg" in layer_tree
        assert "no CRS" in layer_tree

    def test_the_fields_it_reads_are_real(self, layer_tree) -> None:
        session = (ROOT / "app" / "session.py").read_text(encoding="utf-8")
        # Splitting at the first blank line stops inside the docstring, before any
        # field -- the same slip this suite made on the Job dataclass.
        declared = session.split("class MapLayer")[1].split("\nclass ")[0]
        assert "kind:" in declared, "this is not the MapLayer dataclass"
        for field in set(re.findall(r"\blayer\.(\w+)", layer_tree)):
            assert f"{field}:" in declared, f"MapLayer has no {field}"


class TestNoWorkspaceAppendsDecorativeLayers:
    def test_call_sites_pass_a_kind_filter_or_nothing(self, workspaces) -> None:
        """`extra` let a workspace append rows for layers that were never registered.
        A filter over real layers is the honest version of the same intent."""
        for call in re.findall(r"layerTree\(([^)]*)\)", workspaces):
            argument = call.strip()
            assert argument == "" or argument.startswith("["), (
                f"layerTree called with something other than a kind filter: {argument}"
            )
            if argument.startswith("["):
                assert '{' not in argument, "a call site is still passing literal rows"
