"""Mission > Export <format> in the cockpit writes that format, not all six.

The native menu's per-format items (QGroundControl .plan, QGC WPL .waypoints, DJI WPML
.kmz, Litchi CSV, KML, GeoJSON) were mapped to the cockpit's generic Export verb, which
ignores payload.format: "Export DJI WPML .kmz" wrote .geojson, .kml, .kmz, .plan,
.waypoints and four Litchi CSV parts. The default window already exported the one asked
for. Driven through workspace.html in headless Chromium against the real Api; the menu
items are read from app/shell.py so a new format is covered when it is added.
"""

from __future__ import annotations

import re
from pathlib import Path

from cockpit_harness import Cockpit, make_api

AOI = [[-81.7520, 41.3030], [-81.7480, 41.3030], [-81.7480, 41.3060], [-81.7520, 41.3060]]
SHELL_PY = Path(__file__).resolve().parents[1] / "app" / "shell.py"
SUFFIX = {"qgc_plan": ".plan", "qgc_wpl": ".waypoints", "dji_wpml": ".kmz",
          "litchi": ".csv", "kml": ".kml", "geojson": ".geojson"}


def test_each_export_menu_item_writes_only_its_format(tmp_path):
    formats = re.findall(r'act\("mission\.export", \{"format": "(\w+)"\}\)',
                         SHELL_PY.read_text(encoding="utf-8"))
    assert sorted(formats) == sorted(SUFFIX), formats

    api = make_api(tmp_path)
    assert api.set_aoi(AOI)["ok"]
    assert api.plan_mission({"template": "grid", "altitude_m": 60})["ok"]
    with Cockpit(api, tmp_path / "profile") as cockpit:
        seen = cockpit.run(f"""
            const out = {{}};
            for (const format of {formats!r}) {{
              window.odk.onMenu({{ action: "mission.export", payload: {{ format }} }});
              out[format] = await ODK.waitFor(() =>
                ODK.toasts().find((t) => t.startsWith(`Exported ${{format}}:`)), 15000);
            }}
            out.workspace = ODK.$(".nav-item.active").textContent;
            return out;
        """)
    exports = [c for c in cockpit.calls if c["method"] == "export_mission"]
    assert [c["args"] for c in exports] == [[[f]] for f in formats], exports
    assert seen["workspace"] == "Mission Planning", seen
    for fmt in formats:
        toast = seen[fmt]
        assert toast, (fmt, seen)
        written = [p for p in re.split(r",\s*", toast.split(":", 1)[1]) if p.strip()]
        assert written and all(p.strip().endswith(SUFFIX[fmt]) for p in written), (fmt, toast)
        assert all(Path(p.strip()).is_file() for p in written), (fmt, toast)
