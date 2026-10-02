"""Every mission template, uploaded through the Api and read back from ArduCopter.

For each template the app offers (the planner's own list plus the special mission
types), plan over an Aukerman-like site beside the SITL home, upload with
``Api.upload_mission`` exactly as the UIs do, then download the mission back from the
autopilot with MISSION_REQUEST_LIST and check it twice:

* item by item against what the app sent -- command, frame, coordinates, altitude, and
  every parameter ArduPilot stores for that command (yaw, gimbal, hold time, shutter,
  speed);
* viewpoint by viewpoint against the *plan*, by replaying ArduPilot's mission semantics
  over the stored items: the gimbal pitch, yaw, dwell and shutter the aircraft will
  actually have at each planned viewpoint.

The fence the Api uploaded beside the mission is downloaded and compared too.

Opt-in: ``pytest -m sitl`` (see tools/sitl/run_docker.py).
"""

from __future__ import annotations

import json
import math

import pytest

from ._support import (
    LATLON_TOL_DEG,
    all_templates,
    connect_api,
    evidence_dir,
    item_mismatches,
    make_api,
    plan_mismatches,
    plan_template,
    reconstruct,
)

pytestmark = pytest.mark.sitl


@pytest.fixture(scope="module")
def template_api(tmp_path_factory, sitl_connection):
    api = make_api(tmp_path_factory.mktemp("templates"))
    connect_api(api, sitl_connection)
    try:
        yield api
    finally:
        api.disconnect_vehicle()


def _record(template: str, payload: dict) -> None:
    path = evidence_dir("templates") / f"{template}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")


@pytest.mark.parametrize("template", all_templates())
def test_template_uploads_and_reads_back_as_planned(template_api, template):
    from mission import exporters

    api = template_api
    planned = plan_template(api, template)
    assert planned["ok"], f"{template} could not be planned: {planned.get('error')}"
    plan = api._session.mission_plan_dict
    sent = [item.to_dict() for item in exporters.build_mission_items(plan)]
    fence_sent = [item.to_dict() for item in exporters.build_fence_items(plan)]

    upload = api.upload_mission()
    record = {"template": template, "items_sent": len(sent), "fence_sent": len(fence_sent),
              "upload": {k: upload.get(k) for k in ("ok", "error", "items", "upload")}}
    assert upload["ok"], f"{template}: upload refused: {upload.get('error')} {upload.get('upload')}"
    assert upload["upload"]["mission"]["success"] is True

    client = api._session._drone_client
    received = client.download_mission(timeout_s=30.0)
    item_problems = item_mismatches(sent, received)
    plan_problems = plan_mismatches(plan, received)
    vehicle = reconstruct(received)
    record.update({
        "items_on_vehicle": len(received),
        "viewpoints_on_vehicle": len(vehicle.viewpoints),
        "shutters_on_vehicle": sum(1 for v in vehicle.viewpoints if v.trigger),
        "yaw_locked_on_vehicle": sum(1 for v in vehicle.viewpoints if v.yaw_deg is not None),
        "dwell_on_vehicle_s": sorted({v.dwell_s for v in vehicle.viewpoints}),
        "gimbal_pitches_on_vehicle": sorted({round(v.gimbal[0], 2) for v in vehicle.viewpoints
                                             if v.gimbal is not None}),
        "item_mismatches": item_problems[:50],
        "plan_mismatches": plan_problems[:50],
        "first_items_sent": sent[:8],
        "first_items_on_vehicle": received[:8],
    })

    fence_problems: list[str] = []
    if fence_sent:
        assert upload["upload"]["fence"]["success"] is True, upload["upload"]["fence"]
        fence_back = client.download_geofence(timeout_s=30.0)
        record["fence_on_vehicle"] = len(fence_back)
        if len(fence_back) != len(fence_sent):
            fence_problems.append(f"fence: sent {len(fence_sent)}, vehicle holds {len(fence_back)}")
        for a, b in zip(fence_sent, fence_back):
            if (int(a["command"]) != int(b["command"])
                    or abs(float(a["param1"]) - float(b["param1"])) > 1e-6
                    or abs(a["lat"] - b["lat"]) > LATLON_TOL_DEG
                    or abs(a["lon"] - b["lon"]) > LATLON_TOL_DEG):
                fence_problems.append(f"fence vertex {a['seq']}: {a} -> {b}")
        record["fence_mismatches"] = fence_problems
    _record(template, record)

    assert not item_problems, f"{template}: vehicle stored something else:\n" + "\n".join(item_problems[:20])
    assert not plan_problems, f"{template}: vehicle will not fly the plan:\n" + "\n".join(plan_problems[:20])
    assert not fence_problems, "\n".join(fence_problems[:20])
    assert all(math.isfinite(float(v.alt)) for v in vehicle.viewpoints)
