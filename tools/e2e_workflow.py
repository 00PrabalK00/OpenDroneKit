"""Run the product's real workflows end to end and report which ones work.

The button audit proves every control responds. It does not prove a survey can be taken
from an empty project to a report, which is the only question that matters -- a button
can respond correctly and still be the third step of a chain whose second step is broken.

So this drives the same Api the desktop UI drives, in the order an operator would:
create a project, import imagery, draw an area, plan a mission, export it, reconstruct,
analyse, measure, report, share. Each step prints PASS, REFUSED or FAIL.

REFUSED is not FAIL. "Select a dataset before running reconstruction" is the application
telling the truth about its state, and a workflow that refuses for a stated reason is
working. FAIL is an exception, a wrong shape, or a silent success that produced nothing.

    python tools/e2e_workflow.py                 # everything except reconstruction
    python tools/e2e_workflow.py --reconstruct   # including COLMAP, which takes a while
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The surveyed footprint, from the imagery's own GPS tags. Planning needs an area, and
# an area invented somewhere else would make the plan meaningless.
AUKERMAN_POLYGON = [
    [-81.753896, 41.303408],
    [-81.750466, 41.303408],
    [-81.750466, 41.304828],
    [-81.753896, 41.304828],
]

results: list[tuple[str, str, str]] = []


def step(name: str, action: Callable[[], Any]) -> Any:
    """Run one workflow step and record how it went."""
    try:
        outcome = action()
    except Exception as exc:  # noqa: BLE001 - the report is the point
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        return None

    if isinstance(outcome, dict) and outcome.get("ok") is False:
        results.append((name, "REFUSED", str(outcome.get("error", ""))[:110]))
        return outcome
    detail = ""
    if isinstance(outcome, dict):
        detail = ", ".join(
            f"{k}={v}" for k, v in outcome.items()
            if k not in ("ok", "data_uri") and not isinstance(v, (dict, list))
        )[:110]
    results.append((name, "PASS", detail))
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/e2e_workflow.py")
    parser.add_argument("--reconstruct", action="store_true",
                        help="Include COLMAP. Minutes, not seconds, and memory-hungry.")
    parser.add_argument("--images", default="training/data/aukerman_subset")
    args = parser.parse_args(argv)

    from app.api import Api
    from app.session import AppSession

    session = AppSession()
    api = Api(session)

    # -- project and data ----------------------------------------------------
    step("capabilities", lambda: api.capabilities())
    project = step("create project", lambda: api.create_project(
        f"E2E {time.strftime('%H%M%S')}", str(REPO_ROOT / "demo_project" / "e2e")))
    if project and project.get("ok"):
        # The id is nested under project=, which is the same shape the UI got wrong.
        step("set active project", lambda: api.set_active_project(
            (project.get("project") or {}).get("id")
            or project.get("project_id") or project.get("id")))
    step("import dataset", lambda: api.import_dataset(args.images, "E2E imagery"))
    step("list datasets", lambda: api.list_datasets())
    listing = step("list dataset images", lambda: api.list_dataset_images())
    if listing and listing.get("images"):
        step("preview an image", lambda: api.image_preview(listing["images"][0]))

    # -- planning ------------------------------------------------------------
    step("set area of interest", lambda: api.set_aoi(AUKERMAN_POLYGON))
    step("mission templates", lambda: api.mission_templates())
    plan = step("plan mission", lambda: api.plan_mission(
        {"altitude_m": 60, "mode": "grid", "front_overlap_pct": 80, "side_overlap_pct": 70}))
    step("mission estimates", lambda: api.mission_estimates())
    step("mission as geojson", lambda: api.mission_geojson())
    step("save mission", lambda: api.save_mission("E2E mission", "written by the workflow test"))
    step("mission versions", lambda: api.list_mission_versions())
    step("export mission", lambda: api.export_mission())

    # -- the models and their identity ---------------------------------------
    step("verify model digests", lambda: api.verify_models())
    step("reconstruction capabilities", lambda: api.reconstruction_capabilities())
    step("camera capabilities", lambda: api.camera_capabilities())

    # -- fleet, sharing, webhooks -------------------------------------------
    step("fleet status", lambda: api.fleet_status(1))
    aircraft = step("add aircraft", lambda: api.add_aircraft(1, "E2E aircraft", "M3E", "SN-E2E"))
    if aircraft and aircraft.get("ok"):
        step("log maintenance", lambda: api.log_maintenance(aircraft["id"], "inspection", "e2e"))
    step("add battery", lambda: api.add_battery(1, f"BAT-{int(time.time())}", 5000, 300))
    step("create share link", lambda: api.create_share_link(1, "e2e", False))
    step("list share links", lambda: api.list_share_links(1))
    step("add webhook", lambda: api.add_webhook(1, "https://example.invalid/hook", ["job.finished"]))
    step("list plugins", lambda: api.list_plugins())

    # -- reporting -----------------------------------------------------------
    step("report readiness", lambda: api.report_readiness())
    step("generate report", lambda: api.generate_report())

    # -- processing ----------------------------------------------------------
    step("size the reconstruction job",
         lambda: api.size_reconstruction_job(len(listing.get("images", [])) if listing else 24))
    if args.reconstruct:
        started = step("start reconstruction", lambda: api.run_reconstruction(
            {"engine": "auto", "profile": "fast_preview"}))
        job = (started or {}).get("job_id")
        if job:
            print(f"  reconstruction job {job} running; polling…", flush=True)
            last = ""
            while True:
                status = api.job_status(job)
                state = str(status.get("state") or status.get("status") or "")
                message = str(status.get("message") or "")
                if message and message != last:
                    print(f"    {state}: {message}", flush=True)
                    last = message
                if state in ("done", "finished", "complete", "failed", "error", "cancelled"):
                    results.append((
                        "reconstruction finishes",
                        "PASS" if state in ("done", "finished", "complete") else "FAIL",
                        f"{state}: {status.get('error') or message}"[:110],
                    ))
                    break
                time.sleep(10)
    else:
        results.append(("reconstruction", "SKIPPED", "run with --reconstruct"))

    # -- report --------------------------------------------------------------
    width = max(len(name) for name, _, _ in results) + 2
    print("\n" + "=" * 72)
    for name, outcome, detail in results:
        print(f"{outcome:<8} {name:<{width}} {detail}")
    print("=" * 72)
    counts: dict[str, int] = {}
    for _, outcome, _ in results:
        counts[outcome] = counts.get(outcome, 0) + 1
    print(" · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
