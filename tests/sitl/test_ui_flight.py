"""A user's flight, clicked through each UI, flown by real ArduCopter SITL.

The other SITL tests drive ``app.api.Api`` directly. This one drives the page: the real
UI (cockpit ``workspace.html`` and the classic ``index.html``) served against the real
Api the way tests/ui_live_bridge.py does, in headless Edge/Chromium at 1920x1080. It
plans the Aukerman Park grid, connects over pymavlink, runs preflight, uploads, starts,
pauses, resumes and returns to launch with the mouse -- CDP mouse events at each
button's centre, after checking the button is the element under that point -- and
every claim about the aircraft is read from ArduPilot's own SERIAL1, not from the app.

The whole flight is recorded (tests/ui_recorder.py): a video at the speed it happened
and a contact sheet, written to ODK_UI_FLIGHT_OUT (else the test's tmp_path).

Running it needs a browser and a SITL whose home is Aukerman Park. From Windows, with
Docker and the image built from infrastructure/docker/Dockerfile.sitl:

    set ODK_SITL_DOCKER_IMAGE=opendronekit-sitl:local
    set ODK_SITL_HOME=41.3041,-81.7521,300,0
    set ODK_SITL_SPEEDUP=1
    python -m pytest -m sitl tests/sitl/test_ui_flight.py

Inside the SITL container there is no browser, so it skips there and says why.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.sitl

pytest.importorskip("websocket")
pytest.importorskip("requests")

from tools.sitl import launch as sitl_launch  # noqa: E402

from ._support import MavlinkTrace, distance_m, fresh_battery  # noqa: E402

# The observer also reads ArduPilot's copy of the mission, and its landed state.
MavlinkTrace.KEEP = MavlinkTrace.KEEP | {"MISSION_ITEM_INT", "EXTENDED_SYS_STATE"}

# The surveyed footprint of the OpenDroneMap Aukerman dataset, from its images' GPS tags
# (tools/e2e_workflow.py). The SITL home must be inside it: the launch point is in the
# park, as it was for the survey.
from tools.e2e_workflow import AUKERMAN_POLYGON  # noqa: E402

COPTER_MODES = {0: "STABILIZE", 3: "AUTO", 4: "GUIDED", 5: "LOITER", 6: "RTL", 9: "LAND",
                17: "BRAKE"}
MAV_TYPE_GCS = 6
ARMED_FLAG = 0x80

HELPERS = """
window.__toasts = window.__toasts || [];
if (!window.__watching) {
  window.__watching = new MutationObserver((mutations) => mutations.forEach((m) =>
    m.addedNodes.forEach((n) => {
      if (n.classList && n.classList.contains('toast')) window.__toasts.push(n.textContent);
    })));
  window.__watching.observe(document.body, { childList: true, subtree: true });
}
true
"""

STEP_PAUSE_S = float(os.environ.get("ODK_UI_STEP_PAUSE_S", "1.5"))
# Screenshots per second at most. Each is a full 1920x1080 software-rendered frame; the
# cockpit's renderer stopped answering under 10/s on a machine short of memory.
RECORD_FPS = float(os.environ.get("ODK_UI_RECORD_FPS", "4"))


def _home_in_aukerman() -> bool:
    lons = [p[0] for p in AUKERMAN_POLYGON]
    lats = [p[1] for p in AUKERMAN_POLYGON]
    return (min(lons) < sitl_launch.SITL_HOME_LON < max(lons)
            and min(lats) < sitl_launch.SITL_HOME_LAT < max(lats))


# ── The independent view of the aircraft (ArduPilot SERIAL1) ─────────────────

class Vehicle:
    """What ArduPilot says about itself, from the observer trace."""

    def __init__(self, observer, trace: MavlinkTrace) -> None:
        self.observer = observer
        self.trace = trace

    def heartbeat(self) -> dict[str, Any] | None:
        beats = [e for e in self.trace.messages("HEARTBEAT")
                 if int(e.data.get("type", 0)) != MAV_TYPE_GCS]
        return beats[-1].data if beats else None

    def mode(self) -> str:
        beat = self.heartbeat() or {}
        return COPTER_MODES.get(int(beat.get("custom_mode", -1)), str(beat.get("custom_mode")))

    def armed(self) -> bool:
        return bool(int((self.heartbeat() or {}).get("base_mode", 0)) & ARMED_FLAG)

    def _last(self, kind: str) -> dict[str, Any]:
        entries = self.trace.messages(kind)
        return entries[-1].data if entries else {}

    def altitude_m(self) -> float:
        return float(self._last("GLOBAL_POSITION_INT").get("relative_alt", 0)) / 1000.0

    def groundspeed(self) -> float:
        return float(self._last("VFR_HUD").get("groundspeed", 0.0))

    def position(self) -> tuple[float, float]:
        data = self._last("GLOBAL_POSITION_INT")
        return float(data.get("lat", 0)) / 1e7, float(data.get("lon", 0)) / 1e7

    def mission_seq(self) -> int:
        return int(self._last("MISSION_CURRENT").get("seq", -1))

    def state(self) -> dict[str, Any]:
        lat, lon = self.position()
        return {"mode": self.mode(), "armed": self.armed(), "alt_m": round(self.altitude_m(), 1),
                "groundspeed": round(self.groundspeed(), 2), "mission_seq": self.mission_seq(),
                "lat": round(lat, 7), "lon": round(lon, 7)}

    def wait(self, description: str, predicate, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate(self):
                return self.state()
            time.sleep(0.2)
        pytest.fail(f"ArduPilot did not report {description} within {timeout_s:.0f}s; "
                    f"it reports {self.state()}")

    def download_mission(self, timeout_s: float = 300.0) -> list[dict[str, Any]]:
        """The mission ArduPilot holds, read over SERIAL1 -- not through the app."""
        from core.mission_planner_bridge import (
            _NON_POSITIONAL_COMMANDS, _planner_frame, _without_home_slot)

        conn = self.observer
        deadline = time.monotonic() + timeout_s
        count = None
        while count is None and time.monotonic() < deadline:
            mark = self.trace.mark()
            conn.mav.mission_request_list_send(conn.target_system, conn.target_component, 0)
            count = self.trace.wait_for("MISSION_COUNT",
                                        lambda d: int(d.get("mission_type", 0)) == 0, mark, 3.0)
        assert count is not None, "ArduPilot sent no MISSION_COUNT to the observer"
        items = []
        for seq in range(int(count.data["count"])):
            entry = None
            while entry is None and time.monotonic() < deadline:
                mark = self.trace.mark()
                conn.mav.mission_request_int_send(conn.target_system, conn.target_component, seq, 0)
                entry = self.trace.wait_for("MISSION_ITEM_INT",
                                            lambda d, s=seq: int(d["seq"]) == s, mark, 2.0,
                                            poll_s=0.002)
            assert entry is not None, f"ArduPilot did not send mission item {seq}"
            item = entry.data
            positional = int(item["command"]) not in _NON_POSITIONAL_COMMANDS
            items.append({
                "seq": int(item["seq"]), "command": int(item["command"]),
                "frame": _planner_frame(item["frame"]),
                **{f"param{i}": float(item[f"param{i}"]) for i in range(1, 5)},
                "lat": float(item["x"]) / 1e7 if positional else float(item["x"]),
                "lon": float(item["y"]) / 1e7 if positional else float(item["y"]),
                "alt": float(item["z"]), "autocontinue": int(item["autocontinue"]),
            })
        conn.mav.mission_ack_send(conn.target_system, conn.target_component, 0, 0)
        return _without_home_slot(items)


# ── The page, clicked ─────────────────────────────────────────────────────────

class Page:
    """Clicks and types into the page the way a user does, and keeps a step log."""

    def __init__(self, browser, vehicle: Vehicle) -> None:
        self.browser = browser
        self.vehicle = vehicle
        self.started = time.monotonic()
        self.steps: list[dict[str, Any]] = []
        self.marks: list[tuple[float, str]] = []

    def step(self, name: str, **detail: Any) -> None:
        t = round(time.monotonic() - self.started, 1)
        self.marks.append((t, name))
        self.steps.append({"t_s": t, "step": name, "vehicle": self.vehicle.state(), **detail})

    def click(self, locator: str, what: str) -> None:
        """A real mouse click at the centre of `locator`, which must be what is there."""
        found = self.browser.wait_for(
            f"(() => {{ const e = {locator}; if (!e) return null;"
            " e.scrollIntoView({block: 'nearest', inline: 'nearest'});"
            " const r = e.getBoundingClientRect();"
            " if (!r.width || !r.height) return null;"
            " const x = r.left + r.width / 2, y = r.top + r.height / 2;"
            " const hit = document.elementFromPoint(x, y);"
            " return [x, y, !!hit && (hit === e || e.contains(hit)),"
            " hit ? hit.outerHTML.slice(0, 120) : null]; })()", timeout_s=20)
        x, y, on_top, hit = found
        assert on_top, f"{what}: another element covers it ({hit})"
        send = self.browser._send
        send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        send("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
        send("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)

    def type_into(self, locator: str, text: str, what: str) -> None:
        self.click(locator, what)
        # Cleared as select-all + typing over would; number inputs have no select().
        self.browser.js(f"(() => {{ const e = {locator}; e.value = ''; e.focus(); return true; }})()")
        self.browser._send("Input.insertText", text=text)
        self.browser.js(f"(() => {{ const e = {locator};"
                        " e.dispatchEvent(new Event('change', {bubbles: true})); return true; })()")

    def select(self, locator: str, value: str, what: str) -> None:
        """Choose an option of a <select> the user opened (the change event fires)."""
        self.click(locator, what)
        self.browser._send("Input.dispatchKeyEvent", type="keyDown", key="Escape",
                           code="Escape", windowsVirtualKeyCode=27)
        self.browser._send("Input.dispatchKeyEvent", type="keyUp", key="Escape",
                           code="Escape", windowsVirtualKeyCode=27)
        self.browser.js(f"(() => {{ const e = {locator}; e.value = {value!r};"
                        " e.dispatchEvent(new Event('change', {bubbles: true})); return true; })()")

    def toast(self, needle: str, timeout_s: float = 30.0) -> str:
        try:
            found = self.browser.wait_for(
                f"__toasts.slice(window.__seen || 0).find((t) => t.includes({needle!r}))",
                timeout_s=timeout_s)
        except AssertionError:
            seen = self.browser.js("__toasts.slice(window.__seen || 0)")
            raise AssertionError(f"no toast with {needle!r} in {timeout_s:.0f}s; "
                                 f"the page said: {seen}") from None
        self.browser.js("window.__seen = __toasts.length; true")
        return str(found)

    def text(self, selector: str) -> str:
        return str(self.browser.js(f"(document.querySelector({selector!r}) || {{}}).innerText || ''"))

    @staticmethod
    def pause() -> None:
        time.sleep(STEP_PAUSE_S)


def toolbar(name: str) -> str:
    return (f"[...document.querySelectorAll('.toolbar button')]"
            f".find((x) => x.textContent.trim() === {name!r})")


def nav(name: str) -> str:
    return f"[...document.querySelectorAll('.nav-item')].find((x) => x.textContent === {name!r})"


MODAL_PRIMARY = "document.querySelector('.modal-actions .primary')"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flight_out(request, tmp_path) -> Path:
    root = os.environ.get("ODK_UI_FLIGHT_OUT", "").strip()
    out = Path(root) if root else tmp_path
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def ui_stack(sitl_session, tmp_path, flight_out, monkeypatch):
    """The production Api over an Aukerman Park project, served to a 1920x1080 browser."""
    from ui_live_bridge import Browser, chromium_path, serve

    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser to drive the UI (none in the SITL image; "
                    "run this test on a desktop with ODK_SITL_DOCKER_IMAGE).")
    if not _home_in_aukerman():
        pytest.skip("The SITL home is not in Aukerman Park; set "
                    "ODK_SITL_HOME=41.3041,-81.7521,300,0 to fly the Aukerman grid.")
    monkeypatch.chdir(tmp_path)
    project_dir = flight_out / f"aukerman_project_{int(time.time())}"
    api = Api(AppSession(ProjectStore(tmp_path / "projects.db")))
    created = api.create_project("Aukerman Park", str(project_dir))
    assert created.get("ok", True), created
    # The survey area is the project's, as in the demo project: the imagery footprint.
    assert api.set_aoi(AUKERMAN_POLYGON)["ok"]
    server, base = serve(api)
    browser = Browser(executable, window_size=(1920, 1080),
                      log_path=flight_out / f"browser-{int(time.time())}.log")
    try:
        yield api, browser, base, sitl_session.connection_string, project_dir
    finally:
        try:
            api.disconnect_vehicle()
        finally:
            browser.close()
            server.shutdown()


@pytest.fixture
def ui_vehicle(sitl_observer, sitl_trace) -> Vehicle:
    vehicle = Vehicle(sitl_observer, sitl_trace)
    vehicle.wait("a 3D fix and EKF position", lambda v: v.heartbeat() is not None
                 and v.position() != (0.0, 0.0), 60)
    if vehicle.armed():
        pytest.fail(f"The vehicle is armed before the test began: {vehicle.state()}")
    fresh_battery(sitl_observer)
    return vehicle


def _record(page: Page, ui: str, out: Path, recorder, api) -> dict[str, Any]:
    """Write the video, contact sheet and step log. Runs in a finally: a failure to
    encode is recorded, never raised over the flight's own result."""
    recorder.stop()
    problems: list[str] = []
    video = sheet = None
    for attempt in range(2):  # ffmpeg can fail for memory on a loaded machine: once more
        try:
            video = recorder.write_video(out / f"app-flight-{ui}.mp4")
            break
        except Exception as exc:  # noqa: BLE001
            problems.append(f"video attempt {attempt + 1}: {exc}")
            time.sleep(5.0)
    try:
        sheet = recorder.write_contact_sheet(out / f"app-flight-{ui}-contact.jpg", marks=page.marks)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"contact sheet: {exc}")
    record = {"ui": ui, "video": str(video) if video else None,
              "contact_sheet": str(sheet) if sheet else None, "recording": recorder.summary(),
              "recording_problems": problems, "memory": recorder.memory, "steps": page.steps}
    (out / f"app-flight-{ui}-steps.json").write_text(json.dumps(record, indent=1), "utf-8")
    if video:
        shutil.rmtree(recorder.frames_dir, ignore_errors=True)
    return record


def _check_mission_on_vehicle(api, vehicle: Vehicle) -> dict[str, Any]:
    """ArduPilot's copy of the mission is the plan: every viewpoint, altitude, shutter."""
    from mission import exporters

    from ._support import plan_mismatches

    plan = api._session.mission_plan_dict
    held = vehicle.download_mission()
    problems = plan_mismatches(plan, held)
    assert not problems, problems[:10]
    sent = exporters.build_mission_items(api._session.mission_plan)
    assert len(held) == len(sent), (len(held), len(sent))
    return {"items_on_vehicle": len(held), "viewpoints_checked": len(exporters._viewpoints(plan))}


def _fly_and_land(page: Page, vehicle: Vehicle, command, *, n_items: int) -> None:
    """Start, Pause, Resume, RTL, landed -- `command(name)` is the UI's way to send each."""
    command("Start Mission")
    page.step("start clicked")
    vehicle.wait("armed in AUTO", lambda v: v.armed() and v.mode() == "AUTO", 60)
    page.step("ArduPilot armed, AUTO")
    vehicle.wait("climbing past 50 m", lambda v: v.altitude_m() > 50.0, 120)
    page.step("ArduPilot at survey altitude")
    third = max(4, n_items // 3)
    vehicle.wait(f"mission item {third}", lambda v: v.mission_seq() >= third, 600)
    page.step("survey under way", mission_seq=vehicle.mission_seq())

    command("Pause")
    page.step("pause clicked")
    held = vehicle.wait("holding (groundspeed < 0.5 m/s)", lambda v: v.groundspeed() < 0.5, 30)
    seq_paused = vehicle.mission_seq()
    lat0, lon0 = vehicle.position()
    time.sleep(8.0)
    lat1, lon1 = vehicle.position()
    drift = distance_m(lat0, lon0, lat1, lon1)
    assert vehicle.mission_seq() == seq_paused, "the mission advanced while paused"
    assert vehicle.mode() == "AUTO" and vehicle.armed(), vehicle.state()
    assert drift < 2.0, f"drifted {drift:.1f} m while paused"
    page.step("ArduPilot held position for 8 s", drift_m=round(drift, 2), held=held)

    command("Resume")
    page.step("resume clicked")
    # Moving again: away from where it held, or on to the next item. Not a groundspeed
    # threshold -- a hold that falls at the end of a leg resumes into a turn at ~2.8 m/s.
    vehicle.wait("moving again", lambda v: v.mission_seq() > seq_paused
                 or distance_m(lat1, lon1, *v.position()) > 5.0, 30)
    vehicle.wait("the mission advancing", lambda v: v.mission_seq() > seq_paused + 2, 120)
    page.step("ArduPilot resumed the survey", mission_seq=vehicle.mission_seq())

    command("RTL")
    page.step("rtl clicked")
    vehicle.wait("RTL", lambda v: v.mode() == "RTL", 15)
    page.step("ArduPilot in RTL")
    vehicle.wait("landed and disarmed", lambda v: not v.armed() and v.altitude_m() < 1.0, 420)
    lat, lon = vehicle.position()
    home = distance_m(lat, lon, sitl_launch.SITL_HOME_LAT, sitl_launch.SITL_HOME_LON)
    assert home < 5.0, f"landed {home:.1f} m from home"
    page.step("ArduPilot landed and disarmed at home", distance_from_home_m=round(home, 2))


# ── The cockpit ───────────────────────────────────────────────────────────────

def test_cockpit_flies_the_aukerman_grid(ui_stack, ui_vehicle, flight_out):
    from ui_recorder import ScreenRecorder

    api, browser, base, uri, _ = ui_stack
    vehicle = ui_vehicle
    browser.open(f"{base}/workspace.html", settle_s=4)
    browser.js(HELPERS)
    recorder = ScreenRecorder(browser.page_ws_url, flight_out / "frames-cockpit",
                              max_fps=RECORD_FPS, browser_pid=browser.process.pid)
    page = Page(browser, vehicle)
    try:
        page.step("cockpit open")
        # Plan: Mission Planning, 2D mapping from the Mission Types list, Plan.
        page.click(nav("Mission Planning"), "Mission Planning")
        page.pause()
        page.click("[...document.querySelectorAll('.tree-node .label')]"
                   ".find((x) => x.textContent === '2D mapping')", "2D mapping")
        assert browser.js("document.querySelector('select[data-key=type]').value") == "2D mapping"
        page.pause()
        page.click(toolbar("Plan"), "Plan")
        planned = page.toast("Planned")
        plan = api._session.mission_plan_dict
        assert plan["template"] == "grid" and abs(plan["altitude_m"] - 60.0) < 1e-6, plan
        page.step("mission planned", toast=planned)
        page.pause()
        page.pause()

        # Connect over pymavlink to the SITL's SERIAL0.
        page.click(nav("Flight"), "Flight")
        page.pause()
        page.click(toolbar("Connect"), "Connect")
        page.click("[...document.querySelectorAll('.modal-row')]"
                   ".find((x) => x.textContent.includes('MAVLink autopilot'))", "MAVLink autopilot")
        page.type_into("document.querySelector('.modal input')", uri, "connection URI")
        page.pause()
        page.click(MODAL_PRIMARY, "Connect (dialog)")
        connected = page.toast("Connected to", 40)
        assert "via pymavlink" in connected and "SIMULATED" in connected, connected
        page.step("connected", toast=connected)
        browser.wait_for("document.querySelector('.nav-right').innerText.includes('SIMULATED')", 20)
        page.pause()

        # Preflight, then each manual check confirmed, as a pilot records them.
        page.click(toolbar("Preflight"), "Preflight")
        page.step("preflight", toast=page.toast("reflight", 40))
        page.pause()
        for _ in range(12):
            page.click(toolbar("Confirm Checks"), "Confirm Checks")
            outcome = browser.wait_for(
                "(document.querySelector('.modal-row') && 'choose')"
                " || __toasts.slice(window.__seen || 0).find((t) => t.includes('Nothing is waiting'))",
                timeout_s=30)
            if outcome != "choose":
                browser.js("window.__seen = __toasts.length; true")
                break
            page.click("document.querySelector('.modal-row')", "manual check")
            page.toast("Confirmed", 40)
        page.step("manual checks confirmed")
        page.pause()

        # Upload, confirmed; then read back from ArduPilot over SERIAL1.
        page.click(toolbar("Upload"), "Upload")
        page.click(MODAL_PRIMARY, "Upload (confirm)")
        uploaded = page.toast("The vehicle accepted", 90)
        fidelity = _check_mission_on_vehicle(api, vehicle)
        page.step("mission on ArduPilot matches the plan", toast=uploaded, **fidelity)
        page.pause()

        def command(name: str) -> None:
            page.click(toolbar(name), name)
            page.click(MODAL_PRIMARY, f"{name} (confirm)")

        _fly_and_land(page, vehicle, command, n_items=fidelity["items_on_vehicle"])
        # What the screen says at the end, for the record.
        time.sleep(4.0)
        page.step("final screen", top_bar=page.text(".nav-right"),
                  workspace=page.text(".workspace")[:1500])
    finally:
        record = _record(page, "cockpit", flight_out, recorder, api)
    assert record["recording"]["frames"] > 100, record["recording"]


# ── The classic UI ────────────────────────────────────────────────────────────

def test_classic_ui_flies_the_aukerman_grid(ui_stack, ui_vehicle, flight_out):
    """index.html: Plan, Connect, Preflight, Upload, Start, Pause, Resume and RTH are
    buttons on the page (the last four also the desktop shell's native Fly menu)."""
    from ui_recorder import ScreenRecorder

    api, browser, base, uri, _ = ui_stack
    vehicle = ui_vehicle
    browser.open(f"{base}/index.html", settle_s=4)
    recorder = ScreenRecorder(browser.page_ws_url, flight_out / "frames-classic",
                              max_fps=RECORD_FPS, browser_pid=browser.process.pid)
    page = Page(browser, vehicle)
    status = "document.querySelector('#status-message').textContent"
    try:
        page.step("classic UI open")
        page.select("document.querySelector('#mission-template')", "grid", "Template")
        for field, value in (("#mission-altitude", "60"), ("#mission-front", "80"),
                             ("#mission-side", "70")):
            page.type_into(f"document.querySelector('{field}')", value, field)
        page.pause()
        page.click("document.querySelector('#btn-plan')", "Plan Mission")
        browser.wait_for(f"{status}.includes('Planned') || {status}.includes('waypoints')", 30)
        plan = api._session.mission_plan_dict
        assert plan["template"] == "grid" and abs(plan["altitude_m"] - 60.0) < 1e-6, plan
        page.step("mission planned", status=browser.js(status))
        page.pause()

        page.click("document.querySelector('#btn-connect')", "Connect")
        assert browser.js("document.querySelector('#dlg-driver').value") == "pymavlink"
        page.type_into("document.querySelector('#dlg-uri')", uri, "connection URI")
        page.pause()
        page.click("[...document.querySelectorAll('#modal-actions button')]"
                   ".find((x) => x.textContent === 'Connect')", "Connect (dialog)")
        browser.wait_for(f"{status}.includes('Connected')", 40)
        chip = browser.js("document.querySelector('#vehicle-chip').textContent")
        assert chip == "SIMULATED (pymavlink)", chip
        page.step("connected", chip=chip)
        page.pause()

        page.click("document.querySelector('#btn-preflight')", "Preflight")
        browser.wait_for("document.querySelector('#preflight-list').innerText.includes('Ready to start')", 40)
        for _ in range(12):
            if not browser.js("!!document.querySelector('#preflight-list [data-confirm]')"):
                break
            page.click("document.querySelector('#preflight-list [data-confirm]')", "I checked this")
            time.sleep(1.5)
        page.step("preflight confirmed", preflight=page.text("#preflight-list")[:1200])
        page.pause()

        page.click("document.querySelector('#btn-upload')", "Upload")
        browser.wait_for(f"{status}.includes('Vehicle accepted')", 90)
        fidelity = _check_mission_on_vehicle(api, vehicle)
        page.step("mission on ArduPilot matches the plan", status=browser.js(status), **fidelity)
        page.click("document.querySelector('.tab[data-tab=telemetry]')", "Telemetry tab")
        page.pause()

        menu = {"Start Mission": "start", "Pause": "pause", "Resume": "resume", "RTL": "rth"}

        def command(name: str) -> None:
            page.click(f"document.querySelector('#btn-fly-{menu[name]}')", name)
            browser.wait_for(f"{status}.includes('Command {menu[name]}')", 30)

        _fly_and_land(page, vehicle, command, n_items=fidelity["items_on_vehicle"])
        time.sleep(4.0)
        page.step("final screen", chip=browser.js("document.querySelector('#vehicle-chip').textContent"),
                  telemetry=page.text("#telemetry-grid")[:1500])
    finally:
        record = _record(page, "classic", flight_out, recorder, api)
    assert record["recording"]["frames"] > 100, record["recording"]
