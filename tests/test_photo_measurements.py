"""Measuring on a photograph: distance, area and crack length, from the cockpit.

core/measurements.py could always measure in a photograph's pixels -- measure_image_distance,
measure_polygon_area, measure_crack_length -- and nothing in the application called them.
A crack seen on a frame could not be measured on it.

The browser test clicks the photograph the cockpit draws (through the same pixel mapping
Mark GCP uses) in headless Edge/Chromium against a real ``app.api.Api``. The photograph is
drawn with features of known pixel size, so the lengths asserted are what the clicks and
the scale give, not a round trip of a stub. Only the modal answers and the native file
dialog are stood in for, as in test_cockpit_measures_in_a_browser.py.
"""

from __future__ import annotations

import html
import json
import re
import subprocess

import cv2
import numpy as np
import pytest

from browser_evidence import dump_dom_command
from test_cockpit_measures_in_a_browser import _serve

from app.api import Api
from app.session import AppSession
from app.store import ProjectStore

W, H = 1200, 900


@pytest.fixture
def photo_project(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    frame = np.full((H, W, 3), 170, np.uint8)
    cv2.line(frame, (100, 100), (600, 100), (0, 0, 255), 3)   # a 500 px reference
    # A 45-degree crack, 424 px across and down: 424 * sqrt(2) = 599.6 px long. Counting
    # its skeleton's pixels (one per diagonal step) gave 424 px, 29% short.
    cv2.line(frame, (200, 300), (624, 724), (20, 20, 20), 5)
    cv2.imwrite(str(photos / "wall.jpg"), frame)
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (200, 300), (624, 724), 255, 5)
    cv2.imwrite(str(tmp_path / "crack_mask.png"), mask)

    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Photo survey", str(tmp_path / "project"))["ok"]
    assert api.import_dataset(str(photos), "Wall")["ok"]
    return api, tmp_path / "crack_mask.png"


def test_image_measurements_are_scaled_stored_and_refused_honestly(photo_project):
    api, mask = photo_project
    # Scale from a reference of known length: 500 px is 250 mm -> 0.5 mm/px.
    ref = [[100, 100], [600, 100]]
    d = api.measure_on_image("distance", "wall.jpg", [[100, 400], [400, 800]],
                             None, ref, 250)
    assert d["ok"], d
    assert d["value"] == pytest.approx(0.25)          # 500 px * 0.5 mm
    assert d["scale_source"] == "reference"
    a = api.measure_on_image("area", "wall.jpg", [[0, 0], [200, 0], [200, 100], [0, 100]], 0.5)
    assert a["ok"], a
    assert a["value"] == pytest.approx(0.1 * 0.05)     # 100 mm x 50 mm
    c = api.measure_crack_on_image("wall.jpg", str(mask), 0.5)
    assert c["ok"], c
    assert c["value"] == pytest.approx(0.2998, rel=0.01)  # 599.6 px along the crack * 0.5 mm

    rows = api.list_measurements()["measurements"]
    assert sorted(r["measurement_type"] for r in rows) == ["area", "crack_length", "distance"]
    assert all(r["source_type"] in ("image", "crack_mask") for r in rows)
    assert all(r["details"]["scale_mm_per_px"] > 0 and r["details"]["caveat"] for r in rows)

    # No scale is never guessed; nor is a mask stretched onto the wrong photograph.
    assert "no scale of its own" in api.measure_on_image(
        "distance", "wall.jpg", [[0, 0], [10, 10]])["error"]
    small = mask.parent / "small.png"
    cv2.imwrite(str(small), np.full((10, 10), 255, np.uint8))
    assert "must be drawn on the photograph" in api.measure_crack_on_image(
        "wall.jpg", str(small), 0.5)["error"]
    assert "outside" in api.measure_on_image(
        "distance", "wall.jpg", [[0, 0], [5000, 10]], 0.5)["error"]
    assert "outside the active dataset" in api.measure_on_image(
        "distance", "../crack_mask.png", [[0, 0], [5, 5]], 0.5)["error"]
    assert api.list_measurements()["count"] == 3


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="css/workspace.css"></head>
<body data-result="pending"><div id="app"></div><pre id="out"></pre>
<script src="vendor/maplibre/maplibre-gl.js"></script>
<script>
  window.pywebview = { api: new Proxy({}, { get: (_t, name) => (typeof name !== "string")
    ? undefined : (...args) => fetch("/api/" + name, { method: "POST",
        body: JSON.stringify(args) }).then((r) => r.json()) }) };
</script>
<script type="module">
import { Shell } from "./js/workspace/shell.js";

const PLAN = __PLAN__;
const log = [];
const shell = new Shell(document.getElementById("app"));
const toast = shell.toast.bind(shell);
shell.toast = (text, kind) => { log.push({ text: String(text), kind: kind || "" }); toast(text, kind); };
const answers = [];
const clicks = [];
const context = shell.actionContext.bind(shell);
shell.actionContext = () => ({ ...context(),
  prompt: async () => answers.shift() ?? "",
  choose: async (_title, options) => options[0] });
shell.confirmAll = true;
const settle = (ms) => new Promise((r) => setTimeout(r, ms));

/* A click on the photograph at a pixel of the SOURCE frame: the page position is worked
   out from the <img> the cockpit drew, and the real pickImagePixel maps it back. */
async function deliver(next) {
  await settle(150);
  if (next === "esc") {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    return;
  }
  const img = document.querySelector("img.canvas-product");
  const box = img.getBoundingClientRect();
  const nw = img.naturalWidth, nh = img.naturalHeight;
  const fit = Math.min(box.width / nw, box.height / nh);
  const sx = nw / __W__;
  const clientX = box.left + (box.width - nw * fit) / 2 + next[0] * sx * fit;
  const clientY = box.top + (box.height - nh * fit) / 2 + next[1] * sx * fit;
  img.dispatchEvent(new MouseEvent("click", { clientX, clientY, bubbles: true }));
}
const pick = shell.pickImagePixel.bind(shell);
shell.pickImagePixel = (message) => {
  const picked = pick(message);
  deliver(clicks.shift());
  return picked;
};

try {
  await shell.refreshState();
  shell.open("measurements");
  const results = {};
  for (const step of PLAN) {
    const before = log.length;
    answers.splice(0, answers.length, ...(step.answers || []));
    clicks.splice(0, clicks.length, ...(step.clicks || []));
    await shell.runAction(step.action);
    await settle(100);
    results[step.name] = log.slice(before).map((t) => `[${t.kind}] ${t.text}`);
  }
  const history = await window.pywebview.api.list_measurements();
  results.history = (history.measurements || []).map((m) => [m.measurement_type, m.value]);
  await settle(300);
  results.historyPanel = [...document.querySelectorAll("td")].map((td) => td.textContent).join("|");
  document.getElementById("out").textContent = JSON.stringify(results);
  document.body.dataset.result = "done";
} catch (error) {
  document.body.dataset.result = "error:" + (error && error.message);
}
</script></body></html>"""


def test_photo_measuring_tools_in_the_cockpit(tmp_path, photo_project):
    api, mask = photo_project
    plan = [
        # Scale from the drawn 500 px reference, said to be 250 mm long.
        {"name": "distance", "action": "Photo Distance", "answers": ["", "250"],
         "clicks": [[100, 100], [600, 100], [100, 400], [400, 800]]},
        {"name": "area", "action": "Photo Area", "answers": ["0.5"],
         "clicks": [[300, 200], [700, 200], [700, 400], [300, 400], "esc"]},
        {"name": "crack", "action": "Crack Length", "answers": ["0.5"]},
    ]
    page = PAGE.replace("__PLAN__", json.dumps(plan)).replace("__W__", str(W))
    server, thread = _serve(api, [], str(mask))
    server.RequestHandlerClass.extra_pages = {"/photo.html": page.encode("utf-8")}
    try:
        completed = subprocess.run(dump_dom_command(
            "the photo measurement evidence test", tmp_path / "edge-photo",
            f"http://127.0.0.1:{server.server_port}/photo.html",
            virtual_time_ms=60000, webgl=True,
        ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    dom = completed.stdout
    assert 'data-result="done"' in dom, dom[-3000:]
    results = json.loads(html.unescape(re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)))
    text = {k: " ".join(v) if isinstance(v, list) and v and isinstance(v[0], str) else v
            for k, v in results.items()}
    for step in ("distance", "area", "crack"):
        assert "[error]" not in text[step], (step, text[step])
        assert "Kept in the measurement history" in text[step], (step, text[step])
    assert "from the reference" in text["distance"]
    assert "plane of the reference" in text["distance"]   # the perspective caveat is shown
    assert "over 4 corners" in text["area"]
    assert "mask skeleton" in text["crack"]

    values = dict(results["history"])
    # 500 px at 0.5 mm/px; 400 x 200 px at 0.5 mm/px = 200 x 100 mm; the 599.6 px
    # diagonal crack at 0.5 mm/px.
    assert values["distance"] == pytest.approx(0.25, rel=0.01)
    assert values["area"] == pytest.approx(0.02, rel=0.02)
    assert values["crack_length"] == pytest.approx(0.2998, rel=0.01)
    assert "crack_length" in results["historyPanel"]


def _polyline_mask(points, shape=(3672, 4896), width=5):
    mask = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(points, float)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, 255, width)
    return mask


def _true_length(points):
    return float(np.hypot(*np.diff(np.asarray(points, float), axis=0).T).sum())


def _shapes():
    t = np.linspace(0, 1, 4000)
    rng = np.random.default_rng(7)
    # A wandering crack that never turns back on itself (heading kept within 0.1-1.3 rad).
    heading = 0.7 + 0.6 * np.tanh(0.5 * np.cumsum(rng.normal(0, 0.3, 300)))
    return {
        "horizontal": [[1000, 1800], [2600, 1800]],
        "diagonal": [[1000, 1000], [2600, 2600]],
        "30 degrees": [[1000, 1000], [1000 + 1600 * np.cos(np.pi / 6),
                                      1000 + 1600 * np.sin(np.pi / 6)]],
        "22.5 degrees": [[1000, 1000], [1000 + 1600 * np.cos(np.pi / 8),
                                        1000 + 1600 * np.sin(np.pi / 8)]],
        "sine": np.c_[800 + 1600 * t, 1800 + 300 * np.sin(t * 4 * np.pi)],
        "circle": np.c_[2400 + 800 * np.cos(2 * np.pi * t),
                        1800 + 800 * np.sin(2 * np.pi * t)],
        "random walk": np.c_[1000 + np.cumsum(10 * np.cos(heading)),
                             800 + np.cumsum(10 * np.sin(heading))],
    }


@pytest.mark.parametrize("shape", list(_shapes()))
def test_crack_length_follows_the_crack_at_any_angle(shape):
    """On masks the size of a real 4896x3672 frame, at 1 mm/px, the length is the crack's.

    Counting skeleton pixels read a 45-degree crack 29% short and a sine 12% short;
    weighting each step 1 or sqrt(2) reads 22.5 degrees 8% long. Walking the skeleton's
    branches as polylines is within 1% at every angle.
    """
    from core.measurements import calculate_crack_length_from_mask

    points = _shapes()[shape]
    measured = calculate_crack_length_from_mask(_polyline_mask(points), 1.0)
    assert measured == pytest.approx(_true_length(points) / 1000.0, rel=0.01)


def test_a_branched_crack_counts_every_branch_once():
    from core.measurements import calculate_crack_length_from_mask

    branches = [[[1000, 1000], [2000, 2000]], [[2000, 2000], [3500, 2000]],
                [[2000, 2000], [2000, 3200]]]
    mask = np.zeros((3672, 4896), np.uint8)
    for b in branches:
        mask |= _polyline_mask(b)
    true = sum(_true_length(b) for b in branches) / 1000.0
    assert calculate_crack_length_from_mask(mask, 1.0) == pytest.approx(true, rel=0.01)


def test_the_crack_engine_and_forecaster_measure_along_the_crack(tmp_path):
    """The same pixel count was in core/crack_engine.py, core/propagation.py and detection."""
    from core.crack_engine import extract_crack_geometry
    from core.propagation import CrackPropagationForecaster

    points = [[1000, 1000], [2600, 2600]]
    mask_path = tmp_path / "diag.png"
    cv2.imwrite(str(mask_path), _polyline_mask(points))
    true_px = _true_length(points)
    geometry = extract_crack_geometry(mask_path, 1.0)
    assert geometry.length_px == pytest.approx(true_px, rel=0.01)
    assert geometry.length_m == pytest.approx(true_px / 1000.0, rel=0.01)
    metrics = CrackPropagationForecaster._metrics(_polyline_mask(points), 0)
    assert metrics.estimated_length_px == pytest.approx(true_px, rel=0.01)


@pytest.fixture
def rotated_project(tmp_path):
    """A portrait photograph stored landscape (1200x900) with EXIF Orientation 6.

    Every viewer -- and cv2.imread, which image_preview uses -- shows it 900 wide and
    1200 high, so that is the frame the cockpit's clicks are in.
    """
    from PIL import Image

    photos = tmp_path / "photos"
    photos.mkdir()
    stored = np.full((H, W, 3), 170, np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(stored).save(photos / "portrait.jpg", exif=exif.tobytes())
    mask = np.zeros((W, H), np.uint8)              # drawn at the displayed 900x1200 size
    cv2.line(mask, (100, 1000), (700, 1000), 255, 5)
    cv2.imwrite(str(tmp_path / "portrait_mask.png"), mask)

    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    api = Api(session)
    assert api.create_project("Portrait", str(tmp_path / "project"))["ok"]
    assert api.import_dataset(str(photos), "Portrait")["ok"]
    return api, tmp_path / "portrait_mask.png"


def test_a_rotated_photograph_is_measured_in_the_frame_the_preview_shows(rotated_project):
    api, mask = rotated_project
    preview = api.image_preview("portrait.jpg")
    assert (preview["source_width"], preview["source_height"]) == (H, W)   # 900 x 1200
    # y = 1100 is inside the displayed 900x1200 frame (it was refused against 1200x900).
    d = api.measure_on_image("distance", "portrait.jpg", [[100, 100], [100, 1100]], 0.5)
    assert d["ok"], d
    assert d["value"] == pytest.approx(0.5)
    assert d["measurement"]["details"]["image_width"] == H
    assert "outside the 900x1200" in api.measure_on_image(
        "distance", "portrait.jpg", [[0, 0], [1000, 10]], 0.5)["error"]
    c = api.measure_crack_on_image("portrait.jpg", str(mask), 0.5)
    assert c["ok"], c
    assert c["value"] == pytest.approx(0.3, rel=0.01)


def test_photo_measuring_on_a_rotated_photograph_in_the_cockpit(tmp_path, rotated_project):
    api, mask = rotated_project
    plan = [{"name": "distance", "action": "Photo Distance", "answers": ["0.5"],
             "clicks": [[450, 100], [450, 1100]]},
            {"name": "crack", "action": "Crack Length", "answers": ["0.5"]}]
    # The click helper scales by the source frame's width: the displayed 900, not 1200.
    page = PAGE.replace("__PLAN__", json.dumps(plan)).replace("__W__", str(H))
    server, thread = _serve(api, [], str(mask))
    server.RequestHandlerClass.extra_pages = {"/photo.html": page.encode("utf-8")}
    try:
        completed = subprocess.run(dump_dom_command(
            "the rotated photo measurement test", tmp_path / "edge-rotated",
            f"http://127.0.0.1:{server.server_port}/photo.html",
            virtual_time_ms=60000, webgl=True,
        ), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    dom = completed.stdout
    assert 'data-result="done"' in dom, dom[-3000:]
    results = json.loads(html.unescape(re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)))
    for step in ("distance", "crack"):
        joined = " ".join(results[step])
        assert "[error]" not in joined and "Kept in the measurement history" in joined, joined
    values = dict(results["history"])
    assert values["distance"] == pytest.approx(0.5, rel=0.01)   # 1000 px down the portrait
    assert values["crack_length"] == pytest.approx(0.3, rel=0.01)
