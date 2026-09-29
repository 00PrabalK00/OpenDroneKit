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
    cv2.line(frame, (200, 500), (799, 500), (20, 20, 20), 5)  # a 600 px "crack"
    cv2.imwrite(str(photos / "wall.jpg"), frame)
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (200, 500), (799, 500), 255, 5)
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
    assert c["value"] == pytest.approx(0.3, rel=0.02)  # ~600 px skeleton * 0.5 mm

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
    # 500 px at 0.5 mm/px; 400 x 200 px at 0.5 mm/px = 200 x 100 mm; ~600 px skeleton.
    assert values["distance"] == pytest.approx(0.25, rel=0.01)
    assert values["area"] == pytest.approx(0.02, rel=0.02)
    assert values["crack_length"] == pytest.approx(0.3, rel=0.02)
    assert "crack_length" in results["historyPanel"]
