"""The v1 image-space viewer: a click on the photograph is a pixel of the photograph.

js/odk/imageview.js shows a dataset photograph (Api.image_preview) for Review, GCP
marking and measuring on a photo. It keeps the cockpit's contract -- the picture is
img.canvas-product, and a click maps through the letterboxing and the preview's
downscale to the source frame's full-resolution pixel -- and draws marks in those same
source pixels. Driven in headless Chromium with real mouse events, against the real Api,
on a real photograph (tests/fixtures/concrete_cracked_public_domain.jpg).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ui_v1_map_harness import MapBrowser, MapServer, chromium_path, stage_page

pytestmark = pytest.mark.skipif(chromium_path() is None, reason="No Chromium or Edge to drive.")

# The pending pick, or "no answer" after 20 s rather than a hung browser.
PICKED = "Promise.race([window.__iv.pick, new Promise((r) => setTimeout(() => r('no answer'), 20000))])"
PHOTO = Path(__file__).resolve().parent / "fixtures" / "concrete_cracked_public_domain.jpg"

SCRIPT = r"""
import { mountImageView, imageViewProvider } from "./js/odk/imageview.js";
const out = window.__iv = { toasts: [] };
try {
  out.view = mountImageView(document.getElementById("photo"), {
    notify: (text, kind) => out.toasts.push([kind, text]) });
  out.ready = true;
} catch (error) { out.error = String(error && error.stack || error); }
window.__ivProvider = imageViewProvider;
"""


@pytest.fixture
def viewer(tmp_path):
    import cv2

    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    dataset = tmp_path / "images"
    dataset.mkdir()
    shutil.copy(PHOTO, dataset / PHOTO.name)
    height, width = cv2.imread(str(PHOTO)).shape[:2]
    api = Api(AppSession(store=ProjectStore(tmp_path / "projects.db")))
    api.create_project("Bridge deck", str(tmp_path / "project"))
    assert api.set_active_dataset(str(dataset))["ok"]
    server = MapServer(api, stage_page(SCRIPT, body='<div id="photo"></div>'))
    browser = MapBrowser(chromium_path())
    try:
        browser.viewport(1200, 800)
        browser.open(f"{server.base}/__stage.html", settle_s=0.3)
        browser.wait_for("window.__iv && (window.__iv.ready || window.__iv.error)")
        assert not browser.js("window.__iv.error || ''")
        yield browser, api, (width, height)
    finally:
        browser.close()
        server.close()


def _screen_of(browser, x: float, y: float) -> tuple[float, float]:
    """Where source pixel (x, y) is drawn on screen, from the <img> box and its preview."""
    box = browser.js("""(() => { const i = document.querySelector('img.canvas-product');
      const r = i.getBoundingClientRect(); const c = window.__iv.view.current();
      return { left: r.left, top: r.top, width: r.width, height: r.height,
               nw: i.naturalWidth, nh: i.naturalHeight, crop: c.crop, sw: c.source_width, sh: c.source_height }; })()""")
    fit = min(box["width"] / box["nw"], box["height"] / box["nh"])
    ox = box["left"] + (box["width"] - box["nw"] * fit) / 2
    oy = box["top"] + (box["height"] - box["nh"] * fit) / 2
    cx, cy, cw, ch = box["crop"] or [0, 0, box["sw"], box["sh"]]
    return ox + (x - cx) / cw * box["nw"] * fit, oy + (y - cy) / ch * box["nh"] * fit


def test_a_click_is_a_pixel_of_the_source_photograph(viewer) -> None:
    browser, _api, (width, height) = viewer
    shown = browser.js(f"window.__iv.view.show('{PHOTO.name}').then((p) => p && "
                       "{ name: p.name, sw: p.source_width, sh: p.source_height, w: p.width })")
    assert shown == {"name": PHOTO.name, "sw": width, "sh": height, "w": shown["w"]}
    assert browser.js("document.querySelectorAll('img.canvas-product').length") == 1
    head = browser.js("[...document.querySelectorAll('.odk-imageview-head .odk-map-chip')]"
                      ".filter((c) => !c.hidden).map((c) => c.textContent)")
    assert head == [PHOTO.name, f"{width} × {height} px"]

    target = (width * 0.3, height * 0.6)
    browser.js("window.__iv.pick = window.__iv.view.pickPixel('Click the crack tip'); 0")
    browser.wait_for("document.querySelector('.odk-imageview-hint') && !document.querySelector('.odk-imageview-hint').hidden")
    x, y = _screen_of(browser, *target)
    browser.click(x, y)
    picked = browser.js(PICKED)
    # One screen pixel is (source / drawn) pixels of the photograph.
    tolerance = width / browser.js("document.querySelector('img.canvas-product').getBoundingClientRect().width") + 1
    assert picked["width"] == width and picked["height"] == height
    assert picked["x"] == pytest.approx(target[0], abs=tolerance)
    assert picked["y"] == pytest.approx(target[1], abs=tolerance)

    # A click on the letterbox beside the picture is no point at all.
    box = browser.js("document.querySelector('img.canvas-product').getBoundingClientRect().toJSON()")
    x0, y0 = _screen_of(browser, 0, 0)
    off = (box["left"] + 2, box["top"] + box["height"] / 2) if x0 - box["left"] > 4         else (box["left"] + box["width"] / 2, box["top"] + 2) if y0 - box["top"] > 4 else None
    browser.js("window.__iv.pick = window.__iv.view.pickPixel(); 0")
    browser.wait_for("!document.querySelector('.odk-imageview-hint').hidden")
    if off is not None:
        browser.click(*off)
    else:  # the picture fills the box exactly: Esc is the other way out
        browser.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))")
    assert browser.js(PICKED) is None
    assert browser.js("document.querySelector('.odk-imageview-hint').hidden") is True


def test_marks_and_a_close_up_stay_in_source_pixels(viewer) -> None:
    browser, _api, (width, height) = viewer
    browser.js(f"window.__iv.view.show('{PHOTO.name}')")
    at = (width * 0.25, height * 0.4)
    browser.js(f"window.__iv.view.setMarks([{{ x: {at[0]}, y: {at[1]}, label: 'GCP 1', tone: 'ok' }}])")
    mark = browser.js("""(() => { const c = document.querySelector('.odk-imageview-mark circle').getBoundingClientRect();
      return [c.left + c.width / 2, c.top + c.height / 2]; })()""")
    expected = _screen_of(browser, *at)
    assert mark == pytest.approx(list(expected), abs=1.5)

    # A close-up window of the frame: the same source pixel lands under the same mark.
    side = int(min(width, height) / 4)
    crop = [int(at[0] - side / 2), int(at[1] - side / 2), side, side]
    shown = browser.js(f"window.__iv.view.show('{PHOTO.name}', {crop}).then((p) => p.crop)")
    assert shown == crop
    mark = browser.js("""(() => { const c = document.querySelector('.odk-imageview-mark circle').getBoundingClientRect();
      return [c.left + c.width / 2, c.top + c.height / 2]; })()""")
    assert mark == pytest.approx(list(_screen_of(browser, *at)), abs=1.5)
    browser.js("window.__iv.pick = window.__iv.view.pickPixel(); 0")
    browser.wait_for("!document.querySelector('.odk-imageview-hint').hidden")
    browser.click(*mark)
    picked = browser.js(PICKED)
    # On the close-up one screen pixel is a fraction of a source pixel.
    assert picked["x"] == pytest.approx(at[0], abs=1.5) and picked["y"] == pytest.approx(at[1], abs=1.5)


def test_the_shell_provider_draws_a_preview_it_already_has(viewer) -> None:
    browser, api, (width, height) = viewer
    drawn = browser.js(f"""(async () => {{
      const preview = await window.pywebview.api.image_preview('{PHOTO.name}');
      const node = window.__ivProvider()(document.body, preview);
      document.getElementById('photo').replaceChildren(node);
      for (let i = 0; i < 80 && !node.querySelector('img.canvas-product').naturalWidth; i++)
        await new Promise((r) => setTimeout(r, 25));
      return {{ natural: node.querySelector('img.canvas-product').naturalWidth,
                current: node.odkImageView.current().source_width }};
    }})()""")
    assert drawn["natural"] > 0 and drawn["current"] == width
