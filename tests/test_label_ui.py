"""The labelling canvas is executed in a real browser, not asserted about in Python.

Every other UI claim in this project is verified by loading the page in headless
Chromium and reading the DOM back, because a viewer that is only reasoned about in a
test file is not evidence that it renders. This is the same: pointer events are
dispatched at the canvas, and the boxes that come out are read from the page.

What is being proved is narrow and specific. The registry row for ai.custom_training
said "a user still cannot draw a box in this product". These tests exist to make that
sentence false, and to pin the two ways the drawing could be quietly wrong: a click
stored as a zero-area target, and coordinates saved in screen pixels so the corpus
depends on the size of the window it was drawn in.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading

import pytest

from browser_evidence import dump_dom_command

ROOT = Path(__file__).resolve().parents[1]
LABEL_BOX_JS = ROOT / "app" / "web" / "js" / "label-box.js"

# The canvas is 400x200 CSS pixels over a 400x200 image, so a drag from (40, 20) to
# (140, 120) is exactly a quarter of the width and half the height. Those numbers are
# checked below, which is what makes the normalisation claim testable rather than
# decorative.
HARNESS = """<!doctype html>
<body data-result="pending">
<canvas id="canvas" width="400" height="200" style="width:400px;height:200px"></canvas>
<script src="/label-box.js"></script>
<script>
function drag(canvas, x0, y0, x1, y1) {
  const rect = canvas.getBoundingClientRect();
  const at = (x, y, type) => canvas.dispatchEvent(new PointerEvent(type, {
    clientX: rect.left + x, clientY: rect.top + y, bubbles: true,
  }));
  at(x0, y0, 'pointerdown');
  at((x0 + x1) / 2, (y0 + y1) / 2, 'pointermove');
  at(x1, y1, 'pointerup');
}
window.onload = () => {
  try {
    const canvas = document.getElementById('canvas');
    const labeller = new ODKLabelBox.BoxLabeller(canvas, {classes: ['crack', 'spall']});

    drag(canvas, 40, 20, 140, 120);          // a real box
    labeller.setActiveClass('spall');
    drag(canvas, 200, 40, 300, 140);         // a second, different class
    drag(canvas, 10, 10, 11, 11);            // a click: must be discarded

    const first = labeller.boxes[0];
    const removed = labeller.removeBox(1);
    const payload = labeller.toPayload('/tmp/frame.png', false);

    document.body.dataset.result = [
      'boxes:' + (labeller.boxes.length + 1),   // +1 for the one just removed
      'kept:' + labeller.boxes.length,
      'removed:' + removed,
      'label:' + first.label,
      'x:' + first.x.toFixed(3),
      'y:' + first.y.toFixed(3),
      'w:' + first.width.toFixed(3),
      'h:' + first.height.toFixed(3),
      'payloadpath:' + payload.path,
      'empty:' + payload.confirmed_empty,
    ].join(';');
  } catch (error) {
    document.body.dataset.result = 'error:' + error.message;
  }
};
</script>
</body>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the test output readable
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/label-box.js"):
            body = LABEL_BOX_JS.read_bytes()
            content_type = "text/javascript"
        else:
            body = HARNESS.encode("utf-8")
            content_type = "text/html"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def drawn(tmp_path_factory) -> dict[str, str]:
    """Drive the canvas once and hand every test the same parsed result."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = subprocess.run(
            dump_dom_command(
                "the box labelling evidence test",
                tmp_path_factory.mktemp("label-profile"),
                f"http://127.0.0.1:{server.server_port}/harness.html",
                virtual_time_ms=4000,
                webgl=False,
            ),
            capture_output=True, text=True, timeout=40, check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    marker = 'data-result="'
    start = completed.stdout.index(marker) + len(marker)
    raw = completed.stdout[start:completed.stdout.index('"', start)]
    assert not raw.startswith("error:"), raw
    return dict(part.split(":", 1) for part in raw.split(";"))


class TestAUserCanDrawABox:
    def test_the_drag_produced_a_box(self, drawn) -> None:
        assert int(drawn["boxes"]) == 2, "two drags should produce two boxes"

    def test_the_box_carries_the_active_class(self, drawn) -> None:
        assert drawn["label"] == "crack"

    def test_a_second_class_can_be_selected_and_drawn(self, drawn) -> None:
        # The removed box was the spall one; that it existed to remove is the evidence.
        assert drawn["removed"] == "true"
        assert drawn["kept"] == "1"


class TestTheGeometryIsStoredInAFormThatSurvives:
    def test_coordinates_are_normalised_not_pixels(self, drawn) -> None:
        """40px of a 400px canvas is 0.1, not 40.

        Pixels would make the corpus depend on the window it was drawn in, and every
        trainer resizes its input anyway.
        """
        assert float(drawn["x"]) == pytest.approx(0.1, abs=0.005)
        assert float(drawn["y"]) == pytest.approx(0.1, abs=0.005)

    def test_the_size_is_normalised_per_axis(self, drawn) -> None:
        # 100px wide over 400, 100px tall over 200: different fractions on each axis,
        # so a single scale factor applied to both would fail this.
        assert float(drawn["w"]) == pytest.approx(0.25, abs=0.005)
        assert float(drawn["h"]) == pytest.approx(0.5, abs=0.005)


class TestAClickIsNotABox:
    def test_a_one_pixel_drag_is_discarded(self, drawn) -> None:
        """A degenerate box written to a label file is a target no model can match."""
        assert int(drawn["boxes"]) == 2, "the 1px drag must not have become a third box"


class TestThePayloadMatchesWhatTheApiExpects:
    def test_it_carries_the_image_path(self, drawn) -> None:
        assert drawn["payloadpath"] == "/tmp/frame.png"

    def test_confirmed_empty_is_false_when_boxes_exist(self, drawn) -> None:
        """Otherwise an image with boxes could be posted as a deliberate negative."""
        assert drawn["empty"] == "false"


class TestTheDrawnBoxesReachTheCorpusBuilder:
    """The UI and the builder have to agree, or labelling produces nothing trainable."""

    def test_the_payload_shape_is_what_regions_from_payload_parses(self, drawn) -> None:
        from core.label_sets import regions_from_payload

        regions = regions_from_payload([{
            "path": "frame.png",
            "boxes": [{
                "label": drawn["label"],
                "x": float(drawn["x"]), "y": float(drawn["y"]),
                "width": float(drawn["w"]), "height": float(drawn["h"]),
            }],
            "confirmed_empty": drawn["empty"] == "true",
        }])
        assert regions[0].boxes[0].label == "crack"
        assert regions[0].boxes[0].width == pytest.approx(0.25, abs=0.005)
