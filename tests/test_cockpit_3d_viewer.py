"""The Digital Twin draws the project's own model in 3D, and its tools act on it.

Audit, before this change: "Textured Mesh" and "Point Cloud" only switched the canvas to
a 2D raster preview, Save View answered "the cockpit has no interactive 3D camera", Clip
answered "the 3D viewer is not built", Measure in 3D answered "the interactive viewer is
not built", and the twin's Measure and Annotate armed tools nothing read -- with the mesh,
the saved-view store and the clip store all on disk.

The model here is written the way a reconstruction writes one: Open3D's binary mesh.ply
and core.reconstruction's ASCII cloud, in structure-from-motion units, beside the
geo_anchor.json that carries them into a projected CRS. The page is the real cockpit in
headless Chromium against the real Api; clicks land on the WebGL canvas at the screen
position of a model vertex, as an operator's would.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cockpit_harness import Cockpit, make_api

SCALE = 10.0  # one SfM unit is ten metres in the anchor below


def _write_reconstruction(recon: Path) -> None:
    import open3d as o3d

    from core.geo import GeoAnchor
    from core.reconstruction import _write_ascii_ply

    recon.mkdir(parents=True, exist_ok=True)
    box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.0, depth=0.5)
    box = box.subdivide_midpoint(number_of_iterations=3)
    vertices = np.asarray(box.vertices)
    box.vertex_colors = o3d.utility.Vector3dVector(
        np.column_stack([vertices[:, 0] / 2.0, vertices[:, 1], 1 - vertices[:, 2] * 2]).clip(0, 1))
    box.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(recon / "mesh.ply"), box)  # binary, as COLMAP runs do
    rgb = (np.asarray(box.vertex_colors) * 255).astype(np.uint8)
    _write_ascii_ply(recon / "reconstruction.ply", vertices.astype(np.float32), rgb)
    anchor = GeoAnchor(scale=SCALE, rotation=np.eye(3).tolist(), translation=[0.0, 0.0, 0.0],
                       epsg=32617, origin=[437000.0, 4572000.0, 300.0], inlier_count=12,
                       sample_count=12, rmse_m=0.42, max_residual_m=0.9)
    (recon / "geo_anchor.json").write_text(json.dumps(anchor.to_dict()), encoding="utf-8")


@pytest.fixture
def api(tmp_path):
    api = make_api(tmp_path)
    _write_reconstruction(Path(api._session.project_root()) / "reconstruction")
    return api


# Helpers for the scenario: a vertex's screen position, and a click there.
PICK = """
const viewer3d = await import("/js/workspace/viewer3d.js");
const viewer = await ODK.waitFor(() => viewer3d.currentViewer(), 20000);
if (!viewer) throw new Error("no 3D viewer mounted; canvas says: "
  + ODK.$$(".workspace .canvas .placeholder").map((n) => n.innerText).join(" | "));
const screenOf = (file) => {
  const local = [file[0] - viewer.center[0], file[1] - viewer.center[1], file[2] - viewer.center[2]];
  const { mvp, width, height } = viewer.matrices();
  return viewer.project(local, mvp, width, height);
};
const clickAt = async (file) => {
  const [x, y] = screenOf(file);
  const rect = viewer.overlay.getBoundingClientRect();
  const at = { clientX: rect.left + x, clientY: rect.top + y, button: 0, bubbles: true, pointerId: 1 };
  viewer.overlay.dispatchEvent(new PointerEvent("pointerdown", at));
  viewer.overlay.dispatchEvent(new PointerEvent("pointerup", at));
  await ODK.sleep(400);
};
// Two model vertices that face the default camera: the picked point is the one nearest
// the camera, so the test finds its own targets rather than assuming an orientation.
const frontVertices = () => {
  const { mvp, width, height } = viewer.matrices();
  const out = [];
  for (let i = 0; i < viewer.positions.length; i += 3) {
    const local = [viewer.positions[i], viewer.positions[i + 1], viewer.positions[i + 2]];
    const s = viewer.project(local, mvp, width, height);
    if (!s || s[0] < 20 || s[1] < 20 || s[0] > width - 20 || s[1] > height - 20) continue;
    const hit = viewer.pick(s[0], s[1]);
    if (hit && Math.abs(hit.local[0] - local[0]) + Math.abs(hit.local[1] - local[1]) + Math.abs(hit.local[2] - local[2]) < 1e-6) {
      out.push([local[0] + viewer.center[0], local[1] + viewer.center[1], local[2] + viewer.center[2]]);
    }
  }
  return out;
};
const litPixels = () => {
  const gl = viewer.gl;
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let lit = 0;
  for (let i = 0; i < px.length; i += 4) if (px[i] + px[i + 1] + px[i + 2] > 90) lit += 1;
  return lit;
};
"""


def test_the_twin_draws_the_mesh_and_measures_between_picked_points(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Digital Twin");
            {PICK}
            await ODK.sleep(500);
            const chip = ODK.$('[data-role="scene-summary"]').innerText;
            const lit = litPixels();
            const front = frontVertices();
            // Two front-facing vertices far apart, so the length is not trivially small.
            let a = front[0], b = front[0];
            for (const p of front) for (const q of front) {{
              if (Math.hypot(p[0]-q[0], p[1]-q[1], p[2]-q[2]) > Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2])) {{ a = p; b = q; }}
            }}
            await ODK.press("Measure");
            await clickAt(a);
            await clickAt(b);
            const toast = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Length")), 10000);
            await ODK.sleep(400);
            return {{ chip, lit, a, b, toast, panel: ODK.panel("Measurements") }};
        """)

    assert "mesh.ply" in result["chip"] and "triangles" in result["chip"]
    assert "EPSG:32617" in result["chip"]
    assert result["lit"] > 500, "the WebGL canvas drew nothing"
    # The number on screen is the application's, from the two picked points.
    measured = cockpit.called("measure_on_model")
    assert measured and measured[-1]["ok"], measured
    kind, points = measured[-1]["args"]
    assert kind == "length"
    np.testing.assert_allclose(points[0], result["a"], atol=1e-4)
    np.testing.assert_allclose(points[1], result["b"], atol=1e-4)
    expected = SCALE * float(np.linalg.norm(np.subtract(result["b"], result["a"])))
    assert f"Length {expected:.2f} m" in result["toast"], (expected, result["toast"])
    assert "RMSE 0.42 m" in result["toast"]
    assert f"{expected:.2f} m" in result["panel"]


def test_save_view_stores_the_viewer_camera_and_opens_it_again(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Digital Twin");
            {PICK}
            viewer.yaw = 1.1; viewer.pitch = 0.3; viewer.distance *= 0.8; viewer.render();
            const saved = viewer.cameraState();
            await ODK.press("Save View");
            await ODK.answer("north side");
            const toast1 = ODK.lastToast();
            viewer.fit();
            await ODK.press("Save View");
            await ODK.pick("Open “north side”");
            await ODK.sleep(300);
            return {{ saved, restored: viewer.cameraState(), toast1, toast2: ODK.lastToast() }};
        """)

    views = api.list_views()["views"]
    assert [v["name"] for v in views] == ["north side"]
    np.testing.assert_allclose(views[0]["position"], result["saved"]["position"], atol=1e-4)
    np.testing.assert_allclose(views[0]["target"], result["saved"]["target"], atol=1e-4)
    assert views[0]["is_default"], "the first view saved should open by default"
    assert "Saved view “north side”" in result["toast1"]
    np.testing.assert_allclose(result["restored"]["position"], result["saved"]["position"], atol=1e-3)
    assert "Opened view “north side”" in result["toast2"]


def test_clip_cuts_the_model_through_a_picked_point(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Digital Twin");
            {PICK}
            const before = viewer.scene.face_count;
            // Look straight along +X at the box's middle, so the cut is a known plane.
            viewer.applyView({{ position: [-5, 0.5, 0.25], target: [1, 0.5, 0.25], fov_deg: 50 }});
            await ODK.press("Clip");
            await ODK.pick("New cut");
            const front = frontVertices();
            // The picked vertex nearest x = 1, so about half the box lies in front of it.
            front.sort((p, q) => Math.abs(p[0] - 1) - Math.abs(q[0] - 1));
            await clickAt(front[0]);
            await ODK.answer("half");
            await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Cut")), 15000);
            await ODK.sleep(500);
            return {{ before, after: viewer.scene.face_count, clips: viewer.scene.clips_applied,
                      picked: front[0], chip: ODK.$('[data-role="scene-summary"]').innerText }};
        """)

    clips = api.list_clips()["clips"]
    assert [c["name"] for c in clips] == ["half"]
    plane = clips[0]["plane"]
    np.testing.assert_allclose(plane["point"], result["picked"], atol=1e-4)
    # Looking along +X, the plane keeps what is beyond the picked point.
    np.testing.assert_allclose(plane["normal"], [1, 0, 0], atol=1e-3)
    assert result["clips"] == ["half"] and "clipped: half" in result["chip"]
    assert 0 < result["after"] < result["before"]


def test_annotate_pins_a_finding_to_the_model(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Digital Twin");
            {PICK}
            await ODK.press("Annotate");
            const target = frontVertices()[0];
            await clickAt(target);
            await ODK.answer("Spalling");
            await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Pinned")), 10000);
            await ODK.sleep(800);
            return {{ target, panel: ODK.panel("Findings on the Model") }};
        """)

    found = [a for a in api.find_annotations()["annotations"] if a["source_type"] == "3d"]
    assert len(found) == 1 and found[0]["label"] == "Spalling"
    np.testing.assert_allclose(found[0]["geometry"]["coordinates"], result["target"], atol=1e-4)
    assert "Spalling" in result["panel"]


def test_point_cloud_view_and_measure_in_3d_from_measurements(api, tmp_path) -> None:
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Digital Twin");
            await ODK.press("Point Cloud");
            {PICK}
            await ODK.waitFor(() => viewer3d.currentViewer() && viewer3d.currentViewer().scene.kind === "pointcloud", 10000);
            const cloudChip = ODK.$('[data-role="scene-summary"]').innerText;
            await ODK.nav("Measurements");
            await ODK.press("Measure in 3D");
            const armed = await ODK.waitFor(() => {{ const v = viewer3d.currentViewer(); return v && v.tool === "measure" ? v : null; }}, 20000);
            return {{ cloudChip, armed: Boolean(armed), kind: armed && armed.scene.kind }};
        """)

    assert "reconstruction.ply" in result["cloudChip"] and "points" in result["cloudChip"]
    assert result["armed"], "Measure in 3D did not open the model with the measure tool armed"


def test_an_unreferenced_model_is_drawn_but_not_measured(api, tmp_path) -> None:
    (Path(api._session.project_root()) / "reconstruction" / "geo_anchor.json").unlink()
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run(f"""
            await ODK.nav("Digital Twin");
            {PICK}
            const front = frontVertices();
            await ODK.press("Measure");
            await clickAt(front[0]);
            await clickAt(front[front.length - 1]);
            const toast = await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Measure:") && t.includes("georeference")), 10000);
            return {{ chip: ODK.$('[data-role="scene-summary"]').innerText, toast }};
        """)

    assert "not georeferenced" in result["chip"]
    assert "structure-from-motion units" in result["toast"]
    assert not any(c["ok"] for c in cockpit.called("measure_on_model"))


def test_a_project_without_a_model_says_so(tmp_path) -> None:
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile") as cockpit:
        result = cockpit.run("""
            await ODK.nav("Digital Twin");
            await ODK.sleep(1500);
            await ODK.press("Save View");
            return { canvas: ODK.$$(".workspace .canvas .placeholder").map((n) => n.innerText).join(" | "),
                     toast: ODK.lastToast() };
        """)
    assert "No mesh in this project yet" in result["canvas"] or "No point cloud in this project yet" in result["canvas"]
    assert "No 3D model on screen" in result["toast"]
