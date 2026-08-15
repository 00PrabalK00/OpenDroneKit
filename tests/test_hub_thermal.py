"""Thermal Hub evidence from core radiometry through a real browser."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from core.india_geospatial import IndiaPackRefused
from core.india_thermal import ThermalRegistration, write_radiometric_thermal_map
from core.thermal import Calibration, ThermalImage, kelvin_to_raw, raw_to_celsius


ROOT = Path(__file__).parents[1]
VIEWERS = ROOT / "app" / "web" / "js" / "hub-viewers.js"


@pytest.fixture
def thermal_artifact(tmp_path):
    expected = np.array([[20.0, 25.0, 35.0], [22.0, 30.0, 42.0]], dtype=np.float64)
    calibration = Calibration(
        r1=21106.77, r2=0.012545258, b=1501.0, f=1.0, o=-7340.0,
        emissivity=1.0, reflected_temp_c=20.0, transmission=1.0,
    )
    raw = kelvin_to_raw(expected + 273.15, calibration)
    image = ThermalImage(
        celsius=raw_to_celsius(raw, calibration), calibration=calibration,
        source="radiometric-sidecar.json",
    )
    package = write_radiometric_thermal_map(
        image, tmp_path / "thermal", epsg=32643,
        west=500000.0, north=2000000.0, pixel_size=1.0,
        registration=ThermalRegistration(
            method="calibrated_homography", rgb_width=3, rgb_height=2,
            residual_px=0.35, validated_by="checkerboard field calibration",
        ),
    )
    manifest_path = Path(package.artifact_paths[1])
    return package, expected, json.loads(manifest_path.read_text(encoding="utf-8"))


class TestThermalMapArtifact:
    def test_core_temperature_measurements_reach_geotiff_and_hub_json_unchanged(self, thermal_artifact):
        package, expected, manifest = thermal_artifact
        with rasterio.open(package.artifact_paths[0]) as raster:
            assert raster.crs.to_epsg() == 32643
            assert raster.tags()["ODK_VALUE_UNIT"] == "celsius"
            assert raster.tags()["ODK_INTERPOLATED"] == "false"
            assert raster.read(1) == pytest.approx(expected.astype(np.float32), abs=0.01)

        assert manifest["type"] == "odk-thermal-map"
        assert manifest["unit"] == "celsius"
        assert manifest["temperature_source"] == "radiometric_counts_via_core.thermal"
        assert manifest["interpolated"] is False
        assert np.asarray(manifest["values"]).reshape(2, 3) == pytest.approx(expected, abs=0.01)
        assert manifest["registration"]["validated"] is True
        assert manifest["registration"]["residual_px"] == pytest.approx(0.35)

    def test_metric_thermal_package_refuses_a_geographic_crs(self, tmp_path):
        calibration = Calibration(r1=21106.77, r2=0.012545258, b=1501, f=1, o=-7340)
        image = ThermalImage(np.full((2, 2), 25.0), calibration)
        with pytest.raises(IndiaPackRefused, match="projected CRS"):
            write_radiometric_thermal_map(
                image, tmp_path / "thermal", epsg=4326,
                west=77.0, north=13.0, pixel_size=0.0001,
            )


class TestThermalProjectionContract:
    def test_real_thermal_artifact_is_sampled_onto_matching_projected_geometry(self, thermal_artifact, tmp_path):
        _, _, manifest = thermal_artifact
        manifest_path = tmp_path / "thermal.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        scene_path = tmp_path / "scene.json"
        scene_path.write_text(json.dumps({
            "type": "odk-scene-3d", "units": "m", "crs_epsg": 32643,
            "coordinate_frame": "local_projected", "origin_xy": [500000, 2000000],
            "vertices": [0.5, -0.5, 0, 1.5, -0.5, 0, 0.5, -1.5, 1],
            "colors": [1, 1, 1, 1, 1, 1, 1, 1, 1], "indices": [0, 1, 2],
        }), encoding="utf-8")
        script = """
const fs=require('fs'), v=require(process.argv[1]);
const thermal=v.parseThermalMap(JSON.parse(fs.readFileSync(process.argv[2])));
const scene=v.parseScene(JSON.parse(fs.readFileSync(process.argv[3])));
const model=v.projectThermalOntoScene(scene,thermal);
let mismatch=''; try{v.projectThermalOntoScene({...scene,crs_epsg:32644},thermal);}catch(error){mismatch=error.message;}
let unregistered=''; try{v.validateRgbThermalRegistration({...thermal,registration:null},3,2);}catch(error){unregistered=error.message;}
process.stdout.write(JSON.stringify({temps:model.temperatures_c,projection:model.thermal_projection,mismatch,unregistered}));
"""
        completed = subprocess.run(
            ["node", "-e", script, str(VIEWERS), str(manifest_path), str(scene_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=20, check=True,
        )
        result = json.loads(completed.stdout)
        assert result["temps"] == pytest.approx([20.0, 25.0, 22.0], abs=0.01)
        assert result["projection"]["sampled_vertices"] == 3
        assert result["projection"]["sampling"] == "nearest_measured_cell"
        assert result["projection"]["interpolated"] is False
        assert "CRS values do not match" in result["mismatch"]
        assert "explicit validated registration" in result["unregistered"]


class _ThermalBrowserHandler(BaseHTTPRequestHandler):
    viewer_source = VIEWERS.read_bytes()
    thermal_payload = {}

    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path == "/hub-viewers.js":
            body, content_type = self.viewer_source, "text/javascript"
        elif self.path == "/harness.html":
            payload = json.dumps(self.thermal_payload).replace("</", "<\\/")
            body = f"""<!doctype html><body data-result="pending">
<canvas id="map" width="300" height="200"></canvas>
<div id="comparison"><canvas id="rgb" width="300" height="200"></canvas><canvas id="thermal" width="300" height="200"></canvas></div>
<canvas id="model" width="300" height="200"></canvas><script src="/hub-viewers.js"></script><script>
try{{
 const map=ODKHubViewers.parseThermalMap({payload});
 const mapViewer=new ODKHubViewers.OdkThermalCanvas(document.getElementById('map')); mapViewer.load(map);
 const rgbSource=document.createElement('canvas'); rgbSource.width=3; rgbSource.height=2;
 const rgbContext=rgbSource.getContext('2d'); rgbContext.fillStyle='#708090'; rgbContext.fillRect(0,0,3,2);
 const comparison=new ODKHubViewers.OdkThermalComparison(document.getElementById('comparison'),document.getElementById('rgb'),document.getElementById('thermal'));
 comparison.load(rgbSource,map); comparison.setMode('swipe'); comparison.setSwipe(0.4); const clip=comparison.thermalViewer.canvas.style.clipPath;
 comparison.setMode('side-by-side'); const side=document.getElementById('comparison').classList.contains('side-by-side');
 comparison.setMode('overlay'); comparison.setOpacity(0.4); comparison.setView({{zoom:2,panX:3,panY:-2}});
 const scene={{type:'odk-scene-3d',units:'m',crs_epsg:32643,coordinate_frame:'local_projected',origin_xy:[500000,2000000],vertices:[0.5,-0.5,0,1.5,-0.5,0,0.5,-1.5,1],colors:[1,1,1,1,1,1,1,1,1],indices:[0,1,2]}};
 const projected=ODKHubViewers.projectThermalOntoScene(scene,map);
 const model=new ODKHubViewers.OdkWebGLViewer(document.getElementById('model')); model.loadScene(projected);
 const alpha=mapViewer.context.getImageData(50,50,1,1).data[3]; const linked=comparison.viewState().zoom===comparison.thermalViewer.viewState().zoom;
 document.body.dataset.result=`map:${{alpha>0}};side:${{side}};clip:${{clip.startsWith('inset')}};mode:${{document.getElementById('comparison').dataset.mode}};opacity:${{comparison.thermalViewer.canvas.style.opacity}};linked:${{linked}};sampled:${{projected.thermal_projection.sampled_vertices}};vertices:${{model.vertexCount}}`;
}}catch(error){{document.body.dataset.result='error:'+error.message;}}
</script></body>""".encode("utf-8")
            content_type = "text/html"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestThermalRealBrowser:
    def test_map_comparison_modes_linked_zoom_and_3d_projection_render(self, thermal_artifact, tmp_path):
        _, _, manifest = thermal_artifact
        _ThermalBrowserHandler.thermal_payload = manifest
        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        if not edge.is_file():
            command = shutil.which("msedge") or shutil.which("chromium") or shutil.which("google-chrome")
            assert command, "A Chromium browser is required for thermal viewer evidence."
            edge = Path(command)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ThermalBrowserHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            completed = subprocess.run([
                str(edge), "--headless=new", "--enable-webgl", "--ignore-gpu-blocklist",
                "--use-angle=swiftshader", "--disable-software-rasterizer=false",
                f"--user-data-dir={tmp_path / 'thermal-edge'}", "--virtual-time-budget=4000",
                "--dump-dom", f"http://127.0.0.1:{server.server_port}/harness.html",
            ], capture_output=True, text=True, timeout=30, check=True)
            assert 'data-result="map:true;side:true;clip:true;mode:overlay;opacity:0.4;linked:true;sampled:3;vertices:3"' in completed.stdout
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

