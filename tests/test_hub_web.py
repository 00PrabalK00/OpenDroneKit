"""Hub panels tested with real files, real HTTP and a real headless browser."""

from __future__ import annotations

from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading


ROOT = Path(__file__).parents[1]
WEB = ROOT / "app" / "web"


class _Ids(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: set[str] = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "script" and values.get("src"):
            self.scripts.add(values["src"])


class TestHubPanels:
    def test_hub_exposes_every_registry_panel_and_local_viewer_script(self):
        parser = _Ids()
        parser.feed((WEB / "hub.html").read_text(encoding="utf-8"))
        assert {
            "panel-projects", "panel-assets", "panel-maps", "panel-offline",
            "panel-viewer2d", "panel-viewer3d", "panel-pointcloud",
            "panel-thermal", "panel-digital-twin", "panel-timeline",
        } <= parser.ids
        assert {
            "js/map.js", "js/hub-api.js", "js/hub-viewers.js", "js/hub.js"
        } <= parser.scripts
        assert "hub.html" in (WEB / "index.html").read_text(encoding="utf-8")

    def test_viewer_module_parses_real_scene_cloud_twin_and_map_sources(self, tmp_path):
        scene = tmp_path / "scene.json"
        scene.write_text(json.dumps({
            "type": "odk-scene-3d", "units": "m", "crs_epsg": 32643,
            "vertices": [0, 0, 0, 1, 0, 0, 0, 1, 0],
            "colors": [1, 0, 0, 0, 1, 0, 0, 0, 1], "indices": [0, 1, 2],
            "overlays": [{"kind": "defects", "path": "defects.geojson"}],
        }), encoding="utf-8")
        cloud = tmp_path / "cloud.json"
        cloud.write_text(json.dumps({
            "points": [[0, 0, 0, 255, 0, 0], [1, 2, 3, 0, 255, 0]]
        }), encoding="utf-8")
        twin = tmp_path / "digital_twin.json"
        twin.write_text(json.dumps({
            "id": "site-1", "name": "Solar plant", "crs_epsg": 32643,
            "artifacts": {"orthomosaic": "ortho.tif", "point_cloud": ["cloud-1.json", "cloud-2.json"]},
            "surveys": [{"date": "2026-08-01"}, {"date": "2026-08-14", "measured_change": {"area_m2": 12}}],
            "annotations": [{"id": 1}], "defects": [{"id": 2}],
        }), encoding="utf-8")
        module = WEB / "js" / "hub-viewers.js"
        script = """
const fs=require('fs'); const v=require(process.argv[1]);
const scene=v.parseScene(JSON.parse(fs.readFileSync(process.argv[2])));
const cloud=v.parsePointChunk(JSON.parse(fs.readFileSync(process.argv[3])));
const twin=v.parseDigitalTwin(JSON.parse(fs.readFileSync(process.argv[4])));
const wms=v.providerTileUrl({kind:'wms',name:'survey',url:'http://localhost/wms',layers:'ortho'});
const wmts=v.providerTileUrl({kind:'wmts',name:'survey',url:'http://localhost/{TileMatrix}/{TileCol}/{TileRow}.png'});
process.stdout.write(JSON.stringify({scene,cloud,twin,wms,wmts}));
"""
        completed = subprocess.run(
            ["node", "-e", script, str(module), str(scene), str(cloud), str(twin)],
            cwd=ROOT, capture_output=True, text=True, timeout=20, check=True,
        )
        result = json.loads(completed.stdout)
        assert result["scene"]["primitive"] == "triangles"
        assert result["scene"]["crs_epsg"] == 32643
        assert result["cloud"]["count"] == 2
        assert len(result["twin"]["artifacts"]) == 3
        assert "BBOX={bbox-epsg-3857}" in result["wms"]
        assert result["wmts"].endswith("/{z}/{x}/{y}.png")


class _HubHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.headers.get("Authorization") == "Bearer hub-token"
        routes = {
            "/organizations/3/projects": [{"id": 4, "name": "Metro corridor"}],
            "/organizations/3/assets": [{"id": 8, "name": "Pier 8", "geometry": {"type": "Point", "coordinates": [77, 13]}}],
            "/tiles/providers": {"providers": {"street": {"max_zoom": 19}}},
            "/tiles/status": {"providers": {"street": {"tiles": 24, "bytes": 2048, "zoom_levels": [14]}}, "total_tiles": 24, "bytes": 2048},
            "/tiles/cache/cache-1": {"job_id": "cache-1", "done": True, "fetched": 4, "failed": 0, "percent": 100},
        }
        self._send(200 if self.path in routes else 404, routes.get(self.path, {"detail": "not found"}))

    def do_POST(self):
        assert self.headers.get("Authorization") == "Bearer hub-token"
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tiles/cache?organization_id=3":
            assert payload == {"provider": "street", "west": 77, "south": 12, "east": 78, "north": 13, "min_zoom": 14, "max_zoom": 14}
            self._send(202, {"job_id": "cache-1", "tiles": 4, "status": "caching"})
        else:
            self._send(404, {"detail": "not found"})


class TestHubRestClient:
    def test_projects_assets_and_offline_cache_use_real_http(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            module = WEB / "js" / "hub-api.js"
            base = f"http://127.0.0.1:{server.server_port}"
            script = """
const {HubApi}=require(process.argv[1]);
(async()=>{const api=new HubApi(process.argv[2],'hub-token');
const projects=await api.listProjects(3); const assets=await api.listAssets(3);
const before=await api.tileStatus(); const providers=await api.tileProviders();
const job=await api.cacheTiles({provider:'street',west:77,south:12,east:78,north:13,min_zoom:14,max_zoom:14},3);
const progress=await api.cacheStatus(job.job_id);
process.stdout.write(JSON.stringify({projects,assets,before,providers,job,progress}));})().catch(e=>{console.error(e);process.exit(1)});
"""
            completed = subprocess.run(
                ["node", "-e", script, str(module), base], cwd=ROOT,
                capture_output=True, text=True, timeout=20, check=True,
            )
            result = json.loads(completed.stdout)
            assert result["projects"][0]["name"] == "Metro corridor"
            assert result["assets"][0]["geometry"]["type"] == "Point"
            assert result["before"]["total_tiles"] == 24
            assert result["progress"]["fetched"] == 4
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class _TileSourceHandler(BaseHTTPRequestHandler):
    """A real local XYZ source used to prove download then offline replay."""

    tile = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )

    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path != "/0/0/0.png":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.tile)))
        self.end_headers()
        self.wfile.write(self.tile)


class TestHubOfflineTiles:
    def test_real_xyz_tile_is_downloaded_and_served_without_the_source(self, tmp_path, monkeypatch):
        from services.api.routers import tiles

        monkeypatch.setenv("ODK_STORAGE_PATH", str(tmp_path / "storage"))
        server = ThreadingHTTPServer(("127.0.0.1", 0), _TileSourceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setitem(tiles.PROVIDERS, "local-test", {
                "url": f"http://127.0.0.1:{server.server_port}/{{z}}/{{x}}/{{y}}.png",
                "max_zoom": 0,
                "attribution": "Local test source",
            })
            request = tiles.CacheRequest(
                provider="local-test", west=-1, south=-1, east=1, north=1,
                min_zoom=0, max_zoom=0,
            )
            job = tiles.CacheJob(job_id="real-local", provider="local-test", total=1)
            tiles._fetch_area(job, request)
            assert job.done and job.fetched == 1 and job.failed == 0
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        # The upstream server is now gone: this response can only come from disk.
        response = tiles.serve_tile(0, 0, 0, provider="local-test")
        assert response.status_code == 200
        assert response.headers["x-odk-tile"] == "cached"
        assert response.body == _TileSourceHandler.tile


class _BrowserHandler(BaseHTTPRequestHandler):
    viewer_source = (WEB / "js" / "hub-viewers.js").read_bytes()
    map_source = (WEB / "js" / "map.js").read_bytes()
    maplibre_source = (WEB / "vendor" / "maplibre" / "maplibre-gl.js").read_bytes()
    draw_source = (WEB / "vendor" / "maplibre" / "maplibre-gl-draw.js").read_bytes()

    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path == "/hub-viewers.js":
            body, content_type = self.viewer_source, "text/javascript"
        elif self.path == "/map.js":
            body, content_type = self.map_source, "text/javascript"
        elif self.path == "/maplibre-gl.js":
            body, content_type = self.maplibre_source, "text/javascript"
        elif self.path == "/maplibre-gl-draw.js":
            body, content_type = self.draw_source, "text/javascript"
        elif self.path == "/chunk-1.json":
            body, content_type = json.dumps({"points": [[0, 0, 0, 255, 0, 0]]}).encode(), "application/json"
        elif self.path == "/chunk-2.json":
            body, content_type = json.dumps({"points": [[1, 1, 1, 0, 255, 0]]}).encode(), "application/json"
        elif self.path == "/harness.html":
            body = b"""<!doctype html><body data-result="pending"><canvas id="scene" width="320" height="240"></canvas><canvas id="cloud" width="320" height="240"></canvas><script src="/hub-viewers.js"></script><script>
window.addEventListener('load',async()=>{try{
 const scene=new ODKHubViewers.OdkWebGLViewer(document.getElementById('scene'));
 scene.loadScene({type:'odk-scene-3d',units:'m',vertices:[0,0,0,1,0,0,0,1,0],colors:[1,0,0,0,1,0,0,0,1],indices:[0,1,2]}); scene.setClipping(-1);
 const cloud=new ODKHubViewers.OdkWebGLViewer(document.getElementById('cloud'));
 const progress=await cloud.loadPointManifest({type:'odk-point-cloud-manifest',units:'m',chunks:[{url:'/chunk-1.json'},{url:'/chunk-2.json'}]});
 document.body.dataset.result=`scene:${scene.vertexCount};points:${progress.loaded_points};chunks:${progress.loaded_chunks}`;
}catch(error){document.body.dataset.result='error:'+error.message;}});
</script></body>"""
            content_type = "text/html"
        elif self.path == "/map-harness.html":
            body = b"""<!doctype html><body data-result="pending"><div id="map" style="width:640px;height:400px"></div><script src="/maplibre-gl.js"></script><script src="/maplibre-gl-draw.js"></script><script src="/map.js"></script><script>
const survey={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'Polygon',coordinates:[[[77.59,12.97],[77.60,12.97],[77.60,12.98],[77.59,12.97]]]},properties:{survey:'T1'}}]};
const localStyle={version:8,sources:{},layers:[{id:'background',type:'background',paint:{'background-color':'#0d1117'}}]};
const viewer=new OdkMap('map',{center:[77.595,12.975],zoom:14,style:localStyle,onReady:()=>{try{viewer.addVector('survey-a',survey,0.4);viewer.addVector('survey-b',survey,0.6);viewer.setLayerOpacity('survey-a',0.25);viewer.setTool('measure-distance');document.body.dataset.result=`layers:${viewer.vectorLayers.size};tool:${viewer.tool}`;}catch(error){document.body.dataset.result='error:'+error.message;}}});
</script></body>"""
            content_type = "text/html"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestHubRealBrowser:
    def test_webgl_scene_and_progressive_point_chunks_render_in_edge(self, tmp_path):
        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        if not edge.is_file():
            edge_command = shutil.which("msedge") or shutil.which("chromium") or shutil.which("google-chrome")
            assert edge_command, "A Chromium browser is required for the Hub WebGL evidence test."
            edge = Path(edge_command)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _BrowserHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            completed = subprocess.run([
                str(edge), "--headless=new", "--enable-webgl", "--ignore-gpu-blocklist",
                "--use-angle=swiftshader", "--disable-software-rasterizer=false",
                f"--user-data-dir={tmp_path / 'edge-profile'}", "--virtual-time-budget=4000",
                "--dump-dom", f"http://127.0.0.1:{server.server_port}/harness.html",
            ], capture_output=True, text=True, timeout=30, check=True)
            assert 'data-result="scene:3;points:2;chunks:2"' in completed.stdout
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_2d_viewer_loads_real_geojson_layers_and_measurement_tool(self, tmp_path):
        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        if not edge.is_file():
            edge_command = shutil.which("msedge") or shutil.which("chromium") or shutil.which("google-chrome")
            assert edge_command, "A Chromium browser is required for the Hub map evidence test."
            edge = Path(edge_command)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _BrowserHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            completed = subprocess.run([
                str(edge), "--headless=new", "--enable-webgl", "--ignore-gpu-blocklist",
                "--use-angle=swiftshader", f"--user-data-dir={tmp_path / 'map-profile'}",
                "--virtual-time-budget=7000", "--dump-dom",
                f"http://127.0.0.1:{server.server_port}/map-harness.html",
            ], capture_output=True, text=True, timeout=40, check=True)
            assert 'data-result="layers:2;tool:measure-distance"' in completed.stdout
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
