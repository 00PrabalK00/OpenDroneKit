"""The classic window frames what is imported and searches with no network.

Audit (classic UI, the default launch): after importing a layer the map stayed at the
world view (z2), so the layer was a dot nobody could find and an area of interest drawn
at that zoom covered 2.7 billion hectares. Place search sent every query to
nominatim.openstreetmap.org and, with no network, answered only "URLError ... blocked".

These drive index.html in headless Chromium against the real Api with every host but
loopback unresolvable, which is the field laptop with no signal.
"""

from __future__ import annotations

import json
import socket

from cockpit_harness import Cockpit, make_api

SITE = [[-81.7535, 41.3032], [-81.7505, 41.3032], [-81.7505, 41.3052], [-81.7535, 41.3052], [-81.7535, 41.3032]]


def _layer(tmp_path):
    path = tmp_path / "site_boundary.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": "boundary"},
        "geometry": {"type": "Polygon", "coordinates": [SITE]}}]}), encoding="utf-8")
    return path


def test_an_imported_layer_is_framed(tmp_path) -> None:
    api = make_api(tmp_path)
    with Cockpit(api, tmp_path / "profile", page="index.html", window_size="1600,900") as cockpit:
        cockpit.picks["file"] = str(_layer(tmp_path))
        result = cockpit.run("""
            // Every status line, so the test waits for the map itself rather than for
            // whichever message happens to be showing.
            const seen = [ODK.$("#status-message").textContent];
            new MutationObserver(() => seen.push(ODK.$("#status-message").textContent))
              .observe(ODK.$("#status-message"), { childList: true, characterData: true, subtree: true });
            // 'Map ready' and 'OpenDroneKit ready.' are ambient: with no network the
            // basemap fallback note is newer, so they are logged rather than shown.
            await ODK.waitFor(() => ODK.$("#console-log").textContent.includes("Map ready"), 30000);
            await ODK.waitFor(() => window.odk && window.odk.ready, 30000);
            const before = ODK.$("#status-scale").textContent;
            window.odk.onMenu({ action: "data.import_vector" });
            const said = await ODK.waitFor(() => {
              const t = ODK.$("#status-message").textContent;
              return t.startsWith("Layer added") ? t : null; }, 15000);
            // The zoom readout is written when the camera stops moving.
            const after = await ODK.waitFor(() => ODK.$("#status-scale").textContent, 20000);
            return { before, said, after };
        """)
    assert result["said"] == "Layer added and framed: site_boundary", result
    zoom = float(result["after"].lstrip("z"))
    assert zoom >= 14, f"the map did not frame the layer: {result}"


def test_place_search_offline_finds_the_project_and_says_the_remote_search_did_not_happen(
        tmp_path, monkeypatch) -> None:
    # No network for the application either: every name but loopback fails to resolve,
    # as it does on a laptop with no signal. The search itself runs for real.
    real = socket.getaddrinfo

    def no_network(host, *args, **kwargs):
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise socket.gaierror(11001, "getaddrinfo failed")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    api = make_api(tmp_path)
    layer = api.add_layer_from_file(str(_layer(tmp_path)))
    assert layer["ok"], layer
    with Cockpit(api, tmp_path / "profile", page="index.html", window_size="1600,900") as cockpit:
        result = cockpit.run("""
            // Every status line, so the test waits for the map itself rather than for
            // whichever message happens to be showing.
            const seen = [ODK.$("#status-message").textContent];
            new MutationObserver(() => seen.push(ODK.$("#status-message").textContent))
              .observe(ODK.$("#status-message"), { childList: true, characterData: true, subtree: true });
            // 'Map ready' and 'OpenDroneKit ready.' are ambient: with no network the
            // basemap fallback note is newer, so they are logged rather than shown.
            await ODK.waitFor(() => ODK.$("#console-log").textContent.includes("Map ready"), 30000);
            await ODK.waitFor(() => window.odk && window.odk.ready, 30000);
            ODK.$("#place-search").value = "site_bound";
            ODK.$("#btn-place-search").click();
            const rows = await ODK.waitFor(() => ODK.$$("#place-results li[data-index]").length ? ODK.$$("#place-results li") : null, 30000);
            const text = (rows || []).map((r) => r.innerText);
            ODK.$("#place-results li[data-index]").click();
            await ODK.waitFor(() => ODK.$("#status-scale").textContent, 20000);
            return { text, seen, scale: ODK.$("#status-scale").textContent };
        """)
    joined = " | ".join(result["text"])
    assert "site_boundary" in joined and "vector layer" in joined, joined
    # The remote search really was attempted and really failed, and the list says so.
    assert "could not be reached" in joined and "nothing was looked up online" in joined
    assert "Moved to site_boundary" in result["seen"]
    assert float(result["scale"].lstrip("z")) >= 14
