"""Temperature Range sets the range a real radiometric frame is drawn in.

Audit: the button answered "scale_thermal is implemented, but it takes a radiometric
raster and no thermal product is selectable in this build" and did nothing. It now takes
a radiometric sidecar from disk (raw counts plus the camera's Planck constants, which is
what core/thermal reads), asks for the range mode, and shows the Api's answer, including
the warning when the range saturates the hottest pixels.

The frame is written here as counts that convert to a known temperature field: a 20 C
roof with one 80 C flue. The page is the real cockpit against the real Api.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cockpit_harness import Cockpit, make_api
from core.thermal import Calibration, kelvin_to_raw

METADATA = {
    "PlanckR1": 17096.453, "PlanckR2": 0.046642166, "PlanckB": 1428.0,
    "PlanckF": 1.0, "PlanckO": -342.0,
    "Emissivity": 1.0, "ReflectedApparentTemperature": 20.0,
}


def _frame(tmp_path):
    celsius = np.full((40, 40), 20.0)
    celsius[:20] += np.linspace(0, 3, 40)          # a gentle 3 C gradient on the roof
    celsius[30:32, 30:32] = 80.0                   # the flue
    raw = kelvin_to_raw(celsius + 273.15, Calibration.from_metadata(METADATA))
    path = tmp_path / "FLIR0001.json"
    path.write_text(json.dumps({"raw": np.asarray(raw).tolist(), "metadata": METADATA}),
                    encoding="utf-8")
    return path


def _press(api, tmp_path, frame, choice, answers=()):
    with Cockpit(api, tmp_path / "profile") as cockpit:
        cockpit.picks["file"] = str(frame)
        typed = "".join(f"await ODK.answer({json.dumps(a)});" for a in answers)
        result = cockpit.run(f"""
            await ODK.nav("Thermal");
            await ODK.press("Temperature Range");
            await ODK.pick({json.dumps(choice)});
            {typed}
            return {{ toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Range") || t.startsWith("Temperature Range:")), 10000) }};
        """)
    return result["toast"], cockpit.called("scale_thermal")


def test_a_manual_range_below_the_flue_says_it_hides_the_hottest(tmp_path) -> None:
    api = make_api(tmp_path)
    frame = _frame(tmp_path)
    toast, calls = _press(api, tmp_path, frame, "Manual (enter min and max)", ["18", "26"])
    assert calls and calls[-1]["ok"], calls
    assert calls[-1]["args"][:4] == [str(frame), "manual", 18, 26]
    expected = api.scale_thermal(str(frame), "manual", 18, 26)
    assert toast.startswith("Range 18.0 to 26.0 °C (manual, ironbow)"), toast
    assert f"the frame spans {expected['scale']['data_min_c']:.1f} to 80.0 °C" in toast
    assert expected["warning"] in toast and "saturates 4 pixel(s)" in toast


def test_an_automatic_range_reports_the_api_range(tmp_path) -> None:
    api = make_api(tmp_path)
    frame = _frame(tmp_path)
    toast, calls = _press(api, tmp_path, frame, "Automatic (1st to 99th percentile)")
    assert calls[-1]["ok"] and calls[-1]["args"][1] == "auto"
    s = api.scale_thermal(str(frame), "auto")["scale"]
    assert toast.startswith(f"Range {s['min_c']:.1f} to {s['max_c']:.1f} °C"), toast


def test_a_thermal_jpeg_is_refused_with_the_reason(tmp_path) -> None:
    api = make_api(tmp_path)
    jpeg = tmp_path / "FLIR0002.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")
    with Cockpit(api, tmp_path / "profile") as cockpit:
        cockpit.picks["file"] = str(jpeg)
        result = cockpit.run("""
            await ODK.nav("Thermal");
            await ODK.press("Temperature Range");
            await ODK.pick("Anomaly (mean ± 2σ)");
            return { toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Temperature Range:")), 10000) };
        """)
    assert "exiftool" in result["toast"] and "palette, not temperatures" in result["toast"]


def test_the_temperature_map_a_thermal_run_wrote_is_accepted(tmp_path) -> None:
    """The toolkit's own product, thermal_temperature.tif, was refused as "not a JSON"."""
    from core.india_thermal import write_radiometric_thermal_map
    from core.thermal import load_radiometric_file

    image = load_radiometric_file(_frame(tmp_path))
    written = write_radiometric_thermal_map(image, tmp_path / "thermal", epsg=32617,
                                            west=500000.0, north=4570000.0, pixel_size=0.1)
    geotiff = next(Path(p) for p in written.artifact_paths if str(p).endswith("thermal_temperature.tif"))
    api = make_api(tmp_path)
    toast, calls = _press(api, tmp_path, geotiff, "Manual (enter min and max)", ["18", "26"])
    assert calls and calls[-1]["ok"], calls
    # The same field as the sidecar it was written from, so the same answer.
    expected = api.scale_thermal(str(_frame(tmp_path)), "manual", 18, 26)
    assert toast.startswith("Range 18.0 to 26.0 °C (manual, ironbow)"), toast
    assert f"the frame spans {expected['scale']['data_min_c']:.1f} to 80.0 °C" in toast
    assert "saturates 4 pixel(s)" in toast


def _press_and_answer_unit(api, tmp_path, raster, unit):
    with Cockpit(api, tmp_path / "profile") as cockpit:
        cockpit.picks["file"] = str(raster)
        result = cockpit.run(f"""
            await ODK.nav("Thermal");
            await ODK.press("Temperature Range");
            await ODK.pick("Automatic (1st to 99th percentile)");
            await ODK.pick({json.dumps(unit)});
            return {{ toast: await ODK.waitFor(() => ODK.toasts().find((t) => t.startsWith("Range") || t.startsWith("Temperature Range:")), 10000) }};
        """)
    return result["toast"], cockpit.called("scale_thermal")


def test_a_raster_that_is_not_a_temperature_map_is_refused(tmp_path, surface_pair) -> None:
    """A DSM says nothing about temperature: the user is asked, and declining reads nothing."""
    api = make_api(tmp_path)
    toast, calls = _press_and_answer_unit(api, tmp_path, surface_pair["dsm"],
                                          "Not temperatures / not sure")
    assert len(calls) == 1 and not calls[-1]["ok"], calls
    assert "does not say it holds temperatures" in calls[-1]["error"]
    assert api.scale_thermal(str(surface_pair["dsm"]), "auto")["needs_unit_confirmation"] is True
    assert toast.startswith("Temperature Range:") and "not confirmed" in toast, toast


def test_a_raster_whose_unit_the_user_confirms_is_read(tmp_path, surface_pair) -> None:
    api = make_api(tmp_path)
    toast, calls = _press_and_answer_unit(api, tmp_path, surface_pair["dsm"], "Degrees Celsius")
    assert calls[-1]["ok"] and calls[-1]["args"][-1] == "celsius", calls
    assert toast.startswith("Range "), toast
    direct = api.scale_thermal(str(surface_pair["dsm"]), "auto", confirmed_unit="celsius")
    assert direct["provenance"]["unit_source"] == "confirmed_by_user"

