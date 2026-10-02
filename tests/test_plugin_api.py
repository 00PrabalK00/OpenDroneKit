"""The plugin API: versions, manifests, failure isolation and the per-kind contracts.

Discovery is exercised through ``importlib.metadata`` itself: a real ``*.dist-info``
folder with an ``entry_points.txt`` is written onto ``sys.path``, so the plugins under
test are found exactly the way an installed package would be.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from opendronekit.errors import PluginError
from opendronekit.plugins import (
    PLUGIN_API_LEVEL,
    PluginHost,
    PluginKind,
    PluginManifest,
    PluginRegistry,
    PluginSpec,
    api_compatibility,
)


def _install_fake_distribution(root: Path, dist: str, module: str, source: str,
                               entry_points: dict[str, str]) -> None:
    """A minimal installed distribution: a module plus its dist-info metadata."""
    (root / f"{module}.py").write_text(source, encoding="utf-8")
    info = root / f"{dist.replace('-', '_')}-0.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.0.1\n",
                                   encoding="utf-8")
    lines = ["[opendronekit.plugins]"] + [f"{k} = {v}" for k, v in entry_points.items()]
    (info / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def site_dir(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield tmp_path
    for name in [m for m in sys.modules if m.startswith("odk_test_plugin")]:
        del sys.modules[name]


GOOD = '''
from opendronekit.plugins import PluginKind, PluginManifest, PluginSpec

class Section:
    title = "Good section"
    def build(self, context):
        return {"paragraphs": ["project " + context["project"]["name"]]}

manifest = PluginManifest(name="odk-test-good", version="1.2.3", api_version="1.0",
    plugins=(PluginSpec(PluginKind.REPORT_SECTION, "good_section", Section),))
'''

BROKEN_IMPORT = "raise RuntimeError('this plugin is broken at import')\n"

FUTURE = '''
from opendronekit.plugins import PluginKind, PluginSpec

def manifest():
    # Built lazily so the incompatible level is reported as this plugin's failure.
    return PluginSpec(PluginKind.EXPORTER, "future", object, api_version="1.9")
'''


class TestApiVersion:
    def test_level_is_major_minor(self):
        assert PLUGIN_API_LEVEL == "1.1"

    @pytest.mark.parametrize("requested,accepted", [
        ("1", True), ("1.0", True), ("1.1", True), ("1.2", False), ("2", False),
        ("0.9", False), ("one", False), ("", False),
    ])
    def test_same_major_and_no_newer_minor(self, requested, accepted):
        assert api_compatibility(requested)[0] is accepted

    def test_refusal_names_both_levels(self):
        ok, why = api_compatibility("2.0")
        assert not ok and "2.0" in why and PLUGIN_API_LEVEL in why

    def test_spec_and_manifest_refuse_an_incompatible_level(self):
        with pytest.raises(ValueError, match="newer than this host"):
            PluginSpec(PluginKind.EXPORTER, "x", object, api_version="1.5")
        with pytest.raises(ValueError, match="major versions differ"):
            PluginManifest("pkg", "1.0.0", "2",
                           (PluginSpec(PluginKind.EXPORTER, "x", object),))


class TestManifest:
    def test_manifest_versions_fill_the_plugins_that_do_not_state_one(self):
        m = PluginManifest("pkg", "0.4.0", "1.1", (
            PluginSpec(PluginKind.EXPORTER, "a", object),
            PluginSpec(PluginKind.EXPORTER, "b", object, version="9.9"),
        ))
        assert [s.version for s in m.plugins] == ["0.4.0", "9.9"]
        described = m.describe()
        assert described["name"] == "pkg" and described["api_version"] == "1.1"

    @pytest.mark.parametrize("kwargs,match", [
        ({"name": "", "version": "1.0"}, "name"),
        ({"name": "p", "version": ""}, "version"),
        ({"name": "p", "version": "latest"}, "version"),
    ])
    def test_a_manifest_needs_a_name_and_a_version(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PluginManifest(api_version="1.1",
                           plugins=(PluginSpec(PluginKind.EXPORTER, "x", object),), **kwargs)

    def test_an_empty_manifest_is_refused(self):
        with pytest.raises(ValueError, match="no plugins"):
            PluginManifest("p", "1.0", "1.1", ())


class TestFailureIsolation:
    def test_broken_plugins_are_reported_and_skipped_and_the_good_one_loads(self, site_dir):
        _install_fake_distribution(site_dir, "odk-test-good", "odk_test_plugin_good", GOOD,
                                   {"good": "odk_test_plugin_good:manifest"})
        _install_fake_distribution(site_dir, "odk-test-broken", "odk_test_plugin_broken",
                                   BROKEN_IMPORT, {"broken": "odk_test_plugin_broken:manifest"})
        _install_fake_distribution(site_dir, "odk-test-future", "odk_test_plugin_future",
                                   FUTURE, {"future": "odk_test_plugin_future:manifest"})
        _install_fake_distribution(site_dir, "odk-test-missing", "odk_test_plugin_missing",
                                   "x = 1\n", {"missing": "odk_test_plugin_missing:nothing_here"})

        host = PluginHost(PluginRegistry())
        report = host.load()            # must not raise

        assert [s.name for s in report.loaded] == ["good_section"]
        assert report.loaded[0].version == "1.2.3"
        failed = {f.source.split(" = ")[0]: f for f in report.failures}
        assert set(failed) == {"broken", "future", "missing"}
        assert failed["broken"].stage == "load"
        assert "this plugin is broken at import" in failed["broken"].error
        assert "odk-test-broken" in failed["broken"].distribution
        assert failed["future"].stage == "manifest" and "1.9" in failed["future"].error
        assert failed["missing"].stage == "load"

        status = host.status()
        assert {row["name"] for row in status["loaded"]} == {"good_section"}
        assert len(status["failed"]) == 3

    def test_a_duplicate_name_is_a_failure_not_a_crash(self, site_dir):
        _install_fake_distribution(site_dir, "odk-test-good", "odk_test_plugin_good", GOOD,
                                   {"one": "odk_test_plugin_good:manifest",
                                    "two": "odk_test_plugin_good:manifest"})
        report = PluginHost(PluginRegistry()).load()
        assert [s.name for s in report.loaded] == ["good_section"]
        assert report.failures[0].stage == "register"
        assert "already registered" in report.failures[0].error

    def test_the_legacy_discovery_call_no_longer_raises(self, site_dir):
        _install_fake_distribution(site_dir, "odk-test-broken", "odk_test_plugin_broken",
                                   BROKEN_IMPORT, {"broken": "odk_test_plugin_broken:manifest"})
        registry = PluginRegistry()
        assert registry.discover_entry_points() == []
        assert registry.failures and registry.failures[0].stage == "load"


def _host_with(*specs: PluginSpec) -> PluginHost:
    host = PluginHost(PluginRegistry())
    host.load(entry_points=[])
    for spec in specs:
        host.registry.register(spec)
    return host


class _RaisingFactory:
    def __init__(self):
        raise OSError("weights folder missing")


class TestContracts:
    def test_a_factory_that_raises_is_a_plugin_error(self):
        host = _host_with(PluginSpec(PluginKind.EXPORTER, "bad", _RaisingFactory))
        with pytest.raises(PluginError) as caught:
            host.export("bad", {"waypoints": [[0, 0, 10]]}, Path("."))
        assert caught.value.stage == "create" and "weights folder missing" in caught.value.reason
        assert host.status()["failed"][-1]["stage"] == "create"

    def test_an_object_missing_its_contract_is_refused(self):
        host = _host_with(PluginSpec(PluginKind.EXPORTER, "half", lambda: object()))
        with pytest.raises(PluginError, match="needs export, suffix"):
            host.export("half", {}, Path("."))

    def test_an_unknown_plugin_names_what_is_installed(self):
        host = _host_with()
        with pytest.raises(PluginError, match="no exporter plugin named 'nope'"):
            host.export("nope", {}, Path("."))

    def test_an_exporter_that_claims_a_file_it_did_not_write_is_refused(self, tmp_path):
        class Liar:
            suffix = ".txt"

            def export(self, plan, path):
                return str(path)            # writes nothing

        host = _host_with(PluginSpec(PluginKind.EXPORTER, "liar", Liar))
        with pytest.raises(PluginError, match="no non-empty file"):
            host.export("liar", {"waypoints": []}, tmp_path)

    def test_an_exporter_gets_a_copy_of_the_plan(self, tmp_path):
        class Mutator:
            suffix = ".txt"

            def export(self, plan, path):
                plan["waypoints"].clear()
                Path(path).write_text("x", encoding="utf-8")
                return str(path)

        plan = {"waypoints": [[1.0, 2.0, 3.0]]}
        host = _host_with(PluginSpec(PluginKind.EXPORTER, "mut", Mutator))
        assert host.export("mut", plan, tmp_path, "m") == [tmp_path / "m.txt"]
        assert plan["waypoints"] == [[1.0, 2.0, 3.0]]

    def _detector(self, detections, *, key="toy", digest="a" * 64):
        class Toy:
            model_key = key
            model_sha256 = digest

            def detect(self, image):
                return detections

        return _host_with(PluginSpec(PluginKind.MODEL, "toy", Toy))

    def test_a_detector_without_a_digest_is_refused(self):
        host = self._detector([], digest="")
        with pytest.raises(PluginError) as caught:
            host.detect("toy", np.zeros((10, 10, 3), np.uint8))
        assert caught.value.stage == "provenance"

    @pytest.mark.parametrize("bad", [
        {"label": "crack", "score": 1.5, "bbox_xyxy": [0, 0, 5, 5]},
        {"label": "crack", "score": 0.5, "bbox_xyxy": [0, 0, 50, 5]},
        {"label": "", "score": 0.5, "bbox_xyxy": [0, 0, 5, 5]},
        {"score": 0.5, "bbox_xyxy": [0, 0, 5, 5]},
    ])
    def test_detections_outside_the_contract_are_refused(self, bad):
        host = self._detector([bad])
        with pytest.raises(PluginError) as caught:
            host.detect("toy", np.zeros((10, 20, 3), np.uint8))
        assert caught.value.stage == "output"

    def test_valid_detections_carry_the_model_identity(self):
        host = self._detector([{"label": "crack", "score": 0.8, "bbox_xyxy": [1, 2, 9, 8],
                                "extra": "kept"}])
        found = host.detect("toy", np.zeros((10, 20, 3), np.uint8))
        assert found.model_used == "plugin:toy" and found.model_sha256 == "a" * 64
        assert found.detections[0]["extra"] == "kept" and found.image_size == (20, 10)

    def test_a_mission_template_must_name_a_builtin_template(self):
        class Recursive:
            def plan_options(self, aoi, **params):
                return {"template": "plugin:other"}

        host = _host_with(PluginSpec(PluginKind.MISSION_TYPE, "rec", Recursive))
        with pytest.raises(PluginError, match="built-in planner template"):
            host.mission_options("rec", [[0, 0], [0.001, 0], [0, 0.001]])

    def test_a_report_section_gets_a_copy_and_must_say_something(self):
        class Empty:
            title = "Empty"

            def build(self, context):
                context["project"]["name"] = "changed"
                return {"paragraphs": []}

        host = _host_with(PluginSpec(PluginKind.REPORT_SECTION, "empty", Empty))
        context = {"project": {"name": "real"}}
        with pytest.raises(PluginError, match="no paragraphs"):
            host.report_section("empty", context)
        assert context["project"]["name"] == "real"


class TestTheApplicationListsPlugins:
    def test_capabilities_and_list_plugins_show_loaded_and_failed(self, site_dir, tmp_path,
                                                                  monkeypatch):
        import opendronekit.plugins as plugins_module
        from app.api import Api
        from app.session import AppSession
        from app.store import ProjectStore

        _install_fake_distribution(site_dir, "odk-test-good", "odk_test_plugin_good", GOOD,
                                   {"good": "odk_test_plugin_good:manifest"})
        _install_fake_distribution(site_dir, "odk-test-broken", "odk_test_plugin_broken",
                                   BROKEN_IMPORT, {"broken": "odk_test_plugin_broken:manifest"})
        monkeypatch.setattr(plugins_module, "host", PluginHost(PluginRegistry()))

        api = Api(AppSession(ProjectStore(tmp_path / "state" / "projects.db")))
        caps = api.capabilities()
        assert caps["ok"] is True
        listing = caps["capabilities"]["plugins"]
        assert listing["api_version"] == PLUGIN_API_LEVEL
        assert [p["name"] for p in listing["loaded"]] == ["good_section"]
        assert listing["loaded"][0]["package"] == "odk-test-good"
        assert listing["failed"][0]["stage"] == "load"

        listed = api.list_plugins()
        assert listed["ok"] is True
        assert listed["plugins"][0]["name"] == "good_section"
        assert listed["plugins"][0]["version"] == "1.2.3"
        assert "broken at import" in listed["failed"][0]["error"]
