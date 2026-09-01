"""The Developers screen reported a production stack as healthy, and showed an API key.

Five components, all green:

    REST API                 healthy     8000
    Processing workers       3 online    celery/redis
    PostgreSQL + PostGIS     healthy     geometry stored as GeoJSON text
    Object storage           healthy     s3 / minio
    Observability            healthy     /metrics

This is a desktop application. It runs jobs on threads inside its own process and stores
everything in SQLite and files on the local disk. None of those five services was
contacted, and none of them was running. A green health table is acted on: someone reads
it and concludes the platform is up.

It is also the clearest evidence for checking shapes over keeping lists. An earlier sweep
banned the string "redis 7.4". This row says "celery/redis" and passed straight through
it.

The Authentication panel showed a key: `odk_live_8f2a…`, with scopes and 1,204 requests
today. No key was ever issued. A credential-shaped string is worse than a wrong number,
because it tells the reader they *have* a key -- so the next thing they do is go looking
for the rest of it. And the request count implies traffic, and therefore a service.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js"


def strip_comments(js: str) -> str:
    out, i = [], 0
    while i < len(js):
        if js.startswith("/*", i):
            end = js.find("*/", i + 2)
            i = len(js) if end == -1 else end + 2
            continue
        if js.startswith("//", i):
            end = js.find("\n", i)
            i = len(js) if end == -1 else end
            continue
        out.append(js[i])
        i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def developers() -> str:
    source = WORKSPACES.read_text(encoding="utf-8")
    block = source.split("const developers = {")[1].split("\nconst settings")[0]
    return strip_comments(block)


CLAIMED = [
    ("celery", "a task broker this build does not run"),
    ("PostgreSQL", "a database this build does not use"),
    ("PostGIS", "a database extension this build does not use"),
    ("minio", "object storage this build does not use"),
    ("3 online", "worker processes that do not exist"),
    ("odk_live", "an API key that was never issued"),
    ("1,204", "a request count implying a running service"),
]


@pytest.mark.parametrize("needle,why", CLAIMED, ids=[n for n, _ in CLAIMED])
def test_no_infrastructure_is_claimed_that_is_not_there(developers, needle, why) -> None:
    assert needle not in developers, f"{needle} is still rendered -- {why}"


def test_nothing_on_this_screen_is_hardcoded_healthy(developers) -> None:
    """The word is a verdict. It may only appear where something was asked."""
    for match in re.finditer(r'"healthy"', developers):
        window = developers[max(0, match.start() - 600):match.start()]
        assert "live({" in window, "a component is reported healthy without being checked"


class TestItDescribesThisInstallation:
    def test_infrastructure_reads_capabilities(self, developers) -> None:
        block = developers.split('title: "Infrastructure"')[1][:2400]
        assert 'calls: ["capabilities"]' in block

    def test_the_capability_keys_are_real(self, developers) -> None:
        """engine_capabilities() reports pycolmap, colmap_binary, pycolmap_cuda,
        dense_stereo and open3d. There is no `colmap` key -- the first draft read one,
        and would have reported "missing" on a machine with COLMAP installed."""
        engine = (ROOT / "core" / "reconstruction_colmap.py").read_text(encoding="utf-8")
        api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
        block = developers.split('title: "Infrastructure"')[1][:2400]
        for field in set(re.findall(r"\bcaps\.(\w+)", block)):
            produced = (f'"{field}"' in engine.split("def engine_capabilities")[1][:1200]
                        or f'caps["{field}"]' in api)
            assert produced, f"nothing puts {field} in the capabilities payload"

    def test_it_says_jobs_run_in_process(self, developers) -> None:
        block = developers.split('title: "Infrastructure"')[1][:2400]
        assert "in-process" in block or "threads" in block


class TestAuthenticationShowsNoCredential:
    def test_it_renders_no_key_shaped_string(self, developers) -> None:
        block = developers.split('title: "Authentication"')[1][:1400]
        assert not re.search(r'"[a-z]{2,}_[a-z]{2,}_[0-9a-f]{4}', block), (
            "a credential-shaped string is rendered"
        )

    def test_it_says_where_keys_actually_live(self, developers) -> None:
        """Removing the fake key must not leave the panel silent about how auth works."""
        block = developers.split('title: "Authentication"')[1][:1400]
        assert "REST service" in block


class TestTheRestReadsTheRegistries:
    @pytest.mark.parametrize("panel,call", [
        ("Webhooks", "list_webhooks"),
        ("Plugins", "list_plugins"),
        ("Event Stream", "audit_log"),
    ])
    def test_panel_asks(self, developers, panel, call) -> None:
        block = developers.split(f'title: "{panel}"')[1][:1600]
        assert f'"{call}"' in block

    def test_a_webhook_lists_all_its_events(self, developers) -> None:
        """The record's column is `events`, a list. Reading the singular `event` would
        have printed an em dash for every correctly configured webhook."""
        ops = (ROOT / "app" / "desktop_ops.py").read_text(encoding="utf-8")
        shape = ops.split("def list_webhooks")[1].split("\ndef ")[0]
        assert '"events"' in shape
        block = developers.split('title: "Webhooks"')[1][:1600]
        assert "hook.events" in block
