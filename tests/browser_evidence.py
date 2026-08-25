"""One place that knows how to start a headless Chromium for the evidence tests.

Five tests across four files render a real page and read the DOM back, because a
viewer that is asserted about in Python and never executed in a browser is not
evidence of anything. They were each spelling out their own argument list, which is
how they came to disagree with each other and with CI.

The sandbox flags are the reason this module exists. GitHub's Ubuntu runners forbid
unprivileged user namespaces, so Chromium's zygote sandbox aborts -- SIGABRT, exit 6,
before a single page loads. The failure looks like a broken viewer and is nothing of
the kind. Making those tests skip instead would be worse than leaving them red: a skip
counts as no evidence, so it would silently downgrade every row that names them.

--no-sandbox is safe here and only here: the pages are local fixtures served from a
loopback port by the test itself, so there is no untrusted content for the sandbox to
contain.
"""

from __future__ import annotations

import shutil
from pathlib import Path

WINDOWS_EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

# Chromium cannot start a zygote sandbox on a GitHub runner, and /dev/shm in a
# container is too small for the shared memory it expects.
SANDBOX_FLAGS = ["--no-sandbox", "--disable-dev-shm-usage"]

# Runners have no GPU, so WebGL has to come from a software renderer or the context
# never gets created and the harness reports a scene that never rendered.
WEBGL_FLAGS = [
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--use-angle=swiftshader",
    "--disable-software-rasterizer=false",
]


def chromium(reason: str) -> Path:
    """The browser to render with, or a failure naming what went unverified."""
    if WINDOWS_EDGE.is_file():
        return WINDOWS_EDGE
    found = (
        shutil.which("msedge")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )
    assert found, f"A Chromium browser is required for {reason}."
    return Path(found)


def dump_dom_command(
    reason: str,
    profile_dir: Path | str,
    url: str,
    *,
    virtual_time_ms: int,
    webgl: bool = True,
) -> list[str]:
    """The argv that loads `url` and prints the settled DOM to stdout.

    --virtual-time-budget is what makes the run deterministic: the page's timers are
    advanced as fast as they can be evaluated rather than in wall-clock, so a loaded
    runner cannot produce a half-rendered DOM and a mystery assertion failure.
    """
    argv = [str(chromium(reason)), "--headless=new", *SANDBOX_FLAGS,
            f"--user-data-dir={profile_dir}"]
    if webgl:
        argv += WEBGL_FLAGS
    argv += [f"--virtual-time-budget={virtual_time_ms}", "--dump-dom", url]
    return argv
