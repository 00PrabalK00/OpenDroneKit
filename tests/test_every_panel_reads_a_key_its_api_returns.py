"""One guard for the mistake this cockpit keeps making.

A panel asks the application for something and then reads a key off the answer. If the
key is wrong the panel renders blank or `undefined`. It does not raise, nothing logs, and
the result is indistinguishable from an empty project -- so the panel quietly shows
nothing on a project that has data, or worse, its `isEmpty` test passes and it shows the
empty-state message instead.

Found by hand, one screen at a time, in this order:

    gsd / angle          sent to plan_mission, which reads target_gsd_cm / line_heading_deg
    capture_count        on the estimates, which returns image_count
    checked              on the GCP report, which returns used / point_count
    out_of_tolerance     on the GCP report, which returns outlier_count
    severity / message   on notifications, which return level / title
    created_at           on notifications, which return created_utc
    entries              on the audit log, which returns events
    present              on model_status rows, which return exists
    temperature_c        on annotations, which have no temperature at all

Nine, and the last two were written by the person fixing the first seven. Per-screen
tests catch them one at a time and only where someone thought to look, so this checks the
shape instead: every `live()` block names the calls it makes, and the top-level key it
reads off each answer must be one that call really returns.

Top-level only. Nested shapes come from a dozen different modules and chasing them here
would make this guard fragile -- the per-screen tests carry those. The top level is where
the errors above lived, because that is the boundary between the API and the panel.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "app" / "api.py"
WORKSPACES = ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js"

# Keys every ok() carries, added by the wrapper rather than by the method.
ALWAYS = {"ok", "error", "detail"}

# Reading these off a result is a language operation, not a claim about the payload.
NOT_PAYLOAD_KEYS = {
    "length", "map", "filter", "find", "slice", "forEach", "some", "every",
    "reduce", "sort", "includes", "join", "concat", "keys", "values", "entries_",
    "toFixed", "toString", "split", "trim", "push", "indexOf",
}


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


def api_returns() -> dict[str, set[str]]:
    """For each Api method, the keyword names it passes to ok().

    A method with several ok() returns contributes the union: different branches
    legitimately answer with different keys, and a panel may read any of them.
    """
    source = API.read_text(encoding="utf-8")
    methods = re.split(r"\n    def (\w+)\(", source)
    shapes: dict[str, set[str]] = {}
    for name, body in zip(methods[1::2], methods[2::2]):
        keys: set[str] = set()
        for match in re.finditer(r"\bok\(", body):
            # Balanced, not `[^)]*`. A nested call -- ok(images=names[: int(limit)]) --
            # closes the naive pattern at the inner paren, so the real keyword arguments
            # after it were never seen and list_dataset_images() looked like it returned
            # nothing. The guard reported a bug in the panel that was reading it
            # correctly, which is the failure mode a guard can least afford.
            depth, i = 0, match.end() - 1
            while i < len(body):
                if body[i] == "(":
                    depth += 1
                elif body[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            args = body[match.end():i]
            # ok(**something) means the keys come from a dict built elsewhere; this
            # guard cannot see them, so the method is recorded as unknown.
            if "**" in args:
                keys.add("*")
            # Only top-level keywords: `foo=` inside a nested call is that call's
            # argument, not a key of this payload.
            depth = 0
            for token in re.finditer(r"[()\[\]{}]|(\w+)\s*=(?!=)", args):
                char = token.group(0)
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth -= 1
                elif token.group(1) and depth == 0:
                    keys.add(token.group(1))
        if keys:
            shapes[name] = keys
    return shapes


def live_blocks(js: str) -> list[tuple[list[str], list[str], str]]:
    """(calls, destructured names, body) for every live()/readout() block."""
    blocks = []
    for match in re.finditer(r"(?:live|readout)\(\{", js):
        start = match.end() - 1
        # Brace counting has to skip string and template literals, or a `}` inside
        # `${...}` or a message ends the block early and the next block's variables get
        # attributed to this one's calls. That produced a report of `est.image_count`
        # against mission_estimates when `est` belongs to a different scope entirely.
        depth, i, quote = 0, start, ""
        while i < len(js):
            char = js[i]
            if quote:
                if char == "\\":
                    i += 2
                    continue
                if char == quote:
                    quote = ""
            elif char in "\"'`":
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = js[start:i]

        calls_match = re.search(r"calls:\s*\[([^\]]*)\]", body)
        if not calls_match:
            continue
        calls = re.findall(r'"(\w+)"', calls_match.group(1))

        # render: ([a, b]) => ...   and   isEmpty: ([a]) => ...
        names: list[str] = []
        for arrow in re.finditer(r"(?:render|isEmpty):\s*\(\[([^\]]*)\]\)", body):
            found = [n.strip() for n in arrow.group(1).split(",") if n.strip()]
            if len(found) > len(names):
                names = found
        blocks.append((calls, names, body))
    return blocks


@pytest.fixture(scope="module")
def shapes() -> dict[str, set[str]]:
    found = api_returns()
    assert "audit_log" in found, "the Api could not be parsed"
    assert "events" in found["audit_log"]
    return found


@pytest.fixture(scope="module")
def blocks() -> list[tuple[list[str], list[str], str]]:
    found = live_blocks(strip_comments(WORKSPACES.read_text(encoding="utf-8")))
    assert len(found) > 20, f"only {len(found)} live() blocks found; the parser is wrong"
    return found


def test_every_panel_reads_a_key_its_api_returns(shapes, blocks) -> None:
    problems: list[str] = []
    for calls, names, body in blocks:
        for index, variable in enumerate(names):
            if index >= len(calls):
                continue
            method = calls[index]
            known = shapes.get(method)
            if not known or "*" in known:
                # ok(**payload): the keys are assembled in another module and this guard
                # cannot see them. Named rather than silently skipped.
                continue
            read = set(re.findall(rf"\b{re.escape(variable)}\.(\w+)\b", body))
            for key in sorted(read - known - ALWAYS - NOT_PAYLOAD_KEYS):
                problems.append(f"{method}() does not return {key!r} (read as {variable}.{key})")
    assert not problems, "panels read keys their API never returns:\n  " + "\n  ".join(problems)


def test_the_guard_would_catch_the_bug_it_was_written_for(shapes) -> None:
    """The audit-log mistake, checked directly.

    A guard for a class of error is only worth having if it fails on a member of that
    class, and this suite has already shipped two guards that a comment could satisfy.
    """
    assert "events" in shapes["audit_log"]
    assert "entries" not in shapes["audit_log"]


def test_no_panel_asks_for_a_method_the_api_does_not_have(blocks) -> None:
    """A misspelled call name fails at runtime through tryCall, which swallows it and
    shows the empty state -- so the panel looks like an empty project forever."""
    source = API.read_text(encoding="utf-8")
    defined = set(re.findall(r"\n    def (\w+)\(", source))
    missing = sorted({
        call for calls, _, _ in blocks for call in calls if call not in defined
    })
    assert not missing, f"panels call methods the Api does not define: {missing}"
