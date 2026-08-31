"""Exercise the running desktop application and report what each control really does.

Every earlier probe ran in a browser, and the browser and WebView2 disagree in ways that
matter: window.prompt exists in one and not the other, which left twenty-six buttons
silently dead while every test passed. The native menu bar was worse -- app/shell.py
dispatched to window.odk.onMenu, the cockpit never defined it, and evaluate_js against an
undefined property is a no-op, so thirty-three menu items did nothing and logged nothing.

Neither bug was findable from source or from a browser. Both are obvious the moment the
real window is driven. So this drives the real window, over the DevTools protocol:

    ODK_UI=cockpit WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS="--remote-debugging-port=9222" \
        python main.py
    python tools/audit_desktop_ui.py --markdown docs/UI_AUDIT.md

For every workspace it clicks every toolbar button, then dismisses whatever dialog opens,
and records one of:

    api        it called the application and reported a result
    dialog     it asked for input, which is the correct behaviour for that verb
    refused    it declined, with the reason -- a refusal is a working feature
    view       it changed what the canvas shows
    broken     it raised, and the exception text became the user-facing result
    silent     nothing happened and nothing was said, which is the bug worth finding

`silent` and `broken` are the failing outcomes. A refusal is not a failure: "select a
dataset first" is the application telling the truth about its state.

`broken` was added after New Folder was recorded as `api` while answering
"ValueError: tif is not a valid file filter". An exception IS a response, so the audit
counted it among the working controls and reported zero problems. A control that can
only throw is as dead as one that does nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

CDP_URL = "http://127.0.0.1:9222/json"


class Page:
    """A CDP connection to the application window."""

    def __init__(self, url: str = CDP_URL) -> None:
        import requests
        import websocket

        tabs = requests.get(url, timeout=5).json()
        pages = [tab for tab in tabs if tab.get("type") == "page"]
        if not pages:
            raise SystemExit(
                "No page found. Start the app with remote debugging:\n"
                '  $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS="--remote-debugging-port=9222"'
            )
        self.socket = websocket.create_connection(
            pages[0]["webSocketDebuggerUrl"], timeout=30, suppress_origin=True
        )
        self.counter = 0

    def js(self, expression: str) -> Any:
        self.counter += 1
        self.socket.send(json.dumps({
            "id": self.counter,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True, "awaitPromise": True},
        }))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") != self.counter:
                continue
            result = message.get("result", {})
            if "exceptionDetails" in result:
                return {"error": result["exceptionDetails"].get("text", "exception")}
            return result.get("result", {}).get("value")


CLICK_ONE = """
(async () => {
  const label = %s;
  const button = [...document.querySelectorAll('.tbtn')].find(b => b.textContent === label);
  if (!button) return {outcome: 'missing'};

  const before = (([...document.querySelectorAll('.toast')].pop()) || {}).textContent || '';
  const viewBefore = document.querySelector('.region.centre .canvas-product') ? 'image' : 'other';

  button.click();
  await new Promise(r => setTimeout(r, %d));

  const modal = document.querySelector('.modal-backdrop');
  const modalTitle = (document.querySelector('.modal-title') || {}).textContent || null;
  if (modal) {
    // Dismiss it: the audit is about whether the control responds, not about creating
    // rows in the user's database.
    const cancel = [...document.querySelectorAll('.modal-actions .tbtn')]
      .find(b => /Cancel|No|Close|Done/.test(b.textContent));
    if (cancel) cancel.click(); else modal.remove();
    await new Promise(r => setTimeout(r, 200));
    return {outcome: 'dialog', detail: modalTitle};
  }

  const after = (([...document.querySelectorAll('.toast')].pop()) || {}).textContent || '';
  const viewAfter = document.querySelector('.region.centre .canvas-product') ? 'image' : 'other';
  if (viewAfter !== viewBefore) return {outcome: 'view', detail: viewAfter};
  if (after && after !== before) {
    // A raw exception is a response, so it is not silent -- and it was being counted as
    // `api`, which reads as working. New Folder answered
    // "ValueError: tif is not a valid file filter" and the audit called it a pass.
    // A control that can only ever throw is as dead as one that does nothing; the
    // difference is that this one says so and was still not counted.
    // \bError\b does not match inside "TypeError", and anchoring to the start of the
    // string fails because the toast is prefixed with the control's own name --
    // "Match Captures: TypeError: ...". The first version of this check made both
    // mistakes at once and filed a real crash under `refused`, which reads as a working
    // feature. Match the exception class anywhere in the message instead.
    if (/\w*(Error|Exception)\b\s*:|Traceback/.test(after)) {
      return {outcome: 'broken', detail: after.slice(0, 120)};
    }
    const refused = /first|Select|Open a project|not available|no |No /.test(after);
    return {outcome: refused ? 'refused' : 'api', detail: after.slice(0, 80)};
  }
  return {outcome: 'silent'};
})()
"""


def audit(page: Page, settle_ms: int) -> list[dict[str, Any]]:
    workspaces = page.js(
        "[...document.querySelectorAll('.nav-item')].map(n => n.textContent.trim())"
    )
    rows: list[dict[str, Any]] = []

    for workspace in workspaces:
        opened = page.js(f"""(async () => {{
          const nav = [...document.querySelectorAll('.nav-item')]
            .find(n => n.textContent.trim() === {json.dumps(workspace)});
          if (!nav) return null;
          nav.click();
          await new Promise(r => setTimeout(r, 700));
          return {{
            active: (document.querySelector('.nav-item.active') || {{}}).textContent,
            buttons: [...document.querySelectorAll('.tbtn')].map(b => b.textContent),
            panels: document.querySelectorAll('.panel').length,
          }};
        }})()""")
        if not opened:
            rows.append({"workspace": workspace, "button": "(open)", "outcome": "missing"})
            continue
        if opened.get("active", "").strip() != workspace:
            rows.append({"workspace": workspace, "button": "(open)", "outcome": "silent",
                         "detail": "nav click did not change the active workspace"})
            continue

        for label in opened["buttons"]:
            if label in ("Layout ▾", "⛶ Canvas"):
                continue
            result = page.js(CLICK_ONE % (json.dumps(label), settle_ms)) or {}
            rows.append({
                "workspace": workspace,
                "button": label,
                "outcome": result.get("outcome", "unknown"),
                "detail": (result.get("detail") or "").strip(),
                "panels": opened["panels"],
            })
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1

    lines = [
        "# Desktop UI audit",
        "",
        "Generated by `tools/audit_desktop_ui.py` against the running application over the",
        "DevTools protocol. Not a browser: the browser and WebView2 disagree in ways that",
        "have already hidden two whole classes of dead control from every other test.",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(rows)} controls exercised.",
        "",
        "| outcome | count | meaning |",
        "|---|---:|---|",
        f"| api | {counts.get('api', 0)} | called the application and reported a result |",
        f"| dialog | {counts.get('dialog', 0)} | asked for input, which is correct for that verb |",
        f"| refused | {counts.get('refused', 0)} | declined and said why — a working feature |",
        f"| view | {counts.get('view', 0)} | changed what the canvas shows |",
        f"| broken | {counts.get('broken', 0)} | **threw an exception at the user** |",
        f"| silent | {counts.get('silent', 0)} | **did nothing and said nothing — the bug worth finding** |",
        "",
    ]

    failing = [r for r in rows if r["outcome"] in ("silent", "broken")]
    if failing:
        lines += ["## Controls that do not work", "",
                  "| workspace | control | outcome | note |", "|---|---|---|---|"]
        lines += [f"| {r['workspace']} | {r['button']} | {r['outcome']} | "
                  f"{(r.get('detail') or '').replace('|', chr(92) + '|')} |" for r in failing]
        lines.append("")
    else:
        lines += ["Every control responded, and none of them by raising.", ""]

    lines += ["## Every control", "", "| workspace | control | outcome | detail |", "|---|---|---|---|"]
    for row in rows:
        detail = (row.get("detail") or "").replace("|", "\\|")
        lines.append(f"| {row['workspace']} | {row['button']} | {row['outcome']} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/audit_desktop_ui.py")
    parser.add_argument("--markdown", help="Write the report here.")
    parser.add_argument("--settle-ms", type=int, default=900,
                        help="How long to wait after each click before reading the result.")
    args = parser.parse_args(argv)

    rows = audit(Page(), args.settle_ms)
    report = render(rows)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"wrote {args.markdown}")

    failing = [row for row in rows if row["outcome"] in ("silent", "broken")]
    silent = sum(1 for row in failing if row["outcome"] == "silent")
    broken = sum(1 for row in failing if row["outcome"] == "broken")
    print(f"{len(rows)} controls exercised, {silent} silent, {broken} raising")
    for row in failing:
        print(f"  {row['outcome'].upper()} {row['workspace']}/{row['button']}: "
              f"{(row.get('detail') or '')[:90]}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
