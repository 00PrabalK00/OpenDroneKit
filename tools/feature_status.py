"""Compute feature status from test results, and refuse to take anyone's word for it.

A feature reaches ``verified`` only when every test it names actually passes in this
run. Anything claimed higher than the evidence supports is downgraded, and the reason
is printed. This is the mechanism that stops a feature being ticked off by assertion.

    python tools/feature_status.py              # verify and print a summary
    python tools/feature_status.py --markdown   # regenerate docs/FEATURES.md
    python tools/feature_status.py --json       # machine-readable status
    python tools/feature_status.py --strict     # exit non-zero on any downgrade
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "docs" / "features"))

from registry import ALL_FEATURES, PRODUCTS, Feature  # noqa: E402

RANK = {"not_started": 0, "in_progress": 1, "implemented": 2, "verified": 3}
ORDER = ["verified", "implemented", "in_progress", "not_started"]


def run_tests() -> tuple[set[str], set[str], str]:
    """Run the suite and return (passed node ids, failed node ids, raw report)."""
    report_path = REPO_ROOT / ".feature_report.json"
    command = [
        sys.executable, "-m", "pytest",
        "--no-header", "-q",
        f"--junit-xml={REPO_ROOT / '.feature_report.xml'}",
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True
    )

    passed: set[str] = set()
    failed: set[str] = set()
    xml_path = REPO_ROOT / ".feature_report.xml"
    if xml_path.exists():
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        for case in tree.iter("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            # junit classname is dotted: tests.test_geo.TestUmeyamaSimilarity
            parts = classname.split(".")
            if not parts:
                continue
            file_part = "/".join(parts[:2]) + ".py" if len(parts) >= 2 else parts[0]
            rest = "::".join(parts[2:]) if len(parts) > 2 else ""
            node = f"{file_part}::{rest}::{name}" if rest else f"{file_part}::{name}"
            broken = case.find("failure") is not None or case.find("error") is not None
            (failed if broken else passed).add(node)
        xml_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    return passed, failed, completed.stdout + completed.stderr


def _matches(selector: str, node_ids: set[str]) -> list[str]:
    """A selector may name a file, a class, or an exact test."""
    normalised = selector.replace("\\", "/")
    return [node for node in node_ids if node.startswith(normalised)]


def evaluate(feature: Feature, passed: set[str], failed: set[str]) -> tuple[str, str]:
    """Return the earned status and an explanation of any downgrade."""
    if not feature.tests:
        if feature.claimed == "verified":
            return "implemented", "claimed verified but names no tests"
        return feature.claimed, ""

    matched_pass = [m for selector in feature.tests for m in _matches(selector, passed)]
    matched_fail = [m for selector in feature.tests for m in _matches(selector, failed)]
    unmatched = [s for s in feature.tests if not _matches(s, passed | failed)]

    if unmatched:
        downgraded = min(feature.claimed, "implemented", key=lambda s: RANK[s])
        return downgraded, f"test selector matched nothing: {', '.join(unmatched)}"
    if matched_fail:
        return "in_progress", f"{len(matched_fail)} named test(s) failing"
    if matched_pass:
        # Evidence exists and passes. A feature is never promoted above what its
        # author claimed -- passing tests cannot turn an admitted gap into done.
        if RANK[feature.claimed] >= RANK["implemented"]:
            return "verified", ""
        return feature.claimed, "tests pass but the feature is not claimed complete"
    return feature.claimed, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/feature_status.py")
    parser.add_argument("--markdown", action="store_true", help="Rewrite docs/FEATURES.md")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable status")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any downgrade")
    parser.add_argument("--no-tests", action="store_true", help="Skip the test run (claims only)")
    args = parser.parse_args(argv)

    if args.no_tests:
        passed, failed, output = set(), set(), "(test run skipped)"
    else:
        passed, failed, output = run_tests()

    results = []
    downgrades = []
    for feature in ALL_FEATURES:
        status, reason = evaluate(feature, passed, failed)
        results.append((feature, status, reason))
        if RANK[status] < RANK[feature.claimed]:
            downgrades.append((feature, status, reason))

    counts = Counter(status for _, status, _ in results)
    total = len(results)

    if args.json:
        print(json.dumps({
            "total": total,
            "counts": dict(counts),
            "features": [
                {"id": f.id, "title": f.title, "product": f.product,
                 "category": f.category, "claimed": f.claimed, "status": s,
                 "reason": r, "tests": list(f.tests)}
                for f, s, r in results
            ],
        }, indent=2))
        return 1 if (args.strict and downgrades) else 0

    if args.markdown:
        write_markdown(results, counts, total)
        print(f"Wrote {REPO_ROOT / 'docs' / 'FEATURES.md'}")

    print(f"\nOpenDroneKit feature status  ({total} specified capabilities)")
    print("-" * 64)
    for status in ORDER:
        share = counts.get(status, 0) / total * 100
        print(f"  {status:<14} {counts.get(status, 0):>4}   {share:5.1f}%")

    if downgrades:
        print(f"\n{len(downgrades)} claim(s) not supported by evidence:")
        for feature, status, reason in downgrades:
            print(f"  {feature.id:<28} {feature.claimed} -> {status}   ({reason})")

    if failed:
        print(f"\n{len(failed)} test(s) failing.")

    return 1 if (args.strict and (downgrades or failed)) else 0


def write_markdown(results, counts, total) -> None:
    lines: list[str] = []
    lines.append("# OpenDroneKit feature status")
    lines.append("")
    lines.append(
        "Generated by `python tools/feature_status.py --markdown`. Do not edit by hand: "
        "a feature is marked **verified** only when the tests it names actually pass, and "
        "any claim the evidence does not support is downgraded automatically."
    )
    lines.append("")
    lines.append("| Status | Count | Share |")
    lines.append("|---|---:|---:|")
    for status in ORDER:
        lines.append(f"| {status} | {counts.get(status, 0)} | {counts.get(status, 0) / total * 100:.1f}% |")
    lines.append(f"| **total** | **{total}** | |")
    lines.append("")
    lines.append("## Definition of done")
    lines.append("")
    lines.append(
        "A capability is only `verified` when a test exercises it against real inputs and "
        "passes. `implemented` means the code exists but nothing proves it works, which is "
        "deliberately an uncomfortable place for a feature to sit."
    )
    lines.append("")

    grouped: dict[str, list] = defaultdict(list)
    for feature, status, reason in results:
        grouped[feature.category].append((feature, status, reason))

    for category in sorted(grouped):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Feature | Product | Status | Evidence |")
        lines.append("|---|---|---|---|")
        for feature, status, reason in sorted(grouped[category], key=lambda r: -RANK[r[1]]):
            mark = {"verified": "**verified**", "implemented": "implemented",
                    "in_progress": "in progress", "not_started": "not started"}[status]
            evidence = ", ".join(f"`{t}`" for t in feature.tests) if feature.tests else "-"
            if reason:
                evidence = f"{evidence}<br>_{reason}_" if feature.tests else f"_{reason}_"
            note = f"<br>{feature.notes}" if feature.notes else ""
            lines.append(f"| {feature.title}{note} | {feature.product} | {mark} | {evidence} |")
        lines.append("")

    lines.append("## Products")
    lines.append("")
    for key, description in PRODUCTS.items():
        lines.append(f"- `{key}` — {description}")
    lines.append("")

    (REPO_ROOT / "docs" / "FEATURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
