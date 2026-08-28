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


def classify(case) -> str:
    """What a junit <testcase> actually says: passed, failed, or skipped.

    Skipped is its own answer, and getting that wrong was a real defect here: a case was
    read as broken only when it carried <failure> or <error>, so every SKIPPED test
    landed in the passed set and counted as evidence for the feature that named it.

    This project skips a lot on purpose -- weights that are gitignored, PostGIS that is
    not running, SITL that needs a container -- and each of those skips was silently
    promoting a row it had proved nothing about. "Status is computed from passing tests
    and a skip is not a pass" was the claim; this makes it true.
    """
    if case.find("skipped") is not None:
        return "skipped"
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    return "passed"


def node_id(case) -> str:
    """Rebuild a pytest node id from a junit <testcase>.

    junit gives a dotted classname and a bare name:

        classname="tests.sitl.test_mission_upload.TestUpload"  name="test_home"
        -> tests/sitl/test_mission_upload.py::TestUpload::test_home

    The dots are ambiguous -- nothing marks where the directories stop and the module
    begins -- so the module file is found by its name. Test modules are called test_*.py,
    and the LAST segment matching that is the file; everything after it is the class.

    Taking a fixed two segments instead, as this used to, is right only for a test
    directly under tests/. Anything nested came out as `tests/sitl.py::...`, a path that
    exists nowhere, so a selector naming a nested test matched nothing and its feature
    could never be earned. tests/sitl is exactly that case.
    """
    parts = [p for p in case.get("classname", "").split(".") if p]
    name = case.get("name", "")
    if not parts:
        return name
    module_end = next(
        (i for i in range(len(parts) - 1, -1, -1) if parts[i].startswith("test_")),
        min(1, len(parts) - 1),
    )
    file_part = "/".join(parts[: module_end + 1]) + ".py"
    rest = "::".join(parts[module_end + 1:])
    return f"{file_part}::{rest}::{name}" if rest else f"{file_part}::{name}"


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
            node = node_id(case)
            outcome = classify(case)
            if outcome == "passed":
                passed.add(node)
            elif outcome == "failed":
                failed.add(node)
            # A skipped test joins neither set. Its selector then matches nothing, and
            # evaluate() downgrades the row for lack of evidence rather than crediting a
            # test that never ran.
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


def read_report(xml_path: Path) -> tuple[set[str], set[str]]:
    """Read passed and failed node ids from a junit XML written elsewhere.

    This exists for evidence that cannot be produced on the machine asking the question.
    fl.sitl is the case: its tests need ArduPilot in a container, they skip under a plain
    pytest, and a skip is not a pass -- so on any laptop the row is honestly not_started
    no matter how many times the container goes green.

    CI runs that container. Passing its junit report in here is how the run that DID
    happen gets counted, rather than having the status hard-coded to what someone
    believes the container would do.
    """
    import xml.etree.ElementTree as ET

    passed: set[str] = set()
    failed: set[str] = set()
    tree = ET.parse(xml_path)
    for case in tree.iter("testcase"):
        node = node_id(case)
        outcome = classify(case)
        if outcome == "passed":
            passed.add(node)
        elif outcome == "failed":
            failed.add(node)
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/feature_status.py")
    parser.add_argument("--markdown", action="store_true", help="Rewrite docs/FEATURES.md")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable status")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any downgrade")
    parser.add_argument("--no-tests", action="store_true", help="Skip the test run (claims only)")
    parser.add_argument(
        "--extra-report", type=Path, action="append", default=[],
        help="junit XML from a run this machine cannot perform (e.g. the SITL container)",
    )
    args = parser.parse_args(argv)

    if args.no_tests:
        passed, failed, output = set(), set(), "(test run skipped)"
    else:
        passed, failed, output = run_tests()

    # Evidence from elsewhere is merged in, and a failure there still counts as a
    # failure: an outside report can promote a row only by passing, never by being
    # quieter than the local run.
    for extra in args.extra_report:
        extra_passed, extra_failed = read_report(extra)
        passed |= extra_passed
        failed |= extra_failed
        output += f"\n(merged {len(extra_passed)} passed, {len(extra_failed)} failed from {extra})"
    passed -= failed

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
        # A laptop is wrong in BOTH directions, which is why this refuses rather than
        # warns. It has the trained weights, so the model rows go up; it has no SITL
        # container and no live PostGIS, so those rows go DOWN -- and a published document
        # reading "PostGIS: implemented" would be a false demotion of something CI proves
        # every run. Publishing that is worse than publishing nothing, because a reader
        # cannot tell a real gap from a missing environment.
        missing = [f.id for f, _s, reason in results if "selector matched nothing" in reason]
        if missing and not args.extra_report:
            print(
                "Refusing to write docs/FEATURES.md: "
                f"{len(missing)} row(s) have no evidence on this machine "
                f"({', '.join(missing[:4])}{', ...' if len(missing) > 4 else ''}).\n"
                "Those tests run elsewhere -- SITL in its container, PostGIS against a "
                "live instance -- and writing the document here would demote rows that CI "
                "verifies on every run.\n"
                "Pass the junit reports from those runs with --extra-report, or let CI "
                "generate it.",
                file=sys.stderr,
            )
            return 1
        write_markdown(results, counts, total, args.extra_report)
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


def evidence_base(extra_reports=()) -> str:
    """State what this count was computed against, because it is not the same everywhere.

    Continuous integration reaches a lower number than a development machine does, and
    the difference is not a disagreement: the trained weights are gigabytes and are not
    in the repository, so the rows that exercise a real model cannot be verified where
    the model is absent. Both counts are honest about their own evidence.

    What was NOT honest was publishing the higher one with no note, so a reader who
    cloned the repository and ran the same command got a smaller number than the document
    claimed, with nothing to explain it. A count that only one machine can reproduce has
    to say so on its face.
    """
    models = REPO_ROOT / "models"
    weights = sorted(models.rglob("*.onnx")) if models.is_dir() else []
    if not weights:
        return (
            "Computed **without the trained model weights**, which are gigabytes and are "
            "not in the repository. Rows that exercise a real model sit at `implemented` "
            "here: the code exists and nothing present can prove it runs. This is the "
            "number continuous integration reaches, and the one a fresh clone reproduces."
        )
    size_gb = sum(path.stat().st_size for path in weights) / 1e9
    line = (
        f"Computed **with {len(weights)} trained models installed** ({size_gb:.1f} GB, not "
        "in the repository). Continuous integration has no weights and so reaches a lower "
        "verified count -- the model rows sit at `implemented` there. That gap is the "
        "weights, not a disagreement about the evidence; run `python "
        "tools/feature_status.py` on a machine without `models/` to see the reproducible "
        "floor."
    )
    if extra_reports:
        # No single machine can produce this document. The weights are here and the SITL
        # container and live PostGIS are not; on the runner it is the other way round.
        # Saying which reports were merged is what makes the count checkable rather than
        # something the reader has to take on trust.
        merged = ", ".join(f"`{Path(report).name}`" for report in extra_reports)
        line += (
            f" Evidence from runs this machine cannot perform was merged in: {merged}. "
            "No single machine holds all of it -- the weights are here and the SITL "
            "container and live PostGIS are not, and on the runner it is the other way "
            "round."
        )
    return line


def write_markdown(results, counts, total, extra_reports=()) -> None:
    lines: list[str] = []
    lines.append("# OpenDroneKit feature status")
    lines.append("")
    lines.append(
        "Generated by `python tools/feature_status.py --markdown`. Do not edit by hand: "
        "a feature is marked **verified** only when the tests it names actually pass, and "
        "any claim the evidence does not support is downgraded automatically."
    )
    lines.append("")
    lines.append(evidence_base(extra_reports))
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
