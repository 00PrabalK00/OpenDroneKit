"""Run the code of a tutorial in docs/tutorials exactly as a reader would.

    python tools/run_tutorial.py docs/tutorials/01-aukerman-end-to-end.md --workdir out/t1

Fenced blocks are handled by their info string:

* ```` ```python ```` -- executed, in order, as one script (later blocks see earlier names);
* ```` ```<lang> file=<relative path> ```` -- written to ``<workdir>/<relative path>`` before
  anything runs (a plugin's ``pyproject.toml`` and module, say);
* everything else (``bash``, ``text``...) is shown to the reader only. Shell steps that the
  code depends on are repeated by the caller (``tests/test_tutorials.py``).

The script runs with ``<workdir>`` as the working directory and the repository on
``sys.path``. Output goes to stdout; the exit code is the script's.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.S | re.M)


@dataclass(frozen=True)
class Block:
    lang: str
    file: str | None
    code: str


def blocks(markdown: str) -> list[Block]:
    found = []
    for match in _FENCE.finditer(markdown):
        info = match.group(1).strip().split()
        lang = info[0] if info else ""
        file = next((part.split("=", 1)[1] for part in info[1:] if part.startswith("file=")), None)
        found.append(Block(lang, file, match.group(2)))
    return found


def materialise(markdown: str, workdir: Path) -> tuple[list[Path], str]:
    """Write the ``file=`` blocks under ``workdir``; return them and the python script."""
    written, script = [], []
    for block in blocks(markdown):
        if block.file:
            target = (workdir / block.file).resolve()
            if workdir.resolve() not in target.parents:
                raise ValueError(f"{block.file} points outside the working folder")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(block.code, encoding="utf-8")
            written.append(target)
        elif block.lang == "python":
            script.append(block.code)
    return written, "\n".join(script)


def run(tutorial: Path, workdir: Path, *, env: dict[str, str] | None = None,
        preamble: str = "", timeout: float = 3600) -> subprocess.CompletedProcess:
    workdir.mkdir(parents=True, exist_ok=True)
    _, script = materialise(tutorial.read_text(encoding="utf-8"), workdir)
    path = workdir / f"_{tutorial.stem}.py"
    path.write_text(preamble + "\n" + script, encoding="utf-8")
    full_env = dict(os.environ, **(env or {}))
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + [p for p in [full_env.get("PYTHONPATH", "")] if p])
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run([sys.executable, str(path)], cwd=workdir, env=full_env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tutorial", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--write-files-only", action="store_true",
                        help="Only write the tutorial's file= blocks (a plugin package, say).")
    args = parser.parse_args(argv)
    workdir = args.workdir.resolve()
    if args.write_files_only:
        written, _ = materialise(args.tutorial.read_text(encoding="utf-8"), workdir)
        for path in written:
            print("wrote", path)
        return 0
    # The script runs inside the working folder; resolve the defaults from where the
    # reader is, as if they had typed the code in the repository root themselves.
    env = {"ODK_SAMPLE_DATA": os.environ.get("ODK_SAMPLE_DATA") or str(Path("sample_data").resolve()),
           "ODK_TUTORIAL_DIR": os.environ.get("ODK_TUTORIAL_DIR") or str(workdir)}
    result = run(args.tutorial, workdir, env=env)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
