"""Run the SITL tests inside the ArduPilot container.

fl.sitl has sat at not_started with a complete harness behind it, because
``sim_vehicle.py`` is Linux-only in practice and the tests skip without it. Skipping is
the correct behaviour -- feature status comes from PASSING tests, and a skip proves
nothing -- but it means the flight code has never met a real autopilot.

This builds the image once and mounts the working tree, so a run always exercises the
current code rather than a snapshot baked at build time.

    python -m tools.sitl.run_docker --build
    python -m tools.sitl.run_docker
    python -m tools.sitl.run_docker -- -k mission_upload
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "opendronekit-sitl"
DOCKERFILE = ROOT / "infrastructure" / "docker" / "Dockerfile.sitl"


def docker_available() -> str:
    binary = shutil.which("docker")
    if not binary:
        raise SystemExit(
            "docker is not on PATH. SITL needs Linux, and this container is how it runs "
            "on a Windows or macOS host."
        )
    probe = subprocess.run([binary, "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(
            "docker is installed but not responding. Start Docker Desktop and retry.\n"
            + probe.stderr.strip()[:400]
        )
    return binary


def image_exists(binary: str) -> bool:
    probe = subprocess.run([binary, "image", "inspect", IMAGE],
                           capture_output=True, text=True)
    return probe.returncode == 0


def build(binary: str, ref: str) -> None:
    """Build the image, streaming output.

    Not quiet on purpose: this compiles ArduPilot and takes 15-25 minutes, and a silent
    twenty-minute wait is indistinguishable from a hang.
    """
    print(f"building {IMAGE} (ArduPilot {ref}); this compiles from source", flush=True)
    result = subprocess.run(
        [binary, "build", "-f", str(DOCKERFILE), "-t", IMAGE,
         "--build-arg", f"ARDUPILOT_REF={ref}", str(ROOT)],
    )
    if result.returncode != 0:
        raise SystemExit("Image build failed; the SITL tests cannot run.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.sitl.run_docker")
    parser.add_argument("--build", action="store_true", help="Rebuild the image first.")
    parser.add_argument("--ref", default="Copter-4.5.7",
                        help="ArduPilot tag to build. Pinned so behaviour cannot drift.")
    parser.add_argument("--shell", action="store_true",
                        help="Open a shell in the container instead of running tests.")
    parser.add_argument("pytest_args", nargs="*",
                        help="Extra arguments passed through to pytest.")
    args = parser.parse_args(argv)

    binary = docker_available()
    if args.build or not image_exists(binary):
        build(binary, args.ref)

    mount = f"{ROOT}:/workspace"
    command = [binary, "run", "--rm", "-v", mount, "-w", "/workspace"]
    if args.shell:
        command += ["-it", IMAGE, "bash"]
    else:
        command += [IMAGE, "python3", "-m", "pytest", "-m", "sitl", "tests/sitl", "-v",
                    *args.pytest_args]

    print(" ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode == 0:
        print(
            "\nSITL tests passed against a real ArduPilot. Re-run "
            "tools/feature_status.py: fl.sitl should now have passing evidence behind it "
            "rather than a skip.",
            flush=True,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
