"""A real PX4 autopilot in Docker, for the PX4 integration tests.

PX4 v1.15.4 SITL from a pinned, prebuilt image, run with PX4's own SIH physics (the
``sihsim_quadx`` airframe, 10040) so no Gazebo is started: the flight controller is the
unmodified PX4 firmware build, and the simulator runs inside it.

PX4 sends MAVLink to the machine running the tests, the way it does to a ground station
on a real network:

* UDP 14540 -- PX4's "offboard/API" link, which MAVSDK listens on;
* UDP 14550 -- PX4's "GCS" link, which QGroundControl and the pymavlink driver listen on.

The image's own entrypoint resolves ``host.docker.internal`` to whatever comes first,
which on Docker Desktop for Windows is an IPv6 address PX4 refuses ("invalid partner
ip"), so the partner is resolved here to IPv4 explicitly.

    python -m tools.sitl.px4_docker start   # start and wait until PX4 is ready
    python -m tools.sitl.px4_docker stop
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

PX4_IMAGE = os.environ.get("ODK_PX4_IMAGE", "jonasvautherin/px4-gazebo-headless:1.15.4")
PX4_VERSION = "v1.15.4"
PX4_AIRFRAME = "sihsim_quadx"
API_PORT = 14540   # MAVSDK
GCS_PORT = 14550   # QGroundControl / pymavlink
# Every docker CLI call is bounded: a wedged Docker Desktop otherwise hangs the test run
# forever on `docker logs` instead of failing it.
DOCKER_CALL_TIMEOUT_S = 30.0

# Run inside the container: resolve the host to IPv4, point both MAVLink links at it
# with the image's own edit script, and start the prebuilt px4 binary with SIH.
_CONTAINER_SCRIPT = (
    'H="${ODK_PX4_PARTNER_IP:-}"; '
    '[ -z "$H" ] && H=$(getent ahostsv4 host.docker.internal | awk \'NR==1{print $1}\'); '
    '[ -z "$H" ] && H=$(ip route | awk \'/default/ {print $3}\'); '
    'echo "odk: MAVLink partner $H"; '
    'source /root/edit_rcS.bash "$H" "$H" && '
    'cd /root/Firmware/build/px4_sitl_default/rootfs && '
    f'PX4_SIM_MODEL={PX4_AIRFRAME} exec ../bin/px4 -d ../etc'
)


class PX4Unavailable(RuntimeError):
    """Docker or the PX4 image is not present, so PX4 cannot be run here."""


class PX4StartupError(RuntimeError):
    """PX4 was started but did not become ready."""


def _docker() -> str:
    binary = shutil.which("docker")
    if binary is None:
        raise PX4Unavailable("the docker CLI is not on PATH")
    try:
        probe = subprocess.run([binary, "info", "--format", "{{.ServerVersion}}"],
                               capture_output=True, text=True, timeout=DOCKER_CALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise PX4Unavailable("the Docker daemon did not answer `docker info` within "
                             f"{DOCKER_CALL_TIMEOUT_S:.0f} s") from exc
    # Docker Desktop with a stopped engine answers 0 with an empty version.
    if probe.returncode != 0 or not probe.stdout.strip():
        raise PX4Unavailable(f"the Docker daemon is not running: {probe.stderr.strip()[:200]}")
    return binary


def image_digest(image: str = PX4_IMAGE) -> str:
    docker = _docker()
    found = subprocess.run([docker, "image", "inspect", image, "--format",
                            "{{index .RepoDigests 0}}"], capture_output=True, text=True,
                           timeout=DOCKER_CALL_TIMEOUT_S)
    if found.returncode != 0:
        raise PX4Unavailable(f"the PX4 image {image} is not present (docker pull {image})")
    return found.stdout.strip()


class PX4SITL:
    """One PX4 SITL container, started fresh and removed on stop."""

    READY_TEXT = "Ready for takeoff"

    def __init__(self, image: str = PX4_IMAGE, name: str | None = None) -> None:
        self.image = image
        self.name = name or f"odk-px4-sitl-{os.getpid()}"
        self.digest = ""
        self._docker = ""

    def start(self, timeout_s: float = 120.0) -> "PX4SITL":
        self._docker = _docker()
        self.digest = image_digest(self.image)
        subprocess.run([self._docker, "rm", "-f", self.name], capture_output=True)
        env = os.environ.copy()
        env["MSYS_NO_PATHCONV"] = "1"   # Git Bash would rewrite the container paths
        command = [self._docker, "run", "-d", "--rm", "--name", self.name,
                   "--entrypoint", "bash"]
        if os.environ.get("ODK_PX4_PARTNER_IP"):
            command += ["-e", f"ODK_PX4_PARTNER_IP={os.environ['ODK_PX4_PARTNER_IP']}"]
        command += [self.image, "-c", _CONTAINER_SCRIPT]
        try:
            started = subprocess.run(command, capture_output=True, text=True, env=env,
                                     timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise PX4StartupError("docker run did not return within 120 s") from exc
        if started.returncode != 0:
            raise PX4StartupError(f"docker run failed: {started.stderr.strip()}")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            log = self.logs()
            if self.READY_TEXT in log:
                return self
            if not self.running():
                raise PX4StartupError("PX4 exited during startup:\n" + log[-3000:])
            time.sleep(1.0)
        tail = self.logs()[-3000:]
        self.stop()
        raise PX4StartupError(f"PX4 was not ready within {timeout_s:.0f} s:\n{tail}")

    def running(self) -> bool:
        try:
            found = subprocess.run([self._docker, "inspect", "-f", "{{.State.Running}}",
                                    self.name], capture_output=True, text=True,
                                   timeout=DOCKER_CALL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return True   # a slow daemon is not an exited PX4; the deadline still applies
        return found.stdout.strip() == "true"

    def logs(self) -> str:
        try:
            found = subprocess.run([self._docker, "logs", self.name], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace",
                                   timeout=DOCKER_CALL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return ""
        return (found.stdout or "") + (found.stderr or "")

    def stop(self) -> None:
        if self._docker:
            try:
                subprocess.run([self._docker, "rm", "-f", self.name], capture_output=True,
                               timeout=60)
            except subprocess.TimeoutExpired:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--name", default="odk-px4-sitl")
    args = parser.parse_args(argv)
    sitl = PX4SITL(name=args.name)
    if args.action == "stop":
        sitl._docker = _docker()
        sitl.stop()
        return 0
    try:
        sitl.start()
    except (PX4Unavailable, PX4StartupError) as exc:
        print(f"PX4 SITL: {exc}", file=sys.stderr)
        return 1
    print(f"PX4 {PX4_VERSION} ({PX4_AIRFRAME}) ready in container {sitl.name}, "
          f"image {sitl.digest}. MAVSDK: udpin://0.0.0.0:{API_PORT}; "
          f"pymavlink/QGC: udpin:0.0.0.0:{GCS_PORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
