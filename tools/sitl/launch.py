"""Lifecycle-managed ArduPilot Copter SITL launcher for integration tests.

The launcher uses ``sim_vehicle.py`` rather than a protocol stub. It starts
ArduCopter plus MAVProxy, exposes independent client and observer streams, and
does not return until ArduPilot reports a usable EKF and 3D GPS fix.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Sequence


os.environ.setdefault("MAVLINK20", "1")

SITL_HOME_LAT = -35.363261
SITL_HOME_LON = 149.165230
SITL_HOME_ALT_M = 584.0
SITL_HOME_HEADING_DEG = 0.0


class SITLUnavailable(RuntimeError):
    """The optional local SITL toolchain is not installed."""


class SITLStartupError(RuntimeError):
    """SITL was selected, but failed to become flight-ready."""


def _split_command(value: str) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    candidate = Path(raw.strip('"')).expanduser()
    if candidate.exists():
        return [str(candidate)]
    return shlex.split(raw, posix=True)


def _normalise_command(command: Sequence[str]) -> tuple[list[str], Path | None]:
    parts = [str(part) for part in command if str(part)]
    if not parts:
        raise SITLUnavailable("No ArduPilot SITL command was configured.")

    executable = Path(parts[0]).expanduser()
    resolved = executable if executable.is_file() else None
    if resolved is None:
        found = shutil.which(parts[0])
        if found:
            resolved = Path(found)
            parts[0] = found
    else:
        parts[0] = str(resolved.resolve())
    if resolved is None:
        raise SITLUnavailable(
            f"ArduPilot SITL command {parts[0]!r} was not found. Set ARDUPILOT_HOME "
            "or ARDUPILOT_SITL_COMMAND."
        )

    if os.name == "nt" and resolved.suffix.lower() == ".py":
        parts.insert(0, sys.executable)
        script = resolved
    else:
        script = resolved if resolved.name.startswith("sim_vehicle") else None

    source_root: Path | None = None
    if script is not None:
        script = script.resolve()
        try:
            if script.parent.name == "autotest" and script.parent.parent.name == "Tools":
                source_root = script.parents[2]
        except IndexError:
            pass
    return parts, source_root


def discover_sim_vehicle(
    command: Sequence[str] | str | None = None,
) -> tuple[list[str], Path | None]:
    """Resolve ``sim_vehicle.py`` and, when known, its ArduPilot source root."""

    if isinstance(command, str):
        return _normalise_command(_split_command(command))
    if command is not None:
        return _normalise_command(command)

    configured = os.environ.get("ARDUPILOT_SITL_COMMAND", "").strip()
    if configured:
        return _normalise_command(_split_command(configured))

    ardupilot_home = os.environ.get("ARDUPILOT_HOME", "").strip()
    if ardupilot_home:
        script = Path(ardupilot_home).expanduser() / "Tools" / "autotest" / "sim_vehicle.py"
        if not script.is_file():
            raise SITLUnavailable(
                f"ARDUPILOT_HOME is set to {ardupilot_home!r}, but {script} does not exist."
            )
        return _normalise_command([str(script)])

    for name in ("sim_vehicle.py", "sim_vehicle"):
        found = shutil.which(name)
        if found:
            return _normalise_command([found])
    raise SITLUnavailable(
        "sim_vehicle.py is not on PATH and ARDUPILOT_HOME is not set. "
        "See docs/SITL.md for the optional ArduPilot installation."
    )


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class ArduPilotSITL:
    """Start one isolated ArduCopter SITL and tear down its process tree.

    ``start()`` returns the URI for OpenDroneKit's production client. The observer
    URI receives the same autopilot traffic independently, so tests can verify
    ``MISSION_ITEM_REACHED`` without competing with the client's listener.
    """

    def __init__(
        self,
        *,
        command: Sequence[str] | str | None = None,
        startup_timeout_s: float | None = None,
        instance: int | None = None,
        speedup: int | None = None,
        rebuild: bool | None = None,
    ) -> None:
        self._configured_command = command
        self.startup_timeout_s = float(
            startup_timeout_s
            if startup_timeout_s is not None
            else os.environ.get("ODK_SITL_START_TIMEOUT_S", "240")
        )
        self.instance = int(
            instance if instance is not None else os.environ.get("ODK_SITL_INSTANCE", "0")
        )
        self.speedup = max(
            1, int(speedup if speedup is not None else os.environ.get("ODK_SITL_SPEEDUP", "2"))
        )
        self.rebuild = (
            bool(rebuild)
            if rebuild is not None
            else not _truthy_environment("ODK_SITL_NO_REBUILD")
        )
        self.connection_port = _free_udp_port()
        self.observer_port = _free_udp_port()
        while self.observer_port == self.connection_port:
            self.observer_port = _free_udp_port()
        self.connection_string = f"udpin:127.0.0.1:{self.connection_port}"
        self.observer_connection_string = f"udpin:127.0.0.1:{self.observer_port}"
        self.process: subprocess.Popen | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._log_handle = None
        self.log_path: Path | None = None
        self._atexit_registered = False

    def _build_command(self) -> tuple[list[str], Path | None]:
        base, source_root = discover_sim_vehicle(self._configured_command)
        command = [
            *base,
            "-v", "ArduCopter",
            "-f", "quad",
            "-I", str(self.instance),
            "-w",
            "-l",
            f"{SITL_HOME_LAT},{SITL_HOME_LON},{SITL_HOME_ALT_M},{SITL_HOME_HEADING_DEG}",
            "--speedup", str(self.speedup),
            "--out", f"127.0.0.1:{self.connection_port}",
            "--out", f"127.0.0.1:{self.observer_port}",
            "--mavproxy-args=--daemon",
        ]
        if not self.rebuild:
            command.append("--no-rebuild")
        extra = os.environ.get("ODK_SITL_EXTRA_ARGS", "").strip()
        if extra:
            command.extend(_split_command(extra))
        return command, source_root

    def start(self) -> str:
        """Launch SITL, wait for EKF/GPS readiness, and return its client URI."""

        if self.process is not None and self.process.poll() is None:
            return self.connection_string
        try:
            from pymavlink import mavutil  # noqa: F401 - availability gate before spawning
        except ImportError as exc:
            raise SITLUnavailable(
                "pymavlink is not installed; install the SITL test dependencies."
            ) from exc

        command, source_root = self._build_command()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="opendronekit-sitl-")
        work_dir = Path(self._temporary_directory.name)
        self.log_path = work_dir / "sitl.log"
        self._log_handle = self.log_path.open("w", encoding="utf-8", errors="replace")
        command.extend(["--use-dir", str(work_dir)])

        environment = os.environ.copy()
        environment["MAVLINK20"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        popen_kwargs: dict[str, object] = {
            "cwd": str(source_root) if source_root is not None else None,
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": self._log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            popen_kwargs["start_new_session"] = True

        try:
            self.process = subprocess.Popen(command, **popen_kwargs)
        except OSError as exc:
            self._close_resources()
            raise SITLStartupError(f"Could not start ArduPilot SITL: {exc}") from exc

        atexit.register(self.stop)
        self._atexit_registered = True
        try:
            self._wait_for_ekf_ready()
        except Exception:
            self.stop()
            raise
        return self.connection_string

    def _request_readiness_messages(self, connection) -> None:
        from pymavlink import mavutil

        for message_id in (24, 33, 193):
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(message_id),
                200_000.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

    def _wait_for_ekf_ready(self) -> None:
        from pymavlink import mavutil

        required_flags = (
            int(getattr(mavutil.mavlink, "EKF_ATTITUDE", 1))
            | int(getattr(mavutil.mavlink, "EKF_VELOCITY_HORIZ", 2))
            | int(getattr(mavutil.mavlink, "EKF_VELOCITY_VERT", 4))
            | int(getattr(mavutil.mavlink, "EKF_POS_HORIZ_REL", 8))
            | int(getattr(mavutil.mavlink, "EKF_POS_HORIZ_ABS", 16))
            | int(getattr(mavutil.mavlink, "EKF_POS_VERT_ABS", 32))
        )
        deadline = time.monotonic() + max(1.0, self.startup_timeout_s)
        status_text: deque[str] = deque(maxlen=12)
        ekf_flags = 0
        gps_fix = 0
        connection = mavutil.mavlink_connection(
            self.connection_string, autoreconnect=True, source_system=254
        )
        try:
            heartbeat_seen = False
            requested = False
            while time.monotonic() < deadline:
                if self.process is None or self.process.poll() is not None:
                    code = None if self.process is None else self.process.returncode
                    raise SITLStartupError(
                        f"ArduPilot SITL exited before readiness (exit code {code}).\n"
                        f"{self._log_tail()}"
                    )
                message = connection.recv_match(blocking=True, timeout=1.0)
                if message is None:
                    continue
                message_type = message.get_type()
                if message_type == "HEARTBEAT":
                    heartbeat_seen = True
                    if not requested:
                        self._request_readiness_messages(connection)
                        requested = True
                elif message_type == "GPS_RAW_INT":
                    gps_fix = max(gps_fix, int(getattr(message, "fix_type", 0) or 0))
                elif message_type == "EKF_STATUS_REPORT":
                    ekf_flags = int(getattr(message, "flags", 0) or 0)
                elif message_type == "STATUSTEXT":
                    status_text.append(str(getattr(message, "text", "")).strip("\x00"))

                if heartbeat_seen and gps_fix >= 3 and (ekf_flags & required_flags) == required_flags:
                    return

            details = "; ".join(status_text) or "no STATUSTEXT received"
            raise SITLStartupError(
                "ArduPilot SITL did not report EKF/GPS readiness within "
                f"{self.startup_timeout_s:.0f}s (GPS fix={gps_fix}, "
                f"EKF flags=0x{ekf_flags:x}; {details}).\n{self._log_tail()}"
            )
        finally:
            connection.close()
            time.sleep(0.2)

    def _log_tail(self, line_count: int = 40) -> str:
        if self._log_handle is not None:
            try:
                self._log_handle.flush()
            except OSError:
                pass
        if self.log_path is None or not self.log_path.exists():
            return "SITL log is unavailable."
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"Could not read SITL log: {exc}"
        return "\n".join(lines[-line_count:])

    def stop(self) -> None:
        """Stop sim_vehicle and its child autopilot/MAVProxy processes."""

        process = self.process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    process.wait(timeout=8.0)
                except (OSError, subprocess.TimeoutExpired):
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=8.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        self.process = None
        if self._atexit_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:
                pass
            self._atexit_registered = False
        self._close_resources()

    def _close_resources(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        if self._temporary_directory is not None:
            try:
                self._temporary_directory.cleanup()
            except OSError:
                pass
            self._temporary_directory = None

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch an EKF-ready ArduCopter SITL instance.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only resolve sim_vehicle.py; do not launch it.",
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            command, _ = discover_sim_vehicle()
            print(" ".join(command))
            return 0
        launcher = ArduPilotSITL()
        with launcher as connection_string:
            print(connection_string, flush=True)
            print("SITL is ready; press Ctrl-C to stop.", flush=True)
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    except (SITLUnavailable, SITLStartupError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
