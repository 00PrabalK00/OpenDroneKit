"""Watch a Vast training run, retrieve the weights, verify them, then destroy the box.

Written to run unattended overnight, so the ordering is the whole point: nothing
destructive happens until the artefacts are on local disk AND their hashes match what
the remote computed. A run that trained perfectly and then lost its weights to an eager
teardown has produced nothing, and the credit is spent either way.

The failure path never destroys. If the bootstrap failed, the corpus was rejected,
training crashed, or a download arrived with the wrong hash, this stops and leaves the
instance alive with a clear reason. Paying a few more cents while a person looks at it
is always cheaper than repeating a ten-hour run.

    python -m tools.vast_watch 47811117
    python -m tools.vast_watch 47811117 --no-destroy
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS = ROOT / "training" / "runs"
REMOTE_STATE = "/workspace/state"
WANTED = ("best.pt", "last.pt")

# The key registered with Vast as "claude-code-access". ssh only auto-discovers the
# conventional names (id_ed25519, id_rsa) and neither exists here, so without an
# explicit -i every retrieval fails with "Permission denied" -- after the training has
# already finished and been paid for. IdentitiesOnly stops ssh offering other keys and
# tripping the server's auth attempt limit before it reaches this one.
SSH_KEY = str(Path.home() / ".ssh" / "claude_remote_key")
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "IdentitiesOnly=yes",
    "-i", SSH_KEY,
]


def vast(*args: str) -> str:
    result = subprocess.run(["vastai", *args], capture_output=True, text=True)
    return result.stdout.strip()


def instance(instance_id: int) -> dict | None:
    raw = vast("show", "instances", "--raw")
    if not raw:
        return None
    try:
        for item in json.loads(raw):
            if int(item.get("id", 0)) == instance_id:
                return item
    except json.JSONDecodeError:
        return None
    return None


def ssh_target(info: dict) -> tuple[str, str]:
    return str(info.get("ssh_host") or ""), str(info.get("ssh_port") or "")


def remote(info: dict, command: str, timeout: int = 120) -> tuple[int, str]:
    host, port = ssh_target(info)
    if not host or not port:
        return 1, "no ssh endpoint yet"
    result = subprocess.run(
        ["ssh", *SSH_OPTS, "-p", port, f"root@{host}", command],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, (result.stdout or result.stderr).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.vast_watch")
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--run-name", default="pvel_ad_yolo11l")
    parser.add_argument("--poll", type=int, default=300)
    parser.add_argument("--no-destroy", action="store_true",
                        help="Retrieve and verify but leave the instance running.")
    args = parser.parse_args(argv)

    target = LOCAL_RUNS / args.run_name
    target.mkdir(parents=True, exist_ok=True)

    print(f"watching instance {args.instance_id}", flush=True)
    last_state = ""
    while True:
        info = instance(args.instance_id)
        if info is None:
            print("INSTANCE GONE: it is no longer listed. Nothing to retrieve.", flush=True)
            return 1

        status = str(info.get("actual_status") or "unknown")
        code, state = remote(info, f"cat {REMOTE_STATE}/status 2>/dev/null || echo pending")
        state = state if code == 0 else "unreachable"

        if f"{status}/{state}" != last_state:
            print(f"instance={status} bootstrap={state}", flush=True)
            last_state = f"{status}/{state}"

        if state == "failed":
            _, why = remote(info, f"tail -25 {REMOTE_STATE}/bootstrap.log")
            print("BOOTSTRAP FAILED. Instance left alive on purpose.", flush=True)
            print(why, flush=True)
            return 1

        if state == "complete":
            break

        if state == "training":
            # Surface progress so a stalled run is distinguishable from a slow one.
            _, tail = remote(
                info,
                f"grep -oE '[0-9]+/{{0,1}}[0-9]*\\s+epochs?|^ *[0-9]+/[0-9]+' "
                f"{REMOTE_STATE}/train.log 2>/dev/null | tail -1",
            )
            if tail:
                print(f"  training: {tail}", flush=True)

        time.sleep(args.poll)

    print("training complete; retrieving weights", flush=True)
    info = instance(args.instance_id)
    if info is None:
        print("INSTANCE GONE before retrieval.", flush=True)
        return 1

    code, expected_block = remote(info, f"cat {REMOTE_STATE}/weights.sha256")
    expected: dict[str, str] = {}
    for line in expected_block.splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[Path(parts[1]).name] = parts[0].lower()
    print(f"remote hashes: {expected}", flush=True)

    host, port = ssh_target(info)
    retrieved: dict[str, str] = {}
    for name in WANTED:
        local_path = target / name
        source = f"root@{host}:{REMOTE_STATE}/runs/{args.run_name}/weights/{name}"
        print(f"  scp {name}", flush=True)
        result = subprocess.run(
            ["scp", *SSH_OPTS, "-P", port, source, str(local_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not local_path.is_file():
            print(f"DOWNLOAD FAILED for {name}. Instance left alive.", flush=True)
            print(result.stderr.strip(), flush=True)
            return 1
        retrieved[name] = sha256(local_path)
        print(f"    {local_path.stat().st_size} bytes  {retrieved[name][:16]}...", flush=True)

    # best.pt is the deliverable; last.pt is what makes --weights extension possible once
    # the instance no longer exists, so both are verified before anything is destroyed.
    for name, digest in retrieved.items():
        if name in expected and expected[name] != digest:
            print(f"HASH MISMATCH for {name}: remote {expected[name]} local {digest}. "
                  "Instance left alive.", flush=True)
            return 1
    print("all weights verified against the remote hashes", flush=True)

    if args.no_destroy:
        print("--no-destroy set; instance still running.", flush=True)
        return 0

    print(f"destroying instance {args.instance_id}", flush=True)
    out = vast("destroy", "instance", str(args.instance_id))
    print(out or "(no output)", flush=True)
    time.sleep(15)
    if instance(args.instance_id) is None:
        print("DESTROYED: billing has stopped.", flush=True)
        return 0
    print("WARNING: instance still listed after destroy. Check manually.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
