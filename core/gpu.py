"""Which GPUs this machine has, and which of them OpenDroneKit can actually use.

Two different questions, kept apart on purpose:

* **Present** -- the display adapters Windows reports (Win32_VideoController), and the
  NVIDIA devices `nvidia-smi` lists with their driver and memory. This is hardware.
* **Usable** -- the onnxruntime execution providers this build was compiled with
  (CUDA / DirectML / TensorRT ...) and whether torch sees CUDA. A GPU that is present
  but has no usable provider runs nothing faster, and saying "GPU: yes" in that case
  would be a claim the next inference run contradicts.

Every probe is optional and bounded: a missing `nvidia-smi`, PowerShell, onnxruntime or
torch yields an empty answer with the reason, never an exception and never a guess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Callable

Runner = Callable[[list[str], float], "subprocess.CompletedProcess[str] | None"]

GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
)

_CACHE: dict[str, Any] | None = None


def _run(cmd: list[str], timeout: float) -> "subprocess.CompletedProcess[str] | None":
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return None


def vendor_of(name: str) -> str:
    low = (name or "").lower()
    for key, vendor in (("nvidia", "NVIDIA"), ("geforce", "NVIDIA"), ("quadro", "NVIDIA"),
                        ("radeon", "AMD"), ("amd", "AMD"), ("intel", "Intel"),
                        ("arc", "Intel"), ("microsoft basic", "Microsoft"),
                        ("qualcomm", "Qualcomm"), ("adreno", "Qualcomm"), ("apple", "Apple")):
        if key in low:
            return vendor
    return "unknown"


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits`."""
    gpus: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            memory = int(float(parts[2]))
        except ValueError:
            memory = None
        gpus.append({"name": parts[0], "vendor": "NVIDIA", "driver": parts[1],
                     "memory_mb": memory, "source": "nvidia-smi"})
    return gpus


def parse_video_controllers(text: str) -> list[dict[str, Any]]:
    """Parse the JSON PowerShell emits for Win32_VideoController (one object or a list)."""
    try:
        payload = json.loads(text or "null")
    except ValueError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    adapters: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict) or not row.get("Name"):
            continue
        ram = row.get("AdapterRAM")
        # AdapterRAM is a uint32 and saturates at 4 GB, so it is a floor, not a size.
        memory = int(ram) // (1024 * 1024) if isinstance(ram, (int, float)) and ram > 0 else None
        adapters.append({"name": str(row["Name"]), "vendor": vendor_of(str(row["Name"])),
                         "driver": str(row.get("DriverVersion") or ""),
                         "memory_mb": memory, "memory_is_floor": memory is not None and memory >= 4095,
                         "source": "Win32_VideoController"})
    return adapters


def onnxruntime_providers() -> dict[str, Any]:
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "version": "", "providers": [], "gpu_providers": [],
                "reason": f"onnxruntime is not importable: {exc}"}
    providers = list(ort.get_available_providers())
    return {"installed": True, "version": getattr(ort, "__version__", ""),
            "providers": providers,
            "gpu_providers": [p for p in providers if p in GPU_PROVIDERS],
            "reason": ""}


def torch_cuda() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {"installed": False, "cuda": False, "device": "", "reason": "torch is not installed"}
    try:
        available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if available else ""
    except Exception as exc:  # noqa: BLE001
        return {"installed": True, "cuda": False, "device": "", "reason": str(exc)}
    return {"installed": True, "cuda": available, "device": device, "reason": ""}


def detect_gpus(runner: Runner | None = None, *, use_cache: bool = True) -> dict[str, Any]:
    """Hardware present, inference providers usable, and a one-line summary of both."""
    global _CACHE
    if use_cache and runner is None and _CACHE is not None:
        return _CACHE
    run = runner or _run

    adapters: list[dict[str, Any]] = []
    notes: list[str] = []
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader,nounits"], 15.0)
    nvidia = parse_nvidia_smi(smi.stdout) if smi is not None and smi.returncode == 0 else []
    if smi is None:
        notes.append("nvidia-smi not found (no NVIDIA driver, or not on PATH).")
    adapters.extend(nvidia)

    if sys.platform == "win32":
        ps = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                  "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,"
                  "AdapterRAM | ConvertTo-Json -Compress"], 20.0)
        if ps is not None and ps.returncode == 0:
            seen = {g["name"].lower() for g in nvidia}
            for adapter in parse_video_controllers(ps.stdout):
                if adapter["name"].lower() not in seen:
                    adapters.append(adapter)
        else:
            notes.append("Win32_VideoController could not be queried.")

    ort = onnxruntime_providers()
    tch = torch_cuda()
    physical = [a for a in adapters if a["vendor"] != "Microsoft"]
    usable = bool(ort["gpu_providers"]) or bool(tch["cuda"])
    if not physical:
        summary = "No GPU detected; inference runs on the CPU."
    elif usable:
        via = ", ".join(ort["gpu_providers"] + (["torch CUDA"] if tch["cuda"] else []))
        summary = f"{physical[0]['name']} usable for inference via {via}."
    else:
        summary = (f"{physical[0]['name']} present, but this build has no GPU execution "
                   "provider (onnxruntime is the CPU build), so inference runs on the CPU.")
    result = {
        "adapters": adapters,
        "onnxruntime": ort,
        "torch": tch,
        "gpu_present": bool(physical),
        "gpu_usable_for_inference": usable,
        "summary": summary,
        "notes": notes,
    }
    if runner is None:
        _CACHE = result
    return result
