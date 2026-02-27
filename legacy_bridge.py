"""Utilities to discover and reuse legacy assets from this workspace."""

from __future__ import annotations

from pathlib import Path
import shutil


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def toolkit_root() -> Path:
    return Path(__file__).resolve().parent


def models_root() -> Path:
    return toolkit_root() / "models"


def find_swin_unet_checkpoint() -> Path | None:
    target = models_root() / "legacy" / "swin_unet_best_model.pth"
    if target.exists():
        return target

    root = workspace_root()
    candidate = root / "CConCrack_SwinUNet_Final" / "checkpoints" / "best_model.pth"
    if not candidate.exists():
        return None

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
        if target.exists():
            return target
    except Exception:
        pass
    return target if target.exists() else None


def legacy_cracksim_path() -> Path | None:
    root = workspace_root()
    candidate = root / "cracksim.py"
    return candidate if candidate.exists() else None


def legacy_viewer_path() -> Path | None:
    root = workspace_root()
    candidate = root / "viewer.py"
    return candidate if candidate.exists() else None


def legacy_summary() -> dict:
    ckpt = find_swin_unet_checkpoint()
    cracksim = legacy_cracksim_path()
    viewer = legacy_viewer_path()
    return {
        "swin_unet_checkpoint": str(ckpt) if ckpt else "",
        "cracksim": str(cracksim) if cracksim else "",
        "viewer": str(viewer) if viewer else "",
    }
