# Legacy Manifest

This manifest documents which parts of the old workspace are intentionally reused.

- `cc/cracksim.py`
  - Used as optional FEniCSx bridge through `final_toolkit.core.propagation.run_fenicsx_phasefield`.
- `cc/CConCrack_SwinUNet_Final/checkpoints/best_model.pth`
  - Referenced as optional segmentation checkpoint for future deep-model integration.
- `cc/viewer.py`
  - Kept as legacy 3D viewer reference.

Not included in final runtime path by default:

- legacy `cc/ui.py` monolith
- COLMAP worker orchestration modules
- temporary/backup UI copies
