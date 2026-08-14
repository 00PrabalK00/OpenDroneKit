'''Download reusable foundation/domain checkpoints with local provenance.

These files are initializers or third-party baselines. Downloading one does not make
it an OpenDroneKit task-trained model; only models/manifests/model_provenance.json
may make that claim after evaluation.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from training.datasets.download import download_http, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / 'models' / 'pretrained'


@dataclass(frozen=True)
class PretrainedAsset:
    id: str
    filename: str
    url: str
    architecture: str
    license: str
    role: str
    redistribution: str
    task_trained: bool = False


ASSETS: dict[str, PretrainedAsset] = {
    'dinov2_vitb14': PretrainedAsset(
        id='dinov2_vitb14',
        filename='dinov2_vitb14_pretrain.pth',
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        architecture='DINOv2 ViT-B/14',
        license='Apache-2.0',
        role='Shared semantic encoder and anomaly feature backbone.',
        redistribution='Allowed with Apache-2.0 notices.',
    ),
    'yolo11x': PretrainedAsset(
        id='yolo11x',
        filename='yolo11x.pt',
        url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
        architecture='Ultralytics YOLO11x detection',
        license='AGPL-3.0 or Ultralytics Enterprise',
        role='Offline high-accuracy initialization for road, power and rail detectors.',
        redistribution='Licence gate: AGPL-compliant deployment or Enterprise terms required.',
    ),
    'yolo11l_seg': PretrainedAsset(
        id='yolo11l_seg',
        filename='yolo11l-seg.pt',
        url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt',
        architecture='Ultralytics YOLO11l instance segmentation',
        license='AGPL-3.0 or Ultralytics Enterprise',
        role='Initialization for solar-module, plant and tree instances.',
        redistribution='Licence gate: AGPL-compliant deployment or Enterprise terms required.',
    ),
    'weedsgalore_deeplabv3plus': PretrainedAsset(
        id='weedsgalore_deeplabv3plus',
        filename='weedsgalore_deeplabv3plus_ckpts.zip',
        url='https://doidata.gfz.de/weedsgalore_e_celikkan_2024/ckpts.zip',
        architecture='DeepLabv3+ RGB and multispectral benchmark bundle',
        license='Checkpoint licence not explicit; upstream code Apache-2.0 and training data CC BY 4.0',
        role='Domain-specific agriculture baseline, not an India-calibrated production model.',
        redistribution='Do not redistribute until the checkpoint licence is clarified.',
        task_trained=True,
    ),
}


def fetch(names: list[str], root: Path = DEFAULT_ROOT) -> dict[str, dict]:
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for name in names:
        if name not in ASSETS:
            raise KeyError(f'Unknown pretrained asset: {name}')
        spec = ASSETS[name]
        destination = root / spec.filename
        print(f'[{name}] {spec.url}', flush=True)
        download_http(spec.url, destination, progress=lambda line: print(line, flush=True))
        records[name] = {
            **asdict(spec),
            'path': str(destination.relative_to(REPO_ROOT)),
            'bytes': destination.stat().st_size,
            'sha256': sha256_file(destination),
            'status': 'downloaded',
        }

    manifest_path = root / 'download_manifest.json'
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            existing = {}
    existing.update(records)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    print(f'Manifest: {manifest_path}', flush=True)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Download approved pretrained assets.')
    parser.add_argument('names', nargs='*')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.list:
        for spec in ASSETS.values():
            print(f'{spec.id:<32} {spec.license:<42} {spec.role}')
        return 0
    names = list(ASSETS) if args.all else args.names
    if not names:
        parser.error('choose one or more asset ids, or pass --all')
    fetch(names, args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
