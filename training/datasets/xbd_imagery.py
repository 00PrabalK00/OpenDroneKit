'''Complete OpenEarthMap with the imagery it does not redistribute.

Twenty-two OpenEarthMap regions have labels and no images. That is not a broken download:
OpenEarthMap sources those regions from xBD and its readme says plainly that the xBD RGB
images are not included, with xbd_files.csv giving the mapping. Until they are supplied,
the indexer sees a region with no images and walks past it -- so the labels sit on disk
being counted as nothing.

One of them is Gorakhpur, Uttar Pradesh: forty-nine tiles of Indian imagery at 0.3 m per
pixel, exhaustively labelled for all six classes. The existing India holdout is four
SpaceNet 7 tiles at 2.91-4.77 m/px labelled for buildings alone, so it can say nothing
about road, vegetation, water or bare land, and it tests a satellite scale the product
never flies. Gorakhpur can.

GEOREFERENCING IS THE WHOLE POINT. The xBD tiles ship as PNG, which carries no
geotransform, and the trainer resamples every source to a common ground scale using
exactly that. A PNG renamed to .tif would be refused by the loader -- correctly -- so this
writes real GeoTIFFs using the transforms xBD publishes alongside the imagery. Every tile
carries its own; none is assumed.

    python -m training.datasets.xbd_imagery "D:/xview dataset" \\
        --openearthmap training/data/openearthmap_mixed
'''

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np


# Every one of these archives ships named .tar and all four are gzip. Sniffing the
# magic beats trusting the extension: tarfile raises 'invalid header' otherwise, which
# reads as a corrupt 82 GB download rather than a mislabelled one.
GZIP_MAGIC = bytes([0x1F, 0x8B])


class XbdImageryError(RuntimeError):
    pass


def read_mapping(openearthmap_root: Path) -> dict[str, tuple[str, str]]:
    '''xBD PNG name -> (region, OpenEarthMap tile name).'''
    mapping_path = openearthmap_root / 'xbd_files.csv'
    if not mapping_path.is_file():
        raise XbdImageryError(
            f'No xbd_files.csv at {mapping_path}. It ships with OpenEarthMap and is the '
            'only record of which xBD tile becomes which region tile.'
        )
    mapping: dict[str, tuple[str, str]] = {}
    with mapping_path.open(encoding='utf-8') as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0].strip():
                continue
            png, target = row[0].strip(), row[1].strip()
            region = target.rsplit('_', 1)[0]
            mapping[png] = (region, target)
    if not mapping:
        raise XbdImageryError(f'{mapping_path} lists no tiles.')
    return mapping


def read_geotransforms(archive: Path) -> dict[str, Any]:
    '''The per-tile transform and CRS xBD publishes, keyed by PNG name.'''
    if not archive.is_file():
        raise XbdImageryError(
            f'No {archive.name}. Without it the tiles have no georeferencing, and the '
            'trainer cannot bring them to a common ground scale -- it would refuse them, '
            'which is the correct behaviour and not a workaround to disable.'
        )
    with tarfile.open(archive, 'r:gz') as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith('.json')), None)
        if member is None:
            raise XbdImageryError(f'{archive} contains no JSON.')
        payload = tar.extractfile(member)
        if payload is None:
            raise XbdImageryError(f'Could not read {member.name} from {archive}.')
        return json.loads(payload.read().decode('utf-8'))


def write_geotiff(destination: Path, image: np.ndarray, transform: list[float], crs_wkt: str) -> None:
    '''Write a real GeoTIFF, because the ground scale is read back off this file.'''
    import rasterio
    from rasterio.transform import Affine

    if image.ndim != 3 or image.shape[2] < 3:
        raise XbdImageryError(f'Expected an RGB tile, got shape {image.shape}.')
    bands = image[:, :, :3]
    # xBD publishes GDAL's six-element form: (west, x_size, x_skew, north, y_skew, y_size).
    west, x_size, x_skew, north, y_skew, y_size = transform
    affine = Affine(x_size, x_skew, west, y_skew, y_size, north)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination, 'w', driver='GTiff',
        height=bands.shape[0], width=bands.shape[1], count=3,
        dtype=bands.dtype, crs=crs_wkt, transform=affine,
        compress='DEFLATE',
    ) as sink:
        for index in range(3):
            sink.write(bands[:, :, index], index + 1)


def ingest(
    xview_root: Path,
    openearthmap_root: Path,
    *,
    regions: tuple[str, ...] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    mapping = read_mapping(openearthmap_root)
    if regions:
        wanted = set(regions)
        mapping = {k: v for k, v in mapping.items() if v[0] in wanted}
        if not mapping:
            raise XbdImageryError(f'No mapped tiles for regions {sorted(wanted)}.')

    geotransforms = read_geotransforms(xview_root / 'xview_geotransforms.json.tgz')

    outstanding = {}
    for png, (region, target) in mapping.items():
        destination = openearthmap_root / region / 'images' / target
        if destination.is_file() and not overwrite:
            continue
        outstanding[png] = (region, target)

    written: list[str] = []
    missing_transform: list[str] = []
    skipped_no_label: list[str] = []

    # Streamed rather than extracted. The four archives are 31 GB and hold 22,068 tiles;
    # 1,162 are wanted. Unpacking everything to copy 5 per cent of it would cost half the
    # remaining disk for no benefit.
    archives = sorted(xview_root.glob('*_images_labels_targets.tar')) + sorted(
        xview_root.glob('tier3.tar')
    )
    if not archives:
        raise XbdImageryError(f'No xBD image archives found in {xview_root}.')

    for archive in archives:
        if not outstanding:
            break
        # Mode from the file's MAGIC, not its name: every one of these ships as .tar
        # and all four are gzip. Trusting the extension raises 'invalid header' and
        # reads as a corrupt download rather than a mislabelled one.
        with archive.open('rb') as probe:
            compressed = probe.read(2) == GZIP_MAGIC
        with tarfile.open(archive, 'r|gz' if compressed else 'r|') as tar:
            for member in tar:
                if not outstanding:
                    break
                name = Path(member.name).name
                if name not in outstanding or not member.isfile():
                    continue
                region, target = outstanding.pop(name)

                # A tile whose label is absent is not useful and must not be written: the
                # indexer would then see an image with no label and report a corpus fault
                # that this tool created.
                if not (openearthmap_root / region / 'labels' / target).is_file():
                    skipped_no_label.append(target)
                    continue

                entry = geotransforms.get(name)
                if not entry:
                    missing_transform.append(name)
                    continue
                transform, crs_wkt = entry[0], entry[1]

                payload = tar.extractfile(member)
                if payload is None:
                    missing_transform.append(name)
                    continue
                image = np.asarray(Image.open(io.BytesIO(payload.read())).convert('RGB'))
                write_geotiff(
                    openearthmap_root / region / 'images' / target,
                    image, transform, crs_wkt,
                )
                written.append(f'{region}/{target}')
                if len(written) % 100 == 0:
                    print(f'  wrote {len(written)} tiles', flush=True)

    report = {
        'written': len(written),
        'regions': sorted({name.split('/')[0] for name in written}),
        'still_missing': sorted(outstanding.values()),
        'missing_geotransform': missing_transform,
        'skipped_no_label': skipped_no_label,
    }
    if outstanding:
        # Named rather than swallowed: a partly-populated region trains on a subset while
        # looking complete, and the count is the only place that would ever show it.
        print(
            f'{len(outstanding)} mapped tile(s) were not found in any archive. '
            'They are listed in the report.',
            flush=True,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Add the xBD RGB imagery OpenEarthMap does not redistribute.'
    )
    parser.add_argument('xview_root', type=Path, help='Directory holding the xView2 archives.')
    parser.add_argument(
        '--openearthmap', type=Path,
        default=Path('training/data/openearthmap_mixed'),
    )
    parser.add_argument(
        '--region', action='append', default=[], dest='regions',
        help='Only this region. Repeatable. Default: every mapped region.',
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    report = ingest(
        args.xview_root, args.openearthmap,
        regions=tuple(args.regions), overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2)[:4000])
    return 0 if report['written'] or not report['still_missing'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
