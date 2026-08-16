'''Rasterize semantic sources and expose leakage-safe training tiles.'''

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np


IGNORE_INDEX = 255


class SemanticTileError(ValueError):
    pass


def _transform_coordinates(value: Any, transformer: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(
        isinstance(item, (int, float)) for item in value[:2]
    ):
        x, y = transformer.transform(float(value[0]), float(value[1]))
        return [x, y, *[float(item) for item in value[2:]]]
    if isinstance(value, (list, tuple)):
        return [_transform_coordinates(item, transformer) for item in value]
    return value


def _geojson_crs(payload: dict[str, Any]) -> str:
    crs = payload.get('crs') or {}
    name = str((crs.get('properties') or {}).get('name') or '')
    if name:
        if 'EPSG::' in name:
            return f'EPSG:{name.rsplit("EPSG::", 1)[1]}'
        if 'EPSG:' in name.upper():
            return f'EPSG:{name.upper().rsplit("EPSG:", 1)[1]}'
        return name
    return 'EPSG:4326'


def rasterize_geojson_polygons(
    image_path: str | Path,
    label_path: str | Path,
    *,
    class_id: int,
    background_id: int = 0,
) -> np.ndarray:
    import rasterio
    from rasterio.features import rasterize
    from pyproj import CRS, Transformer

    payload = json.loads(Path(label_path).read_text(encoding='utf-8'))
    with rasterio.open(image_path) as image:
        if image.crs is None:
            raise SemanticTileError(f'Training image has no CRS: {image_path}')
        source_crs = CRS.from_user_input(_geojson_crs(payload))
        target_crs = CRS.from_user_input(image.crs)
        transformer = None
        if source_crs != target_crs:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometries = []
        for feature in payload.get('features', []):
            geometry = dict(feature.get('geometry') or {})
            if not geometry.get('type') or not geometry.get('coordinates'):
                continue
            if transformer is not None:
                geometry['coordinates'] = _transform_coordinates(
                    geometry['coordinates'], transformer
                )
            geometries.append((geometry, int(class_id)))
        return rasterize(
            geometries,
            out_shape=(image.height, image.width),
            transform=image.transform,
            fill=int(background_id),
            dtype='uint16',
        )


def read_semantic_sample(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import rasterio

    image_path = Path(sample['image'])
    label_path = Path(sample['label'])
    with rasterio.open(image_path) as source:
        if source.count < 3:
            raise SemanticTileError(f'Semantic RGB image has fewer than three bands: {image_path}')
        image = source.read((1, 2, 3))
        source_dtype = np.dtype(source.dtypes[0])
        height, width = source.height, source.width
        # Band 4 on SpaceNet 7 is a validity mask, not decoration: a mosaic tile that
        # does not fully cover its footprint marks the uncovered pixels transparent.
        # Reading only RGB leaves those pixels looking like dark ground, and because the
        # building labels contain no polygons there, they are learned as background --
        # so the model is taught that no-data is a confident negative.
        invalid = None
        if source.count >= 4:
            alpha = source.read(4)
            if np.any(alpha == 0):
                invalid = alpha == 0
    if np.issubdtype(source_dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(source_dtype).max)
    else:
        image = image.astype(np.float32)
        if image.size and float(np.nanmax(image)) > 1.0:
            image /= 255.0
    image = np.nan_to_num(np.clip(image, 0.0, 1.0), copy=False)

    label_format = str(sample.get('label_format') or 'raster_class_ids')
    if label_format == 'geojson_polygons':
        label = rasterize_geojson_polygons(
            image_path,
            label_path,
            class_id=int(sample['class_id']),
            background_id=int(sample.get('background_id', 0)),
        )
    elif label_format == 'raster_class_ids':
        with rasterio.open(label_path) as source:
            label = source.read(1)
            if source.height != height or source.width != width:
                raise SemanticTileError('Semantic raster label dimensions do not match the image.')
    else:
        raise SemanticTileError(f'Unsupported semantic label_format: {label_format!r}')
    label = np.asarray(label)
    class_map = sample.get('class_map')
    if class_map is not None:
        if not isinstance(class_map, dict) or not class_map:
            raise SemanticTileError('Semantic class_map must be a non-empty mapping.')
        remapped = np.full(label.shape, IGNORE_INDEX, dtype=np.uint16)
        for source_id, target_id in class_map.items():
            remapped[label == int(source_id)] = int(target_id)
        label = remapped
    if invalid is not None:
        # Marked ignore rather than dropped: the pixels still occupy their position in
        # the tile, and the loss simply does not score them.
        label = np.asarray(label).copy()
        if label.shape == invalid.shape:
            label[invalid] = IGNORE_INDEX
    return image, label


class SemanticTileDataset:
    '''PyTorch dataset over one split of a built semantic corpus manifest.'''

    def __init__(
        self,
        corpus_manifest: str | Path,
        split: str,
        *,
        tile_size: int = 518,
        augment: bool = False,
    ) -> None:
        if split not in {'train', 'validation', 'test'}:
            raise SemanticTileError(f'Unknown semantic split: {split!r}')
        if tile_size < 16:
            raise SemanticTileError('tile_size must be at least 16.')
        manifest_path = Path(corpus_manifest)
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        # Relative sample paths resolve against the manifest, not the working directory.
        # A corpus built on one machine and read on another -- which is the whole point
        # of packing one for a hosted GPU -- carries relative paths, and resolving those
        # against CWD makes the corpus work or fail depending on where python was
        # started from. That failure arrives as 'file not found' several minutes into a
        # rented run.
        root = manifest_path.parent
        self.samples = []
        for item in payload.get('samples', []):
            if item.get('split') != split:
                continue
            item = dict(item)
            for key in ('image', 'label'):
                value = Path(str(item.get(key, '')))
                if not value.is_absolute():
                    item[key] = str((root / value).resolve())
            self.samples.append(item)
        if not self.samples:
            raise SemanticTileError(f'Corpus has no samples in split {split!r}.')
        classes = list((payload.get('schema') or {}).get('classes') or [])
        self.class_to_channel = {int(item['id']): index for index, item in enumerate(classes)}
        if len(self.class_to_channel) < 2:
            raise SemanticTileError('Semantic corpus schema needs at least two classes.')
        self.tile_size = int(tile_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        sample = self.samples[index]
        image, source_mask = read_semantic_sample(sample)
        mask = np.full(source_mask.shape, IGNORE_INDEX, dtype=np.int64)
        for class_id, channel in self.class_to_channel.items():
            mask[source_mask == class_id] = channel
        height, width = mask.shape
        pad_h = max(0, self.tile_size - height)
        pad_w = max(0, self.tile_size - width)
        if pad_h or pad_w:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=IGNORE_INDEX)
            height, width = mask.shape
        if self.augment:
            top = random.randint(0, height - self.tile_size)
            left = random.randint(0, width - self.tile_size)
        else:
            top = (height - self.tile_size) // 2
            left = (width - self.tile_size) // 2
        image = image[:, top:top + self.tile_size, left:left + self.tile_size]
        mask = mask[top:top + self.tile_size, left:left + self.tile_size]
        if self.augment and random.random() < 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
        if self.augment and random.random() < 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]
        return {
            'image': torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            'mask': torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            'sample_id': str(sample['id']),
            'site_id': str(sample['site_id']),
            'capture_date': str(sample['capture_date']),
        }
