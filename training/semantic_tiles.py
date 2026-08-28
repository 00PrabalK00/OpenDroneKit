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


def read_semantic_sample(
    sample: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    no_data = np.zeros(label.shape, dtype=bool)
    if invalid is not None:
        # Marked ignore rather than dropped: the pixels still occupy their position in
        # the tile, and the loss simply does not score them.
        label = np.asarray(label).copy()
        if label.shape == invalid.shape:
            label[invalid] = IGNORE_INDEX
            no_data = invalid
    # Returned separately because the two reasons a pixel is IGNORE are not the same
    # evidence. An unlabelled pixel in an exhaustively annotated tile is evidence the
    # class is absent; a no-data pixel is evidence of nothing at all, and treating it as
    # a negative would teach the model that transparent mosaic edges are confidently
    # not-building.
    return image, label, no_data


def sample_gsd(sample: dict[str, Any]) -> float | None:
    """Ground sample distance of this tile, in metres per pixel.

    Read from the file's own geotransform rather than assumed per source, because the
    assumption was wrong for a year: the registry note called SpaceNet 7 "0.5 m satellite
    imagery" when it is a Planet mosaic at roughly 3.8 m. Web Mercator pixels are
    inflated by 1/cos(latitude), so the raw transform overstates the ground size and has
    to be corrected before the two sources can be compared at all.
    """
    import math

    import rasterio

    try:
        with rasterio.open(sample['image']) as source:
            pixel = abs(source.transform.a)
            epsg = source.crs.to_epsg() if source.crs else None
            if epsg == 3857:
                latitude = math.degrees(
                    2 * math.atan(math.exp(source.bounds.top / 6378137.0)) - math.pi / 2
                )
                return pixel * math.cos(math.radians(latitude))
            if epsg == 4326:
                latitude = (source.bounds.top + source.bounds.bottom) / 2
                return pixel * 111320.0 * math.cos(math.radians(latitude))
            return pixel
    except Exception:  # noqa: BLE001 - an unreadable transform is not fatal here
        return None


def resample_to_scale(image, mask, no_data, factor: float):
    """Resize a crop so one pixel means the same distance on the ground in every source.

    The corpus mixes OpenEarthMap at 0.5 m with SpaceNet 7 at about 3.8 m, and the tiler
    cut a fixed 518-pixel window from both. That window is 259 m across on one source and
    nearly 2 km on the other -- the same tensor shape standing for ground areas that
    differ by seven and a half times.

    So the model saw buildings at two incompatible scales and learned a prior that fits
    neither: on coarse imagery a building is a two-pixel blob, and applying that prior to
    fine imagery produces exactly the failure measured on the holdout -- building
    predicted over 0.244 of the frame against a labelled 0.092.

    Images interpolate; masks must not. A label is a class id, and averaging class 1 with
    class 3 yields class 2, which is a different thing entirely -- so masks and the
    no-data plane go through nearest neighbour.
    """
    import cv2

    if abs(factor - 1.0) < 1e-3:
        return image, mask, no_data

    height, width = mask.shape
    target = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    # Shrinking wants area averaging; growing wants a smooth interpolant.
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    resized = np.stack([
        cv2.resize(band, target, interpolation=interpolation) for band in image
    ])
    mask = cv2.resize(mask.astype(np.int32), target, interpolation=cv2.INTER_NEAREST)
    no_data = cv2.resize(
        no_data.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return resized, mask.astype(np.int64), no_data


class SemanticTileDataset:
    '''PyTorch dataset over one split of a built semantic corpus manifest.'''

    def __init__(
        self,
        corpus_manifest: str | Path,
        split: str,
        *,
        tile_size: int = 518,
        augment: bool = False,
        target_gsd: float | None = None,
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
        # Metres per pixel every sample is brought to before it reaches the model. None
        # keeps the old behaviour of cropping a fixed pixel count, which mixed a 259 m
        # window with a 2 km one and called both a tile.
        self.target_gsd = float(target_gsd) if target_gsd else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        sample = self.samples[index]
        image, source_mask, no_data = read_semantic_sample(sample)
        mask = np.full(source_mask.shape, IGNORE_INDEX, dtype=np.int64)
        for class_id, channel in self.class_to_channel.items():
            mask[source_mask == class_id] = channel
        # Cut the window by GROUND extent, not by pixel count, then bring it to the
        # tile size. Cropping 518 native pixels from a 3.8 m mosaic took in two
        # kilometres where the same crop of a 0.5 m tile takes in 259 metres, and the
        # model was asked to learn one notion of "building" across both.
        #
        # Crop first and resample after: the alternative resizes a whole 1024-pixel tile
        # by seven and a half before throwing most of it away, which is sixty million
        # pixels of work per sample for a 518-pixel result.
        crop_size = self.tile_size
        scale = 1.0
        if self.target_gsd:
            native = sample.get('_gsd')
            if native is None:
                native = sample['_gsd'] = sample_gsd(sample)
            if native and native > 0:
                scale = self.target_gsd / native
                crop_size = max(16, int(round(self.tile_size * scale)))

        height, width = mask.shape
        pad_h = max(0, crop_size - height)
        pad_w = max(0, crop_size - width)
        if no_data.shape != mask.shape:
            no_data = np.zeros(mask.shape, dtype=bool)
        if pad_h or pad_w:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=IGNORE_INDEX)
            # Padding is invented pixels, so it carries no evidence either way and must
            # never become a negative.
            no_data = np.pad(no_data, ((0, pad_h), (0, pad_w)), constant_values=True)
            height, width = mask.shape
        if self.augment:
            top = random.randint(0, height - crop_size)
            left = random.randint(0, width - crop_size)
        else:
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
        image = image[:, top:top + crop_size, left:left + crop_size]
        mask = mask[top:top + crop_size, left:left + crop_size]
        no_data = no_data[top:top + crop_size, left:left + crop_size]
        if crop_size != self.tile_size:
            image, mask, no_data = resample_to_scale(
                image, mask, no_data, self.tile_size / crop_size
            )
        if self.augment and random.random() < 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
            no_data = no_data[:, ::-1]
        if self.augment and random.random() < 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]
            no_data = no_data[::-1, :]

        # An exhaustively annotated tile says something about its unlabelled pixels: the
        # annotators drew every instance of these classes they saw, so a pixel outside
        # every polygon is evidence that class is ABSENT there. It is not evidence of
        # what the pixel is, which is why this cannot be folded into the mask as a
        # background label -- there is no basis for calling it road rather than water.
        negative_classes = np.zeros(len(self.class_to_channel), dtype=np.float32)
        for class_id in sample.get('exhaustive_class_ids') or ():
            channel = self.class_to_channel.get(int(class_id))
            if channel is not None:
                negative_classes[channel] = 1.0
        negative_mask = (mask == IGNORE_INDEX) & ~no_data if negative_classes.any() \
            else np.zeros(mask.shape, dtype=bool)

        return {
            'image': torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            'mask': torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            # Where a class is known absent, and which classes that applies to.
            'negative_mask': torch.from_numpy(np.ascontiguousarray(negative_mask)),
            'negative_classes': torch.from_numpy(negative_classes),
            'sample_id': str(sample['id']),
            'site_id': str(sample['site_id']),
            'capture_date': str(sample['capture_date']),
        }
