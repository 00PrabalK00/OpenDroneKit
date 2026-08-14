'''Shared, geospatial semantic-segmentation inference.

The engine is deliberately model-agnostic at runtime. A trained DINOv2 + decoder
can be exported to ONNX and passed through :class:`ONNXSemanticPredictor`, while
tests and research adapters can implement the small ``predict(tile)`` protocol.
Foundation weights alone are rejected: an encoder checkpoint is not a task model.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from . import geo


CLASS_NODATA = 65535


class SemanticInferenceRefused(RuntimeError):
    '''Raised when an inference request cannot make a defensible result.'''


@dataclass(frozen=True)
class SemanticClass:
    id: int
    name: str
    color_rgb: tuple[int, int, int]
    background: bool = False

    def __post_init__(self) -> None:
        if not 0 <= int(self.id) < CLASS_NODATA:
            raise ValueError(f'Class id must be between 0 and {CLASS_NODATA - 1}.')
        if not self.name.strip():
            raise ValueError('Semantic class names cannot be empty.')
        if len(self.color_rgb) != 3 or any(not 0 <= int(v) <= 255 for v in self.color_rgb):
            raise ValueError('color_rgb must contain three values between 0 and 255.')

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['color_rgb'] = list(self.color_rgb)
        return value


@dataclass(frozen=True)
class SemanticSchema:
    id: str
    version: str
    classes: tuple[SemanticClass, ...]

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.version.strip():
            raise ValueError('Semantic schema id and version are required.')
        if len(self.classes) < 2:
            raise ValueError('A semantic schema needs at least two classes.')
        ids = [item.id for item in self.classes]
        names = [item.name.casefold() for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError('Semantic class ids must be unique.')
        if len(names) != len(set(names)):
            raise ValueError('Semantic class names must be unique.')

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'version': self.version,
            'nodata': CLASS_NODATA,
            'classes': [item.to_dict() for item in self.classes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'SemanticSchema':
        return cls(
            id=str(value['id']),
            version=str(value['version']),
            classes=tuple(
                SemanticClass(
                    id=int(item['id']),
                    name=str(item['name']),
                    color_rgb=tuple(int(v) for v in item['color_rgb']),
                    background=bool(item.get('background', False)),
                )
                for item in value['classes']
            ),
        )


@dataclass(frozen=True)
class SemanticModelMetadata:
    key: str
    version: str
    architecture: str
    checkpoint_sha256: str
    schema_id: str
    schema_version: str
    task_trained: bool
    training_origin: str
    validation_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.version.strip() or not self.architecture.strip():
            raise ValueError('Model key, version and architecture are required.')
        digest = self.checkpoint_sha256.casefold()
        if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
            raise ValueError('checkpoint_sha256 must be a 64-character SHA-256 digest.')
        if not self.schema_id.strip() or not self.schema_version.strip():
            raise ValueError('Model schema id and version are required.')
        if not self.training_origin.strip():
            raise ValueError('training_origin is required for model provenance.')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'SemanticModelMetadata':
        if not isinstance(value.get('task_trained'), bool):
            raise ValueError('task_trained must be an explicit boolean.')
        return cls(
            key=str(value['key']),
            version=str(value['version']),
            architecture=str(value['architecture']),
            checkpoint_sha256=str(value['checkpoint_sha256']),
            schema_id=str(value['schema_id']),
            schema_version=str(value['schema_version']),
            task_trained=bool(value.get('task_trained', False)),
            training_origin=str(value['training_origin']),
            validation_metrics={
                str(k): float(v) for k, v in dict(value.get('validation_metrics', {})).items()
            },
        )


@dataclass(frozen=True)
class SemanticInferenceConfig:
    tile_size: int = 518
    overlap: int = 126
    device: str = 'cuda'
    allow_cpu: bool = False
    max_cpu_pixels: int = 4_000_000
    min_polygon_area_m2: float = 1.0
    polygonize_background: bool = False
    input_bands: tuple[int, int, int] = (1, 2, 3)

    def __post_init__(self) -> None:
        if self.tile_size < 16:
            raise ValueError('tile_size must be at least 16 pixels.')
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError('overlap must be non-negative and smaller than tile_size.')
        if self.max_cpu_pixels <= 0:
            raise ValueError('max_cpu_pixels must be positive.')
        if self.min_polygon_area_m2 < 0:
            raise ValueError('min_polygon_area_m2 cannot be negative.')
        if len(self.input_bands) != 3 or min(self.input_bands) < 1:
            raise ValueError('input_bands must name three one-based raster bands.')


@dataclass(frozen=True)
class SemanticInferencePackage:
    class_map_path: str
    confidence_path: str
    polygons_path: str
    manifest_path: str

    def artifact_paths(self) -> list[str]:
        return [
            self.class_map_path,
            self.confidence_path,
            self.polygons_path,
            self.manifest_path,
        ]


class SemanticPredictor(Protocol):
    device: str

    def predict(self, tile_chw_01: np.ndarray) -> np.ndarray:
        '''Return class logits shaped ``(classes, height, width)``.'''


def load_semantic_manifest(path: str | Path) -> tuple[SemanticSchema, SemanticModelMetadata, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if int(payload.get('manifest_schema_version', 0)) != 1:
        raise SemanticInferenceRefused('Unsupported semantic model manifest schema version.')
    schema = SemanticSchema.from_dict(dict(payload['schema']))
    model = SemanticModelMetadata.from_dict(dict(payload['model']))
    return schema, model, dict(payload.get('inference', {}))


class ONNXSemanticPredictor:
    '''ONNX adapter for a trained full semantic model, including its decoder.'''

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = 'cuda',
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SemanticInferenceRefused('onnxruntime is required for semantic inference.') from exc

        requested = device.casefold()
        available = ort.get_available_providers()
        if requested.startswith('cuda'):
            if 'CUDAExecutionProvider' not in available:
                raise SemanticInferenceRefused(
                    'CUDA semantic inference was requested but ONNX Runtime has no CUDA provider.'
                )
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = 'cuda'
        elif requested == 'cpu':
            providers = ['CPUExecutionProvider']
            self.device = 'cpu'
        else:
            raise SemanticInferenceRefused(f'Unsupported semantic inference device: {device!r}.')

        path = Path(model_path)
        if not path.is_file():
            raise SemanticInferenceRefused(f'Semantic ONNX model does not exist: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self._std = np.asarray(std, dtype=np.float32)[:, None, None]

    def predict(self, tile_chw_01: np.ndarray) -> np.ndarray:
        normalised = (np.asarray(tile_chw_01, dtype=np.float32) - self._mean) / self._std
        output = self._session.run(None, {self._input_name: normalised[None]})[0]
        return np.asarray(output[0] if output.ndim == 4 else output, dtype=np.float32)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_weight(tile_size: int) -> np.ndarray:
    axis = np.hanning(tile_size).astype(np.float32)
    return np.maximum(np.outer(axis, axis), 1e-3)


def _normalise_input(tile: np.ndarray, dtype: str) -> np.ndarray:
    data = tile.astype(np.float32, copy=False)
    kind = np.dtype(dtype)
    if np.issubdtype(kind, np.integer):
        data = data / float(np.iinfo(kind).max)
    elif data.size and float(np.nanmax(data)) > 1.0:
        data = data / 255.0
    return np.nan_to_num(np.clip(data, 0.0, 1.0), copy=False)


def _predict_logits(
    predictor: SemanticPredictor | Callable[[np.ndarray], np.ndarray],
    tile: np.ndarray,
    class_count: int,
) -> np.ndarray:
    function = getattr(predictor, 'predict', predictor)
    logits = np.asarray(function(tile), dtype=np.float32)
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 3 or logits.shape[0] != class_count:
        raise SemanticInferenceRefused(
            f'Predictor returned {logits.shape}; expected ({class_count}, height, width).'
        )
    if logits.shape[1:] != tile.shape[1:]:
        import cv2

        logits = np.stack(
            [
                cv2.resize(channel, tile.shape[1:][::-1], interpolation=cv2.INTER_LINEAR)
                for channel in logits
            ]
        )
    if not np.isfinite(logits).all():
        raise SemanticInferenceRefused('Predictor returned non-finite semantic logits.')
    return logits


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    return abs(
        sum(
            float(ring[i][0]) * float(ring[(i + 1) % len(ring)][1])
            - float(ring[(i + 1) % len(ring)][0]) * float(ring[i][1])
            for i in range(len(ring))
        )
    ) / 2.0


def _geometry_area(geometry: dict[str, Any]) -> float:
    coordinates = geometry.get('coordinates', [])
    if geometry.get('type') == 'Polygon':
        return max(0.0, _ring_area(coordinates[0]) - sum(_ring_area(r) for r in coordinates[1:]))
    if geometry.get('type') == 'MultiPolygon':
        return sum(
            max(0.0, _ring_area(poly[0]) - sum(_ring_area(r) for r in poly[1:]))
            for poly in coordinates
        )
    return 0.0


def _write_raster(
    path: Path,
    array: np.ndarray,
    *,
    crs: Any,
    transform: Any,
    dtype: str,
    nodata: int | float,
    tags: dict[str, str],
    categorical: bool,
) -> str:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        'driver': 'COG',
        'height': int(array.shape[0]),
        'width': int(array.shape[1]),
        'count': 1,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'nodata': nodata,
        'compress': 'DEFLATE',
    }
    try:
        with rasterio.open(path, 'w', **profile) as target:
            target.write(np.asarray(array, dtype=dtype), 1)
            target.update_tags(**tags)
    except Exception:
        path.unlink(missing_ok=True)
        profile.update({'driver': 'GTiff', 'tiled': True, 'blockxsize': 256, 'blockysize': 256})
        with rasterio.open(path, 'w', **profile) as target:
            target.write(np.asarray(array, dtype=dtype), 1)
            target.update_tags(**tags)
            factors = [factor for factor in (2, 4, 8, 16) if min(array.shape) // factor >= 1]
            if factors:
                method = rasterio.enums.Resampling.nearest if categorical else rasterio.enums.Resampling.average
                target.build_overviews(factors, method)
    return str(path)


def _run_semantic_inference_impl(
    orthomosaic_path: str | Path,
    output_dir: str | Path,
    *,
    schema: SemanticSchema,
    model: SemanticModelMetadata,
    predictor: SemanticPredictor | Callable[[np.ndarray], np.ndarray],
    config: SemanticInferenceConfig | None = None,
) -> SemanticInferencePackage:
    '''Run tiled inference and emit class/confidence rasters, polygons and provenance.'''
    geo.require('rasterio')
    import rasterio
    from rasterio.features import shapes
    from rasterio.windows import Window

    cfg = config or SemanticInferenceConfig()
    if not model.task_trained:
        raise SemanticInferenceRefused(
            'The selected checkpoint is an untrained foundation initializer, not a semantic task model.'
        )
    if not model.validation_metrics:
        raise SemanticInferenceRefused(
            'The semantic model manifest has no validation metrics.'
        )
    if (model.schema_id, model.schema_version) != (schema.id, schema.version):
        raise SemanticInferenceRefused(
            'Model and requested semantic schema versions do not match.'
        )
    predictor_digest = getattr(predictor, 'checkpoint_sha256', None)
    if predictor_digest and str(predictor_digest).casefold() != model.checkpoint_sha256.casefold():
        raise SemanticInferenceRefused(
            'Semantic model file SHA-256 does not match its manifest.'
        )

    source_path = Path(orthomosaic_path)
    if not source_path.is_file():
        raise SemanticInferenceRefused(f'Orthomosaic does not exist: {source_path}')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    temp_paths = [
        out / '.semantic_scores.dat',
        out / '.semantic_weights.dat',
        out / '.semantic_classes.dat',
        out / '.semantic_confidence.dat',
        out / '.semantic_valid.dat',
    ]

    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise SemanticInferenceRefused('Orthomosaic has no CRS; semantic GIS output would be unlocated.')
        if not source.crs.is_projected:
            raise SemanticInferenceRefused(
                'Semantic polygon areas require an orthomosaic in a projected CRS.'
            )
        if max(cfg.input_bands) > source.count:
            raise SemanticInferenceRefused(
                f'Orthomosaic has {source.count} band(s); RGB semantic inference needs {max(cfg.input_bands)}.'
            )

        height, width = source.height, source.width
        runtime_device = str(getattr(predictor, 'device', cfg.device)).casefold()
        if runtime_device.startswith('cpu'):
            if not cfg.allow_cpu:
                raise SemanticInferenceRefused(
                    'CPU semantic inference is disabled by default; use a GPU or explicitly allow CPU.'
                )
            if height * width > cfg.max_cpu_pixels:
                raise SemanticInferenceRefused(
                    f'Orthomosaic has {height * width:,} pixels, above the CPU limit of '
                    f'{cfg.max_cpu_pixels:,}.'
                )

        class_count = len(schema.classes)
        scores = np.memmap(temp_paths[0], mode='w+', dtype='float32', shape=(class_count, height, width))
        weights = np.memmap(temp_paths[1], mode='w+', dtype='float32', shape=(height, width))
        scores[:] = 0
        weights[:] = 0
        blend = _blend_weight(cfg.tile_size)
        starts_y = _tile_starts(height, cfg.tile_size, cfg.overlap)
        starts_x = _tile_starts(width, cfg.tile_size, cfg.overlap)

        for top in starts_y:
            for left in starts_x:
                actual_h = min(cfg.tile_size, height - top)
                actual_w = min(cfg.tile_size, width - left)
                window = Window(left, top, actual_w, actual_h)
                raw = source.read(cfg.input_bands, window=window)
                tile = _normalise_input(raw, source.dtypes[cfg.input_bands[0] - 1])
                valid = source.dataset_mask(window=window) > 0
                if actual_h != cfg.tile_size or actual_w != cfg.tile_size:
                    tile = np.pad(
                        tile,
                        ((0, 0), (0, cfg.tile_size - actual_h), (0, cfg.tile_size - actual_w)),
                        mode='edge',
                    )
                logits = _predict_logits(predictor, tile, class_count)[:, :actual_h, :actual_w]
                tile_weight = blend[:actual_h, :actual_w] * valid.astype(np.float32)
                scores[:, top:top + actual_h, left:left + actual_w] += logits * tile_weight
                weights[top:top + actual_h, left:left + actual_w] += tile_weight

        scores.flush()
        weights.flush()
        if not np.any(weights > 0):
            raise SemanticInferenceRefused('Orthomosaic contains no valid pixels.')

        class_map = np.memmap(temp_paths[2], mode='w+', dtype='uint16', shape=(height, width))
        confidence = np.memmap(temp_paths[3], mode='w+', dtype='float32', shape=(height, width))
        valid_map = np.memmap(temp_paths[4], mode='w+', dtype='uint8', shape=(height, width))
        class_map[:] = CLASS_NODATA
        confidence[:] = -1.0
        valid_map[:] = 0
        class_values = np.asarray([item.id for item in schema.classes], dtype=np.uint16)
        counts = {item.id: 0 for item in schema.classes}
        confidence_sums = {item.id: 0.0 for item in schema.classes}

        chunk_rows = max(1, min(1024, height))
        for row in range(0, height, chunk_rows):
            stop = min(height, row + chunk_rows)
            local_weight = np.asarray(weights[row:stop])
            local_valid = local_weight > 0
            averaged = np.asarray(scores[:, row:stop]) / np.maximum(local_weight[None], 1e-12)
            averaged -= averaged.max(axis=0, keepdims=True)
            probabilities = np.exp(averaged)
            probabilities /= probabilities.sum(axis=0, keepdims=True)
            channel = probabilities.argmax(axis=0)
            local_classes = class_values[channel]
            local_confidence = probabilities.max(axis=0)
            class_map[row:stop][local_valid] = local_classes[local_valid]
            confidence[row:stop][local_valid] = local_confidence[local_valid]
            valid_map[row:stop] = local_valid.astype(np.uint8)
            for item in schema.classes:
                selected = local_valid & (local_classes == item.id)
                counts[item.id] += int(selected.sum())
                confidence_sums[item.id] += float(local_confidence[selected].sum())

        class_map.flush()
        confidence.flush()
        valid_map.flush()
        try:
            unit_factor = float(source.crs.linear_units_factor[1])
        except Exception:
            unit_factor = 1.0
        pixel_area_m2 = abs(float(source.transform.a * source.transform.e - source.transform.b * source.transform.d)) * unit_factor ** 2
        stats = {
            item.id: {
                'class_id': item.id,
                'class_name': item.name,
                'pixel_count': counts[item.id],
                'area_m2': counts[item.id] * pixel_area_m2,
                'mean_confidence': confidence_sums[item.id] / counts[item.id] if counts[item.id] else None,
            }
            for item in schema.classes
        }

        tags = {
            'ODK_MODEL_KEY': model.key,
            'ODK_MODEL_VERSION': model.version,
            'ODK_MODEL_SHA256': model.checkpoint_sha256,
            'ODK_SCHEMA_ID': schema.id,
            'ODK_SCHEMA_VERSION': schema.version,
        }
        class_path = _write_raster(
            out / 'semantic_classes.tif', class_map, crs=source.crs,
            transform=source.transform, dtype='uint16', nodata=CLASS_NODATA,
            tags=tags, categorical=True,
        )
        confidence_path = _write_raster(
            out / 'semantic_confidence.tif', confidence, crs=source.crs,
            transform=source.transform, dtype='float32', nodata=-1.0,
            tags=tags, categorical=False,
        )

        by_id = {item.id: item for item in schema.classes}
        features: list[dict[str, Any]] = []
        for geometry, value in shapes(
            np.asarray(class_map), mask=np.asarray(valid_map, dtype=bool), transform=source.transform
        ):
            class_id = int(value)
            item = by_id.get(class_id)
            if item is None or (item.background and not cfg.polygonize_background):
                continue
            area_m2 = _geometry_area(geometry) * unit_factor ** 2
            if area_m2 < cfg.min_polygon_area_m2:
                continue
            features.append({
                'type': 'Feature',
                'geometry': geometry,
                'properties': {
                    'class_id': class_id,
                    'class_name': item.name,
                    'area_m2': area_m2,
                    'class_mean_confidence': stats[class_id]['mean_confidence'],
                    'model_key': model.key,
                    'model_version': model.version,
                    'schema_id': schema.id,
                    'schema_version': schema.version,
                },
            })

        epsg = source.crs.to_epsg()
        if epsg is None:
            raise SemanticInferenceRefused('Orthomosaic CRS has no EPSG identifier for GeoJSON export.')
        polygons_path = geo.write_geojson(
            out / 'semantic_polygons.geojson',
            features,
            epsg=epsg,
            properties={
                'model': model.to_dict(),
                'schema': schema.to_dict(),
            },
        )
        manifest_payload = {
            'status': 'completed',
            'source': {
                'orthomosaic': str(source_path),
                'crs': source.crs.to_string(),
                'epsg': epsg,
                'width': width,
                'height': height,
                'transform': list(source.transform)[:6],
            },
            'model': model.to_dict(),
            'schema': schema.to_dict(),
            'inference': asdict(cfg),
            'classes': [stats[item.id] for item in schema.classes],
            'polygon_count': len(features),
            'artifacts': {
                'class_map': class_path,
                'confidence': confidence_path,
                'polygons': polygons_path,
            },
        }

    manifest_path = out / 'semantic_manifest.json'
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding='utf-8')
    return SemanticInferencePackage(
        class_map_path=class_path,
        confidence_path=confidence_path,
        polygons_path=polygons_path,
        manifest_path=str(manifest_path),
    )


def run_semantic_inference(
    orthomosaic_path: str | Path,
    output_dir: str | Path,
    *,
    schema: SemanticSchema,
    model: SemanticModelMetadata,
    predictor: SemanticPredictor | Callable[[np.ndarray], np.ndarray],
    config: SemanticInferenceConfig | None = None,
) -> SemanticInferencePackage:
    '''Run semantic inference and always release temporary disk-backed score maps.'''
    try:
        return _run_semantic_inference_impl(
            orthomosaic_path,
            output_dir,
            schema=schema,
            model=model,
            predictor=predictor,
            config=config,
        )
    finally:
        # CPython releases the implementation frame before this block. Collecting is
        # still needed on Windows, where an open memmap prevents unlinking the file.
        gc.collect()
        output = Path(output_dir)
        for path in output.glob('.semantic_*.dat') if output.exists() else ():
            try:
                path.unlink()
            except OSError:
                pass
