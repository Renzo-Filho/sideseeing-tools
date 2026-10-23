"""Route-scoped predictions for the viewer.

Reads SAM3 manifests written by SideSeeingWorkspace (preds/sam3/<prompt>/<route>/<chunk>/)
and older outputs: preds/<model>/detections.csv with mask PNGs, and Project Sidewalk style
preds/predictions.csv. Parsed files are cached by modification time, so new results appear
without restarting the server.
"""
import colorsys
import glob
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MASK_NAMES = ('{stem}-mask.png', '{stem}_mask.png', '{stem}-mask.jpg')
WIDE_FILES = ('predictions.csv', 'predictions.general.csv')


def _mtime(path):
    return Path(path).stat().st_mtime_ns


def _detection(class_name, scores=(), boxes=(), mask=None):
    return dict(class_name=str(class_name), scores=list(scores), boxes=list(boxes), mask=mask)


@lru_cache(maxsize=4096)
def _manifest(path, mtime):
    """{frame name: detection} from one SAM3 manifest; masks stay on disk."""
    manifest_path = Path(path)
    found = {}

    for frame in json.loads(manifest_path.read_text())['frames']:
        name = frame['instances_file']

        if frame['status'] != 'processed' or Path(name).name != name:
            continue

        instances = manifest_path.parent / name
        with np.load(instances, allow_pickle=False) as arrays:
            scores, boxes = arrays['scores'].tolist(), arrays['boxes'].tolist()

        mask = ('npz', str(instances)) if scores else None
        found[Path(frame['image_path']).name] = _detection(frame['prompt'], scores, boxes, mask)

    return found


@lru_cache(maxsize=256)
def _legacy_csv(path, mtime):
    """Rows of an older detections.csv with detections, with the route parsed from relative_path."""
    rows = pd.read_csv(path).rename(columns={'prompt': 'class_name'})
    rows = rows[rows['num_detections'] > 0].copy()

    if 'relative_path' in rows.columns:
        rows['folder'] = rows['relative_path'].astype(str).str.split('/').str[0]
        rows['route'] = rows['folder'].str.replace(r'-frames$', '', regex=True).str.replace(r'^data_', '', regex=True)
    else:
        # ponytail: without relative_path, rows match any route with that frame name.
        rows['folder'] = ''
        rows['route'] = None

    return rows


@lru_cache(maxsize=16)
def _wide_csv(path, mtime):
    """(frame name, class) pairs from a Project Sidewalk style CSV with one 0/1 column per class."""
    rows = pd.read_csv(path)

    if 'image' not in rows.columns:
        return ()

    melted = rows.melt(id_vars='image', var_name='class_name', value_name='presence')
    return tuple(
        (Path(str(image)).name, class_name)
        for image, class_name in melted.loc[melted['presence'] > 0, ['image', 'class_name']].itertuples(index=False)
    )


def _numbers(text):
    return [float(value) for value in str(text).split(';') if value] if pd.notna(text) else []


def _boxes(text):
    return [[float(value) for value in box.split(',')] for box in str(text).split(';') if box] if pd.notna(text) else []


def _legacy_mask(csv_dir, row):
    stem = Path(row.image_name).stem

    for folder in (row.folder, f'{row.route}-frames', ''):
        for pattern in MASK_NAMES:
            path = csv_dir / folder / pattern.format(stem=stem)
            if path.is_file():
                return ('png', str(path))

    return None


def route_predictions(preds_dir, route, frame_names):
    """{frame name: [detection]} for the given frames of one route.

    A detection is a dict with class_name, scores, boxes ([x_min, y_min, x_max, y_max] in source
    pixels) and mask, a ('npz' | 'png', path) pair or None. Treat the result as read-only.
    """
    preds_dir = Path(preds_dir)
    sources = (
        sorted(preds_dir.glob(f'sam3/*/{glob.escape(route)}/*/manifest.json'))
        + sorted(preds_dir.glob('*/detections.csv'))
        + [path for path in (preds_dir / name for name in WIDE_FILES) if path.is_file()]
    )
    return _route_predictions(route, tuple(frame_names), tuple((str(path), _mtime(path)) for path in sources))


@lru_cache(maxsize=64)
def _route_predictions(route, frame_names, sources):
    found = {name: [] for name in frame_names}

    for path, mtime in sources:
        path = Path(path)

        if path.name == 'manifest.json':
            for name, detection in _manifest(str(path), mtime).items():
                if name in found:
                    found[name].append(detection)

        elif path.name == 'detections.csv':
            rows = _legacy_csv(str(path), mtime)
            rows = rows[((rows['route'] == route) | rows['route'].isna()) & rows['image_name'].isin(found.keys())]

            for row in rows.itertuples():
                scores, boxes = _numbers(getattr(row, 'scores', None)), _boxes(getattr(row, 'boxes', None))
                found[row.image_name].append(_detection(row.class_name, scores, boxes, _legacy_mask(path.parent, row)))

        else:
            for name, class_name in _wide_csv(str(path), mtime):
                if name in found:
                    found[name].append(_detection(class_name))

    return found


def class_color(class_name, all_classes):
    """A consistent RGBA color for a class, spreading hues across the classes in all_classes."""
    if class_name.lower() == 'sidewalk':
        return (255, 165, 0, 255)
    if class_name.lower() == 'crosswalk':
        return (0, 255, 0, 255)

    dynamic_classes = [c for c in all_classes if c.lower() not in ('sidewalk', 'crosswalk')]

    if class_name in dynamic_classes:
        index = dynamic_classes.index(class_name)
        # Golden ratio hue spacing, alternating saturation and lightness
        hue = (index * 0.618033988749895) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.45 + (index % 3) * 0.1, 0.8 + (index % 2) * 0.15)
    else:
        value = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
        r, g, b = colorsys.hls_to_rgb((value % 360) / 360.0, 0.45 + ((value >> 8) % 20) / 100.0, 0.7 + (value % 30) / 100.0)

    return (int(r * 255), int(g * 255), int(b * 255), 255)


@lru_cache(maxsize=256)
def _overlay_png(sources):
    overlay = None

    for kind, path, _, color in sources:
        if kind == 'npz':
            with np.load(path, allow_pickle=False) as arrays:
                mask = arrays['masks'].any(axis=0)
        else:
            with Image.open(path) as image:
                mask = np.asarray(image.convert('L')) > 0

        if overlay is None:
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask] = color

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, 'PNG')
    return buffer.getvalue()


def mask_overlay(detections, classes, all_classes):
    """PNG bytes with the masks of the requested classes in their colors, or None without masks."""
    sources = tuple(
        (kind, path, _mtime(path), class_color(detection['class_name'], all_classes))
        for detection in detections
        if detection['mask'] and detection['class_name'] in classes
        for kind, path in [detection['mask']]
    )
    return _overlay_png(sources) if sources else None
