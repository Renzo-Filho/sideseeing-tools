"""Route-scoped predictions for the viewer.

Reads SAM3 manifests written by SideSeeingWorkspace (preds/sam3/<prompt>/<route>/<chunk>/)
and older outputs: preds/<model>/detections.csv with mask PNGs, and Project Sidewalk style
preds/predictions.csv. Parsed files are cached by modification time, so new results appear
without restarting the server.
"""
import colorsys
import glob
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


# One hue per group (orange, blue, green, purple, red, teal, yellow, pink); prompts in a group are its shades.
GROUP_HUES = (0.08, 0.6, 0.33, 0.78, 0.0, 0.5, 0.15, 0.9)
FILL_ALPHA = 110  # surfaces are drawn faintly so the objects on them stay visible
OUTLINE_WIDTH = 3


def prompt_groups(workspace_dir, classes):
    """[{name, fill, classes}] from the workspace's optional prompt_groups.json, keeping only classes present.

    The file holds {"groups": [{"name": ..., "fill": true|false, "prompts": [...]}]}; classes in no group
    go to a last group, "Other" (or "All" when there is no file).
    """
    path = Path(workspace_dir) / 'prompt_groups.json'
    configured = json.loads(path.read_text())['groups'] if path.is_file() else []
    present = set(classes)
    groups = [
        dict(name=str(group['name']), fill=bool(group.get('fill')), classes=[c for c in group['prompts'] if c in present])
        for group in configured
    ]
    groups = [group for group in groups if group['classes']]
    grouped = {name for group in groups for name in group['classes']}
    rest = [name for name in classes if name not in grouped]

    if rest:
        groups.append(dict(name='Other' if groups else 'All', fill=False, classes=rest))
    return groups


def group_colors(groups):
    """{class: RGBA}: one hue per group, from dark to light shades for the classes in it."""
    colors = {}

    for index, group in enumerate(groups):
        hue = GROUP_HUES[index % len(GROUP_HUES)]
        count = len(group['classes'])

        for position, name in enumerate(group['classes']):
            lightness = 0.32 + 0.38 * position / max(count - 1, 1)
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.85)
            colors[name] = (int(r * 255), int(g * 255), int(b * 255), 255)
    return colors


def _outline(mask, width=OUTLINE_WIDTH):
    """The inner border of a boolean mask, `width` pixels thick."""
    inner = mask

    for _ in range(width):
        padded = np.pad(inner, 1)
        inner = inner & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~inner


@lru_cache(maxsize=256)
def _overlay_png(sources, min_score):
    overlay = None

    for kind, path, _, color, fill in sources:
        if kind == 'npz':
            with np.load(path, allow_pickle=False) as arrays:
                masks = arrays['masks'][arrays['scores'] >= min_score]
                shape = arrays['masks'].shape[1:]
        else:
            with Image.open(path) as image:
                masks = (np.asarray(image.convert('L')) > 0)[None]
            shape = masks.shape[1:]

        if overlay is None:
            overlay = np.zeros((*shape, 4), dtype=np.uint8)
        if fill:
            overlay[masks.any(axis=0)] = (*color[:3], FILL_ALPHA)
        else:
            for mask in masks:  # one border per instance, so touching objects stay apart
                overlay[_outline(mask)] = color

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, 'PNG')
    return buffer.getvalue()


def mask_overlay(detections, classes, colors, filled=(), min_score=0.0):
    """PNG bytes with the masks of the requested classes, or None without masks.

    Classes in `filled` are faint fills drawn first; the others are outlines on top. Instances scoring
    below `min_score` are left out (older PNG masks have no per-instance scores and are always drawn).
    """
    sources = sorted(
        (
            (kind, path, _mtime(path), colors[detection['class_name']], detection['class_name'] in filled)
            for detection in detections
            if detection['mask'] and detection['class_name'] in classes
            for kind, path in [detection['mask']]
        ),
        key=lambda source: not source[4],
    )
    return _overlay_png(tuple(sources), float(min_score)) if sources else None
