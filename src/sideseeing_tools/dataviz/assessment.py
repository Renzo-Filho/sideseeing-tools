"""Sidewalk assessment events for the viewer map, from a workspace's geomatching/ folder."""
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd

FRAME_NAME = re.compile(r'^(.*)_(\d+)(?:_ms)?\.(?:jpg|png)$')


def load_events(workspace_dir):
    path = Path(workspace_dir) / 'geomatching' / 'map_events.csv'

    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=['route'])

    events = pd.read_csv(path)
    # Older workspaces named routes "data_<route>"
    events['route'] = events['instance_name'].astype(str).str.replace(r'^data_', '', regex=True)
    return events


def routes_with_events(workspace_dir):
    return sorted(load_events(workspace_dir)['route'].unique())


def _event_frames(frame_names, start_image, end_image):
    start, end = FRAME_NAME.match(str(start_image)), FRAME_NAME.match(str(end_image))

    if not start or not end:
        return []

    first, last = int(start[2]), int(end[2])
    matches = (FRAME_NAME.match(name) for name in frame_names)
    return [m[0] for m in matches if m and m[1] == start[1] and first <= int(m[2]) <= last]


def _route_path(workspace_dir, route):
    gpkg = Path(workspace_dir) / 'geomatching' / 'routes_gpkg' / f'{route}.gpkg'

    if not gpkg.is_file():
        return []

    import geopandas as gpd

    path = []
    for geometry in gpd.read_file(gpkg).geometry.dropna():
        for line in getattr(geometry, 'geoms', [geometry]):
            path.extend([y, x] for x, y in line.coords)
    return path


def route_assessment(workspace_dir, route, frame_names):
    """Map center, event markers with their frame URLs, and the map-matched path of one route."""
    events = load_events(workspace_dir)
    events = events[events['route'] == route].drop_duplicates('event_id')

    if events.empty:
        return None

    points = [
        dict(
            latitude=float(event.center_latitude),
            longitude=float(event.center_longitude),
            type=str(event.feature),
            event_id=str(event.event_id),
            length_meters=float(event.length_meters),
            frames=[
                f'/api/dataviz/image/{quote(route)}/{quote(name)}'
                for name in _event_frames(frame_names, event.start_image, event.end_image)
            ],
        )
        for event in events.itertuples()
    ]

    return dict(
        center=[points[0]['latitude'], points[0]['longitude']],
        points=points,
        path=_route_path(workspace_dir, route),
    )
