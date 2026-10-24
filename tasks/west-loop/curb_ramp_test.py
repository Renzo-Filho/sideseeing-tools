"""Compare the curb ramp phrasings on D1 S1 (tasks/west-loop-run.md, Phase 2).

    python curb_ramp_test.py WORKSPACE_COPY NETWORK_REPO

WORKSPACE_COPY holds preds/sam3/<prompt>/<route>/, transcripts/ and data/<route>/{gps.csv,metadata.json}
(a copy of the desktop-01 workspace); NETWORK_REPO is sideseeing-network, for the OSM snapshot and its
helpers. Prints one row per phrasing:

- mentions_with_detection: share of spoken curb ramp mentions with a detection from 10 s before to 5 s
  after the mention. Narration is not exhaustive, so this is a recall proxy, not recall.
- negatives_with_detection: share of frames at least 50 m from any mapped crosswalk with a detection;
  curb ramps sit at crossings, so these are likely false positives.
"""
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROUTE = '2026-10-19-15-43-14-478'
PHRASINGS = ['curb ramp', 'curb cut', 'wheelchair ramp', 'ramp', 'curbramp']
MENTION = r'\bcur\w*\s*(?:ramp|wrem|rem|rim)\w*'
WINDOW = (-10, 5)
NEGATIVE_M = 50
UTM = 'EPSG:32616'


def detections(ws, prompt):
    """[(t, best score or nan)] for every frame of the route, from the prompt's manifests."""
    rows = []
    for manifest in sorted((ws / 'preds' / 'sam3' / prompt.replace(' ', '-') / ROUTE).glob('chunk-*/manifest.json')):
        for frame in json.loads(manifest.read_text())['frames']:
            best = np.nan
            if frame['detections']:
                with np.load(manifest.parent / frame['instances_file']) as arrays:
                    best = float(arrays['scores'].max())
            rows.append((frame['timestamp_s'], best))
    return pd.DataFrame(rows, columns=['t', 'score'])


def main(ws, network):
    sys.path.insert(0, str(network / 'network'))
    from routes import ways_frame

    meta = json.loads((ws / 'data' / ROUTE / 'metadata.json').read_text())['time']
    start = pd.Timestamp(meta['mediaStartDateTime'])
    gps = pd.read_csv(ws / 'data' / ROUTE / 'gps.csv', parse_dates=['datetime_utc']).sort_values('datetime_utc')
    gps['t'] = (gps.datetime_utc - start).dt.total_seconds()
    xy = gpd.GeoSeries(gpd.points_from_xy(gps.longitude, gps.latitude), crs=4326).to_crs(UTM)
    osm = json.loads(sorted((network / 'data' / 'west_loop').glob('osm_*.json'))[-1].read_text())
    ways = ways_frame(osm['elements'])
    crosswalks = ways[ways.footway.eq('crossing') | ways.highway.eq('crossing')].geometry.union_all()

    segments = pd.DataFrame(json.loads((ws / 'transcripts' / 'whisper-turbo' / f'{ROUTE}.json').read_text())['segments'])
    mentions = segments[segments.text.str.lower().str.contains(MENTION)].start.values

    rows = []
    for prompt in PHRASINGS:
        d = detections(ws, prompt)
        if d.empty:
            rows.append(dict(phrasing=prompt, frames=0))
            continue
        hit = d.score.notna()
        points = gpd.GeoSeries(gpd.points_from_xy(np.interp(d.t, gps.t, xy.x), np.interp(d.t, gps.t, xy.y)), crs=UTM)
        negative = points.distance(crosswalks).values >= NEGATIVE_M
        found = [hit[(d.t >= t + WINDOW[0]) & (d.t <= t + WINDOW[1])].any() for t in mentions]
        rows.append(dict(
            phrasing=prompt, frames=len(d), frames_with_detection=int(hit.sum()),
            mentions_with_detection=np.mean(found), negatives_with_detection=hit[negative].mean(),
            negative_frames=int(negative.sum()), median_best_score=d.score.median(),
        ))
    table = pd.DataFrame(rows).set_index('phrasing')
    print(f'{len(mentions)} spoken mentions; window {WINDOW} s; negatives ≥ {NEGATIVE_M} m from mapped crosswalks')
    print(table.round(3).to_string())
    return table


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
