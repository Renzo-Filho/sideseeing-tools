import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sideseeing_tools import report_geo, report_sensors
from sideseeing_tools.export import Report
from sideseeing_tools.geolocation import position_at, smooth_track

FIXTURE_DATASET = Path(__file__).parent.parent / 'fixtures' / 'dataset'


def test_sensor_export_skips_absent_groups_and_preserves_values(tmp_path):
    frame = pd.DataFrame({'Time (s)': [0, 1], 'x': [10, 20]})
    instance = SimpleNamespace(name='route', sensors1={'light': frame}, sensors3=None, sensors6=None)
    dataset = SimpleNamespace(iterator=[instance])

    paths = report_sensors.process_sensors_data(dataset, str(tmp_path))
    data = pd.read_json(tmp_path / 'sensors_route.json').iloc[0]['data']

    assert paths == {'route': 'data/sensors_route.json'}
    assert data[0]['x'] == [0, 1]
    assert data[0]['y'] == [10, 20]


def test_smoothing_removes_single_fix_jumps_and_interpolates_in_time():
    times = pd.date_range('2026-10-19 20:00', periods=6, freq='s')
    track = smooth_track(pd.DataFrame({'Datetime UTC': times, 'latitude': [0, 0, 9, 0, 0, 0.0],
                                       'longitude': [0, 1, 2, 3, 4, 5.0]}))
    latitude, longitude = position_at(track, [times[2] + pd.Timedelta(seconds=0.5), times[0] - pd.Timedelta(seconds=10)])

    assert track.latitude.tolist() == [0.0] * 6
    assert latitude[0] == 0.0 and longitude[0] == pytest.approx(2.5)
    assert np.isnan(latitude[1]) and np.isnan(longitude[1])


def test_geo_export_adds_the_smoothed_track(tmp_path):
    gps = pd.DataFrame({'latitude': [41.87, 41.87, 41.99, 41.87, 41.87], 'longitude': [-87.65] * 5})
    dataset = SimpleNamespace(iterator=[SimpleNamespace(name='a', geolocation_points=gps, geolocation_center=[41.87, -87.65])])

    report_geo.process_geo_data(dataset, str(tmp_path))
    data = json.loads((tmp_path / 'geo_a.json').read_text())

    assert data['path'][2] == [41.99, -87.65]
    assert data['smoothed_path'] == [[41.87, -87.65]] * 5


@pytest.mark.parametrize('generate_metadata', [False, True])
def test_report_only_writes_source_metadata_when_requested(tmp_path, generate_metadata):
    source = tmp_path / 'dataset'
    shutil.copytree(FIXTURE_DATASET, source)
    metadata_path = source / 'metadata.csv'
    metadata_path.unlink()
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    output = tmp_path / 'report'

    Report().generate_report(source, output, generate_metadata=generate_metadata)

    assert metadata_path.exists() == generate_metadata
    after = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    if generate_metadata:
        after.pop(Path('metadata.csv'))
        assert 'video_fps' in pd.read_csv(metadata_path)
    assert after == before
    assert (output / 'index.html').exists()
    assert (output / 'static' / 'js' / 'custom.js').exists()
