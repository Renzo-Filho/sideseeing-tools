import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sideseeing_tools import report_geo, report_sensors
from sideseeing_tools.export import Report

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


def test_geo_export_adds_the_map_matched_route_when_given(tmp_path):
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import LineString

    gps = pd.DataFrame({'latitude': [41.87, 41.88], 'longitude': [-87.65, -87.64]})
    dataset = SimpleNamespace(iterator=[
        SimpleNamespace(name=name, geolocation_points=gps, geolocation_center=[41.875, -87.645]) for name in ('a', 'b')])
    routes = tmp_path / 'routes'
    routes.mkdir()
    gpd.GeoDataFrame(geometry=[LineString([(-87.651, 41.871), (-87.641, 41.881)])], crs='EPSG:4326').to_file(
        routes / 'a.gpkg', driver='GPKG')

    report_geo.process_geo_data(dataset, str(tmp_path / 'plain'))
    report_geo.process_geo_data(dataset, str(tmp_path / 'matched'), str(routes))
    read = lambda folder, name: json.loads((tmp_path / folder / f'geo_{name}.json').read_text())

    assert 'corrected_path' not in read('plain', 'a')
    assert read('matched', 'a')['corrected_path'] == [[41.871, -87.651], [41.881, -87.641]]
    assert read('matched', 'b')['corrected_path'] == []
    assert read('matched', 'a')['path'] == read('plain', 'a')['path']


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
