import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sideseeing_tools.dataviz import api, create_app
from sideseeing_tools.vision.results import save_segmentations

FIXTURE = Path(__file__).resolve().parent.parent / 'fixtures' / 'dataset' / 'instance-001'
FRAME = 'frame_00000.jpg'  # both routes use this name


def save_manifest(ws, route, prompt):
    image_path = ws / 'frames' / f'{route}-frames' / FRAME
    masks = np.zeros((1, 40, 60), dtype=bool)
    masks[0, 10:20, 10:30] = True
    record = dict(image_path=str(image_path), prompt=prompt, masks=masks,
                  scores=np.array([0.9]), boxes=np.array([[10.0, 10.0, 29.0, 19.0]]), parameters={})
    save_segmentations([record], ws / 'preds' / 'sam3' / prompt / route / 'chunk-0000', source_id=route, timestamps_s=[0.0])


@pytest.fixture
def ws(tmp_path):
    for route in ('route-a', 'route-b'):
        frames = tmp_path / 'frames' / f'{route}-frames'
        frames.mkdir(parents=True)
        cv2.imwrite(str(frames / FRAME), np.zeros((40, 60, 3), np.uint8))

    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'route-a').symlink_to(FIXTURE)
    save_manifest(tmp_path, 'route-a', 'curb ramp')

    legacy = tmp_path / 'preds' / 'sam3-pole'
    (legacy / 'route-b-frames').mkdir(parents=True)
    pd.DataFrame([dict(image_name=FRAME, relative_path=f'route-b-frames/{FRAME}', prompt='pole',
                       num_detections=1, scores='0.7000', boxes='1,2,3,4')]).to_csv(legacy / 'detections.csv', index=False)
    mask = np.zeros((40, 60), np.uint8)
    mask[:5, :5] = 255
    cv2.imwrite(str(legacy / 'route-b-frames' / 'frame_00000-mask.png'), mask)
    return tmp_path


@pytest.fixture
def client(ws):
    return TestClient(create_app(ws))


def test_repeated_frame_names_stay_in_their_route(client):
    route_a = client.get('/api/dataviz/data/route-a').json()
    route_b = client.get('/api/dataviz/data/route-b').json()

    assert route_a['mask_classes'] == ['curb ramp'] and route_b['mask_classes'] == ['pole']
    assert route_a['bboxes'][FRAME][0] | {'color': None} == dict(
        class_name='curb ramp', confidence=pytest.approx(0.9), color=None, xmin=10.0, ymin=10.0, xmax=29.0, ymax=19.0)
    assert route_b['bboxes'][FRAME][0]['confidence'] == pytest.approx(0.7)


def test_new_results_appear_without_restart(client, ws):
    assert client.get('/api/dataviz/data/route-b').json()['mask_classes'] == ['pole']

    save_manifest(ws, 'route-b', 'tree')

    assert client.get('/api/dataviz/data/route-b').json()['mask_classes'] == ['pole', 'tree']


def test_mask_overlay_colors_requested_classes(client):
    response = client.get(f'/api/dataviz/mask/{FRAME}', params=dict(classes='curb ramp', instance='route-a'))
    alpha = np.asarray(Image.open(io.BytesIO(response.content)))[..., 3]

    assert alpha.shape == (40, 60)
    assert np.count_nonzero(alpha) == 200 and alpha[15, 20] == 255


@pytest.mark.parametrize('url', [
    '/api/dataviz/image/route-a/..%2F..%2Fdata%2Froute-a%2Fgps.csv',
    '/api/dataviz/image/..%2Fdata/' + FRAME,
    '/api/dataviz/image/unknown/' + FRAME,
    '/api/dataviz/thumb/route-a/frames.json',
    f'/api/dataviz/mask/{FRAME}?classes=pole&instance=..',
    '/report/..%2Fdata%2Froute-a%2Fgps.csv',
    '/static/..%2F..%2Fworkspace.py',
])
def test_requests_cannot_leave_their_folders(client, url):
    assert client.get(url).status_code == 404


def test_viewer_lists_routes_and_events(client, ws):
    (ws / 'geomatching').mkdir()
    pd.DataFrame([dict(event_id='EVT-00001', start_image=FRAME, end_image=FRAME, center_latitude=-23.5,
                       center_longitude=-46.7, feature='pole', length_meters=3.0,
                       instance_name='data_route-b')]).to_csv(ws / 'geomatching' / 'map_events.csv', index=False)

    page = client.get('/').text
    assessment = client.get('/api/viewer/assessment/route-b').json()

    assert '"name": "route-a"' in page and '"name": "route-b"' in page and 'href="report/"' not in page
    assert assessment['points'][0]['frames'] == [f'/api/dataviz/image/route-b/{FRAME}']
    assert client.get('/api/viewer/assessment/route-a').status_code == 404


def test_extraction_runs_in_the_background(client, ws, monkeypatch):
    monkeypatch.setattr(api, 'EXTRACT_STEP', 1200)
    (ws / 'frames' / 'route-a-frames' / FRAME).unlink()

    assert client.post('/api/dataviz/extract/route-a').json()['status'] == 'processing'
    assert client.get('/api/dataviz/data/route-a').json()['frames'] == [
        'route-a_00000.jpg', 'route-a_01200.jpg', 'route-a_02400.jpg']
    assert client.post('/api/dataviz/extract/route-b').status_code == 404
