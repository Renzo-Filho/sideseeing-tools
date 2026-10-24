import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from sideseeing_tools import workspace as workspace_module
from sideseeing_tools.sideseeing import SideSeeingDS
from sideseeing_tools.workspace import SideSeeingWorkspace, frame_timestamp, list_frames

FIXTURE = Path(__file__).resolve().parent.parent / 'fixtures' / 'dataset'
STEP = 600  # five frames from the 84 s fixture video


class FakeSegmenter:
    """Stands in for Sam3Segmenter: one 10x20 mask in the top-left corner of every frame."""

    provenance = dict(model_id='fake')

    def __init__(self):
        self.calls = []

    def segment(self, image_paths, prompt, batch_size=1):
        self.calls.append((prompt, len(image_paths)))
        records = []

        for path in map(Path, image_paths):
            height, width = cv2.imread(str(path)).shape[:2]
            masks = np.zeros((1, height, width), dtype=bool)
            masks[0, :10, :20] = True
            records.append(dict(
                image_path=str(path.resolve()), prompt=prompt, masks=masks,
                scores=np.array([0.9]), boxes=np.array([[0.0, 0.0, 19.0, 9.0]]), parameters={},
            ))

        return records


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    raw = tmp_path_factory.mktemp('raw')
    for route in ('route-a', 'route-b'):
        shutil.copytree(FIXTURE / 'instance-001', raw / route)
    shutil.copy(FIXTURE / 'metadata.csv', raw / 'metadata.csv')
    return SideSeeingDS(str(raw))


@pytest.fixture(scope='module')
def extracted(dataset, tmp_path_factory):
    ws = SideSeeingWorkspace(tmp_path_factory.mktemp('ws'))
    ws.setup_data_directory(dataset)
    ws.extract_frames(dataset, STEP)
    return ws


def file_hashes(paths):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_frame_timestamp():
    assert frame_timestamp('route_00030.jpg', 30) == 1.0
    assert frame_timestamp('2025-11-09-07-23-58-838_00005_ms.jpg', 30) == 5.0
    assert frame_timestamp('cover.jpg', 30) is None


def test_data_directory_loads_as_dataset(extracted):
    linked = SideSeeingDS(str(extracted.data_dir))

    assert sorted(linked.instances) == ['route-a', 'route-b']
    assert (extracted.data_dir / 'metadata.csv').is_symlink()


def test_extraction_records_step_and_refuses_another(dataset, extracted):
    frames_dir = extracted.route_frames_dir('route-a')
    record = json.loads((frames_dir / 'frames.json').read_text())

    assert [path.name for path in list_frames(frames_dir)] == [f'route-a_{i:05d}.jpg' for i in range(0, 2518, STEP)]
    assert record['step'] == STEP and record['fps'] == pytest.approx(30.0)
    assert not workspace_module.extract_route_frames(dataset.instances['route-a'].video, frames_dir, 'route-a_', STEP)

    with pytest.raises(ValueError, match='extracted with step 600'):
        extracted.extract_frames(dataset, 300)


def test_interrupted_extraction_publishes_nothing(dataset, tmp_path, monkeypatch):
    def fail_midway(source, target, step, prefix):
        (Path(target) / f'{prefix}00000.jpg').write_bytes(b'partial')
        raise RuntimeError('interrupted')

    monkeypatch.setattr(workspace_module.media, 'extract_frames', fail_midway)
    ws = SideSeeingWorkspace(tmp_path)

    with pytest.raises(RuntimeError):
        ws.extract_frames(dataset, STEP)

    assert list(ws.frames_dir.iterdir()) == []


def test_segmentation_resumes_missing_chunks_and_feeds_events(dataset, extracted):
    segmenter = FakeSegmenter()
    extracted.generate_segmentation(dataset, ['curb ramp'], segmenter=segmenter, chunk_size=2)
    route_dir = extracted.preds_dir / 'sam3' / 'curb-ramp' / 'route-a'
    manifest = json.loads((route_dir / 'chunk-0001' / 'manifest.json').read_text())

    assert sorted(path.name for path in route_dir.iterdir()) == ['chunk-0000', 'chunk-0001', 'chunk-0002']
    assert manifest['source_id'] == 'route-a'
    assert [frame['timestamp_s'] for frame in manifest['frames']] == pytest.approx([40.0, 60.0])

    shutil.rmtree(route_dir / 'chunk-0001')
    segmenter.calls.clear()
    extracted.generate_segmentation(dataset, ['curb ramp'], segmenter=segmenter, chunk_size=2)

    assert segmenter.calls == [('curb ramp', 2)]

    extracted.generate_sidewalk_assessment_events(dataset)
    events = pd.read_csv(extracted.geomatching_dir / 'map_events.csv')

    assert events[['instance_name', 'feature', 'start_image', 'end_image']].values.tolist() == [
        ['route-a', 'curb ramp', 'route-a_00000.jpg', 'route-a_02400.jpg'],
        ['route-b', 'curb ramp', 'route-b_00000.jpg', 'route-b_02400.jpg'],
    ]


def test_load_detections_prefers_manifests_over_legacy_csv(dataset, extracted):
    extracted.generate_segmentation(dataset, ['pole'], segmenter=FakeSegmenter(), chunk_size=5)
    legacy_dir = extracted.preds_dir / 'sam3-pole'
    legacy_dir.mkdir()
    pd.DataFrame([
        dict(image_name='route-a_00000.jpg', relative_path='route-a-frames/route-a_00000.jpg', prompt='pole', num_detections=0),
        dict(image_name='old_00001.jpg', relative_path='data_route-c/old_00001.jpg', prompt='pole', num_detections=2),
    ]).to_csv(legacy_dir / 'detections.csv', index=False)

    detections = extracted.load_detections()
    poles = detections[detections['prompt'] == 'pole'].set_index(['route', 'image_name'])

    assert poles.loc[('route-a', 'route-a_00000.jpg'), 'num_detections'] == 1
    assert poles.loc[('route-c', 'old_00001.jpg'), 'num_detections'] == 2


def test_anonymization_writes_copies_and_keeps_sources(dataset, extracted):
    sources = list_frames(extracted.route_frames_dir('route-b'))
    before = file_hashes(sources)

    extracted.anonymize_frames(dataset, instances=[dataset.instances['route-b']], segmenter=FakeSegmenter())
    redacted_dir = extracted.redacted_dir / 'route-b-frames'
    record = json.loads((redacted_dir / 'redaction.json').read_text())

    assert file_hashes(sources) == before
    assert sorted(path.name for path in list_frames(redacted_dir)) == sorted(before)
    assert {frame['masked_pixels'] for frame in record['frames']} == {200}
    assert record['prompts'] == ['face', 'license plate'] and record['model'] == {'model_id': 'fake'}


class FakeWhisper:
    """Stands in for a Whisper model; raises `error` instead when given one."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def transcribe(self, path, language=None, **options):
        self.calls.append(Path(path).parent.name)
        if self.error:
            raise self.error
        segment = dict(start=1.0, end=2.5, text=' Curb ramp here.', avg_logprob=-0.2, no_speech_prob=0.1)
        return dict(language='en', text=segment['text'], segments=[segment])


def test_transcription_writes_timed_segments_once(dataset, tmp_path):
    ws = SideSeeingWorkspace(tmp_path)
    whisper = FakeWhisper()
    ws.transcribe(dataset, model_name='tiny', model=whisper)
    record = json.loads((ws.transcripts_dir / 'whisper-tiny' / 'route-a.json').read_text())

    assert whisper.calls == ['route-a', 'route-b']
    assert record['media_start_time'] == '2024-01-06T14:59:48.138000'
    assert record['segments'] == [dict(start=1.0, end=2.5, text='Curb ramp here.', avg_logprob=-0.2, no_speech_prob=0.1)]
    assert record['language'] == 'en' and record['parameters'] == dict(condition_on_previous_text=False, language=None)

    whisper.calls.clear()
    ws.transcribe(dataset, model_name='tiny', model=whisper)

    assert whisper.calls == []


def test_transcription_skips_missing_audio_but_not_other_errors(dataset, tmp_path):
    ws = SideSeeingWorkspace(tmp_path)
    ws.transcribe(dataset, model_name='tiny', model=FakeWhisper(RuntimeError('Failed to load audio: no stream')))

    assert list((ws.transcripts_dir / 'whisper-tiny').iterdir()) == []

    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        ws.transcribe(dataset, model_name='tiny', model=FakeWhisper(RuntimeError('CUDA out of memory')))


def test_build_workspace_runs_each_stage_once(dataset, tmp_path, monkeypatch):
    ws = SideSeeingWorkspace(tmp_path)
    calls = []
    stages = ['setup_data_directory', 'extract_frames', 'anonymize_frames', 'generate_segmentation',
              'generate_map_matched_routes', 'generate_sidewalk_assessment_events', 'transcribe']
    for stage in stages:
        monkeypatch.setattr(ws, stage, lambda *args, stage=stage, **kwargs: calls.append(stage))

    ws.build_workspace(dataset, ['sidewalk'], anonymize=True, segmenter=FakeSegmenter(), whisper_model='tiny')

    assert calls == stages
