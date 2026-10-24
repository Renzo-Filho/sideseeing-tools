"""One stage of the West Loop run (tasks/west-loop-run.md): python westloop_step.py STAGE

Each stage is a separate process so GPU memory is released between stages. Work already in the
workspace is skipped by the library, so a stage can be rerun after an interruption. A failure in
one session is logged and the stage continues with the next one; the exit code is 1 if any failed.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace
from sideseeing_tools.workspace import release_gpu_memory

RAW = Path(os.environ.get('WESTLOOP_RAW', Path.home() / 'sideseeing/raw/chicago/west-loop'))
WORKSPACE = Path(os.environ.get('WESTLOOP_WORKSPACE', Path.home() / 'sideseeing/workspaces/west-loop'))
SAM3 = Path.home() / 'models/sam3-offline'
LOGS = Path(os.environ.get('WESTLOOP_LOGS', Path.home() / 'sideseeing/logs/westloop'))
STEP = 30

PROMPTS = [
    # surface and crossings, with five curb ramp phrasings to compare
    'sidewalk', 'crosswalk', 'curb', 'curb ramp', 'curb cut', 'wheelchair ramp', 'ramp', 'curbramp', 'tactile paving',
    'driveway', 'alley', 'stairs',
    # crossing control
    'traffic light', 'pedestrian signal', 'stop sign',
    # obstacles and street furniture
    'pole', 'fire hydrant', 'trash bin', 'bench', 'bollard', 'tree', 'planter', 'parking meter', 'bicycle rack', 'bus stop',
    # temporary obstructions
    'construction barrier', 'traffic cone', 'scaffolding',
    # surface condition
    'pothole', 'puddle', 'manhole cover', 'sidewalk crack',
    # context
    'person', 'car', 'bicycle',
]
PRIVATE = ('human face', 'license plate')

VOCABULARY = 'Sidewalk audit narration: curb ramp, crosswalk, obstacle, pothole, signal, sidewalk, pole, tree.'
WHISPER = dict(language='en', initial_prompt=VOCABULARY)
# name: (model, transcribe options)
WHISPER_VARIANTS = {
    'whisper-turbo': ('turbo', dict()),
    'turbo-silence': ('turbo', dict(word_timestamps=True, hallucination_silence_threshold=2.0)),
    'turbo-vad': ('turbo', dict(vad=True)),
    'large-v3-vad': ('large-v3', dict(vad=True)),
    # added after the first comparison: the silence guard kept the narration, VAD lost most of it
    'large-v3-silence': ('large-v3', dict(word_timestamps=True, hallucination_silence_threshold=2.0)),
}

HALF = {'large-v3'}

failures = []


def each(stage, ds, work):
    """Run work(instance) for every session, logging failures and carrying on."""
    for instance in ds.iterator:
        start = time.time()
        try:
            work(instance)
        except Exception as error:
            failures.append(f'{stage} {instance.name}: {error!r}')
            print(f'[run] FAILED {stage} {instance.name}: {error!r}', flush=True)
            traceback.print_exc()
            release_gpu_memory()
        else:
            print(f'[run] {stage} {instance.name}: {time.time() - start:.0f} s', flush=True)


def segmenter():
    from sideseeing_tools.vision.segmentation import MODEL_REVISION, Sam3Segmenter
    return Sam3Segmenter(str(SAM3), MODEL_REVISION, local_files_only=True)


def main(stage):
    ds = SideSeeingDS(str(RAW))
    ws = SideSeeingWorkspace(str(WORKSPACE))
    print(f'[run] {stage}: {len(ds.instances)} sessions: {sorted(ds.instances)}', flush=True)

    if stage == 'link':
        ws.setup_data_directory(ds)

    elif stage == 'whisper':
        import whisper
        for model_name in dict.fromkeys(model for model, _ in WHISPER_VARIANTS.values()):
            # openai-whisper keeps float32 weights; large-v3 needs ~10 GB that way, so it runs in half precision.
            # Whisper's LayerNorm computes in float32 with its own weights, so those layers stay float32.
            # Turbo stays as in the pilot.
            model = whisper.load_model(model_name, device='cpu' if model_name in HALF else None)
            if model_name in HALF:
                model = model.half()
                for module in model.modules():
                    if isinstance(module, torch.nn.LayerNorm):
                        module.float()
                model = model.to('cuda')
            for name, (variant_model, options) in WHISPER_VARIANTS.items():
                if variant_model == model_name:
                    each(f'whisper {name}', ds, lambda instance: ws.transcribe(
                        ds, [instance], model_name=model_name, model=model, name=name, **WHISPER, **options))
            del model
            release_gpu_memory()

    elif stage == 'frames':
        each('frames', ds, lambda instance: ws.extract_frames(ds, STEP, [instance]))

    elif stage == 'segment':
        model = segmenter()
        each('segment', ds, lambda instance: ws.generate_segmentation(
            ds, PROMPTS, [instance], segmenter=model, private_prompts=PRIVATE))

    elif stage == 'redact':
        model = segmenter()  # only used for chunks the segment stage did not finish
        each('redact', ds, lambda instance: ws.anonymize_frames(ds, [instance], prompts=PRIVATE, segmenter=model))

    elif stage == 'events':
        ws.generate_sidewalk_assessment_events(ds)

    elif stage == 'report':
        ws.generate_report()

    elif stage == 'summary':
        summary = dict(
            sessions={
                name: dict(
                    frames=len(list(ws.route_frames_dir(name).glob('*.jpg'))),
                    redacted=(ws.redacted_dir / f'{name}-frames' / 'redaction.json').exists(),
                    prompts_complete=sum(
                        len(list((ws.preds_dir / 'sam3' / slug / name).glob('chunk-*/manifest.json')))
                        == -(-len(list(ws.route_frames_dir(name).glob('*.jpg'))) // 16)
                        for slug in [p.name for p in (ws.preds_dir / 'sam3').iterdir()]
                    ),
                    transcripts=sorted(p.parent.name for p in ws.transcripts_dir.glob(f'*/{name}.json')),
                )
                for name in sorted(ds.instances)
            },
            disk=subprocess.run(['du', '-sh', str(WORKSPACE)], capture_output=True, text=True).stdout.split()[0],
        )
        (LOGS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))

    else:
        raise SystemExit(f'Unknown stage {stage}')


if __name__ == '__main__':
    stage = sys.argv[1]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    main(stage)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f'[stage] {stage}: {time.time() - start:.0f} s, peak GPU memory {peak:.2f} GiB, failures {len(failures)}', flush=True)
    for failure in failures:
        print(f'[run] FAILURE {failure}', flush=True)
    sys.exit(1 if failures else 0)
