# SideSeeingWorkspace

A workspace is one folder per study. It links to the raw recordings without copying or modifying them, and keeps everything derived from them in fixed places: extracted frames, SAM3 predictions, redacted frames, map-matched routes, sidewalk events and the report. The [viewer](doc-dataviz.md) and the report both read this layout, so you only pass them the workspace folder.

```bash
pip install "sideseeing-tools[workspace,vision]"
```

`workspace` covers the viewer, map matching and events; `vision` adds SAM3 (PyTorch and Transformers). The `facebook/sam3` checkpoint is gated: accept its terms at <https://huggingface.co/facebook/sam3> and run `hf auth login` before the first run, or use a local checkpoint (see below).

## Layout

```text
workspace/
├── data/                              links to the raw routes, plus metadata.csv
│   ├── metadata.csv
│   └── route01/ -> /raw/route01/
├── frames/
│   └── route01-frames/
│       ├── route01_00000.jpg          every step-th video frame, named by frame index
│       └── frames.json                step, real video fps, source video
├── frames-redacted/                   only after anonymize_frames()
│   └── route01-frames/
│       ├── route01_00000.jpg
│       └── redaction.json             prompts, mode, model, masked pixels per frame
├── preds/
│   └── sam3/
│       └── curb-ramp/                 one folder per prompt
│           └── route01/
│               └── chunk-0000/
│                   ├── manifest.json  prompt, model revision, source hashes, timestamp of each frame
│                   └── frame-000000.npz  masks, boxes and scores of every detected instance
├── geomatching/
│   ├── routes_gpkg/route01.gpkg       OSRM map-matched route
│   └── map_events.csv                 sidewalk events built from detections and GPS
├── transcripts/
│   └── whisper-small/route01.json     timed Whisper segments, after transcribe()
└── report/                            static report, after generate_report()
```

`data/` can be loaded on its own with `SideSeeingDS('workspace/data')`.

## Build everything

```python
from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace

ds = SideSeeingDS('/path/to/raw/dataset')
workspace = SideSeeingWorkspace('/path/to/workspace')

workspace.build_workspace(ds, prompts=['sidewalk', 'curb ramp'], extract_step=30, anonymize=True)
workspace.generate_report()
```

`build_workspace` runs each stage once: link data, extract frames, write redacted copies (when `anonymize=True`), segment, map-match and build events. [examples/build_workspace.py](examples/build_workspace.py) does the same from the command line.

## Step by step

```python
from sideseeing_tools.vision.segmentation import MODEL_REVISION, Sam3Segmenter

routes = [ds.instances['route01'], ds.instances['route02']]  # or omit `instances` for all routes
segmenter = Sam3Segmenter()  # or Sam3Segmenter('/path/to/sam3-offline', MODEL_REVISION, local_files_only=True)

workspace.setup_data_directory(ds)
workspace.extract_frames(ds, extract_step=30, instances=routes)
workspace.anonymize_frames(ds, instances=routes, segmenter=segmenter)          # optional
workspace.generate_segmentation(ds, ['crosswalk'], instances=routes, batch_size=1, segmenter=segmenter)
workspace.generate_map_matched_routes(ds, instances=routes)                     # calls the public OSRM server
workspace.generate_sidewalk_assessment_events(ds)
workspace.transcribe(ds, instances=routes, model_name='small', language='en')   # optional, needs [transcription]
workspace.generate_report()                                                     # writes report/
```

Create one `Sam3Segmenter` and pass it to every step to load the model once.

## Transcripts

`transcribe()` runs Whisper on each route's video and writes `transcripts/whisper-<model>/<route>.json`: the detected language, the parameters, the video's `media_start_time` (UTC) and a list of segments with `start` and `end` in seconds from the video start, `text`, `avg_logprob` and `no_speech_prob`. Add a segment's `start` to `media_start_time` to place it on the GPS track and frame timeline. High `no_speech_prob` segments are often Whisper inventing text during silence.

It needs `pip install sideseeing-tools[transcription]` and FFmpeg on the `PATH`. `condition_on_previous_text` defaults to `False` to limit repeated text across long silences; other Whisper options pass through. Videos without readable audio are skipped with a warning, and existing transcripts are kept. On an 8 GB GPU, free SAM3 first (`build_workspace` does this) before loading a larger Whisper model.

## Interrupted and repeated runs

- **Frames.** A route's folder appears only after its extraction finishes, so an interrupted extraction leaves nothing behind. Running again with the same step skips finished routes; a different step raises an error. Delete `frames/<route>-frames/` to extract again.
- **Segmentation.** Frames are segmented in chunks (`chunk_size`, 16 frames by default), and each chunk folder appears only when complete. A rerun processes only missing chunks. Delete a prompt or route folder under `preds/sam3/` to redo it.
- **Redaction.** Each route's redacted folder appears only when complete; existing folders are skipped. Source frames are never modified.

## Timing

Timestamps come from the frame index and the fps recorded in `frames.json`, and are stored in each manifest. Events align them with GPS by adding them to the recording's media start time. Frames named `..._00012_ms.jpg` come from older extractions and count seconds.

## Older workspaces

Workspaces made before this layout still work in the viewer and in event generation:

- `preds/sam3-<prompt>/detections.csv` with mask PNGs in `preds/sam3-<prompt>/<route>-frames/`; routes come from `relative_path`, with or without the `-frames` suffix.
- Route names with a `data_` prefix in `map_events.csv`.
- Frame folders without `frames.json`: they are kept as they are, and their fps is read from the video.
- `metadata.csv` at the workspace root: `generate_report()` links it into `data/`.

## Tips

- **GPU memory.** Start with `batch_size=1` on an 8 GB GPU. A chunk's masks stay in memory until the chunk is saved; lower `chunk_size` for 4K video.
- **Privacy.** Redaction masks can miss faces or plates; `redaction.json` is marked `processed-not-privacy-validated` until someone reviews the output. The viewer serves unredacted frames, so keep it on `127.0.0.1`.
- **Disk.** Keep `use_symlinks=True` (the default) to avoid copying raw videos. On Windows, creating symlinks may require administrator rights or developer mode.
