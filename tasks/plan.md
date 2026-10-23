# Plan: evaluate the workspace and add video analysis

## Goal

Keep the useful `data/`, `frames/`, `preds/`, and `geomatching/` workspace layout while making runs reproducible, resumable, and consumable by both the report and dataviz modules. Add SAM3 and Whisper as optional analysis capabilities for videos. This is a plan; model integration and structural changes are not implemented yet.

## Current architecture

`SideSeeingDS` reads the raw route dataset. `SideSeeingWorkspace` links that data, extracts frames, invokes the image-based Hugging Face SAM3 `Segmenter`, and writes masks and a detections CSV. `Report` generates the HTML report. The FastAPI dataviz server reads the workspace and overlays predictions. Whisper is not integrated. The separate `sam3.py` and `sam3-offline/` in the working tree are untracked experiments and should be preserved.

## Findings to validate in a baseline audit

- `build_workspace()` runs its whole pipeline twice (`workspace.py`).
- `geopandas`, `shapely`, `requests`, `jinja2`, and `Pillow` are imported by package modules but are absent from core dependencies in `pyproject.toml`; package imports are eager.
- Frame extraction skips a route whenever its frames folder contains any file. Segmentation skips when a mask exists, even if the matching CSV row is missing. Both can leave incomplete runs that appear complete.
- Anonymization rewrites extracted frames in place, making provenance and comparisons difficult.
- The prediction adapter drops SAM3 scores, boxes, and `relative_path`; the viewer then keys images largely by basename, which can mix routes with repeated filenames.
- The viewer caches prediction files and indexes without invalidation, so a running server may miss new results.
- The API path check uses a string prefix and does not resolve symlinks. The server listens on all interfaces by default and exposes frames, so safe path handling and local binding matter.
- The existing `test_workspace.py` and `test_dataviz.py` are manual scripts; the older dataset and media modules have automated tests. CI publishes but does not run tests first.

## Working design

1. Preserve the existing top-level workspace folders initially. Define a versioned artifact record keyed by route ID, source video, frame index or timestamp, model, prompt, parameters, and output path. Choose the exact file format after examining representative data.
2. Keep model weights and caches outside Git. SAM3 and Whisper remain optional dependencies; the core library should import without them.
3. Run GPU jobs sequentially on desktop-01's 8 GB RTX 3070. Start with SAM3 image inference at batch size 1 and Whisper `small`. Measure memory and throughput on a short representative clip before choosing video tracking or larger Whisper models.
4. Give SAM3 image segmentation and SAM3 video tracking separate backends behind a small interface. The present `Segmenter` is image inference, not temporal tracking. Transformers also provides a SAM3 video API; evaluate it first in the current environment. The official SAM3 repository provides an alternative video predictor for MP4 or JPEG folders, but its `numpy<2` requirement may conflict with other current packages.
5. Add Whisper as a transcription module with timestamped segments and model/language metadata. Connect those results to the same route and video timeline used by the report and dataviz modules.

## Implementation order

1. Create a small, consented test fixture and record baseline package imports, tests, workspace output, and a short GPU run.
2. Correct dependency declarations and document the host Conda environment.
3. Specify the workspace artifact contract and compatibility strategy for existing workspaces.
4. Repair workspace orchestration and partial-run recovery.
5. Secure file serving and make the local server bind to loopback by default.
6. Add a small SAM3 end-to-end slice: frame to persisted artifact to dataviz overlay.
7. Add a small Whisper end-to-end slice: video to timestamped transcript artifact.
8. Display route-scoped SAM3 results and transcript segments in dataviz/report.
9. Add focused automated tests and a test gate before publishing.

Detailed acceptance criteria and verification are in [todo.md](todo.md).

## Checkpoints

- After tasks 1–3: review the artifact schema against one real route and existing outputs.
- After tasks 4–5: verify restart/resume and path handling with temporary fixtures.
- After tasks 6–8: review SAM3 mask alignment and Whisper timestamps on a short real video, then decide whether video tracking is useful enough for the 8 GB GPU.
- After task 9: run the test suite and package build before release.

## Risks and open decisions

- SAM3 model access is gated; a Hugging Face account must be granted access. Model loading or video tracking may exceed 8 GB VRAM. Test a short clip first and record the measured limit.
- Symlinked raw datasets and frames may contain sensitive images or audio. Decide whether anonymized exports should be the default for sharing.
- Decide whether transcripts belong in a new `transcripts/` folder or under model-specific `preds/` output, and whether JSON plus SRT or a tabular format best serves research analysis.
- Decide whether the initial goal is sparse image segmentation, continuous SAM3 tracking, or both; they answer different research questions and have different GPU costs.
