# West Loop run: SAM3 prompts, redaction and Whisper for every session

Checklist, 24 October 2026. Runs on utk-desktop-01; results stay there. Nothing is pushed to GitHub.

**Unattended mode (agreed 24 October).** Renzo is away during the run, so no phase waits for him: the curb ramp variants and the Whisper variants all run on every session, and the choices marked **You** are made afterwards from the comparison reports. Public OSRM map matching is skipped for the new sessions (it sends collectors' GPS to a third party and only gives a car-profile road match; it takes about 2 min per session if wanted later). One resumable script on desktop-01 runs everything, so a VPN drop does not stop it.

## Agreed

- Run **all** prompts of the list below on every West Loop session in RustFS.
- D1 S1 (`2026-10-19-15-43-14-478`) keeps its `sidewalk`, `curb ramp` and `crosswalk` results; only new prompts are added. Other sessions run everything.
- `human face` and `license plate` are used only for redaction: redacted frames for every session, served by every viewer tab.
- Curb ramp: `curb ramp`, `curb cut`, `wheelchair ramp`, `ramp` and `curbramp` all run on every session (with one encoding per frame they add ~0.5 s per frame); the comparison on D1 S1's 63 spoken mentions picks the one to keep.
- Whisper: test fixes for hallucination first; Renzo reviews the result before one configuration is used for all sessions.
- The dataviz tab needs a design that stays readable with ~30 prompts.

## Prompt list

| Group | Prompts |
|---|---|
| Surface and crossings | sidewalk, crosswalk, curb, curb ramp, curb cut, wheelchair ramp, ramp, curbramp, tactile paving, driveway, alley, stairs |
| Crossing control | traffic light, pedestrian signal, stop sign |
| Obstacles and furniture | pole, fire hydrant, trash bin, bench, bollard, tree, planter, parking meter, bicycle rack, bus stop |
| Temporary obstructions | construction barrier, traffic cone, scaffolding |
| Surface condition | pothole, puddle, manhole cover, sidewalk crack |
| Context | person, car, bicycle |
| Privacy (redaction only) | human face, license plate |

Scripts: `tasks/west-loop/` (copied to `~/sideseeing/` on desktop-01). `run_westloop.sh` runs `fetch_all.sh` and then `westloop_step.py` stage by stage; `whisper_review.py` builds the review packet; `prompt_groups.json` groups the prompts for the viewer. Logs: `~/sideseeing/logs/westloop/` (`status`, `run.log`, `gpu.csv`, `summary.json`).

## Phase 0: inventory (no GPU)

- [x] 8 sessions in `rustfs:sideseeing/chicago/west-loop/`, 291.5 min, 43 GB (19 Oct: 2, 20 Oct: 2, 22 Oct: 4).
- [x] Estimate: ~17,500 frames at step 30; 37 prompts at 4.8 s per frame, ~23–25 h of SAM3; ~1 h of Whisper; ~75 GB of disk (758 GB free).
- [x] Fetched the 7 missing captures (presigned links, 12 h): 63 files, sizes and MD5s match; the two videos without an MD5 in RustFS open fully, with durations matching their metadata.

## Phase 1: encode each frame once (built into sideseeing-tools)

- [x] Approved.
- [x] `Sam3Segmenter.segment_prompts()` encodes each batch once and runs every prompt on it; `segment()` is the one-prompt case. `generate_segmentation` runs all prompts a chunk is missing in one pass; output layout unchanged (e82ff6c).
- [x] Privacy masks go to `preds/private/` (never read by events, viewer or report); `anonymize_frames` reuses them instead of running SAM3 again.
- [x] `transcribe(name=..., vad=True)`: one folder per variant, Silero VAD before Whisper.
- [x] Unit tests (193 pass).
- [x] On desktop-01: identical masks, scores and boxes to the old per-prompt path on 20 frames × 5 prompts (0 differing pixels).
- [x] 37 prompts: 4.82 s per frame (old path: 28.6 s), peak 4.02 GiB, ~52 MiB of masks per frame in RAM.
- [x] Smoke test of every stage on the fixture recording; `large-v3` runs in half precision (float32 weights need ~10 GB).

## Phase 2: curb ramp prompt test

- [ ] Test set from D1 S1: frames from 10 s before to 5 s after each of the 63 spoken curb ramp mentions (~900 frames), plus ~300 mid-block frames ≥ 50 m from any mapped crosswalk as negatives.
- [ ] The 5 variants come from the full run (no separate experiment).
- [x] Result on D1 S1 (`curb_ramp_test.py`; 63 spoken mentions, 113 frames ≥ 50 m from any mapped crosswalk):

  | Prompt | Frames with a detection | Mentions found | Mid-block frames with one | Median score |
  |---|---:|---:|---:|---:|
  | `curb ramp` | 15 | 2% | 0% | 0.61 |
  | `curb cut` | 1,947 | 100% | 72% | 0.74 |
  | `wheelchair ramp` | 32 | 3% | 7% | 0.59 |
  | `ramp` | 10 | 3% | 2% | 0.56 |
  | `curbramp` | 0 | 0% | 0% | – |
  | **`tactile paving`** | 577 | **95%** | **0%** | 0.85 |
  | `curb` | 2,225 | 100% | 86% | 0.77 |

  No curb ramp phrasing works. `curb cut` fires more mid-block than near crosswalks at every score threshold (0.5–0.95), so it is not tracking ramps. `tactile paving` is a strong proxy: Chicago's ramps carry detectable warning plates. Spoken mentions are not exhaustive, so "mentions found" is a recall proxy.
- [ ] Galleries of `tactile paving` and `curb cut` masks on redacted frames, to confirm what each one segments.
- [ ] **You:** adopt `tactile paving` as the curb ramp signal? Later option: SAM3 also accepts image exemplars (boxes around a few real ramps) instead of text.

## Phase 3: Whisper hallucination test

- [ ] Variants on every session, one transcript folder each (all with the vocabulary prompt, English, `condition_on_previous_text=False`):
  - `whisper-turbo`: current configuration (baseline);
  - `turbo-silence`: + `hallucination_silence_threshold=2` with `word_timestamps=True` (built in);
  - `turbo-vad`: Silero VAD first, Whisper only on speech regions (`silero-vad` approved);
  - `large-v3-vad`: `large-v3` with VAD;
  - `large-v3-silence`: `large-v3` with the silence guard (added after the first comparison);
  - post-filters on `avg_logprob`, `compression_ratio` and `no_speech_prob`, applied in the comparison, not as runs.
- [x] The three turbo variants ran on all 8 sessions in the main run. `large-v3` failed there: `.half()` breaks Whisper's float32 LayerNorm. Fixed by keeping LayerNorm in float32 (tested on D1 S1's first 90 s, peak 3.1 GiB); `after_run.sh` runs both `large-v3` variants once the main run finishes.
- [x] First automatic comparison (`review/whisper-v1/comparison.csv`), all sessions:

  | Variant | Segments | Speech time | Flagged | "curb ramp" | "crosswalk" |
  |---|---:|---:|---:|---:|---:|
  | `whisper-turbo` | 1,867 | 183.9 min | 376 | 451 | 374 |
  | `turbo-silence` | 1,708 | 55.5 min | 92 | 455 | 390 |
  | `turbo-vad` | 1,184 | 49.2 min | 318 | 47 | 68 |

  The baseline claims 184 min of speech in 292 min of recording; the silence guard cuts flagged segments 4× and keeps the narration; Silero VAD with default settings drops most of it (likely voice under street noise). Flags are automatic (`avg_logprob` < −1, non-Latin script), not verified.
- [x] Review packet v1 (three turbo variants, 64 windows, 8 per session): `sideseeing-network/data/west_loop_review/whisper-v1/review.html` on the laptop. v2 with the `large-v3` variants after the run.
- [ ] **You:** review, then pick the configuration.

## Phase 4: freeze

- [ ] Final prompt list and Whisper configuration recorded in `tasks/desktop-01-pilot.md`.
- [x] Public OSRM map matching skipped for the new sessions (unattended default); **You:** run it later?

## Phase 5: full run (detached, logged, monitored)

- [ ] One study workspace, `~/sideseeing/workspaces/west-loop`, with all sessions.
- [x] Frames, step 30, for new sessions (1,087 s for 7 sessions, no failures).
- [ ] Redaction (face, license plate) for all sessions, including D1 S1.
- [ ] Segmentation: all prompts; D1 S1 only the new ones. Started 24 Oct 04:00 (desktop clock). Measured in the run: 80 s per 16-frame chunk with 34 prompts (5.0 s per frame including saving), process memory steady at ~12 GB; expected end ~25 Oct morning.
- [ ] Whisper: every variant on every session, one folder each; D1 S1's current `whisper-turbo` transcript is kept as the baseline.
- [ ] Map matching (per Phase 4), events, report.
- [ ] Record duration and peak GPU memory per stage and session.

## Phase 6: viewer

- [ ] Viewer and report serve redacted frames whenever they exist, in every tab; test that no route falls back to originals silently.
- [ ] Dataviz design for ~30 prompts (see `tasks/visual-analytics-plan.md`): mock-up first. **You:** approve, then implement.

## Phase 7: review and close

- [ ] Spot-check redaction on each session (marked `processed-not-privacy-validated` until reviewed).
- [ ] Update the analysis notebook for all sessions, if wanted.
- [ ] Update docs, commit locally.
