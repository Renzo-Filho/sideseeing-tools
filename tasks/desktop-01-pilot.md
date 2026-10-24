# Pilot on utk-desktop-01: SAM3 and Whisper on one West Loop capture

Goal: run the workspace pipeline (frames, SAM3, Whisper) on one real capture fetched from RustFS, measure time and GPU memory on the RTX 3070 (8 GB), and check results in the viewer before scaling to the other captures.

| Input | Value |
|---|---|
| Capture | `rustfs:sideseeing/chicago/west-loop/2026-10-19-15-43-14-478` |
| Video | 44.2 min, 1920x1080 at about 30 fps, 6.7 GB; stereo 44.1 kHz audio |
| Speech | English collector narration |
| Results | Stay on desktop-01; view through an SSH tunnel |

Measured from the laptop on 23 October 2026 (rclone listing, `metadata.json`, video header through a presigned link).

## Steps

1. **Access.** SSH keys from the laptop to `utk` and `utk-desktop-01` (`~/.ssh/config` entries added on the laptop). Check `ssh utk-desktop-01 hostname`.
2. **Probe desktop-01.** Free disk under `$HOME` (needs about 7 GB raw + about 1 GB of frames at step 30), GPU idle (`nvidia-smi`), the `sideseeing` env, `ffmpeg`, and whether `data.urbantk.org` is reachable.
3. **Code.** Bring the branch with the workspace, viewer and transcription changes onto desktop-01's checkout (not yet pushed; see open questions). Then `pip install -e '.[vision,workspace,transcription,dev]'`, `pip check` and `pytest -q`.
4. **SAM3 weights.** Copy the laptop's `sam3-offline/` (3.2 GB, float32, saved with transformers 5.9) to `~/models/sam3-offline/` and pass `--model ~/models/sam3-offline`, or run `hf auth login` with a token that has access to `facebook/sam3`.
5. **RustFS.** Install rclone in the env, configure a read-only `rustfs` remote, and copy the capture to `~/sideseeing/raw/chicago/west-loop/2026-10-19-15-43-14-478/` (see [research environment](../docs/research-environment.md#fetch-captures-from-rustfs)).
6. **Quick pass** (about 90 frames): `--step 900 -p sidewalk --whisper small --language en --skip-map-matching` into `~/sideseeing/workspaces/west-loop-quick`. Record the `[stage]` lines.
7. **Full pass** (about 2,650 frames): `--step 30 -p sidewalk "curb ramp" crosswalk --whisper small --language en` into `~/sideseeing/workspaces/west-loop`, in `tmux` with a log file. Use the quick pass to estimate duration first.
8. **Review.** Open the viewer through the tunnel; check mask alignment on frames, event placement on the map, and transcript timestamps against the video at a few spoken moments.

## Measurements

Measured on 23 October 2026. "PyTorch" is `torch.cuda.max_memory_allocated`; "card" is the `nvidia-smi` maximum, including the CUDA context and about 0.6 GB used by the desktop session.

| Stage | Frames | Duration | Peak GPU memory | Notes |
|---|---:|---:|---:|---|
| Fetch (presigned links, curl) | – | under 2 min for 7.0 GB | – | All 9 MD5s match RustFS |
| SAM3 weights copy (laptop to desktop) | – | 74 s for 3.4 GB | – | SHA-256 identical |
| Extract frames, step 900 | 89 | 6 s | – | Recorded fps 30.03 |
| SAM3 `sidewalk`, batch 1 | 89 | 74 s | 3.87 GiB PyTorch, 5.1 GB card | 86 of 89 frames with detections |
| SAM3 batch 2 / batch 4 | 89 | 0.778 / 0.775 s per frame | 4.57 / 5.96 GiB | No speedup over batch 1 (0.778 s) |
| Whisper `small`, 44 min | – | 42 s | 1.53 GiB | Mishearings and invented fragments |
| Whisper `turbo`, 44 min | – | 102 s | 4.75 GiB | Better; some errors |
| Whisper `turbo` + vocabulary prompt | – | 52 s | 4.75 GiB | Best: consistent "curb ramp", fewest invented fragments |
| Extract frames, step 30 | 2,653 | 158 s | – | |
| SAM3, 3 prompts, batch 1 | 2,653 × 3 | 6,371 s (1 h 46 min) | 3.90 GiB PyTorch | 0.80 s per frame and prompt; 498 chunks |
| Map matching (public OSRM) | – | 119 s | – | First attempt failed: torchvision's bundled libjpeg broke pyogrio; fixed in 7e20eb9 |
| Sidewalk events | – | 3 s | – | 131 events: 67 crosswalk, 59 sidewalk, 5 curb ramp |
| Whisper `turbo` + vocabulary prompt | – | 73 s | 4.59 GiB | 273 segments, none with `no_speech_prob` > 0.6 |

Detections per prompt (frames with at least one instance, of 2,653): sidewalk 2,618 (median best score 0.91), crosswalk 1,528 (0.85), curb ramp 15 (0.61). Curb ramps are mentioned often in the narration, so SAM3 misses most of them with this prompt; crosswalk in 58% of frames needs a visual check for false positives.

Decisions from the quick pass: SAM3 batch 1 (batching does not speed it up on the 3070); Whisper `turbo` with the vocabulary prompt `Sidewalk audit narration: curb ramp, crosswalk, obstacle, pothole, signal, sidewalk, pole, tree.`

## Decisions after the pilot

- SAM3 batch size and chunk size on 8 GB.
- Whisper `small` versus `turbo` for this narration (accuracy on a few known phrases, time, memory).
- Whether SAM3 video tracking adds enough over per-frame segmentation to justify its memory and time.
- Prompt list and step for the remaining West Loop captures and for Brás.

## Open questions

- How should the branch reach desktop-01: push it to the fork (`origin`), or copy it over SSH without publishing?
- Is there a read-only RustFS access key for desktop-01, or should it reuse the laptop's key?
- Which SAM3 prompts matter for West Loop beyond sidewalk, curb ramp and crosswalk?
