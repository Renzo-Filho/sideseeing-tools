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

| Stage | Frames | Duration | Peak GPU memory | Notes |
|---|---:|---:|---:|---|
| Fetch (rclone) | – | | – | |
| Extract frames, step 900 | | | – | |
| SAM3, 1 prompt, batch 1 | | | | |
| Whisper small, 44 min | – | | | |
| Extract frames, step 30 | | | – | |
| SAM3, 3 prompts, batch 1 | | | | |

## Decisions after the pilot

- SAM3 batch size and chunk size on 8 GB.
- Whisper `small` versus `turbo` for this narration (accuracy on a few known phrases, time, memory).
- Whether SAM3 video tracking adds enough over per-frame segmentation to justify its memory and time.
- Prompt list and step for the remaining West Loop captures and for Brás.

## Open questions

- How should the branch reach desktop-01: push it to the fork (`origin`), or copy it over SSH without publishing?
- Is there a read-only RustFS access key for desktop-01, or should it reuse the laptop's key?
- Which SAM3 prompts matter for West Loop beyond sidewalk, curb ramp and crosswalk?
