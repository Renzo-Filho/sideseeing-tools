# Workspace and video analysis tasks

## Task 1: Capture a representative baseline

**Description:** Select one short, shareable route fixture and document the current import, test, report, and dataviz behavior.

**Acceptance criteria:**
- [ ] A small fixture or fixture-generation recipe exercises video, frames, and repeated frame names across two routes.
- [ ] Current results and known failures are recorded without changing user data.

**Verification:** Run focused existing tests and inspect the generated workspace and report on the fixture.

**Dependencies:** None. **Likely files:** `tests/`, `tasks/plan.md`. **Scope:** Small.

## Task 2: Make installation reproducible

**Description:** Reconcile declared imports, optional model dependencies, and the Conda install instructions.

**Acceptance criteria:**
- [ ] Fresh core install imports the package without model packages.
- [ ] Vision and transcription extras install and `pip check` succeeds in the documented environment.

**Verification:** Fresh Python 3.12 environment; `python -c 'import sideseeing_tools'`; `python -m pip check`.

**Dependencies:** 1. **Likely files:** `pyproject.toml`, `environment.yml`, `docs/research-environment.md`. **Scope:** Medium.

## Task 3: Define a workspace artifact contract

**Description:** Record route, source video, frame/timestamp, model, prompt, parameters, output path, and run status in a versioned schema while reading existing CSV workspaces.

**Acceptance criteria:**
- [x] Two routes with the same frame basename remain distinct.
- [x] Existing `detections.csv` files still load through a compatibility adapter.
- [ ] The location and format of transcript artifacts are documented.

**Verification:** Contract tests on fixture outputs and a representative existing workspace.

**Dependencies:** 1–2. **Likely files:** `workspace.py`, `dataviz/adapters.py`, `tests/`, `doc-workspace.md`. **Scope:** Medium.

## Task 4: Make workspace steps restartable

**Description:** Remove duplicate orchestration and make extraction and segmentation completion checks cover the full artifact set. Keep source and anonymized frames distinguishable.

**Acceptance criteria:**
- [x] `build_workspace()` executes each requested stage once.
- [x] A run interrupted between mask and CSV writing can be resumed without missing rows or duplicate results.
- [x] Changing extraction parameters triggers a deliberate rebuild or a clear error.

**Verification:** Unit tests with interrupted output and changed step values; compare output counts after two runs.

**Dependencies:** 3. **Likely files:** `workspace.py`, `anonymization.py`, `tests/`. **Scope:** Medium.

## Task 5: Restrict dataviz file access

**Description:** Resolve paths against a trusted root, validate route names, and use loopback as the default server host.

**Acceptance criteria:**
- [x] Sibling-prefix and symlink traversal attempts are rejected.
- [x] Extraction cannot read outside the selected dataset.
- [x] Default server URL is local and documented for SSH port forwarding.

**Verification:** FastAPI route tests for valid and invalid paths; start the server on a test port.

**Dependencies:** 1. **Likely files:** `dataviz/api.py`, `dataviz/server.py`, `tests/`. **Scope:** Medium.

## Task 6: Complete one SAM3 result path

**Description:** Run image SAM3 on one clip's extracted frames and persist masks, boxes, scores, timestamps, and model provenance in the artifact contract. Assess video tracking separately after the GPU measurement.

**Acceptance criteria:**
- [x] One prompt yields route-scoped, reproducible predictions that the dataviz overlay can load.
- [ ] Batch size 1 has a measured peak VRAM and elapsed time on desktop-01.
- [ ] Failed frames are reported and can be retried.

**Verification:** CPU tests with a fake model; GPU smoke test on a short authorized clip.

**Dependencies:** 2–4. **Likely files:** `segmentation.py`, `workspace.py`, `dataviz/adapters.py`, `tests/`. **Scope:** Medium.

## Task 7: Complete one Whisper result path

**Description:** Transcribe one video through a new optional module and write timestamped segments with language, model, and source identity.

**Acceptance criteria:**
- [ ] A transcription can be resumed without rerunning a completed video.
- [ ] Segment timestamps align with the source video timeline.
- [ ] Missing audio and model errors are distinguishable.

**Verification:** Unit tests with a fake transcriber; GPU smoke test with Whisper `small` on a short clip.

**Dependencies:** 2–3. **Likely files:** `transcription.py`, `workspace.py`, `pyproject.toml`, `tests/`. **Scope:** Medium.

## Task 8: Show model output in dataviz

**Description:** Use route-scoped artifact records to show SAM3 confidence/overlays and Whisper segments near the corresponding video time.

**Acceptance criteria:**
- [x] Repeated frame names from different routes never mix overlays.
- [x] New completed artifacts appear without restarting the server.
- [ ] Transcript segments can be inspected by route and timestamp.

**Verification:** API tests plus a browser walkthrough using the fixture.

**Dependencies:** 3, 5–7. **Likely files:** `dataviz/api.py`, `dataviz/visualizer.py`, `dataviz/adapters.py`, templates. **Scope:** Medium.

## Task 9: Add transcript context to the report

**Description:** Let the report refer to timestamped transcript segments when they exist while still rendering older workspaces.

**Acceptance criteria:**
- [ ] Transcript context appears for the selected route and time.
- [ ] Existing reports without transcripts still render.

**Verification:** Generate both kinds of reports from fixtures and inspect the HTML output.

**Dependencies:** 3, 7–8. **Likely files:** `export.py`, report templates, `tests/`. **Scope:** Medium.

## Task 10: Gate releases on tests

**Description:** Add CI checks for Python 3.12 tests and packaging before the publish job.

**Acceptance criteria:**
- [x] Pull requests run focused tests and a package build.
- [x] Publishing depends on the test job passing.

**Verification:** Review the CI workflow and run its commands locally.

**Dependencies:** 4–9. **Likely files:** `.github/workflows/main.yml`, `pyproject.toml`. **Scope:** Small.

## Checkpoints

- [ ] After tasks 1–3: review artifact schema against one real route.
- [ ] After tasks 4–5: verify interrupted runs and route/path security.
- [ ] After tasks 6–9: compare SAM3 masks and Whisper timestamps with the source video.
- [ ] After task 10: tests, build, and documentation are ready for review.
