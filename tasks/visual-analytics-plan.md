# Plan: from viewer to evidence workbench

Draft, 24 October 2026. A proposal for discussion, not agreed scope. It builds on the West Loop pilot (`tasks/desktop-01-pilot.md`), its analysis notebook (`sideseeing-network/network/west_loop_sam3_whisper.ipynb`) and the research design in `sideseeing-network/network/docs/` (`PLAN.md`, `RESEARCH_CRITIQUE_2026-10-23.md`, `HYPOTHESIS_EXPERIMENTS_2026-10-23.md`).

## Goal

The research aims at an **evidence-linked pedestrian passage graph**: passages, ports and transitions, each tied to the observations that support or contradict it, with alternatives kept where the evidence is ambiguous. The viewer should become the place where that evidence is inspected, labelled and tested, not a gallery of model outputs.

Concretely, a researcher should be able to:

1. pick any moment of a recording and see every modality at that moment: frame, masks, GPS, motion events, speech;
2. ask a domain question ("are crosswalk detections near mapped crosswalks?", "does SAM3 see the curb ramps the collector names?") and get an answer that links back to the moments that produced it;
3. label reference intervals (side, port, transition, crossing attachment, unknown) without seeing model output when the protocol requires it;
4. later, see graph hypotheses on the map and follow each element back to its evidence.

## Where we are

**Workspace.** Built by `SideSeeingWorkspace` and `examples/build_workspace.py`, on desktop-01 for the West Loop capture D1 S1 (`~/sideseeing/workspaces/west-loop`, plus the 89-frame `west-loop-quick`). Layout: `data/` (links to raw), `frames/` with `frames.json`, `preds/sam3/<prompt>/<route>/chunk-*/` with manifests, `geomatching/` (OSRM route, `map_events.csv`), `transcripts/whisper-<model>/`, `report/`.

**Viewer** (`python -m sideseeing_tools.dataviz`): video frames, per-frame masks with class toggles and scores, a sidewalk assessment map of events with their frames, and a link to the static report. Read-only, one route and one frame at a time; the transcript is not shown.

**What the pilot showed** (details in `tasks/desktop-01-pilot.md`):

| Finding | Consequence for the tool |
|---|---|
| `sidewalk` and `crosswalk` work; `curb ramp` fails (15 false detections, 1 of 63 spoken mentions found) | Prompt quality must be checkable inside the tool, against labels, not only by eye |
| 57 of 59 sidewalk events are exactly the 45-frame cap | "Events" are runs of detections cut into pieces, not places. They should not be the main unit |
| Event position = mean raw GPS of the frames, snapped to the matched route | The camera's position is used as the object's position, then moved to the street centreline (see GPS section) |
| `Corrected GPS` comes from `router.project-osrm.org/match/v1/foot`. The demo server ignores the profile: `foot` and `driving` both route 676 m in 80 s (30 km/h); a foot router gives 533 s | The "corrected" track is a **car road-centreline** match. It is not a sidewalk position and should not be labelled as one |
| Whisper hallucinations have low `no_speech_prob` and `avg_logprob` < −1 | Transcript quality flags must travel with the text |
| The notebook had to rebuild a per-frame table (time, GPS, scores, mask area) from manifests, and join stops, turns and crossings by time | That join is the missing **read API** of the workspace. The viewer and every notebook need it |

## The workspace's role

**Proposal.** The workspace is the study's **observation store**: everything derived from the recordings, on one time base, with provenance, readable by the viewer, the report and the research notebooks through one API. Graph inference stays in `sideseeing-network` until its format settles; the workspace exports to it and reads its results back for display.

Four layers, each written by one kind of process:

| Layer | Content | Written by | Rule |
|---|---|---|---|
| L0 raw | `data/` links, `metadata.csv` | `setup_data_directory` | Never modified |
| L1 extraction | frames, SAM3 masks, transcripts, motion events (stops, turns) | library steps | Recomputable; each output has a manifest: inputs and their hashes, model and version, parameters |
| L2 alignment | map matching, object positions, detection intervals | library steps | Recomputable; records the method and the **input regime** (street-only, street + aerial, map-assisted) |
| L3 research | reference labels, graph hypotheses, saved analyses | viewer and notebooks | Append-only, with author and time; never overwritten by a recompute |

What this changes in practice:

1. **One time base.** Every record carries `route` and `t` (seconds from video start), with the route's `media_start_time` to get UTC. Frames, masks, transcript segments, IMU events, GPS fixes and labels then join on `(route, t)`. An observation is a `(route, t0, t1, modality)` interval with an ID, the unit the research design calls an Observation.
2. **A read API.** `workspace.frames(route)`, `workspace.detections(route, prompt, min_score)`, `workspace.transcript(route, max_logprob=-1)`, `workspace.gps(route)`, `workspace.motion_events(route)`, and `workspace.timeline(route)` that joins them. The notebook's loading code moves here; the viewer's endpoints call the same functions.
3. **`workspace.status()`**: which stages exist per route, with their parameters and dates. It would have shown at once that the report was missing.
4. **Transcripts need review before use.** D1 S1 has 51 of 273 segments with `avg_logprob` < −1, some invented outright. The West Loop run (`tasks/west-loop-run.md`, Phase 3) tests fixes (built-in silence guard, speech detection before transcription, larger model, filters), and Renzo reviews a sample with its audio before one configuration is adopted. Reviewed labels stay in L3, so later configurations can be scored against them.
5. **Whisper as an observation stream, kept apart.** Transcripts get the same manifest as SAM3. A small extraction step turns segments into **spoken events** (term, interval, quality flags). Decision 14 in `PLAN.md` makes voice an evaluation reference only, so spoken events live in their own layer, are marked `assisted`, and never feed automatic events or graph construction. The viewer shows them as a separate track that can be hidden. Audio stays local; the raw-audio restrictions already in the project apply to anything exported.
6. **Motion events in the workspace.** Stops, turns and crossings are computed in `west_loop.ipynb` today. Port the stop and turn detectors (accelerometer and gravity-projected gyroscope) into the library as an L1 step, so the viewer can show them for every route. Crossing matching uses OSM and the route plan, so it stays a declared map-assisted L2 step.
7. **Export for the graph.** Observations, detection intervals and positions as GeoParquet/GeoJSON tables keyed by observation ID, matching the four records proposed in the critique: observation, passage hypothesis, transition hypothesis, evidence association. The workspace owns the first; `sideseeing-network` owns the other three.

## Visual analytics plan

Five phases. Each phase is useful on its own and builds on the previous one. The first two only need the current pilot data.

### V1. Linked timeline (foundation)

A route page with coordinated views on one cursor and one brushed interval:

- frame player with mask overlay (exists);
- map: raw GPS, the cursor's position, and later the estimated object positions (see GPS section);
- tracks: best score per prompt, mask area, stops and turns, crossings, spoken events, GPS accuracy.

Brushing an interval on any view highlights it in all the others. This is the notebook's §1.1 figure made interactive.

**Acceptance.** From a spoken "curb ramp" at 29:21, one click shows the frame, its curb ramp masks (none) and crosswalk masks, the map position and the nearest crossing.

### Dataviz with many prompts

The West Loop run adds ~30 prompts. Drawing every mask in one colour each makes the frame unreadable. Proposal, to mock up before building:

- **Groups** (surface and crossings, crossing control, obstacles, temporary obstructions, surface condition, context) with one colour family each and a group toggle; prompts inside a group as shades.
- **Presets**: "Walkable surface", "Crossing", "Obstacles", "Everything"; default to one preset, not all layers.
- **Surfaces as faint fills, objects as outlines with labels**, so a sidewalk mask does not hide the pole on it.
- **Per-frame list** of detections sorted by score; hovering one highlights its mask, clicking isolates it.
- **One score threshold** for all layers, plus per-prompt counts for the current frame and route.
- **Search** for a prompt by name.
- Later, the V1 timeline gives each group a track, so a user finds where a prompt fires before opening frames.

### V2. Evidence quality and domain questions

- Score threshold and prompt filters that update every view, plus counts.
- **Question panels**: a saved question is a filter plus a measure plus its linked examples, re-run on any route. Start with the pilot's questions: detection rate against distance to mapped crosswalks; detections during gyroscope crossings; spoken terms against detections nearby; sidewalk share of the image while walking versus crossing. Each result opens its frames.
- **Prompt evaluation**: pick a set of moments (for example the 63 spoken curb ramp mentions), run new prompts on just those frames as a background job on the GPU machine, and compare hit rates. This turns the curb ramp failure into a repeatable test.
- Model and transcript provenance shown next to every value.

### V3. Reference labelling

Interval labels on the timeline, plus point and line labels on the map: physical side, port, transition and type, crossing attachment, obstacle, `unknown`. Requirements from the research design:

- **Blind mode** hides model output, detections and spoken events while labelling;
- two annotators per motif, with disagreements kept and adjudication recorded;
- labels are L3 records with author, time and protocol version, and are never edited in place;
- labels split by site (intersection or corridor), so training and held-out sites never mix.

The label vocabulary (ports, transition types) belongs to the research and is defined in `sideseeing-network` as a config file the viewer loads, so the tool stays generic.

### V4. Hypothesis workbench

Views for the three hypotheses in `HYPOTHESIS_EXPERIMENTS_2026-10-23.md`, once their records exist:

- **H1, modalities**: at one motif, what GPS alone, GPS + IMU, GPS + video and all three conclude, next to the reference;
- **H2, alternatives**: one motif's alternative graphs side by side, a route query across them, and whether one fixed route is supported in every alternative; risk–coverage curves with each point linked to its sites;
- **H3, revisits**: candidate revisit sites ranked by each policy, with their estimated cost.

These start as notebooks; a view moves into the viewer only when a notebook version is used repeatedly.

### V5. Graph view

When the graph exists: passages, ports and transitions on the map, coloured by provenance (street, aerial, both) and state (supported, contradicted, unresolved, outside coverage); alternative groups shown as alternatives, never merged; clicking an element lists its evidence intervals and opens them in the V1 timeline. Several routes and days shown together, each keeping its identity.

## Future study: GPS correction and detection placement

Two separate problems that the assessment map currently mixes up:

1. **Where was the camera?** (trajectory: side of street, crossings)
2. **Where is the detected object?** A frame from one sidewalk can show the opposite sidewalk or a crosswalk 20 m ahead.

**Immediate fixes, small:**

- Label the current layer "Road-matched (OSRM car profile)", not "Corrected GPS", in the report and viewer.
- In the assessment tab, place events at the camera position (raw GPS) and draw the road match only as context.
- Ask before sending traces to a public server. A collector's GPS is personal data; `map_matching` should name its endpoint and allow a self-hosted one.

**Study.** Compare trajectory methods on D1 and D2 against the gyroscope crossings and, later, labelled references:

| Method | Input regime | Note |
|---|---|---|
| Raw GPS, 5-fix median | street-only | Baseline |
| OSRM demo (current) | map-assisted, car network | Current behaviour |
| Valhalla `trace_attributes` with pedestrian costing, self-hosted on desktop-01 | map-assisted, street network | A real foot profile, no third party |
| HMM matching to OSM **sidewalk** lines (`footway=sidewalk`, crossings) | map-assisted, sidewalk network | Gives a side, not just a street |
| The above + IMU turn and crossing events as constraints | map-assisted + motion | Tests H1's "motion resolves when, not where" |
| The notebook's corrected track | map-assisted + route plan | Upper reference; uses the route plan, so not a method |

**Object placement**, the part that serves the assessment tab: project each mask onto the ground. We already have the camera pitch from gravity (the notebook's §8), the mask's lowest row and horizontal centre (`sam3_frames.parquet`), the heading from GPS or gyroscope, and the camera's position. Ground-plane projection gives a distance and bearing from the camera, so a position with an error ellipse, and a side relative to the walked street. Measures: distance from crosswalk detections to the nearest mapped crosswalk (current median 3.5 m for event centres, which are camera positions), and side accuracy against labels. Keep camera position, object position and their uncertainty as separate fields. The assessment tab draws all three.

Metrics follow the critique: side identity and abstention, length of wrong-side stretches, crossing attachment, and distance to mapped features only as a map-assisted diagnostic.

## Order and checkpoints

| Step | Work | Depends on |
|---|---|---|
| 1 | Workspace read API, `status()`, manifests for transcripts and map matching; notebook switches to it | – |
| 2 | Relabel road matching; assessment events at camera position; configurable matcher endpoint | – |
| 3 | V1 timeline with spoken-event and motion-event tracks | 1, stop/turn detectors ported |
| 4 | V2 question panels and prompt evaluation | 3 |
| **Checkpoint A** | Review with the research plan: is the timeline the right primitive for labelling? | |
| 5 | Label vocabulary config in `sideseeing-network`; V3 labelling with blind mode | A |
| 6 | GPS study: Valhalla on desktop-01, sidewalk HMM, ground-plane projection | 1, labels for evaluation |
| 7 | Graph export tables; V4 views from notebooks that proved useful | 5, graph records |
| 8 | V5 graph view | graph prototype in `sideseeing-network` |

## Insights

1. **The unit is the time interval, not the frame or the event.** The graph's evidence association points to an interval of a recording. If the viewer is built around intervals from V1, labelling (V3) and the graph view (V5) reuse the same primitive. If it stays built around frames and events, every later step needs a translation layer.
2. **Today's events are a processing artefact.** Detection runs cut at 45 frames are neither places nor observations of one object. Replace them with detection intervals (L1/L2) and let places come from graph association.
3. **The map currently tells a confident story it cannot support.** A car-profile centreline drawn as "Corrected GPS", with detections snapped onto it, hides exactly the ambiguity the research studies: side identity and camera versus object. Showing raw GPS, uncertainty and the input regime is more honest and more useful.
4. **Narration is the cheapest reference we have, and the easiest to misuse.** It found the curb ramp failure in minutes. Keeping it as a visible, separate, assisted layer protects decision 14 while still using it to prioritise review.
5. **Scope.** The critique warns against making an interactive application a prerequisite for the first paper. The tool earns its place as research instrumentation: faster labelling, visible provenance, repeatable checks. A visualization contribution of its own (for example evidence-linked graph inspection under uncertainty) would be a separate decision, worth considering once V3 and V5 exist.

## Open questions

1. Should the generic labelling feature live in `sideseeing-tools` and only the vocabulary in `sideseeing-network`, as proposed, or should labelling stay entirely in the research repo?
2. Is the viewer meant to go upstream to Rafael's repository (PR B), or remain a fork feature while it changes quickly?
3. Self-hosted Valhalla (or OSRM with a foot profile) on desktop-01: acceptable to run there, and which OSM extract?
4. Does the workspace hold several routes of one study (all West Loop sessions), or one workspace per day? The graph needs several routes together.
5. Which questions should be the first V2 panels: the pilot's, or the H1 motif checks?
