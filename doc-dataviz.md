# SideSeeing Viewer

The viewer is a local web server for a [workspace](doc-workspace.md). It complements the static report: the report is a shareable page of summaries, sensors, Wi-Fi and maps, while the viewer inspects model results on the raw frames and needs the server running.

| Section | What it shows |
|---|---|
| Sidewalk Assessment | Map of each route's matched path and sidewalk events; each marker opens the event's frames |
| Video Frames | Extracted frames of a route, with a button to extract them if missing |
| Dataviz | SAM3 masks and boxes with scores over each frame, filterable by class |
| Report | The static report in `report/`, when it exists |

## Run it

```bash
pip install "sideseeing-tools[workspace]"
python -m sideseeing_tools.dataviz -i /path/to/workspace -p 5000
```

Open <http://127.0.0.1:5000>. To include the report, run `SideSeeingWorkspace('/path/to/workspace').generate_report()` first.

From Python, `start_server(workspace_dir, port=5000, host='127.0.0.1')` runs the server and `create_app(workspace_dir)` returns the FastAPI application without starting it.

### On a remote machine

The server listens on `127.0.0.1` only. Forward the port over SSH instead of exposing it; for example, through a jump host:

```bash
ssh -L 5000:localhost:5000 -J user@jump.example.org user@gpu-host
python -m sideseeing_tools.dataviz -i /path/to/workspace   # on gpu-host
```

`--host 0.0.0.0` makes the viewer reachable by anyone on the network, including its unredacted frames.

## What it reads

| Source | Location |
|---|---|
| Routes | Folders in `data/` and `frames/` |
| Frames | `frames/<route>-frames/` (or `frames/<route>/` in older workspaces) |
| SAM3 results | `preds/sam3/<prompt>/<route>/<chunk>/manifest.json` and its `.npz` files |
| Older SAM3 results | `preds/<model>/detections.csv` with masks in `preds/<model>/<route>-frames/<frame>-mask.png` |
| Project Sidewalk labels | `preds/predictions.csv`: an `image` column and one 0/1 column per class |
| Events and paths | `geomatching/map_events.csv` and `geomatching/routes_gpkg/<route>.gpkg` |

Results are read per route, so frames with the same name in different routes never share overlays. Parsed files are cached until they change, so results written while the server runs appear on the next request. The server writes nothing into the workspace except frames requested through Video Frames.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/dataviz/data/{route}` | Frame names, boxes with scores per frame, box and mask classes, class colors |
| `GET /api/dataviz/image/{route}/{frame}` | The frame |
| `GET /api/dataviz/thumb/{route}/{frame}` | A thumbnail of at most 200x200 pixels |
| `GET /api/dataviz/mask/{frame}?instance={route}&classes=a,b` | Transparent PNG with the masks of the given classes |
| `POST /api/dataviz/extract/{route}` | Starts extracting one frame every 30 video frames |
| `GET /api/viewer/assessment/{route}` | Map center, events with frame URLs, and the matched path |

Requests only accept route names present in the workspace and plain frame names, so they cannot reach files outside `frames/` and `report/`.

## Supporting another model

Write its results as SAM3-style manifests with `sideseeing_tools.vision.results.save_segmentations`, or add a reader to `route_predictions` in `src/sideseeing_tools/dataviz/predictions.py` that returns, per frame, detections with a class name, scores, `[x_min, y_min, x_max, y_max]` boxes in source pixels and an optional mask.
