"""JSON and image endpoints for the viewer.

Routes must exist in the workspace and file names must be plain frame names, so a request
can only reach frames under frames/.
"""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image

from sideseeing_tools.dataviz import predictions
from sideseeing_tools.workspace import extract_route_frames, list_frames

router = APIRouter(prefix="/api/dataviz", tags=["Dataviz"])

IMAGE_SUFFIXES = {".jpg", ".png"}  # the frame types listed by workspace.list_frames
EXTRACT_STEP = 30
EMPTY_PNG = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'


def list_routes(workspace_dir):
    """Routes linked in data/ or with extracted frames in frames/."""
    workspace_dir = Path(workspace_dir)
    data_dir, frames_dir = workspace_dir / "data", workspace_dir / "frames"
    routes = {path.name for path in data_dir.iterdir() if path.is_dir()} if data_dir.is_dir() else set()

    if frames_dir.is_dir():
        routes |= {path.name.removesuffix("-frames") for path in frames_dir.iterdir() if path.is_dir()}

    return sorted(routes)


def route_frames_dir(workspace_dir, route):
    frames_dir = Path(workspace_dir) / "frames"
    named = frames_dir / f"{route}-frames"
    # Older workspaces used frames/<route>/
    return named if named.is_dir() or not (frames_dir / route).is_dir() else frames_dir / route


def _workspace_for(request, route):
    workspace_dir = request.app.state.workspace

    if route not in list_routes(workspace_dir):
        raise HTTPException(status_code=404, detail="Unknown route.")

    return workspace_dir


def _frame_path(request, route, filename):
    path = route_frames_dir(_workspace_for(request, route), route) / filename

    if Path(filename).name != filename or path.suffix not in IMAGE_SUFFIXES or not path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found.")

    return path


def _route_predictions(workspace_dir, route):
    frames = [path.name for path in list_frames(route_frames_dir(workspace_dir, route))]
    found = predictions.route_predictions(workspace_dir / "preds", route, frames)
    classes = sorted({detection["class_name"] for detections in found.values() for detection in detections})
    return frames, found, classes


@router.get("/image/{route}/{filename}")
def get_image(route: str, filename: str, request: Request):
    """Serves the raw extracted frame."""
    return FileResponse(_frame_path(request, route, filename))


@lru_cache(maxsize=2048)
def _thumbnail(path, mtime):
    with Image.open(path) as image:
        image.thumbnail((200, 200))
        buffer = BytesIO()
        image.convert("RGB").save(buffer, "JPEG")
        return buffer.getvalue()


@router.get("/thumb/{route}/{filename}")
def get_thumbnail(route: str, filename: str, request: Request):
    """Serves a frame thumbnail of at most 200x200 pixels for the gallery."""
    path = _frame_path(request, route, filename)
    return Response(_thumbnail(str(path), path.stat().st_mtime_ns), media_type="image/jpeg")


@router.get("/mask/{filename}")
def get_mask_overlay(filename: str, request: Request, classes: str = "", instance: str = ""):
    """A transparent PNG with the masks of the requested classes for one frame."""
    _frame_path(request, instance, filename)
    requested = {name.strip() for name in classes.split(",") if name.strip()}
    _, found, all_classes = _route_predictions(request.app.state.workspace, instance)
    overlay = predictions.mask_overlay(found.get(filename, []), requested, all_classes) if requested else None
    return Response(overlay or EMPTY_PNG, media_type="image/png")


@router.get("/data/{route}")
def get_route_data(route: str, request: Request):
    """Frames, boxes with scores, and the box and mask classes of one route."""
    frames, found, classes = _route_predictions(_workspace_for(request, route), route)
    colors = {name: "rgb({},{},{})".format(*predictions.class_color(name, classes)[:3]) for name in classes}
    bboxes = {}

    for name, detections in found.items():
        bboxes[name] = []

        for detection in detections:
            boxes = detection["boxes"]
            scores = detection["scores"] if len(detection["scores"]) == len(boxes) else [None] * len(boxes)

            for score, (x_min, y_min, x_max, y_max) in zip(scores, boxes):
                bboxes[name].append(dict(
                    class_name=detection["class_name"], confidence=score, color=colors[detection["class_name"]],
                    xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max,
                ))

    detections = [detection for found_list in found.values() for detection in found_list]
    return {
        "frames": frames,
        "bboxes": bboxes,
        "box_classes": sorted({d["class_name"] for d in detections if d["boxes"]}),
        "mask_classes": sorted({d["class_name"] for d in detections if d["mask"]}),
        "class_colors": colors,
    }


def _extract(jobs, route, video, frames_dir):
    try:
        extract_route_frames(video, frames_dir, f"{route}_", EXTRACT_STEP)
        jobs[route] = "completed"
    except Exception as error:
        print(f"Frame extraction failed for {route}: {error}")
        jobs[route] = "error"


@router.post("/extract/{route}")
def trigger_extraction(route: str, background_tasks: BackgroundTasks, request: Request):
    """Extracts one frame per EXTRACT_STEP video frames in the background."""
    workspace_dir = _workspace_for(request, route)
    video = workspace_dir / "data" / route / "video.mp4"

    if not video.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found for {route}")

    jobs = request.app.state.extraction_jobs
    if jobs.get(route) == "processing":
        return {"message": "Extraction already in progress", "status": "processing"}

    jobs[route] = "processing"
    background_tasks.add_task(_extract, jobs, route, video, workspace_dir / "frames" / f"{route}-frames")
    return {"message": "Extraction started", "status": "processing"}
