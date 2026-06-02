import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sideseeing_tools.export import Report
from sideseeing_tools import media
from sideseeing_tools.dataviz.api import router as dataviz_router

app = FastAPI(title="SideSeeing Server", version="0.10.1")
app.include_router(dataviz_router)

# In-memory dictionary to track extraction progress
extraction_jobs = {}

@app.on_event("shutdown")
def cleanup_cache():
    import shutil
    output_dir = getattr(app.state, "output_dir", None)
    if output_dir:
        cache_dir = os.path.join(output_dir, "cache")
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"Cleaned up temporary cache directory: {cache_dir}")
            except Exception as e:
                print(f"Failed to clean up cache directory: {e}")

def background_extractor(video_path: str, output_dir: str, instance_name: str, fps: int = 1):
    """
    Runs in a background thread so it doesn't block the FastAPI event loop.
    """
    extraction_jobs[instance_name] = {"status": "processing"}
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Using the existing media.py function
        media.extract_frames(
            source_path=video_path,
            target_dir=output_dir,
            step=30, # Assuming 30fps video, step=30 means 1 frame per sec
            prefix=f"{instance_name}_"
        )
        extraction_jobs[instance_name] = {"status": "completed"}
    except Exception as e:
        extraction_jobs[instance_name] = {"status": "error", "message": str(e)}



# --- FastAPI Routes (Defined globally) ---

@app.get("/")
async def serve_index(request: Request):
    """Serves the main static report HTML."""
    output_dir = getattr(request.app.state, "output_dir", "output")
    index_path = os.path.join(output_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found.")

@app.post("/api/dataviz/extract/{instance_name}")
async def trigger_extraction(instance_name: str, background_tasks: BackgroundTasks, request: Request):
    """Triggers background extraction of frames for a given video."""
    input_dir = request.app.state.input_dir
    frames_dir = request.app.state.frames_dir or os.path.join(request.app.state.output_dir, "extracted_frames")
    
    video_path = os.path.join(input_dir, instance_name, "video.mp4")
    output_dir = os.path.join(frames_dir, f"{instance_name}-frames")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found for {instance_name}")

    # Don't trigger if already extracting or completed
    if extraction_jobs.get(instance_name, {}).get("status") in ["processing", "completed"]:
        return {"message": "Extraction already processing or completed", "status": extraction_jobs[instance_name]["status"]}

    background_tasks.add_task(background_extractor, video_path, output_dir, instance_name)
    return {"message": "Extraction started", "output_dir": output_dir}



# --- Server Start Logic ---

def start_server(input_dir: str, output_dir: str = None, port: int = 5000, host: str = "0.0.0.0"):
    """
    Starts the local FastAPI server. Can be called via CLI or Python script.
    """
    if output_dir is None:
        output_dir = input_dir

    data_dir = os.path.join(input_dir, "data")
    frames_dir = os.path.join(input_dir, "frames")
    ai_dir = os.path.join(input_dir, "preds")
    predictions_csv = os.path.join(ai_dir, "predictions.csv")
    metadata_csv = os.path.join(input_dir, "metadata.csv")

    # Store these globally in the app state
    app.state.input_dir = data_dir
    app.state.output_dir = output_dir
    app.state.ai_dir = ai_dir
    app.state.frames_dir = frames_dir
    app.state.masks_dir = None
    app.state.predictions_csv = predictions_csv
    app.state.metadata_csv = metadata_csv

    index_path = os.path.join(output_dir, "index.html")
    static_dir = os.path.join(output_dir, "static")
    data_dir = os.path.join(output_dir, "data")

    # Safely get geomatching paths
    geomatching_dir = os.path.join(output_dir, "geomatching")
    events_csv_path = os.path.join(geomatching_dir, "map_events.csv")
    gpkg_dir = os.path.join(geomatching_dir, "routes_gpkg")

    # Check if the report exists. If not, generate it.
    if not os.path.exists(index_path) or not os.path.exists(static_dir):
        print(f"Report not found at {output_dir}. Generating base report...")
        r = Report()
        r.generate_report(
            input_dir=input_dir, 
            output_dir=output_dir, 
            metadata_csv=metadata_csv,
            events_csv_path=events_csv_path if os.path.exists(events_csv_path) else None,
            gpkg_dir=gpkg_dir if os.path.exists(gpkg_dir) else None,
            image_dir=frames_dir if os.path.exists(frames_dir) else None
        )
        print("Base report generated successfully.")
    else:
        print(f"Existing report found at {output_dir}. Booting server...")
        # Always update static assets on boot for better developer experience
        Report()._copy_static_assets(output_dir)
        
        # Fast HTML cache-busting update (so we don't need to rebuild the massive JSONs)
        import re
        import time
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read()
                new_buster = str(int(time.time()))
                content = re.sub(r'(\.js\?v=)\d+', r'\g<1>' + new_buster, content)
                content = re.sub(r'(\.css\?v=)\d+', r'\g<1>' + new_buster, content)
                with open(index_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"Failed to update cache-buster: {e}")

    # Safely mount static directories if they exist
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    if os.path.exists(data_dir):
        app.mount("/data", StaticFiles(directory=data_dir), name="data")
        
    frames_export_dir = os.path.join(output_dir, "frames")
    if os.path.exists(frames_export_dir):
        app.mount("/frames", StaticFiles(directory=frames_export_dir), name="frames")
    elif os.path.exists(frames_dir):
        app.mount("/frames", StaticFiles(directory=frames_dir), name="frames")

    print(f"Starting SideSeeing server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

