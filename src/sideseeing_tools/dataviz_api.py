import os
import io
import hashlib
import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

# Initialize the router
router = APIRouter(prefix="/api/vision", tags=["AI Vision"])

def get_safe_path(base_dir: str, filename: str) -> str:
    """Prevents directory traversal attacks by ensuring the path stays within base_dir."""
    if not base_dir:
        raise HTTPException(status_code=400, detail="Directory not configured.")
        
    full_path = os.path.abspath(os.path.join(base_dir, filename))
    base_dir_abs = os.path.abspath(base_dir)
    
    if not full_path.startswith(base_dir_abs) or not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    return full_path

@router.get("/image/{instance_name}/{filename:path}")
def get_raw_image(instance_name: str, filename: str, request: Request):
    """Serves the raw extracted frame."""
    frames_dir = request.app.state.frames_dir
    
    # Append the required suffix to find the correct folder
    target_path = os.path.join(f"{instance_name}-frames", filename)
    img_path = get_safe_path(frames_dir, target_path)
    
    return FileResponse(img_path)

@router.get("/thumb/{instance_name}/{filename:path}")
def get_thumbnail(instance_name: str, filename: str, request: Request):
    """Generates and serves a 200x200 thumbnail for the gallery, with caching."""
    frames_dir = request.app.state.frames_dir
    output_dir = getattr(request.app.state, "output_dir", "output")
    
    # Append the required suffix to find the correct folder
    target_path = os.path.join(f"{instance_name}-frames", filename)
    img_path = get_safe_path(frames_dir, target_path)
    
    # Setup thumb cache directory
    thumbs_dir = os.path.join(output_dir, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)
    
    # Create a cache hash based on the original file path
    path_hash = hashlib.md5(img_path.encode()).hexdigest()
    thumb_path = os.path.join(thumbs_dir, f"{path_hash}.jpg")
    
    if not os.path.exists(thumb_path):
        try:
            with Image.open(img_path) as img:
                img.thumbnail((200, 200))
                img.save(thumb_path, "JPEG")
        except Exception as e:
            print(f"Failed to generate thumbnail for {filename}: {e}")
            return FileResponse(img_path) # Fallback to original image if thumb fails
            
    return FileResponse(thumb_path)

@router.get("/mask/{filename:path}")
def get_mask_overlay(filename: str, request: Request, classes: str = "", instance: str = ""):
    """
    Dynamically generates a transparent PNG overlay of the segmentation masks.
    """
    frames_dir = request.app.state.frames_dir
    ai_dir = request.app.state.ai_dir
    output_dir = getattr(request.app.state, "output_dir", "output")
    
    # Append the required suffix to find the correct base image
    target_path = os.path.join(f"{instance}-frames", filename)
    img_path = get_safe_path(frames_dir, target_path)
    
    # Parse requested classes
    requested_classes = sorted([c.strip() for c in classes.split(",") if c.strip()])
    
    # Setup vis cache directory
    cache_dir = os.path.join(output_dir, "cache_vis")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache based on the image AND the specific combination of classes requested
    h_base = hashlib.md5(img_path.encode()).hexdigest()
    h_filter = hashlib.md5("".join(requested_classes).encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"{h_base}_{h_filter}.png")
    
    if os.path.exists(cache_path):
        return FileResponse(cache_path)

    # Scope to the specific instance in the preds folder
    instance_ai_dir = os.path.join(ai_dir, instance) if ai_dir and instance else None
    
    if not instance_ai_dir or not os.path.exists(instance_ai_dir):
        return _empty_transparent_png()
        
    try:
        from sideseeing_tools.visualizer import generate_mask_vis 
        img_io = generate_mask_vis(img_path, instance_ai_dir, requested_classes)
        
        if img_io:
            with open(cache_path, "wb") as f:
                f.write(img_io.getvalue())
            img_io.seek(0)
            return StreamingResponse(img_io, media_type="image/png")
        else:
            return _empty_transparent_png()
            
    except Exception as e:
        print(f"Mask generation failed: {e}")
        return _empty_transparent_png()

def _empty_transparent_png():
    """Helper to return a 1x1 transparent PNG if no mask data exists."""
    empty_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return StreamingResponse(io.BytesIO(empty_png), media_type="image/png")

@router.get("/data/{instance_name}")
def get_instance_data(instance_name: str, request: Request):
    """
    Returns the JSON payload containing the list of frames, available classes, 
    and bounding boxes for a specific SideSeeing instance.
    """
    frames_dir = getattr(request.app.state, "frames_dir", None)
    ai_dir = getattr(request.app.state, "ai_dir", None)
    
    if not frames_dir:
        raise HTTPException(status_code=400, detail="Frames directory not configured.")

    # 1. Fetch available frames
    instance_frames_dir = os.path.join(frames_dir, f"{instance_name}-frames")
    
    # If frames aren't extracted yet, return empty arrays so the UI knows to show a "No frames" state
    if not os.path.exists(instance_frames_dir):
        return {"frames": [], "bboxes": {}, "classes": []}

    valid_exts = {".jpg", ".jpeg", ".png"}
    frames = sorted([
        f for f in os.listdir(instance_frames_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    # Initialize response structures
    bboxes = {f: [] for f in frames}
    classes = set()

    # 2. Fetch AI Predictions (if configured)
    if ai_dir:
        from sideseeing_tools.visualizer import AI_Visualizer

        instance_ai_dir = os.path.join(ai_dir, instance_name)
        
        try:
            df = AI_Visualizer._get_predictions_df(instance_ai_dir)
            
            if not df.empty:
                # Extract all unique classes
                classes = set(df['class_name'].dropna().unique().tolist())

                # Map predictions to our frames
                for _, row in df.iterrows():
                    img_basename = os.path.basename(str(row['image_name']))
                    
                    # Only process if we actually have this frame extracted
                    if img_basename in bboxes:
                        # Forward-Compatibility: The SAM3 and Project Sidewalk CSVs don't 
                        # contain bounding box coords (xmin/ymin), only masks and labels.
                        # We build this logic to handle future object detection models (like YOLO)
                        # that do provide these columns.
                        if 'xmin' in df.columns and pd.notna(row.get('xmin')):
                            color_tuple = AI_Visualizer._get_color_for_class(str(row['class_name']))
                            
                            bboxes[img_basename].append({
                                "class_name": str(row['class_name']),
                                "confidence": float(row.get('confidence', 1.0)),
                                "xmin": float(row['xmin']),
                                "ymin": float(row['ymin']),
                                "xmax": float(row['xmax']),
                                "ymax": float(row['ymax']),
                                "color": f"rgb({color_tuple[0]},{color_tuple[1]},{color_tuple[2]})"
                            })
        except Exception as e:
            print(f"Error compiling instance data: {e}")

    return {
        "frames": frames,
        "bboxes": bboxes,
        "classes": sorted(list(classes))
    }