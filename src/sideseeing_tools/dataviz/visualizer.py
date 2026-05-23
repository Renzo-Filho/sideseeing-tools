import os
import io
import glob
import numpy as np
from PIL import Image
from functools import lru_cache
from typing import List, Optional
from sideseeing_tools.dataviz.adapters import PredictionAdapter

class Visualizer:
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _build_dataset_index(ai_dir: str):
        """Builds a cached index of detections and masks to avoid slow recursive lookups."""
        index = {
            'detections_csv': [],
            'masks': {}
        }
        if not ai_dir or not os.path.exists(ai_dir):
            return index
            
        for root, _, files in os.walk(ai_dir):
            for file in files:
                if file == "detections.csv":
                    index['detections_csv'].append(os.path.join(root, file))
                elif "mask" in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if file not in index['masks']:
                        index['masks'][file] = []
                    index['masks'][file].append(os.path.join(root, file))
        return index

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_predictions_df(ai_dir: str, predictions_csv: str = None):
        import pandas as pd
        
        dfs = []
        loaded_files = set()
        
        def load_and_append(csv_path):
            if os.path.exists(csv_path):
                abs_path = os.path.abspath(csv_path)
                if abs_path not in loaded_files:
                    try:
                        df = PredictionAdapter.load_and_normalize(abs_path)
                        # Track the exact directory this CSV came from to avoid mask collisions
                        df['mask_base'] = os.path.dirname(abs_path)
                        dfs.append(df)
                        loaded_files.add(abs_path)
                    except Exception as e:
                        print(f"Warning: Failed to load {csv_path}: {e}")

        if predictions_csv:
            load_and_append(predictions_csv)
            
        if ai_dir:
            # Direct files (Standard structure)
            load_and_append(os.path.join(ai_dir, "detections.csv"))
            load_and_append(os.path.join(ai_dir, "predictions.general.csv"))
                
            # Use index for nested detections.csv (Branched structure)
            index = Visualizer._build_dataset_index(ai_dir)
            for file_path in index['detections_csv']:
                load_and_append(file_path)
                
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            print(f"Warning: No supported prediction CSV found in {ai_dir} or {predictions_csv}")
            return pd.DataFrame(columns=['image_name', 'class_name', 'confidence', 'is_mask', 'mask_base'])

    @staticmethod
    @lru_cache(maxsize=5000)
    def _find_mask_file(image_name: str, ai_dir: str, mask_base: str = None) -> Optional[str]:
        """
        Finds the corresponding mask file for a given image.
        Uses the tracked mask_base to prevent filename collisions across different SAM3 classes.
        """
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        expected_names = [f"{base_name}_mask.png", f"{base_name}-mask.jpg", f"{base_name}-mask.png"]
        parent_dir = os.path.basename(os.path.dirname(image_name))
        
        # 1. Strict Path Check: Prioritize the directory where the predictions CSV was found
        if mask_base:
            for expected_mask_name in expected_names:
                # Check expected branched structure: e.g., sam3-sidewalk/route01-frames/route01_0001_mask.png
                direct_path_with_parent = os.path.join(mask_base, parent_dir, expected_mask_name)
                if os.path.exists(direct_path_with_parent):
                    return direct_path_with_parent
                
                # Check flat structure
                direct_path = os.path.join(mask_base, expected_mask_name)
                if os.path.exists(direct_path):
                    return direct_path

        # 2. Fast path: check if it's right in the ai_dir root
        for expected_mask_name in expected_names:
            direct_path = os.path.join(ai_dir, expected_mask_name)
            if os.path.exists(direct_path):
                return direct_path
                
        # 3. Index path (Fallback)
        index = Visualizer._build_dataset_index(ai_dir)
        for expected_mask_name in expected_names:
            matches = index['masks'].get(expected_mask_name, [])
            if matches:
                # Try to disambiguate strictly using the mask_base first
                if mask_base:
                    for match in matches:
                        if match.startswith(mask_base):
                            return match

                if len(matches) == 1:
                    return matches[0]
                
                # If multiple and no mask_base matched, fall back to parent directory disambiguation
                if parent_dir:
                    for match in matches:
                        if parent_dir in match:
                            return match
                return matches[0]
                
        return None

    @staticmethod
    def _get_color_for_class(class_name: str) -> tuple:
        """Generates a consistent RGBA color based on the class name."""
        if class_name.lower() == 'sidewalk':
            return (255, 165, 0, 255) # Orange for sidewalk
        if class_name.lower() == 'crosswalk':
            return (0, 255, 0, 255)   # Green for crosswalk
        
        # Deterministic random color for unknown classes
        import hashlib
        hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        return (r, g, b, 255)

    @staticmethod
    def generate_mask_vis(img_path: str, ai_dir: str, requested_classes: List[str], predictions_csv: str = None) -> Optional[io.BytesIO]:
        """
        Reads the prediction dataframe, finds the relevant mask images,
        and generates a transparent PNG overlay with the colored masks.
        """
        if not os.path.exists(img_path):
            return None
            
        image_basename = os.path.basename(img_path)
        
        # Load the cached DataFrame
        df = Visualizer._get_predictions_df(ai_dir, predictions_csv)
        
        # Filter dataframe for this specific image (using endswith to handle path variations)
        img_preds = df[df['image_name'].str.endswith(image_basename)]
        
        if img_preds.empty:
            return None
            
        overlay = None
        
        for _, row in img_preds.iterrows():
            if not row['is_mask']:
                continue
                
            class_name = str(row['class_name'])
            mask_base = str(row.get('mask_base', ''))
            
            # Skip if the frontend didn't ask for this class
            if requested_classes is not None and class_name not in requested_classes:
                continue
                
            # Pass the mask_base down to disambiguate the file search
            mask_path = Visualizer._find_mask_file(str(row['image_name']), ai_dir, mask_base if mask_base else None)
            
            if mask_path:
                try:
                    # Open the mask (assuming it's a grayscale image where > 0 is the mask)
                    with Image.open(mask_path) as m_img:
                        m_np = np.array(m_img.convert("L"))
                        
                        # Initialize transparent overlay canvas if it doesn't exist
                        if overlay is None:
                            overlay = np.zeros((m_np.shape[0], m_np.shape[1], 4), dtype=np.uint8)
                        
                        color = Visualizer._get_color_for_class(class_name)
                        mask_pixels = m_np > 0
                        
                        # Apply color instantly using boolean indexing
                        overlay[mask_pixels] = color
                except Exception as e:
                    print(f"Error processing mask {mask_path}: {e}")

        # If we successfully created an overlay, encode it to PNG bytes
        if overlay is not None:
            ol_img = Image.fromarray(overlay, "RGBA")
            img_io = io.BytesIO()
            ol_img.save(img_io, 'PNG')
            img_io.seek(0)
            return img_io
            
        return None

# Quick wrapper for the API router to use
def generate_mask_vis(img_path: str, ai_dir: str, requested_classes: List[str], predictions_csv: str = None):
    return Visualizer.generate_mask_vis(img_path, ai_dir, requested_classes, predictions_csv)