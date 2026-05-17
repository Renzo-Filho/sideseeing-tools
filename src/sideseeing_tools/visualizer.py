import os
import io
import glob
import numpy as np
from PIL import Image
from functools import lru_cache
from typing import List, Optional
from sideseeing_tools.dataviz_adapters import PredictionAdapter

class AI_Visualizer:
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_predictions_df(ai_dir: str):
        import pandas as pd
        
        sam3_csv = os.path.join(ai_dir, "detections.csv")
        ps_csv = os.path.join(ai_dir, "predictions.general.csv")
        
        dfs = []
        if os.path.exists(sam3_csv):
            dfs.append(PredictionAdapter.load_and_normalize(sam3_csv))
        if os.path.exists(ps_csv):
            dfs.append(PredictionAdapter.load_and_normalize(ps_csv))
            
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            print(f"Warning: No supported prediction CSV found in {ai_dir}")
            return pd.DataFrame(columns=['image_name', 'class_name', 'confidence', 'is_mask'])

    @staticmethod
    @lru_cache(maxsize=500)
    def _find_mask_file(image_name: str, ai_dir: str) -> Optional[str]:
        """
        Finds the corresponding mask PNG file for a given image.
        Uses caching so we don't spam os.walk on the filesystem.
        """
        # Assuming masks are saved with a _mask.png suffix
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        expected_mask_name = f"{base_name}_mask.png"
        
        # Fast path: check if it's right in the ai_dir
        direct_path = os.path.join(ai_dir, expected_mask_name)
        if os.path.exists(direct_path):
            return direct_path
            
        # Slow path: Recursive search (happens only once per image due to cache)
        matches = glob.glob(os.path.join(ai_dir, "**", expected_mask_name), recursive=True)
        if matches:
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
    def generate_mask_vis(img_path: str, ai_dir: str, requested_classes: List[str]) -> Optional[io.BytesIO]:
        """
        Reads the prediction dataframe, finds the relevant mask images,
        and generates a transparent PNG overlay with the colored masks.
        """
        if not os.path.exists(img_path):
            return None
            
        # Get the relative image name to match the CSV format (e.g., folder/image_0001.jpg)
        # Note: Depending on how frames are extracted, you might just need the basename
        image_basename = os.path.basename(img_path)
        
        # Load the cached DataFrame
        df = AI_Visualizer._get_predictions_df(ai_dir)
        
        # Filter dataframe for this specific image (using endswith to handle path variations)
        img_preds = df[df['image_name'].str.endswith(image_basename)]
        
        if img_preds.empty:
            return None
            
        overlay = None
        
        for _, row in img_preds.iterrows():
            if not row['is_mask']:
                continue
                
            class_name = str(row['class_name'])
            
            # Skip if the frontend didn't ask for this class
            if requested_classes is not None and class_name not in requested_classes:
                continue
                
            mask_path = AI_Visualizer._find_mask_file(str(row['image_name']), ai_dir)
            
            if mask_path:
                try:
                    # Open the mask (assuming it's a grayscale image where > 0 is the mask)
                    with Image.open(mask_path) as m_img:
                        m_np = np.array(m_img.convert("L"))
                        
                        # Initialize transparent overlay canvas if it doesn't exist
                        if overlay is None:
                            overlay = np.zeros((m_np.shape[0], m_np.shape[1], 4), dtype=np.uint8)
                        
                        color = AI_Visualizer._get_color_for_class(class_name)
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
def generate_mask_vis(img_path: str, ai_dir: str, requested_classes: List[str]):
    return AI_Visualizer.generate_mask_vis(img_path, ai_dir, requested_classes)