import pandas as pd
import os

class PredictionAdapter:

    @staticmethod
    def load_and_normalize(csv_path: str) -> pd.DataFrame:
        """
        Reads a CSV, detects its format, and normalizes it to a standard schema.
        Standard Schema: ['image_name', 'class_name', 'confidence', 'is_mask']
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Prediction file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Detect Format 1: Project Sidewalk (Wide Format)
        # columns: image, crosswalk, curbramp, surfaceproblem, obstacle
        if 'image' in df.columns and 'crosswalk' in df.columns:
            return PredictionAdapter._normalize_project_sidewalk(df)

        # Detect Format 2: SAM3 Detections (Long Format)
        # columns: image_name, class_name, num_detections
        elif 'image_name' in df.columns and 'num_detections' in df.columns:
            return PredictionAdapter._normalize_sam3(df)

        else:
            raise ValueError(f"Unrecognized prediction format in {csv_path}. Please check the documentation for supported schemas.")

    @staticmethod
    def _normalize_project_sidewalk(df: pd.DataFrame) -> pd.DataFrame:
        """Converts wide binary format to standard long format."""
        # Melt the dataframe: turn columns into rows
        id_vars = ['image']
        value_vars = [col for col in df.columns if col != 'image']
        
        melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='class_name', value_name='presence')
        
        # Keep only rows where the class was detected (presence == 1)
        detected = melted[melted['presence'] > 0].copy()
        
        # Rename and standardize
        detected.rename(columns={'image': 'image_name'}, inplace=True)
        detected['confidence'] = 1.0 # Assuming 1.0 since it's a binary flag
        detected['is_mask'] = False  # Project sidewalk provides bboxes/flags, not masks
        
        return detected[['image_name', 'class_name', 'confidence', 'is_mask']]

    @staticmethod
    def _normalize_sam3(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes SAM3 detection format."""
        normalized = df[df['num_detections'] > 0].copy()
        
        normalized['confidence'] = 1.0 # Add default if scores aren't present
        normalized['is_mask'] = True   # SAM3 provides segmentation masks
        
        return normalized[['image_name', 'class_name', 'confidence', 'is_mask']]