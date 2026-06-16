import os
import csv
import shutil
from pathlib import Path
from PIL import Image
from sideseeing_tools.sideseeing import SideSeeingDS

class SideSeeingWorkspace:
    """
    A manager class that transforms a raw SideSeeingDS dataset into a 
    standardized processed directory structure ready for downstream tasks 
    like the Dataviz Server.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.frames_dir = self.output_dir / "frames"
        self.preds_dir = self.output_dir / "preds"
        self.geomatching_dir = self.output_dir / "geomatching"
        self.routes_gpkg_dir = self.geomatching_dir / "routes_gpkg"

        # Create base directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.geomatching_dir.mkdir(parents=True, exist_ok=True)
        self.routes_gpkg_dir.mkdir(parents=True, exist_ok=True)

    def setup_data_directory(self, dataset: SideSeeingDS, use_symlinks: bool = True):
        """
        Safely populates the `data/` directory and `metadata.csv` by copying 
        or symlinking from the raw SideSeeingDS dataset. 
        This ensures raw sensor data remains intact for the report module.
        """
        print(f"[Workspace] Setting up workspace data directory safely in {self.data_dir}...")
        
        # Link or copy dataset metadata.csv
        if dataset.metadata_path and os.path.exists(dataset.metadata_path):
            dest_metadata = self.output_dir / "metadata.csv"
            if not dest_metadata.exists():
                if use_symlinks:
                    os.symlink(os.path.abspath(dataset.metadata_path), dest_metadata)
                else:
                    shutil.copy2(dataset.metadata_path, dest_metadata)

        # Link or copy route instances
        for instance in dataset.iterator:
            if not instance.name:
                print(f"[Workspace Warning] Skipping instance with empty name (likely a file in the root dir).")
                continue
            dest_instance_dir = self.data_dir / instance.name
            if not dest_instance_dir.exists():
                if use_symlinks:
                    os.symlink(os.path.abspath(instance.path), dest_instance_dir, target_is_directory=True)
                else:
                    shutil.copytree(instance.path, dest_instance_dir)

        print(f"[Workspace] Data directory setup complete. Linked {dataset.size} routes.")

    def extract_frames(self, dataset: SideSeeingDS, extract_step: int, instances: list = None):
        """
        Extracts video frames from the dataset instances into the standardized frames/ directory.
        
        Args:
            dataset: The SideSeeingDS instance.
            extract_step: The step rate to extract frames.
            instances: Optional list of specific SideSeeingInstance objects to extract. If None, extracts all.
        """
        iterator = instances if instances else dataset.iterator
        
        for instance in iterator:
            if not instance.name:
                continue
                
            instance_frames_dir = self.frames_dir / f"{instance.name}-frames"
            
            if instance_frames_dir.exists() and any(instance_frames_dir.iterdir()):
                print(f"[Workspace] Frames for {instance.name} already exist. Skipping extraction...")
                continue
                
            instance_frames_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[Workspace] Extracting frames for route: {instance.name} (Step: {extract_step})...")
            # SideSeeingInstance.extract_frames(output_dir, step, prefix, left_zeros)
            instance.extract_frames(
                output_dir=str(instance_frames_dir),
                step=extract_step,
                prefix=f"{instance.name}_"
            )
            
        print(f"[Workspace] Frame extraction complete for all requested instances.")

    def generate_segmentation(self, dataset: SideSeeingDS, prompts: list, instances: list = None, batch_size: int = 8):
        """
        Runs the SAM3 Segmenter on the extracted frames for the specified prompts.
        Generates binary masks and detections.csv in the preds/ directory.
        Includes resume capability by skipping existing masks.
        """
        try:
            from sideseeing_tools.segmentation import Segmenter
        except ImportError:
            raise ImportError(
                "Optional dependencies for segmentation are not installed. "
                "Please install them using: pip install sideseeing-tools[vision]"
            )

        segmenter = Segmenter()
        iterator = instances if instances else list(dataset.iterator)
        
        for prompt in prompts:
            prompt_dir = self.preds_dir / f"sam3-{prompt}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = prompt_dir / "detections.csv"
            file_exists = csv_path.exists() and os.path.getsize(csv_path) > 0
            
            with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                if not file_exists:
                    writer.writerow(["image_name", "relative_path", "prompt", "num_detections", "scores", "boxes"])
                
                for instance in iterator:
                    if not instance.name:
                        continue
                        
                    instance_frames_dir = self.frames_dir / f"{instance.name}-frames"
                    if not instance_frames_dir.exists():
                        print(f"[Workspace Warning] Frames directory not found for {instance.name}. Did you run extract_frames() first? Skipping...")
                        continue
                        
                    prompt_instance_frames_dir = prompt_dir / f"{instance.name}-frames"
                    prompt_instance_frames_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Discover frames
                    all_frames = sorted(instance_frames_dir.glob("*.jpg")) + sorted(instance_frames_dir.glob("*.png"))
                    
                    # Filter frames to process (Resume Capability)
                    frames_to_process = []
                    for frame_path in all_frames:
                        expected_mask = prompt_instance_frames_dir / f"{frame_path.stem}-mask.png"
                        if not expected_mask.exists():
                            frames_to_process.append(frame_path)
                            
                    if not frames_to_process:
                        continue
                        
                    print(f"[Workspace] Found {len(frames_to_process)} un-segmented frames for {instance.name} with prompt '{prompt}'. Segmenting...")
                    
                    for i in range(0, len(frames_to_process), batch_size):
                        batch_paths = frames_to_process[i:i+batch_size]
                        batch_images = []
                        valid_paths = []
                        
                        for path in batch_paths:
                            try:
                                batch_images.append(Image.open(path).convert("RGB"))
                                valid_paths.append(path)
                            except Exception as e:
                                print(f"[Workspace Error] Error opening image {path}: {e}")
                                
                        if not batch_images:
                            continue
                            
                        # Run inference
                        print(f"[Workspace] Submitting batch of {len(batch_images)} images to Segmenter...")
                        results = segmenter.segment_batch(batch_images, [prompt] * len(batch_images))
                        
                        # Save results
                        for path, result in zip(valid_paths, results):
                            masks = result["masks"]
                            scores = result["scores"]
                            boxes = result["boxes"]
                            
                            mask_name = f"{path.stem}-mask.png"
                            mask_path = prompt_instance_frames_dir / mask_name
                            
                            if masks.shape[0] > 0:
                                import numpy as np
                                combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                                Image.fromarray(combined_mask).save(mask_path)
                                scores_str = ";".join([f"{s:.4f}" for s in scores])
                                boxes_str = ";".join([f"{b[0]},{b[1]},{b[2]},{b[3]}" for b in boxes if b is not None])
                            else:
                                empty_mask = Image.new("L", batch_images[0].size, 0)
                                empty_mask.save(mask_path)
                                scores_str = ""
                                boxes_str = ""
                                
                            rel_path = f"{instance.name}-frames/{path.name}"
                            writer.writerow([path.name, rel_path, prompt, masks.shape[0], scores_str, boxes_str])
                        
                        csv_file.flush()

    def anonymize_frames(self, dataset: SideSeeingDS, instances: list = None, method: str = "yolo", batch_size: int = 8, blur_radius: int = 15):
        """
        Post-processing step that scans the frames directory, detects sensitive areas (faces/plates),
        and overwrites the frames with blurred versions.
        """
        try:
            from sideseeing_tools.anonymization import Anonymizer
        except ImportError:
            raise ImportError(
                "Optional dependencies for vision are not installed. "
                "Please install them using: pip install sideseeing-tools[vision]"
            )

        anonymizer = Anonymizer(method=method, blur_radius=blur_radius)
        iterator = instances if instances else dataset.iterator

        for instance in iterator:
            if not instance.name:
                continue
                
            instance_frames_dir = self.frames_dir / f"{instance.name}-frames"
            if not instance_frames_dir.exists():
                print(f"[Workspace Warning] Frames directory not found for {instance.name}. Skipping anonymization.")
                continue

            all_frames = sorted(instance_frames_dir.glob("*.jpg")) + sorted(instance_frames_dir.glob("*.png"))
            if not all_frames:
                continue
                
            print(f"[Workspace] Scanning frames for {instance.name}. Found {len(all_frames)} total frames to anonymize using {method.upper()}...")
            
            for i in range(0, len(all_frames), batch_size):
                batch_paths = all_frames[i:i+batch_size]
                batch_images = []
                valid_paths = []
                
                for path in batch_paths:
                    try:
                        batch_images.append(Image.open(path).convert("RGB"))
                        valid_paths.append(path)
                    except Exception as e:
                        print(f"[Workspace Error] Error opening image {path} for anonymization: {e}")
                        
                if not batch_images:
                    continue
                    
                blurred_images = anonymizer.anonymize_batch(batch_images)
                
                for path, blurred_img in zip(valid_paths, blurred_images):
                    # Overwrite original frame with blurred version
                    blurred_img.save(path)

    def generate_map_matched_routes(self, dataset: SideSeeingDS, instances: list = None):
        """
        Generates Map-Matched routes for the dataset using the OSRM API.
        """
        try:
            from sideseeing_tools.mapping import MapMatcher
        except ImportError:
            raise ImportError("mapping module not found.")
            
        print(f"[Workspace] Generating map-matched routes in {self.routes_gpkg_dir}...")
        matcher = MapMatcher()
        iterator = instances if instances else dataset.iterator
        
        for instance in iterator:
            if not instance.name:
                continue
            matcher.match_trace(instance, output_dir=str(self.routes_gpkg_dir))

    def _create_event_dict(self, event_df, event_id, instance_name):
        from sideseeing_tools import utils
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        
        if event_df.empty:
            return None
            
        start_row = event_df.iloc[0]
        end_row = event_df.iloc[-1]
        
        lat_mean = event_df['latitude'].mean()
        lon_mean = event_df['longitude'].mean()
        
        center_latitude = lat_mean
        center_longitude = lon_mean
        
        gpkg_path = self.routes_gpkg_dir / f"{instance_name}.gpkg"
        if gpkg_path.exists():
            try:
                gdf = gpd.read_file(gpkg_path)
                if not gdf.empty and 'geometry' in gdf.columns:
                    route_line = gdf.geometry.unary_union
                    raw_point = Point(lon_mean, lat_mean)
                    snapped_point, _ = nearest_points(route_line, raw_point)
                    center_latitude = snapped_point.y
                    center_longitude = snapped_point.x
            except Exception as e:
                print(f"[Workspace Warning] Error snapping to route for {instance_name}: {e}")
        
        # Haversine distance
        length_meters = utils.calculate_haversine_distance(
            start_row['latitude'], start_row['longitude'],
            end_row['latitude'], end_row['longitude']
        ) * 1000.0 # utils returns km
        
        return {
            'event_id': f"EVT-{event_id:05d}",
            'start_image': start_row['image_name'],
            'end_image': end_row['image_name'],
            'center_latitude': center_latitude,
            'center_longitude': center_longitude,
            'feature': start_row['prompt'],
            'length_meters': length_meters,
            'instance_name': instance_name
        }

    def generate_sidewalk_assessment_events(self, dataset: SideSeeingDS):
        """
        Aggregates SAM3 detections into spatial map events based on continuous video frames,
        syncing timestamps from frames to the raw GPS data.
        """
        import pandas as pd
        import re
        from datetime import timedelta
        
        print("[Workspace] Generating sidewalk assessment events...")
        map_events = []
        event_id_counter = 1
        
        for prompt_dir in self.preds_dir.glob("sam3-*"):
            detections_csv = prompt_dir / "detections.csv"
            if not detections_csv.exists() or os.path.getsize(detections_csv) == 0:
                continue
                
            print(f"[Workspace] Processing detections from {prompt_dir.name}")
            df_det = pd.read_csv(detections_csv)
            
            df_det['instance_name'] = df_det['relative_path'].apply(lambda x: x.split('-frames')[0] if isinstance(x, str) else None)
            
            for instance_name, group in df_det.groupby('instance_name'):
                instance = dataset.instances.get(instance_name)
                if not instance:
                    print(f"[Workspace Warning] Instance {instance_name} not found in dataset. Skipping.")
                    continue
                    
                df_gps = instance.geolocation_points
                if df_gps is None or df_gps.empty:
                    print(f"[Workspace Warning] No GPS data for {instance_name}. Skipping event generation.")
                    continue
                    
                df_gps = df_gps.copy()
                df_gps['Datetime UTC'] = pd.to_datetime(df_gps['Datetime UTC']).dt.tz_localize(None)
                df_gps = df_gps.sort_values('Datetime UTC')
                
                media_start_time = instance.media_start_time
                video_fps = float(instance.metadata.get('video_fps', 30.0))
                
                frame_data = []
                for _, row in group.iterrows():
                    image_name = row['image_name']
                    match = re.search(r'_(\d+)(?:_ms)?\.(?:jpg|png)$', image_name)
                    if not match:
                        continue
                    
                    frame_idx = int(match.group(1))
                    frame_time = media_start_time + timedelta(seconds=(frame_idx / video_fps))
                    
                    num_detections = row.get('num_detections', 0)
                    if num_detections > 0:
                        frame_data.append({
                            'image_name': image_name,
                            'frame_idx': frame_idx,
                            'frame_time': frame_time,
                            'prompt': row['prompt']
                        })
                
                if not frame_data:
                    continue
                    
                df_frames = pd.DataFrame(frame_data)
                df_frames = df_frames.sort_values('frame_time')
                
                df_merged = pd.merge_asof(
                    df_frames,
                    df_gps[['Datetime UTC', 'latitude', 'longitude']],
                    left_on='frame_time',
                    right_on='Datetime UTC',
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=5)
                )
                
                df_merged = df_merged.dropna(subset=['latitude', 'longitude'])
                if df_merged.empty:
                    continue
                    
                df_merged = df_merged.sort_values('frame_idx').reset_index(drop=True)
                
                if len(df_merged) > 1:
                    gaps = df_merged['frame_idx'].diff().dropna()
                    if len(gaps) > 0:
                        common_gap = gaps.value_counts().idxmax()
                    else:
                        common_gap = 30
                else:
                    common_gap = 30
                    
                max_gap = int(common_gap * 1.5)
                
                current_event = []
                for idx, row in df_merged.iterrows():
                    if not current_event:
                        current_event.append(row)
                        continue
                        
                    prev_row = current_event[-1]
                    if (row['frame_idx'] - prev_row['frame_idx']) <= max_gap and row['prompt'] == prev_row['prompt']:
                        current_event.append(row)
                    else:
                        event_df = pd.DataFrame(current_event)
                        event_dict = self._create_event_dict(event_df, event_id_counter, instance_name)
                        if event_dict:
                            map_events.append(event_dict)
                            event_id_counter += 1
                        
                        current_event = [row]
                
                if current_event:
                    event_df = pd.DataFrame(current_event)
                    event_dict = self._create_event_dict(event_df, event_id_counter, instance_name)
                    if event_dict:
                        map_events.append(event_dict)
                        event_id_counter += 1
                        
        if map_events:
            df_out = pd.DataFrame(map_events)
            out_path = self.geomatching_dir / "map_events.csv"
            df_out.to_csv(out_path, index=False)
            print(f"[Workspace] Generated {len(map_events)} sidewalk assessment events in {out_path}")
        else:
            print("[Workspace] No events generated.")

    def build_workspace(self, dataset: SideSeeingDS, prompts: list, extract_step: int = 30, use_symlinks: bool = True, anonymize_method: str = None):
        """
        Master function to execute the entire pipeline on the whole dataset.
        Sets up the directory, extracts frames, generates segmentations, and optionally anonymizes frames.
        """
        self.setup_data_directory(dataset, use_symlinks)
        self.extract_frames(dataset, extract_step)
        
        if anonymize_method:
            self.anonymize_frames(dataset, method=anonymize_method)
            
        self.generate_segmentation(dataset, prompts)
        
        self.generate_map_matched_routes(dataset)
        self.generate_sidewalk_assessment_events(dataset)
        
        print(f"[Workspace] Build pipeline complete! Everything is ready in {self.output_dir}.")
