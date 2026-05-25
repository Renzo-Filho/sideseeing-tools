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

        # Create base directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)

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
            instance_frames_dir = self.frames_dir / f"{instance.name}-frames"
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
        iterator = instances if instances else dataset.iterator
        
        for prompt in prompts:
            prompt_dir = self.preds_dir / f"sam3-{prompt}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = prompt_dir / "detections.csv"
            file_exists = csv_path.exists()
            
            with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                if not file_exists:
                    writer.writerow(["image_name", "relative_path", "prompt", "num_detections", "scores", "boxes"])
                
                for instance in iterator:
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
        print(f"[Workspace] Build pipeline complete! Everything is ready in {self.output_dir}.")
