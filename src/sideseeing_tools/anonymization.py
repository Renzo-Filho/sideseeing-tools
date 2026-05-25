from PIL import Image, ImageFilter
import numpy as np

class Anonymizer:
    """
    A utility class to anonymize images (blurring sensitive regions like faces and vehicles).
    Provides a light option (YOLOv8) and a heavy option (SAM3).
    Lazily loads dependencies.
    """
    def __init__(self, method="yolo", device=None, blur_radius=15):
        """
        Args:
            method (str): 'yolo' (light) or 'sam3' (heavy).
            device (str): 'cuda' or 'cpu'. Auto-detected if None.
            blur_radius (int): The intensity of the Gaussian blur.
        """
        if method not in ["yolo", "sam3"]:
            raise ValueError("Anonymizer method must be either 'yolo' or 'sam3'.")
        
        self.method = method
        self.device = device
        self.blur_radius = blur_radius
        self._model = None

    def _initialize_model(self):
        """
        Lazily initialize the chosen ML model.
        """
        if self._model is not None:
            return

        try:
            import torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            raise ImportError(
                "Optional dependencies for vision are not installed. "
                "Please install them using: pip install sideseeing-tools[vision]"
            )

        if self.method == "yolo":
            try:
                from ultralytics import YOLO
                print(f"Loading YOLOv8 model on {self.device}...")
                self._model = YOLO('yolov8n.pt') # Lightweight YOLO model
            except ImportError:
                raise ImportError(
                    "Ultralytics is not installed. "
                    "Please install it using: pip install sideseeing-tools[vision] or pip install ultralytics"
                )
        elif self.method == "sam3":
            from sideseeing_tools.segmentation import Segmenter
            print(f"[Anonymizer] Loading SAM3 Segmenter for heavy anonymization...")
            self._model = Segmenter(device=self.device)

    def _apply_blur_to_boxes(self, image: Image.Image, boxes: list) -> Image.Image:
        """
        Applies Gaussian blur to specific bounding boxes in an image.
        """
        result_image = image.copy()
        for box in boxes:
            if box is None:
                continue
            x_min, y_min, x_max, y_max = [int(v) for v in box]
            
            # Ensure box is within image bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(image.width, x_max), min(image.height, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue

            # Crop the region, blur it, and paste it back
            box_tuple = (x_min, y_min, x_max, y_max)
            region = result_image.crop(box_tuple)
            blurred_region = region.filter(ImageFilter.GaussianBlur(self.blur_radius))
            result_image.paste(blurred_region, box_tuple)

        return result_image

    def _apply_blur_to_masks(self, image: Image.Image, masks: np.ndarray) -> Image.Image:
        """
        Applies Gaussian blur using exact segmentation masks.
        """
        if masks.shape[0] == 0:
            return image
            
        result_image = image.copy()
        blurred_image = image.filter(ImageFilter.GaussianBlur(self.blur_radius))
        
        # Combine all masks into a single 2D mask
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        mask_image = Image.fromarray(combined_mask).convert("L")
        
        # Paste the blurred image onto the result image using the mask
        result_image.paste(blurred_image, (0, 0), mask_image)
        return result_image

    def anonymize_batch(self, images: list, yolo_classes: list = None, sam3_prompts: list = None):
        """
        Anonymizes a batch of images using the selected model.
        
        Args:
            images: List of PIL.Image.Image objects.
            yolo_classes: List of class IDs to blur for YOLO. 
                          Defaults to [0, 2, 3, 5, 7] (person, car, motorcycle, bus, truck).
            sam3_prompts: List of prompts to blur for SAM3. 
                          Defaults to ["person", "face", "license plate", "car"].
                          
        Returns:
            List of blurred PIL.Image.Image objects.
        """
        if not images:
            return []

        print(f"[Anonymizer] Initializing anonymization for a batch of {len(images)} images...")
        self._initialize_model()
        blurred_images = []

        if self.method == "yolo":
            if yolo_classes is None:
                # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
                yolo_classes = [0, 2, 3, 5, 7]
                
            print(f"[Anonymizer] Running YOLO inference for classes {yolo_classes}...")
            # YOLO batch inference
            results = self._model(images, classes=yolo_classes, verbose=False, device=self.device)
            
            for i, result in enumerate(results):
                # Extract bounding boxes
                boxes = result.boxes.xyxy.cpu().numpy().tolist()
                blurred = self._apply_blur_to_boxes(images[i], boxes)
                blurred_images.append(blurred)

        elif self.method == "sam3":
            if sam3_prompts is None:
                sam3_prompts = ["person", "face", "license plate", "car"]
                
            print(f"[Anonymizer] Running SAM3 inference with prompts: {sam3_prompts}...")
            # SAM3 batch inference (we must prompt for each image)
            # Since segment_batch takes 1 prompt per image, we need to iterate over prompts or images.
            # To find multiple prompts per image, we will iterate over prompts and accumulate masks.
            for img in images:
                all_masks = []
                for prompt in sam3_prompts:
                    res = self._model.segment_image(img, prompt)
                    if res and res["masks"].shape[0] > 0:
                        all_masks.append(res["masks"])
                        
                if all_masks:
                    # Concatenate masks along the first dimension (N, H, W)
                    combined_masks = np.concatenate(all_masks, axis=0)
                    blurred = self._apply_blur_to_masks(img, combined_masks)
                    blurred_images.append(blurred)
                else:
                    blurred_images.append(img)

        return blurred_images

    def anonymize_image(self, image: Image.Image, yolo_classes: list = None, sam3_prompts: list = None):
        """
        Anonymizes a single image.
        """
        results = self.anonymize_batch([image], yolo_classes, sam3_prompts)
        return results[0] if results else None
