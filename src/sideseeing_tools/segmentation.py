import numpy as np

class Segmenter:
    """
    A utility class to run image segmentation using Facebook's SAM3.
    This class lazily loads heavy ML dependencies (torch, transformers) to ensure 
    that they are only required when segmentation is explicitly used.
    """
    def __init__(self, device=None):
        self._model = None
        self._processor = None
        self.device = device

    def _initialize_model(self):
        """
        Lazily initialize the SAM3 model and processor. 
        Will raise an ImportError if the required optional dependencies are not installed.
        """
        if self._model is None:
            try:
                import torch
                from transformers import Sam3Processor, Sam3Model
            except ImportError:
                raise ImportError(
                    "Optional dependencies for segmentation are not installed. "
                    "Please install them using: pip install sideseeing-tools[vision]"
                )
            
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            print(f"Loading SAM3 model on {self.device}...")
            self._processor = Sam3Processor.from_pretrained("facebook/sam3")
            self._model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)

    def segment_batch(self, images, prompts, threshold=0.5, mask_threshold=0.5):
        """
        Segments a batch of images given a list of prompts.
        
        Args:
            images (list of PIL.Image.Image): A list of PIL Image objects to segment.
            prompts (list of str): A list of text prompts corresponding to the images.
            threshold (float): The threshold for the segmentation score.
            mask_threshold (float): The threshold for binarizing the mask.
            
        Returns:
            list of dict: A list of dictionaries containing the following keys:
                - 'masks' (numpy.ndarray): The binary masks.
                - 'scores' (numpy.ndarray): The confidence scores of the masks.
                - 'boxes' (list of list of int): Bounding boxes for each mask in the format [x_min, y_min, x_max, y_max].
        """
        if not images:
            return []

        if len(images) != len(prompts):
            raise ValueError("The number of images and prompts must be the same.")

        self._initialize_model()
        import torch

        # Prepare inputs
        inputs = self._processor(
            images=images, 
            text=prompts, 
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process outputs
        results_list = self._processor.post_process_instance_segmentation(
            outputs, 
            threshold=threshold, 
            mask_threshold=mask_threshold, 
            target_sizes=[img.size[::-1] for img in images]
        )
        
        final_results = []
        for result in results_list:
            masks = result["masks"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            
            # Calculate bounding boxes [x_min, y_min, x_max, y_max] from the boolean masks
            boxes = []
            for mask in masks:
                # np.where returns (y_indices, x_indices) for 2D mask
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                    boxes.append([x_min, y_min, x_max, y_max])
                else:
                    boxes.append(None)
            
            final_results.append({
                "masks": masks,
                "scores": scores,
                "boxes": boxes
            })
            
        return final_results

    def segment_image(self, image, prompt, threshold=0.5, mask_threshold=0.5):
        """
        Segments a single image given a prompt.
        
        Args:
            image (PIL.Image.Image): The PIL Image object to segment.
            prompt (str): The text prompt.
            threshold (float): The threshold for the segmentation score.
            mask_threshold (float): The threshold for binarizing the mask.
            
        Returns:
            dict: A dictionary containing 'masks', 'scores', and 'boxes'.
        """
        results = self.segment_batch([image], [prompt], threshold, mask_threshold)
        return results[0] if results else None
