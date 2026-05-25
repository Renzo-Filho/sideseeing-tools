# Image Segmentation with SideSeeing Tools

The `sideseeing-tools` library now includes an advanced image segmentation module powered by Facebook's SAM3. This feature allows you to extract precise masks and bounding boxes from images using text prompts (e.g., "sidewalk", "pothole", "person").

## Installation

Because machine learning models are large and require heavy dependencies (like `torch` and `transformers`), the segmentation feature is completely optional. 

If you want to use the `Segmenter`, you must install the library with the `vision` extra:

```bash
pip install sideseeing-tools[vision]
```

*Note: If you try to use the `Segmenter` without installing the optional dependencies, it will safely raise an `ImportError` instructing you to install them.*

## Using the Segmenter

The `Segmenter` class lazily loads the SAM3 model. It is designed to be highly efficient, allowing you to segment single images or batches of images.

### 1. Segmenting a Single Image

You can segment a single image by passing a `PIL.Image` and a text prompt to `segment_image()`.

```python
from PIL import Image
from sideseeing_tools.segmentation import Segmenter

# 1. Initialize the Segmenter (Automatically detects CUDA/CPU)
segmenter = Segmenter()

# 2. Load your image
image = Image.open("path/to/frame_001.jpg").convert("RGB")

# 3. Segment the image
result = segmenter.segment_image(image, prompt="sidewalk")

if result:
    print(f"Masks found: {result['masks'].shape[0]}")
    print(f"Scores: {result['scores']}")
    print(f"Bounding Boxes: {result['boxes']}")
```

### 2. Segmenting a Batch of Images (Recommended)

To maximize GPU utilization, it is recommended to process images in batches. Pass a list of images and a corresponding list of prompts.

```python
from PIL import Image
from sideseeing_tools.segmentation import Segmenter

segmenter = Segmenter()

# Load a batch of images
images = [
    Image.open("frame_001.jpg").convert("RGB"),
    Image.open("frame_002.jpg").convert("RGB")
]

# Prompts must match the length of the images list
prompts = ["sidewalk", "pothole"]

# Run batch segmentation
results = segmenter.segment_batch(images, prompts)

for i, res in enumerate(results):
    print(f"Image {i+1} has {len(res['boxes'])} detections.")
```

## Understanding the Output

The `Segmenter` returns a dictionary (or a list of dictionaries for batches) containing:

1. **`masks`**: A `numpy.ndarray` containing the boolean masks. Shape is usually `(N, H, W)` where `N` is the number of detected instances.
2. **`scores`**: A `numpy.ndarray` of confidence scores for each mask.
3. **`boxes`**: A list of bounding boxes in the format `[x_min, y_min, x_max, y_max]`. If a mask is empty, the box will be `None`.

### Example: Saving the Mask

You can easily convert the returned masks into images:

```python
import numpy as np

# Assuming result['masks'] has shape (N, H, W)
if result['masks'].shape[0] > 0:
    # Combine all masks into a single image
    combined_mask = np.any(result['masks'], axis=0).astype(np.uint8) * 255
    mask_image = Image.fromarray(combined_mask)
    mask_image.save("frame_001_mask.png")
```
