# SideSeeingWorkspace Module

The `SideSeeingWorkspace` module acts as a bridge between your raw SideSeeing dataset (`SideSeeingDS`) and downstream tools like the Dataviz Server. It automates the extraction, transformation, and structuring of your data without altering or duplicating your original dataset.

## Key Features

1. **Strict Dataviz Structure Generation**: Automatically generates the exact directory structure expected by the Dataviz Server (`data/`, `frames/`, `preds/`), saving you from manually organizing files.
2. **Safe Data Linking (Symlinks)**: Instead of copying gigabytes of raw video and sensor data, the workspace uses symbolic links to reference the original files. This ensures your existing reporting scripts can seamlessly run on the generated workspace without any changes.
3. **Built-in Anonymization**: Protects PII by automatically detecting and blurring sensitive regions (faces, license plates) using either a fast YOLO model or a highly precise SAM3 model.
4. **Granular Control**: Run the entire pipeline at once or break it down into specific steps (e.g., extracting frames today, running segmentation tomorrow).
5. **Resumable Segmentations**: Segmenting large datasets takes time. If you stop the process halfway, running it again will automatically skip frames that have already been masked, picking up right where it left off.

## Output Structure

The workspace creates the following standardized layout:

```text
your_workspace_dir/
├── metadata.csv             <-- Linked from raw dataset
├── data/
│   └── route01/             <-- Safely linked from raw dataset
│       ├── video.mp4
│       └── sensors.three.csv
├── frames/
│   └── route01-frames/      <-- Extracted frames
│       ├── route01_0001.jpg 
│       └── ...
└── preds/
    └── sam3-crosswalk/      <-- Model predictions
        ├── detections.csv   <-- Contains prompts, scores, and bounding boxes
        └── route01-frames/
            └── route01_0001-mask.png
```

---

## How to Use It

### Method 1: The "All-In-One" Pipeline

If you want to extract frames and generate segmentations in one go, use the `build_workspace()` method.

```python
from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace

# 1. Load the raw dataset
ds = SideSeeingDS('/path/to/raw/multisensor/dataset')

# 2. Initialize the Workspace
workspace = SideSeeingWorkspace('/path/to/output_workspace')

# 3. Build the entire workspace (with automatic fast anonymization)
workspace.build_workspace(
    dataset=ds, 
    prompts=["crosswalk", "curbramp", "pothole"], 
    extract_step=30,      # Extract 1 frame every 30 frames
    use_symlinks=True,    # Use safe linking for raw data
    anonymize_method="yolo" # Blurs persons and cars immediately after extraction
)
```

### Method 2: Step-by-Step Control

If you prefer to run steps individually or only process specific routes, you can use the granular methods:

```python
from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace

ds = SideSeeingDS('/path/to/raw/multisensor/dataset')
workspace = SideSeeingWorkspace('/path/to/output_workspace')

# Step 1: Safely setup the data/ directory (Links original videos and sensors)
workspace.setup_data_directory(dataset=ds, use_symlinks=True)

# (Optional) Select specific routes instead of the whole dataset
my_routes = [ds.instances["route01"], ds.instances["route02"]]

# Step 2: Extract frames only for selected routes
workspace.extract_frames(
    dataset=ds, 
    extract_step=30, 
    instances=my_routes
)

# Step 3: Anonymize the extracted frames to protect PII
# "yolo" is fast and blurs whole persons/cars. "sam3" is slow but precisely masks faces/plates.
workspace.anonymize_frames(
    dataset=ds,
    instances=my_routes,
    method="yolo",
    batch_size=8
)

# Step 4: Run segmentations only for selected routes
# NOTE: Requires `pip install sideseeing-tools[vision]`
workspace.generate_segmentation(
    dataset=ds, 
    prompts=["crosswalk"], 
    instances=my_routes,
    batch_size=8
)
```

## Tips
- **Disk Space**: Always try to leave `use_symlinks=True` (the default) to prevent cloning massive raw datasets. Windows users may need to run their IDE or terminal as Administrator to create symlinks.
- **Hardware Limitations**: If you encounter an "Out of Memory" (OOM) error on your GPU during `generate_segmentation`, simply lower the `batch_size` (e.g., from `8` to `4` or `2`). The process will resume exactly where it crashed.
- **Hugging Face Authentication**: The `facebook/sam3` model used by the Segmenter is a gated repository. Before running the segmentation or heavy anonymization for the first time, you must accept the terms at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and authenticate your terminal by running `huggingface-cli login` using your access token.
