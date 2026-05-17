# SideSeeing AI Vision Integration (Local Server)

## Overview

The SideSeeing AI Vision integration introduces a scalable, dynamic web server for inspecting AI object detections and segmentation masks overlaid on video frames.

To handle massive datasets and thousands of frames without freezing or inflating disk storage, this module transitions the report from a purely static HTML export (Server-Side Rendered) to a dynamic Client-Side Rendered (CSR) application backed by a lightning-fast local **FastAPI** server.

### Key Features:

* **Decoupled Architecture:** Raw data, AI predictions, and extracted frames are kept in separate directories.
* **On-the-fly Image Processing:** Numpy-accelerated mask blending and auto-caching.
* **Format-Agnostic Adapters:** Natively supports both Project Sidewalk (wide-format) and SAM3 (long-format) predictions via the Adapter pattern.
* **Background Frame Extraction:** Asynchronous frame extraction directly from the UI without blocking the server.

---

## 1. Installation & Environment Setup

It is highly recommended to run the SideSeeing server inside an isolated virtual environment.

**Step 1: Create the Virtual Environment**

```bash
python -m venv venv

```

**Step 2: Activate the Environment**

* **Linux / macOS:**
```bash
source venv/bin/activate

```


* **Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat

```


* **Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1

```



**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt

```

---

## 2. Directory Structure Setup

To protect the integrity of your raw data, the server uses a decoupled workspace approach. You should organize your workspace as follows before starting the server:

```text
my_workspace/
│
├── raw_dataset/                <-- Passed as: -i ./raw_dataset
│   └── route01/                <-- (This is your "instance_name")
│       ├── video.mp4
│       └── sensors.csv
│
├── extracted_frames/           <-- Passed as: --frames_dir ./extracted_frames
│   └── route01-frames/         <-- MUST be named exactly {instance_name}-frames
│       ├── route01_0001.jpg    <-- Base images
│       └── route01_0002.jpg
│
└── ai_predictions/             <-- Passed as: --ai_dir ./ai_predictions
    ├── detections.csv              <-- (SAM3 format)
    ├── predictions.general.csv     <-- (Project Sidewalk format)
    │
    └── masks/                      <-- Masks can be in a subfolder, but...
        ├── route01_0001_mask.png   <-- MUST be named {image_base_name}_mask.png
        └── route01_0002_mask.png


workspace/
├── dataset/              
│   ├── route01/
│   └── route02/
│
├── frames/           
│   ├── route01-frames/
│   └── route02-frames/
│
└── preds/           
    ├── route01/                     <-- ALL AI for route01 goes here
    │   ├── predictions.general.csv 
    │   ├── detections.csv     
    │   └── masks/                    
    │       └── route01_0001_mask.png 
    │
    └── route02/                     <-- ALL AI for route02 goes here
        └── detections.csv
```

---

## 3. Running the Server (CLI)

The server is invoked via the `sideseeing_tools.server` module.

**Basic Start (No AI Vision):**

```bash
python -m sideseeing_tools.server -i ./raw_dataset -o ./report_output

```

**Full AI Vision Start:**

```bash
python -m sideseeing_tools.server \
    -i ./raw_dataset \
    -o ./report_output \
    --ai_dir ./ai_predictions \
    --frames_dir ./extracted_frames \
    -p 5000

```

### CLI Arguments Reference

| Argument | Short | Description | Default |
| --- | --- | --- | --- |
| `--input_dir` | `-i` | **(Required)** Path to the raw SideSeeing dataset. | *None* |
| `--output_dir` | `-o` | Path to save/read the base HTML report. | `./output` |
| `--ai_dir` |  | Path to the directory containing CSV prediction files. | `None` |
| `--frames_dir` |  | Path to store/read extracted video frames. | `[output_dir]/extracted_frames` |
| `--port` | `-p` | The port to run the ASGI server on. | `5000` |

Once the server boots, navigate to `http://localhost:5000` in your web browser and click the **AI Vision (Dataviz)** tab.

---

## 4. Frame Extraction Workflow

Because extracting frames from `.mp4` files is CPU-intensive, the server handles this "Just-in-Time".

1. Start the server and navigate to an instance in the UI.
2. If frames have not been extracted for that instance yet, the UI will prompt you.
3. Click the "Extract Frames" button in the UI.
4. The server will trigger a background task using `moviepy`/`cv2` to extract the frames at 1 FPS. You can continue browsing other tabs while this runs.
5. Once complete, refresh the AI Vision tab to view the data.

---

## 5. Programmatic Usage (Python API)

You can easily embed the server boot sequence into automated pipelines or Jupyter Notebooks using the exposed Python API.

```python
from sideseeing_tools.server import start_server

# Define your workspace paths
RAW_DATA = "./data/raw"
AI_PREDS = "./data/models/sam3_run"
FRAMES = "./data/cache/frames"
REPORT_OUT = "./my_custom_report"

# Start the ASGI server synchronously
start_server(
    input_dir=RAW_DATA,
    output_dir=REPORT_OUT,
    ai_dir=AI_PREDS,
    frames_dir=FRAMES,
    port=8080
)

```

## 6. Extending AI Models (Adapter Pattern)

If you need to support a new AI model format (e.g., YOLOv10) in the future, you do **not** need to touch the UI or the image-processing engine.

Simply open `src/sideseeing_tools/dataviz_adapters.py` and add a new detection rule to `load_and_normalize()`. Ensure your new adapter returns a Pandas DataFrame containing the standard columns: `['image_name', 'class_name', 'confidence', 'is_mask', 'xmin', 'ymin', 'xmax', 'ymax']`.