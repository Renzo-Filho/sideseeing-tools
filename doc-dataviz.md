# SideSeeing Dataviz Integration (Local Server)

## Overview

The SideSeeing Dataviz integration introduces a scalable, dynamic web server for inspecting AI object detections and segmentation masks overlaid on video frames.

To handle massive datasets and thousands of frames without freezing or inflating disk storage, this module transitions the report from a purely static HTML export (Server-Side Rendered) to a dynamic Client-Side Rendered (CSR) application backed by a lightning-fast local **FastAPI** server.

### Key Features:
* **Decoupled Architecture:** Raw data, AI predictions, and extracted frames are kept in separate directories.
* **On-the-fly Image Processing:** Numpy-accelerated mask blending and auto-caching.
* **Format-Agnostic Adapters:** Natively supports both Project Sidewalk (wide-format) and SAM3 (long-format) predictions via the Adapter pattern.
* **Background Frame Extraction:** Asynchronous frame extraction directly from the UI without blocking the server.
* **Advanced UI Controls:** Independent Box/Mask filtering tabs, real-time detection statistics, and deep navigation controls (jump by 10, 100, 1000 frames).

---

## 1. Installation & Environment Setup

It is highly recommended to run the SideSeeing server inside an isolated virtual environment.

**Step 1: Create the Virtual Environment**
```bash
python -m venv venv

```

**Step 2: Activate the Environment**

* **Linux / macOS:** `source venv/bin/activate`
* **Windows (Cmd):** `venv\Scripts\activate.bat`
* **Windows (PS):** `.\venv\Scripts\Activate.ps1`

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt

```

---

## 2. Directory Structure Setup

To protect the integrity of your raw data, the server uses a decoupled workspace approach. You should organize your workspace as follows before starting the server:

```text
workspace/
├── dataset/                 <-- Passed as: -i ./dataset
│   └── route01/             <-- (This is your "instance_name")
│       ├── video.mp4
│       └── sensors.three.csv
│
├── frames/                  <-- Passed as: --frames_dir ./frames
│   └── route01-frames/      <-- MUST be named exactly {instance_name}-frames
│       ├── route01_0001.jpg 
│       └── route01_0002.jpg
│
└── preds/                   <-- Passed as: --ai_dir ./preds
    └── route01/             <-- AI data scoped by instance
        ├── predictions.general.csv  <-- (Project Sidewalk format)
        ├── detections.csv           <-- (SAM3 format)
        └── masks/                   
        └── masks/                   
            └── route01_0001_mask.png 

```

### Branched Dataset Structure (Advanced)

If you are working with massive datasets that do not fit the standardized structure (e.g., your extracted frames, segmentation masks, and predictions are spread across entirely different directories), you can use the branched structure flags to map the data without duplicating files.

```text
remote_storage/
├── data/                                 
│   └── 01_image_sequences/               <-- Passed as: --frames_dir
│       └── route01/                      <-- The server falls back to looking directly inside the instance folder
│           └── image_0001.jpg
├── results/
│   └── segmentation/                     <-- Passed as: --masks_dir
│       ├── sam3-crosswalk/               <-- Subdirectories are recursively searched for masks & detections.csv
│       │   ├── route01/
│       │   │   └── image_0001-mask.jpg   <-- Naming: _mask.png, -mask.jpg, or -mask.png
│       │   └── detections.csv            <-- Detections for this class
│       └── sam3-curbramp/
└── analysis/
    └── predictions.csv                   <-- Passed as: --predictions_csv (e.g. Project Sidewalk format)
```

In this mode, the server will intelligently fall back to looking inside `route01/` instead of `route01-frames/` for images. It will also recursively aggregate all `detections.csv` files found within the `masks_dir` and merge them with the global `predictions_csv`.

---

## 3. Running the Server (CLI)

The server is invoked via the `sideseeing_tools.dataviz` module.

**Full Dataviz Start:**

```bash
python -m sideseeing_tools.dataviz \
    -i ./dataset \
    -o ./report_output \
    --ai_dir ./preds \
    --frames_dir ./frames \
    -p 5000

```

### CLI Arguments Reference

| Argument | Short | Description | Default |
| --- | --- | --- | --- |
| `--input_dir` | `-i` | **(Required)** Path to the raw SideSeeing dataset. | *None* |
| `--output_dir` | `-o` | Path to save/read the base HTML report. | `./output` |
| `--ai_dir` |  | Path to the directory containing CSV prediction files. | `None` |
| `--frames_dir` |  | Path to store/read extracted video frames. | `[output_dir]/extracted_frames` |
| `--masks_dir` |  | Optional: Path to a branched directory containing segmentation masks. Recursively searched. | `None` |
| `--predictions_csv` |  | Optional: Direct path to a predictions CSV file (e.g. Project Sidewalk format). | `None` |
| `--port` | `-p` | The port to run the ASGI server on. | `5000` |

Once the server boots, navigate to `http://localhost:5000` and click the **Dataviz** tab.

---

## 4. User Interface & Features

The Dataviz UI provides granular control over how you inspect AI outputs:

1. **Box & Mask Separation (Tabs):** Because bounding boxes and segmentation masks serve different analytical purposes, they are split into separate tabs in the sidebar. Toggling a class in the "Boxes" tab will not affect its mask counterpart, allowing precise visual isolation.
2. **Real-time Statistics:** A dynamic statistics panel calculates the exact number of visible bounding boxes currently rendered on the frame, sorted by frequency.
3. **Advanced Navigation:** Skip through lengthy video captures easily using the footer controls. You can jump forward/backward by 1, 10, 100, or 1000 frames, or type a specific frame number into the center input.
4. **Hover Inspection:** Hovering over a class name in the sidebar will highlight the corresponding bounding boxes on the canvas by increasing their stroke width.

---

## 5. Deep Dive: Data Processing & Architectural Flow

The Dataviz module is built to handle massive datasets dynamically. Instead of generating gigabytes of static HTML and overlaid images beforehand, it processes requests "Just-In-Time". Here is the lifecycle of the data.

### Step 5.1: Ingestion & Normalization (`dataviz/adapters.py`)

AI models output data in wildly varying formats. For example, Project Sidewalk uses a *wide* binary format (columns for each class: `crosswalk`, `curbramp`), whereas SAM3 uses a *long* format.
When the server boots, the `PredictionAdapter` intercepts these CSVs and maps them to a **Unified Schema**:
`['image_name', 'class_name', 'confidence', 'is_mask', 'xmin', 'ymin', 'xmax', 'ymax']`
This allows the core engine to remain entirely agnostic to the AI model used.

### Step 5.2: API Data Compilation (`dataviz/api.py`)

When a user selects an instance (e.g., `route01`) from the dropdown, the frontend calls `/api/dataviz/data/route01`. The backend:

1. Scans the `frames_dir` to build an ordered array of available `.jpg` frames.
2. Reads the normalized AI DataFrame and separates the detected classes into two isolated sets: `box_classes` and `mask_classes` (based on the `is_mask` flag).
3. Groups Bounding Box coordinates by frame, generating consistent deterministic RGB colors for each class using an MD5 hash.
4. Returns a comprehensive JSON payload to initialize the frontend state.

### Step 5.3: Frontend State Management (`dataviz.js`)

The browser receives the JSON payload and builds a local `currentDatavizState` object.
The UI relies on an SVG Layer (for drawing boxes) and an `<img>` tag (for the base frame and mask overlay). The JS logic tracks exactly which classes the user has toggled on/off in the active `Set()`.

### Step 5.4: Dynamic Mask Rendering (`dataviz/visualizer.py`)

Drawing bounding boxes via SVG in the browser is cheap, but rendering complex segmentation masks is expensive. Instead of forcing the browser to calculate pixels, we delegate this to NumPy on the backend.
When the frontend needs a mask, it requests an image URL containing the active classes:
`/api/dataviz/mask/frame_001.jpg?classes=sidewalk,crosswalk`

The Backend intercepts this and:

1. Locates the raw grayscale mask `.png` generated by the AI.
2. Converts it into a NumPy array.
3. Filters out unrequested classes (`requested_classes and class_name not in requested_classes`).
4. Uses blazing-fast boolean indexing (`mask_pixels = m_np > 0`) to inject the generated RGBA colors into a transparent canvas matrix.
5. Encodes the resulting matrix to a PNG byte stream and sends it to the browser.
6. **Caching:** The specific combination of requested classes is hashed (`h_filter = MD5("sidewalk,crosswalk")`). Subsequent requests for this exact visual state are served instantly from disk cache.

---

## 6. Background Frame Extraction

Because extracting frames from `.mp4` files is CPU-intensive, the server handles this asynchronously.

1. Navigate to an instance in the UI.
2. If frames have not been extracted yet, the UI will display a prompt.
3. Click **"Extract Frames"**.
4. The server triggers a `BackgroundTasks` thread using `moviepy`/`cv2` to extract frames at 1 FPS. You can continue browsing other tabs without blocking the web server.
5. The UI will automatically poll the server every 5 seconds and refresh the viewer once the extraction completes.

---

## 7. Programmatic Usage (Python API)

You can embed the server boot sequence into automated pipelines or Jupyter Notebooks:

```python
from sideseeing_tools.dataviz import start_server

# Start the ASGI server synchronously
start_server(
    input_dir="./data/raw",
    output_dir="./report",
    ai_dir="./data/preds",
    frames_dir="./data/frames",
    port=8080
)

```

## 8. Extending AI Models (Adapter Pattern)

If you need to support a new AI model format (e.g., YOLOv10) in the future, you do **not** need to touch the UI, the API, or the image-processing engine.

Simply open `src/sideseeing_tools/dataviz/adapters.py` and add a new detection rule to `load_and_normalize()`. Ensure your new adapter returns a Pandas DataFrame containing the standard columns: `['image_name', 'class_name', 'confidence', 'is_mask', 'xmin', 'ymin', 'xmax', 'ymax']`. The `is_mask` boolean will automatically route the data to the correct UI Tab (Box or Mask).
