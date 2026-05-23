from sideseeing_tools.dataviz import start_server

# Define your workspace paths
RAW_DATA = "/home/renzo/Documents/creativision/SideSeeing-Workspace/dataset"
AI_PREDS = "/home/renzo/Documents/creativision/SideSeeing-Workspace/preds"
FRAMES = "/home/renzo/Documents/creativision/SideSeeing-Workspace/frames"
REPORT_OUT = "/home/renzo/Documents/creativision/SideSeeing-Workspace/out"

# Start the ASGI server synchronously
start_server(
    input_dir=RAW_DATA,
    output_dir=REPORT_OUT,
    ai_dir=AI_PREDS,
    frames_dir=FRAMES,
    port=8080
)

from sideseeing_tools.dataviz import start_server

# Define your strict dataset root path
ROOT_DIR = "/scratch/renzo.filho/data/bras"

# Directory where the report index.html and static files will be saved
REPORT_OUT = f"{ROOT_DIR}/out_dataviz"

# Start the ASGI server using the minimalist approach
start_server(
    input_dir=ROOT_DIR,
    output_dir=REPORT_OUT,
    port=8080
)
