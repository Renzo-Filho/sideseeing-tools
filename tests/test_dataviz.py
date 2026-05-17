from sideseeing_tools.server import start_server

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
