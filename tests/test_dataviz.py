from sideseeing_tools.dataviz import start_server

# Define your strict dataset root path
ROOT_DIR = "/scratch/renzo.filho/dataset/bras"

# Directory where the report index.html and static files will be saved
REPORT_OUT = f"{ROOT_DIR}/out_dataviz"

# Start the ASGI server using the minimalist approach
start_server(
    input_dir=ROOT_DIR,
    output_dir=REPORT_OUT,
    port=5000
)
