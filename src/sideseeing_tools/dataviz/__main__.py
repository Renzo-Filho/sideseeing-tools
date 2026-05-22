import argparse
from .server import start_server

def main():
    parser = argparse.ArgumentParser(description="Serve SideSeeing HTML report dynamically.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to raw dataset.")
    parser.add_argument("-o", "--output_dir", default="output", help="Path to report output.")
    parser.add_argument("--ai_dir", default=None, help="Optional: Path to AI predictions folder.")
    parser.add_argument("--frames_dir", default=None, help="Optional: Path to pre-extracted frames.")
    parser.add_argument("--masks_dir", default=None, help="Optional: Path to directory containing segmentation masks.")
    parser.add_argument("--predictions_csv", default=None, help="Optional: Direct path to a predictions CSV file.")
    parser.add_argument("-p", "--port", type=int, default=5000)

    args = parser.parse_args()
    start_server(args.input_dir, args.output_dir, args.ai_dir, args.frames_dir, args.masks_dir, args.predictions_csv, args.port)

if __name__ == "__main__":
    main()
