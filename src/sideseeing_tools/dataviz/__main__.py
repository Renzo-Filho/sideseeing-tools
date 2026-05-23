import argparse
from .server import start_server

def main():
    parser = argparse.ArgumentParser(description="Serve SideSeeing HTML report dynamically.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to raw dataset.")
    parser.add_argument("-o", "--output_dir", default="output", help="Path to report output.")
    parser.add_argument("-p", "--port", type=int, default=5000)

    args = parser.parse_args()
    start_server(args.input_dir, args.output_dir, args.port)

if __name__ == "__main__":
    main()
