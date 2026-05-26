import argparse
from .server import start_server

def main():
    parser = argparse.ArgumentParser(description="Serve SideSeeing HTML report dynamically.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to raw dataset.")
    parser.add_argument("-o", "--output_dir", default=None, help="Path to report output. Defaults to input_dir.")
    parser.add_argument("-p", "--port", type=int, default=5000)

    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else args.input_dir
    start_server(args.input_dir, output_dir, args.port)

if __name__ == "__main__":
    main()
