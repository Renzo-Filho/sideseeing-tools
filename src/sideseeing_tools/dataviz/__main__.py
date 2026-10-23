import argparse

from .server import start_server


def main():
    parser = argparse.ArgumentParser(description="Serve the SideSeeing viewer for a workspace.")
    parser.add_argument("-i", "--input_dir", required=True, help="Workspace directory created by SideSeeingWorkspace.")
    parser.add_argument("-p", "--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on. 0.0.0.0 exposes unredacted frames to the network.")

    args = parser.parse_args()
    start_server(args.input_dir, args.port, args.host)


if __name__ == "__main__":
    main()
