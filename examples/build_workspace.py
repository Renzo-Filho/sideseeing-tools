"""Build a SideSeeing workspace from a raw dataset, then open it in the viewer.

    python examples/build_workspace.py -i /path/to/raw -o /path/to/workspace -p sidewalk crosswalk
    python -m sideseeing_tools.dataviz -i /path/to/workspace

Requires `pip install sideseeing-tools[workspace,vision]` and access to facebook/sam3,
or a local checkpoint passed with --model.
"""
import argparse

from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace
from sideseeing_tools.vision.segmentation import MODEL_REVISION, Sam3Segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Raw dataset root directory.")
    parser.add_argument("-o", "--output", required=True, help="Workspace directory.")
    parser.add_argument("-s", "--step", type=int, default=30, help="Extract one frame every STEP video frames.")
    parser.add_argument("-p", "--prompts", nargs="+", default=["sidewalk"], help="SAM3 text prompts.")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="SAM3 batch size; start with 1 on 8 GB GPUs.")
    parser.add_argument("-m", "--model", default="facebook/sam3", help="SAM3 model id or local checkpoint directory.")
    parser.add_argument("--anonymize", action="store_true", help="Also write redacted copies of the frames.")
    parser.add_argument("--report", action="store_true", help="Also write the static report to report/.")
    args = parser.parse_args()

    ds = SideSeeingDS(args.input)
    workspace = SideSeeingWorkspace(args.output)
    segmenter = Sam3Segmenter(args.model, revision=MODEL_REVISION, local_files_only=args.model != "facebook/sam3")

    workspace.setup_data_directory(ds)
    workspace.extract_frames(ds, extract_step=args.step)
    if args.anonymize:
        workspace.anonymize_frames(ds, batch_size=args.batch_size, segmenter=segmenter)
    workspace.generate_segmentation(ds, args.prompts, batch_size=args.batch_size, segmenter=segmenter)
    workspace.generate_map_matched_routes(ds)
    workspace.generate_sidewalk_assessment_events(ds)
    if args.report:
        workspace.generate_report()

    print(f"Workspace ready. Open it with: python -m sideseeing_tools.dataviz -i {args.output}")


if __name__ == "__main__":
    main()
