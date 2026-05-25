import argparse
import os
from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace

def main():
    parser = argparse.ArgumentParser(description="Test script to build a SideSeeing Workspace for Dataviz.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the raw MultiSensor dataset root directory.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path where the processed workspace will be created.")
    parser.add_argument("--step", "-s", type=int, default=30, help="Frame extraction step rate (e.g., 30 means 1 frame every 30 frames).")
    parser.add_argument("--prompts", "-p", type=str, nargs="+", default=["sidewalk", "pothole", "curbramp"], help="Prompts for SAM3 segmentation.")
    parser.add_argument("--anonymize", "-a", type=str, choices=["yolo", "sam3", "none"], default="yolo", help="Anonymization method.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    print("="*50)
    print("1. Loading Raw Dataset")
    print("="*50)
    ds = SideSeeingDS(args.input)
    print(f"Loaded dataset: {ds.name} with {ds.size} routes.")

    print("\n" + "="*50)
    print("2. Initializing Workspace")
    print("="*50)
    workspace = SideSeeingWorkspace(args.output)

    # We can either use build_workspace() to do everything at once, 
    # or demonstrate the granular features. Let's do the granular approach to show everything.
    
    print("\n[Step A] Setting up Data Directory (Safe Symlinks)...")
    workspace.setup_data_directory(ds, use_symlinks=True)

    print(f"\n[Step B] Extracting Video Frames (Step: {args.step})...")
    workspace.extract_frames(ds, extract_step=args.step)

    if args.anonymize != "none":
        print(f"\n[Step C] Anonymizing Extracted Frames (Method: {args.anonymize})...")
        print("This will blur persons and vehicles to protect PII.")
        workspace.anonymize_frames(ds, method=args.anonymize)
    else:
        print("\n[Step C] Skipping Anonymization...")

    print(f"\n[Step D] Generating Segmentations for: {args.prompts}...")
    workspace.generate_segmentation(ds, prompts=args.prompts)

    print("\n" + "="*50)
    print("WORKSPACE BUILD COMPLETE!")
    print("="*50)
    print(f"Your workspace is ready at: {args.output}")
    print("\nYou can now start the Dataviz server with:")
    print(f"    python -m sideseeing_tools.dataviz -i {args.output}")

if __name__ == "__main__":
    main()
