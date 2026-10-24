"""Build a SideSeeing workspace from a raw dataset, then open it in the viewer.

    python examples/build_workspace.py -i /path/to/raw -o /path/to/workspace -p sidewalk crosswalk --whisper small
    python -m sideseeing_tools.dataviz -i /path/to/workspace

Prints the duration and peak GPU memory of each stage. Requires
`pip install sideseeing-tools[workspace,vision,transcription]`, and access to facebook/sam3
or a local checkpoint passed with --model.
"""
import argparse
import time

import torch

from sideseeing_tools import SideSeeingDS, SideSeeingWorkspace
from sideseeing_tools.vision.segmentation import MODEL_REVISION, Sam3Segmenter
from sideseeing_tools.workspace import REDACTION_PROMPTS, release_gpu_memory


def stage(name, function, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    function(*args, **kwargs)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"[stage] {name}: {time.time() - start:.0f} s, peak GPU memory {peak:.2f} GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Raw dataset root directory.")
    parser.add_argument("-o", "--output", required=True, help="Workspace directory.")
    parser.add_argument("-s", "--step", type=int, default=30, help="Extract one frame every STEP video frames.")
    parser.add_argument("-p", "--prompts", nargs="*", default=["sidewalk"], help="SAM3 text prompts; none skips SAM3.")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="SAM3 batch size; start with 1 on 8 GB GPUs.")
    parser.add_argument("-m", "--model", default="facebook/sam3", help="SAM3 model id or local checkpoint directory.")
    parser.add_argument("--anonymize", action="store_true", help="Also write redacted copies of the frames.")
    parser.add_argument("--whisper", metavar="MODEL", help="Also transcribe the audio with this Whisper model, e.g. small.")
    parser.add_argument("--language", help="Spoken language for Whisper, e.g. en or pt; detected when omitted.")
    parser.add_argument("--whisper-prompt", help="Text Whisper sees before the audio, e.g. the audit vocabulary.")
    parser.add_argument("--whisper-name", help="Transcript folder name, to keep variants apart; default whisper-MODEL.")
    parser.add_argument("--vad", action="store_true", help="Transcribe only the speech found by Silero VAD.")
    parser.add_argument("--skip-map-matching", action="store_true", help="Do not call the public OSRM server.")
    parser.add_argument("--report", action="store_true", help="Also write the static report to report/.")
    args = parser.parse_args()

    ds = SideSeeingDS(args.input)
    workspace = SideSeeingWorkspace(args.output)

    stage("link data", workspace.setup_data_directory, ds)
    stage("extract frames", workspace.extract_frames, ds, extract_step=args.step)

    if args.prompts or args.anonymize:
        segmenter = Sam3Segmenter(args.model, revision=MODEL_REVISION, local_files_only=args.model != "facebook/sam3")
        private = REDACTION_PROMPTS if args.anonymize else ()
        stage("segment", workspace.generate_segmentation, ds, args.prompts, batch_size=args.batch_size,
              segmenter=segmenter, private_prompts=private)
        if args.anonymize:
            stage("redact frames", workspace.anonymize_frames, ds, prompts=private, segmenter=segmenter)
        del segmenter
        release_gpu_memory()

    if not args.skip_map_matching:
        stage("map-match", workspace.generate_map_matched_routes, ds)
    stage("events", workspace.generate_sidewalk_assessment_events, ds)

    if args.whisper:
        options = dict(initial_prompt=args.whisper_prompt) if args.whisper_prompt else {}
        stage("transcribe", workspace.transcribe, ds, model_name=args.whisper, language=args.language,
              name=args.whisper_name, vad=args.vad, **options)
    if args.report:
        stage("report", workspace.generate_report)

    print(f"Workspace ready. Open it with: python -m sideseeing_tools.dataviz -i {args.output}")


if __name__ == "__main__":
    main()
