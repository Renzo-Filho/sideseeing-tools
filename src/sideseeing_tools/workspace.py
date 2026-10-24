import json
import os
import re
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sideseeing_tools import media
from sideseeing_tools.sideseeing import SideSeeingDS

FRAMES_RECORD = 'frames.json'
FRAME_NAME = re.compile(r'_(\d+)(_ms)?\.(?:jpg|png)$')
# Frames per SAM3 manifest. A chunk's masks stay in memory until saved; lower it for 4K video.
CHUNK_SIZE = 16
# Longest event, in extracted frames. Earlier versions derived it from a 30 fps default.
MAX_FRAMES_PER_EVENT = 45


def prompt_slug(prompt):
    return re.sub(r'[^a-z0-9]+', '-', prompt.lower()).strip('-')


def frame_timestamp(name, fps):
    """Seconds from video start for a frame name written by media.extract_frames.

    Names ending in '_ms' come from older extractions and already count seconds.
    """
    match = FRAME_NAME.search(name)

    if not match:
        return None

    index = int(match[1])
    return float(index) if match[2] else index / fps


def read_frames_record(frames_dir):
    path = Path(frames_dir) / FRAMES_RECORD
    return json.loads(path.read_text()) if path.exists() else None


def list_frames(frames_dir):
    frames_dir = Path(frames_dir)
    return sorted(frames_dir.glob('*.jpg')) + sorted(frames_dir.glob('*.png'))


def extract_route_frames(video_path, frames_dir, prefix, step):
    """Extract every step-th frame, publishing frames_dir only once extraction finished.

    Returns True when frames were extracted and False when complete frames already exist.
    """
    frames_dir = Path(frames_dir)
    record = read_frames_record(frames_dir)

    if record is not None:
        if record['step'] != step:
            raise ValueError(
                f"{frames_dir} was extracted with step {record['step']}, not {step}. "
                'Remove it to extract again.'
            )
        return False

    if frames_dir.exists() and any(frames_dir.iterdir()):
        print(f'[Workspace] Keeping frames extracted before {FRAMES_RECORD} existed: {frames_dir}')
        return False

    frames_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='.frames-', dir=frames_dir.parent) as temporary:
        staging = Path(temporary) / 'frames'
        staging.mkdir()
        # ponytail: media.extract_frames reports read errors instead of raising, so only
        # interruptions (not decoder failures) are kept out of the published folder.
        if not media.extract_frames(str(video_path), str(staging), step, prefix):
            raise ValueError(f'No frames extracted from {video_path}')

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        record = dict(source=str(Path(video_path).resolve()), step=step, fps=fps)
        (staging / FRAMES_RECORD).write_text(json.dumps(record, indent=2) + '\n')

        if frames_dir.exists():
            frames_dir.rmdir()
        staging.rename(frames_dir)

    return True


class SideSeeingWorkspace:
    """
    A manager class that transforms a raw SideSeeingDS dataset into a
    standardized processed directory structure ready for downstream tasks
    like the SideSeeing viewer.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.frames_dir = self.output_dir / "frames"
        self.redacted_dir = self.output_dir / "frames-redacted"
        self.preds_dir = self.output_dir / "preds"
        self.geomatching_dir = self.output_dir / "geomatching"
        self.routes_gpkg_dir = self.geomatching_dir / "routes_gpkg"
        self.report_dir = self.output_dir / "report"

        for directory in (self.data_dir, self.frames_dir, self.preds_dir, self.routes_gpkg_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def route_frames_dir(self, route):
        return self.frames_dir / f"{route}-frames"

    def setup_data_directory(self, dataset: SideSeeingDS, use_symlinks: bool = True):
        """
        Populates `data/` by copying or symlinking the raw routes and their metadata.csv,
        so `data/` can itself be loaded as a dataset (for example, by the report).
        """
        print(f"[Workspace] Setting up workspace data directory safely in {self.data_dir}...")

        sources = [(dataset.metadata_path, self.data_dir / "metadata.csv")]
        sources += [(instance.path, self.data_dir / instance.name) for instance in dataset.iterator if instance.name]

        for source, destination in sources:
            if not os.path.exists(source) or os.path.lexists(destination):
                continue
            if use_symlinks:
                os.symlink(os.path.abspath(source), destination, target_is_directory=os.path.isdir(source))
            elif os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        print(f"[Workspace] Data directory setup complete. Linked {dataset.size} routes.")

    def extract_frames(self, dataset: SideSeeingDS, extract_step: int, instances: list = None):
        """
        Extracts every `extract_step`-th video frame into frames/<route>-frames/.

        Args:
            dataset: The SideSeeingDS instance.
            extract_step: The step rate to extract frames.
            instances: Optional list of specific SideSeeingInstance objects to extract. If None, extracts all.
        """
        for instance in instances or dataset.iterator:
            if not instance.name or instance.video is None:
                print(f"[Workspace Warning] No video for {instance.name}. Skipping extraction...")
                continue

            print(f"[Workspace] Extracting frames for route: {instance.name} (Step: {extract_step})...")
            created = extract_route_frames(
                instance.video, self.route_frames_dir(instance.name), f"{instance.name}_", extract_step
            )
            if not created:
                print(f"[Workspace] Frames for {instance.name} already exist. Skipping extraction...")

        print(f"[Workspace] Frame extraction complete for all requested instances.")

    def _segmenter(self, segmenter):
        if segmenter is not None:
            return segmenter

        from sideseeing_tools.vision.segmentation import Sam3Segmenter
        return Sam3Segmenter()

    def _route_frames(self, route):
        frames = list_frames(self.route_frames_dir(route))
        if not frames:
            print(f"[Workspace Warning] No frames found for {route}. Did you run extract_frames() first? Skipping...")
        return frames

    def _route_fps(self, instance):
        record = read_frames_record(self.route_frames_dir(instance.name))

        if record:
            return record['fps']
        if instance.video is None:
            return 30.0

        capture = cv2.VideoCapture(instance.video)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        return fps

    def generate_segmentation(self, dataset: SideSeeingDS, prompts: list, instances: list = None,
                              batch_size: int = 1, segmenter=None, chunk_size: int = CHUNK_SIZE):
        """
        Runs SAM3 on the extracted frames for each prompt and saves instance masks, boxes,
        scores, timestamps and model provenance to preds/sam3/<prompt>/<route>/chunk-NNNN/.

        Each chunk is published only when complete, so an interrupted run resumes from the
        first missing chunk. Pass a Sam3Segmenter to choose the model or a local checkpoint.
        Requires `pip install sideseeing-tools[vision]` for the default segmenter.
        """
        from sideseeing_tools.vision.results import save_segmentations

        if not prompts:
            return

        segmenter = self._segmenter(segmenter)

        for prompt in prompts:
            for instance in instances or dataset.iterator:
                frames = self._route_frames(instance.name)
                fps = self._route_fps(instance)
                route_dir = self.preds_dir / "sam3" / prompt_slug(prompt) / instance.name

                for start in range(0, len(frames), chunk_size):
                    destination = route_dir / f"chunk-{start // chunk_size:04d}"

                    if (destination / "manifest.json").exists():
                        continue

                    chunk = frames[start:start + chunk_size]
                    print(f"[Workspace] Segmenting {len(chunk)} frames of {instance.name} with prompt '{prompt}'...")
                    timestamps = [frame_timestamp(path.name, fps) for path in chunk]
                    save_segmentations(
                        segmenter.segment(chunk, prompt, batch_size=batch_size),
                        destination,
                        source_id=instance.name,
                        timestamps_s=None if None in timestamps else timestamps,
                    )

    def anonymize_frames(self, dataset: SideSeeingDS, instances: list = None, prompts=("face", "license plate"),
                         mode: str = "blur", batch_size: int = 1, segmenter=None, chunk_size: int = CHUNK_SIZE):
        """
        Writes redacted copies of the extracted frames to frames-redacted/<route>-frames/,
        obscuring SAM3 masks for `prompts`. Source frames are never modified.

        Copies keep the source format, so JPEG frames are re-encoded. The masks are not
        guaranteed to be complete: review the output before treating it as anonymous.
        """
        from sideseeing_tools.privacy.redaction import redact_image

        segmenter = self._segmenter(segmenter)
        self.redacted_dir.mkdir(exist_ok=True)

        for instance in instances or dataset.iterator:
            destination = self.redacted_dir / f"{instance.name}-frames"
            frames = self._route_frames(instance.name)

            if destination.exists() or not frames:
                continue

            print(f"[Workspace] Redacting {len(frames)} frames of {instance.name}...")

            with tempfile.TemporaryDirectory(prefix=".redaction-", dir=self.redacted_dir) as temporary:
                staging = Path(temporary) / "frames"
                staging.mkdir()
                records = []

                for start in range(0, len(frames), chunk_size):
                    chunk = frames[start:start + chunk_size]
                    masks = {str(path.resolve()): [] for path in chunk}

                    for prompt in prompts:
                        for result in segmenter.segment(chunk, prompt, batch_size=batch_size):
                            masks[result["image_path"]].append(result["masks"])

                    for path in chunk:
                        image = cv2.imread(str(path))
                        stack = masks[str(path.resolve())]
                        combined = np.concatenate(stack) if stack else np.zeros((0, *image.shape[:2]), dtype=bool)

                        if not cv2.imwrite(str(staging / path.name), redact_image(image, combined, mode=mode)):
                            raise OSError(f"Could not write redacted frame {path.name}")

                        records.append(dict(file=path.name, masked_pixels=int(np.any(combined, axis=0).sum())))

                record = dict(
                    source=str(self.route_frames_dir(instance.name).resolve()),
                    status="processed-not-privacy-validated",
                    prompts=list(prompts),
                    mode=mode,
                    model=getattr(segmenter, "provenance", None),
                    frames=records,
                )
                (staging / "redaction.json").write_text(json.dumps(record, indent=2) + "\n")
                staging.rename(destination)

    def generate_map_matched_routes(self, dataset: SideSeeingDS, instances: list = None):
        """
        Generates Map-Matched routes for the dataset using the OSRM API.
        """
        from sideseeing_tools.mapping import MapMatcher

        print(f"[Workspace] Generating map-matched routes in {self.routes_gpkg_dir}...")
        matcher = MapMatcher()

        for instance in instances or dataset.iterator:
            if not instance.name:
                continue
            matcher.match_trace(instance, output_dir=str(self.routes_gpkg_dir))

    def _create_event_dict(self, event_df, event_id, instance_name):
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        from sideseeing_tools import utils

        if event_df.empty:
            return None

        start_row = event_df.iloc[0]
        end_row = event_df.iloc[-1]

        raw_lat = event_df['latitude'].mean()
        raw_lon = event_df['longitude'].mean()

        center_latitude = raw_lat
        center_longitude = raw_lon

        gpkg_path = self.routes_gpkg_dir / f"{instance_name}.gpkg"

        if gpkg_path.exists():
            try:
                gdf = gpd.read_file(gpkg_path)
                if not gdf.empty and 'geometry' in gdf.columns:
                    line_geom = gdf.geometry.union_all()
                    snapped_point = nearest_points(line_geom, Point(raw_lon, raw_lat))[0]
                    center_latitude = snapped_point.y
                    center_longitude = snapped_point.x
            except Exception as e:
                print(f"[Workspace Warning] Error snapping to route for {instance_name}: {e}")

        # Haversine distance
        length_meters = utils.calculate_haversine_distance(
            start_row['latitude'], start_row['longitude'],
            end_row['latitude'], end_row['longitude']
        ) * 1000.0 # utils returns km

        return {
            'event_id': f"EVT-{event_id:05d}",
            'start_image': start_row['image_name'],
            'end_image': end_row['image_name'],
            'center_latitude': center_latitude,
            'center_longitude': center_longitude,
            'feature': start_row['prompt'],
            'length_meters': length_meters,
            'instance_name': instance_name
        }

    def load_detections(self):
        """
        Frame-level SAM3 detection counts from preds/sam3 manifests and older
        preds/sam3-<prompt>/detections.csv files, one row per route, frame and prompt.
        """
        rows = []

        for manifest_path in sorted(self.preds_dir.glob("sam3/*/*/*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            for frame in manifest["frames"]:
                rows.append(dict(
                    route=manifest["source_id"] or manifest_path.parent.parent.name,
                    image_name=Path(frame["image_path"]).name,
                    prompt=frame["prompt"],
                    num_detections=frame["detections"],
                    timestamp_s=frame["timestamp_s"],
                ))

        for csv_path in sorted(self.preds_dir.glob("sam3-*/detections.csv")):
            if os.path.getsize(csv_path) == 0:
                continue
            legacy = pd.read_csv(csv_path)
            if "relative_path" not in legacy.columns:
                print(f"[Workspace Warning] {csv_path} has no relative_path column, so its routes are unknown. Skipping.")
                continue
            folder = legacy["relative_path"].astype(str).str.split("/").str[0]
            legacy["route"] = folder.str.replace(r"-frames$", "", regex=True).str.replace(r"^data_", "", regex=True)
            legacy["timestamp_s"] = np.nan
            rows.extend(legacy[["route", "image_name", "prompt", "num_detections", "timestamp_s"]].to_dict("records"))

        columns = ["route", "image_name", "prompt", "num_detections", "timestamp_s"]
        detections = pd.DataFrame(rows, columns=columns)
        # Manifests come first, so they win over a legacy CSV of the same frames.
        return detections.drop_duplicates(subset=["route", "image_name", "prompt"])

    def generate_sidewalk_assessment_events(self, dataset: SideSeeingDS):
        """
        Aggregates SAM3 detections into spatial map events based on continuous video frames,
        syncing timestamps from frames to the raw GPS data.
        """
        print("[Workspace] Generating sidewalk assessment events...")
        map_events = []
        event_id_counter = 1
        detections = self.load_detections()

        for (route, prompt), group in detections.groupby(["route", "prompt"], sort=True):
            instance = dataset.instances.get(route)

            if not instance:
                print(f"[Workspace Warning] Instance {route} not found in dataset. Skipping.")
                continue

            df_gps = instance.geolocation_points
            if df_gps is None or df_gps.empty:
                print(f"[Workspace Warning] No GPS data for {route}. Skipping event generation.")
                continue

            df_gps = df_gps.copy()
            df_gps['Datetime UTC'] = pd.to_datetime(df_gps['Datetime UTC']).dt.tz_localize(None)
            df_gps = df_gps.sort_values('Datetime UTC')

            fps = self._route_fps(instance)
            frame_data = []

            for row in group[group["num_detections"] > 0].itertuples():
                match = FRAME_NAME.search(row.image_name)
                seconds = row.timestamp_s if pd.notna(row.timestamp_s) else frame_timestamp(row.image_name, fps)

                if not match or seconds is None:
                    continue

                frame_data.append({
                    'image_name': row.image_name,
                    'frame_idx': int(match[1]),
                    'frame_time': instance.media_start_time + timedelta(seconds=seconds),
                    'prompt': prompt,
                })

            if not frame_data:
                continue

            df_frames = pd.DataFrame(frame_data).sort_values('frame_time')

            df_merged = pd.merge_asof(
                df_frames,
                df_gps[['Datetime UTC', 'latitude', 'longitude']],
                left_on='frame_time',
                right_on='Datetime UTC',
                direction='nearest',
                tolerance=pd.Timedelta(seconds=5)
            )

            df_merged = df_merged.dropna(subset=['latitude', 'longitude'])
            if df_merged.empty:
                continue

            df_merged = df_merged.sort_values('frame_idx').reset_index(drop=True)

            gaps = df_merged['frame_idx'].diff().dropna()
            common_gap = gaps.value_counts().idxmax() if len(gaps) > 0 else 30
            max_gap = int(common_gap * 5)

            current_event = []
            for _, row in df_merged.iterrows():
                if current_event and (
                    row['frame_idx'] - current_event[-1]['frame_idx'] > max_gap
                    or len(current_event) >= MAX_FRAMES_PER_EVENT
                ):
                    event_dict = self._create_event_dict(pd.DataFrame(current_event), event_id_counter, route)
                    if event_dict:
                        map_events.append(event_dict)
                        event_id_counter += 1
                    current_event = []

                current_event.append(row)

            if current_event:
                event_dict = self._create_event_dict(pd.DataFrame(current_event), event_id_counter, route)
                if event_dict:
                    map_events.append(event_dict)
                    event_id_counter += 1

        if map_events:
            out_path = self.geomatching_dir / "map_events.csv"
            pd.DataFrame(map_events).to_csv(out_path, index=False)
            print(f"[Workspace] Generated {len(map_events)} sidewalk assessment events in {out_path}")
        else:
            print("[Workspace] No events generated.")

    def generate_report(self, **kwargs):
        """
        Writes the static report for data/ to report/. It links back to the viewer, which
        serves it at /report. Keyword arguments go to Report.generate_report.
        """
        from sideseeing_tools.export import Report

        metadata = self.data_dir / "metadata.csv"
        older_metadata = self.output_dir / "metadata.csv"
        if not os.path.lexists(metadata) and older_metadata.exists():
            metadata.symlink_to(older_metadata.resolve())

        kwargs.setdefault("title", f"SideSeeing Report - {self.output_dir.name}")
        kwargs.setdefault("viewer_url", "../")
        kwargs.setdefault("routes_dir", str(self.routes_gpkg_dir))
        Report().generate_report(str(self.data_dir), str(self.report_dir), **kwargs)

    def build_workspace(self, dataset: SideSeeingDS, prompts: list, extract_step: int = 30,
                        use_symlinks: bool = True, anonymize: bool = False, segmenter=None):
        """
        Master function to execute the entire pipeline on the whole dataset: links the data,
        extracts frames, optionally writes redacted copies, segments, map-matches and builds events.
        """
        self.setup_data_directory(dataset, use_symlinks)
        self.extract_frames(dataset, extract_step)

        if prompts or anonymize:
            segmenter = self._segmenter(segmenter)
        if anonymize:
            self.anonymize_frames(dataset, segmenter=segmenter)
        self.generate_segmentation(dataset, prompts, segmenter=segmenter)

        self.generate_map_matched_routes(dataset)
        self.generate_sidewalk_assessment_events(dataset)

        print(f"[Workspace] Build pipeline complete! Everything is ready in {self.output_dir}.")
