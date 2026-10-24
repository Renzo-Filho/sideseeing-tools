import json
import os

from sideseeing_tools.geolocation import smooth_track


def process_geo_data(ds, output_data_dir):
    """Prepares geospatial data (GPS routes), saving one JSON per sample: the phone's fixes as `path` and
    the same track with a running median (geolocation.smooth_track) as `smoothed_path`."""
    print("Exporting geospatial data to JSONs...")
    os.makedirs(output_data_dir, exist_ok=True)

    instance_json_map = {}

    for sample in ds.iterator:
        df_gps = sample.geolocation_points
        center = sample.geolocation_center

        if df_gps is None or df_gps.empty or center is None:
            continue

        try:
            path_data = df_gps[['latitude', 'longitude']].values.tolist()

            smoothed = smooth_track(df_gps)[['latitude', 'longitude']].values.tolist()
            output_data = {"center": center, "path": path_data, "smoothed_path": smoothed}

        except Exception as e:
            print(f"Error extracting GPS data for {sample.name}: {e}")
            continue

        json_filename = f"geo_{sample.name}.json"
        json_save_path = os.path.join(output_data_dir, json_filename)
        json_relative_path = f"data/{json_filename}"

        try:
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f)

            instance_json_map[sample.name] = json_relative_path
        except Exception as e:
            print(f"Error saving geospatial JSON for {sample.name}: {e}")

    return instance_json_map if instance_json_map else None
