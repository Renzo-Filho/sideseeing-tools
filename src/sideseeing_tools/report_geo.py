import json
import os


def corrected_path(gpkg_path):
    """[lat, lon] points of a map-matched route written by SideSeeingWorkspace, or [] when absent."""
    if not os.path.exists(gpkg_path):
        return []

    import geopandas as gpd

    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)

    points = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        lines = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
        for line in lines:
            points.extend([y, x] for x, y in line.coords)
    return points


def process_geo_data(ds, output_data_dir, routes_dir=None):
    """Prepares geospatial data (GPS routes), saving one JSON per sample.

    With `routes_dir`, a sample's map-matched route `<routes_dir>/<sample>.gpkg` is added as `corrected_path`.
    """
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

            output_data = {"center": center, "path": path_data}

            if routes_dir:
                output_data["corrected_path"] = corrected_path(os.path.join(routes_dir, f"{sample.name}.gpkg"))

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
