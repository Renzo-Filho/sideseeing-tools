import os
import requests
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from pathlib import Path
import pandas as pd

class MapMatcher:
    def __init__(self, endpoint="http://router.project-osrm.org/match/v1/foot"):
        self.endpoint = endpoint

    def _chunk_trace(self, coords, chunk_size=10, overlap=2):
        """
        Chunks the coordinates into overlapping segments to respect OSRM API limits.
        """
        chunks = []
        i = 0
        while i < len(coords):
            end = min(i + chunk_size, len(coords))
            chunks.append(coords[i:end])
            if end == len(coords):
                break
            i += chunk_size - overlap
        return chunks

    def match_trace(self, instance, output_dir):
        """
        Takes a SideSeeingInstance, extracts coordinates, and matches them to a 
        routing graph using OSRM, then saves as a GPKG.
        """
        df_gps = instance.geolocation_points
        if df_gps is None or df_gps.empty:
            print(f"[MapMatcher] No GPS data for {instance.name}, skipping.")
            return

        if 'accuracy' in df_gps.columns:
            df_gps['accuracy'] = pd.to_numeric(df_gps['accuracy'], errors='coerce')
            df_gps = df_gps[df_gps['accuracy'] <= 15.0]

        df_gps = df_gps.sort_values(by='Datetime UTC').dropna(subset=['latitude', 'longitude'])
        coords = list(zip(df_gps['longitude'], df_gps['latitude']))

        if len(coords) < 2:
            print(f"[MapMatcher] Not enough GPS points for {instance.name}, skipping.")
            return

        chunks = self._chunk_trace(coords)
        matched_lines = []

        for chunk in chunks:
            coords_str = ";".join([f"{lon},{lat}" for lon, lat in chunk])
            url = f"{self.endpoint}/{coords_str}?geometries=geojson&overview=full"
            try:
                # Need to pass radiuses to avoid match failures on noisy GPS?
                # Using default settings for now
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "Ok" and "matchings" in data:
                        for match in data["matchings"]:
                            geom = match.get("geometry")
                            if geom and geom.get("type") == "LineString":
                                matched_lines.append(LineString(geom["coordinates"]))
                else:
                    print(f"[MapMatcher Warning] OSRM API error: {response.status_code}")
            except Exception as e:
                print(f"[MapMatcher Warning] Request failed: {e}")

        if not matched_lines:
            print(f"[MapMatcher Warning] No matches returned for {instance.name}.")
            return

        # Combine matching lines
        if len(matched_lines) == 1:
            final_line = matched_lines[0]
        else:
            final_line = linemerge(MultiLineString(matched_lines))

        # Save to gpkg
        out_path = Path(output_dir) / f"{instance.name}.gpkg"
        if out_path.exists():
            out_path.unlink()
        gdf = gpd.GeoDataFrame(geometry=[final_line], crs="EPSG:4326")
        gdf.to_file(out_path, driver="GPKG")
        print(f"[MapMatcher] Saved matched route to {out_path}")
