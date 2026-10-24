from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import requests
import reverse_geocode

COMPONENT_MAPPING = {
    'route': 'street',
    'administrative_area_level_2': 'city',
    'administrative_area_level_1': 'state',
    'country': 'country',
}
EARTH_RADIUS_KM = 6371


def inverse_geocode(latitude, longitude, key=None):
    data = {'country': 'Unknown', 'state': 'Unknown', 'city': 'Unknown', 'street': 'Unknown'}

    if not key:
        geo = reverse_geocode.search([[latitude, longitude]])

        if geo:
            first_geo = geo.pop()
            data['country'] = first_geo.get('country', 'Unknown')
            data['city'] = first_geo.get('city', 'Unknown')
    else:
        data = inverse_geocode_from_google(latitude, longitude, key)

    return data


def inverse_geocode_from_google(latitude, longitude, key):
    """Performs reverse geocoding using Google's Geocoding API to find address components."""

    data = {v: 'Unknown' for v in COMPONENT_MAPPING.values()}

    try:
        params = {'latlng': f'{float(latitude)},{float(longitude)}', 'key': key}
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json', params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return data

        for component in results[0].get('address_components', []):
            for component_type, data_key in COMPONENT_MAPPING.items():
                if component_type in component.get('types', []):
                    data[data_key] = component.get('long_name')
    except requests.exceptions.RequestException as e:
        print(f"ERROR. Failed to call Google Geocoding API: {e}")

    return data


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the latitude/longitude distance between two points in km."""

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return EARTH_RADIUS_KM * c


# Fixes in the running median of smooth_track: about 5 s of walking at 1 Hz GPS. It removes jitter and
# single-fix jumps but keeps corners. ponytail: a display smoother, not map matching or side-of-street
# correction (see tasks/visual-analytics-plan.md for that study).
SMOOTHING_WINDOW = 5


def smooth_track(points, window=SMOOTHING_WINDOW):
    """A copy of time-ordered GPS points with latitude and longitude replaced by a centred running median."""
    smoothed = points.copy()
    for column in ('latitude', 'longitude'):
        smoothed[column] = points[column].rolling(window, center=True, min_periods=1).median()
    return smoothed


def position_at(track, times, time_column='Datetime UTC', tolerance_s=5.0):
    """(latitude, longitude) arrays interpolated along a track at `times`.

    Times more than `tolerance_s` before the first fix or after the last are NaN. Gaps between fixes are
    bridged linearly, so positions inside a long GPS gap are approximate.
    """
    def seconds(values):
        return (pd.to_datetime(pd.Series(values)).dt.tz_localize(None) - pd.Timestamp(0)).dt.total_seconds().to_numpy()

    fixes, wanted = seconds(track[time_column]), seconds(times)
    latitude = np.interp(wanted, fixes, track['latitude'].to_numpy())
    longitude = np.interp(wanted, fixes, track['longitude'].to_numpy())
    outside = (wanted < fixes[0] - tolerance_s) | (wanted > fixes[-1] + tolerance_s)
    latitude[outside] = longitude[outside] = np.nan
    return latitude, longitude
