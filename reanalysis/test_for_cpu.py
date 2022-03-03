import dask
import traceback
import json
import track_to_ndarray as lib
import numpy as np
import os
import xarray

with open('../tracks/proc_tracks.json', 'r') as f:
    tracks = json.load(f)

def error_tolerant(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception:
        traceback.print_exc()

def get_some_cyclones(lower, upper):
    cyclone_thunks = []

    for ssid in list(tracks.keys())[lower:upper]:
        track = tracks[ssid]
        cyclone_thunks.append(dask.delayed(
            error_tolerant(lib.track_to_ndarray, track['iso_times'], [(lat, long) for [long, lat] in track['coordinates']], levels=[200,450,650,750,850], degree_window=10)
        ))

    cyclones = dask.compute(*cyclone_thunks)

    return cyclones, list(tracks.keys())[lower:upper]

if __name__ == '__main__':
    y, ssids = get_some_cyclones(1000,1002)

    for i, cyclone in enumerate(y):
        with open(f'/g/data/x77/jm0124/{ssids[i]}', 'wb') as f:
            np.save(f, cyclone)
