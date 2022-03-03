import dask
import traceback
import json
import track_to_ndarray as lib
import numpy as np
import os
import xarray
import time

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

    print("Running dask compute")
    cyclones = dask.compute(*cyclone_thunks)
    print("Finished compute")
    
    return cyclones, list(tracks.keys())[lower:upper]

if __name__ == '__main__':
    o_time = time.perf_counter()
    y, ssids = get_some_cyclones(1010,1014)
    n_time = time.perf_counter()

    for i, cyclone in enumerate(y):
        with open(f'/g/data/x77/jm0124/cyclone_binaries/{ssids[i]}', 'wb') as f:
            np.save(f, cyclone)

    quit()
