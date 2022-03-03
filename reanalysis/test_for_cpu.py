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

def get_some_cyclones():
    cyclone_thunks = []

    for ssid in list(tracks.keys())[1000:1003]:
        track = tracks[ssid]
        cyclone_thunks.append(dask.delayed(
            error_tolerant(lib.track_to_ndarray, track['iso_times'], [(lat, long) for [long, lat] in track['coordinates']], degree_window=10)
        ))

    cyclones = dask.compute(*cyclone_thunks)