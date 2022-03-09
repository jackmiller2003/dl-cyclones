import dask
import dask.bag as db
import traceback
import json
import track_to_ndarray as lib
import numpy as np
import os
import xarray
import time

dask.config.set(scheduler="synchronous")

with open('../tracks/proc_tracks.json', 'r') as f:
    tracks = json.load(f)

def get_user_path() -> str:
    # I'm not sure if there's a general way to know what the username is so we'll hardcode Jack's for now
    # -- probably os.getlogin(), but this wasn't quite working for me!
    return f'/scratch/x77/ahg157'

def job(ssid) -> None:
    "Saves a binary ndarray for a cyclone given the track JSON and unique ID from the IBTrACS database"

    print(f"Getting ndarray for cyclone {ssid}")

    track = tracks[ssid]
    N = None # subset track length

    s_time = time.perf_counter()
    cyclone = lib.track_to_ndarray_xr(
        track['iso_times'][:N],
        [(lat, long) for [long, lat] in track['coordinates'][:N]],
        levels=[200,450,650,750,850],
        degree_window=10
    )

    print(f"Saving ndarray for cyclone {ssid}")
    with open(f'{get_user_path()}/cyclone_binaries/{ssid}', 'wb') as f:
        np.save(f, cyclone)
    e_time = time.perf_counter()

    print(f"Finished processing cyclone {ssid} in {e_time - s_time:.2f} seconds")

def get_some_cyclones_list(list_of_cyclones) -> None:
    cyclone_thunks = []
    list_of_tracks = list(tracks.keys())

    b = db.from_sequence([list_of_tracks[n] for n in list_of_cyclones])
    b.map(job).compute()

if __name__ == '__main__':
    o_time = time.perf_counter()
    cyclones = [513, 688, 952, 2021, 3096]
    get_some_cyclones_list(cyclones)
    n_time = time.perf_counter()
    print(f"Finished processing {len(cyclones)} cyclones in {n_time-o_time:.2f} seconds")
