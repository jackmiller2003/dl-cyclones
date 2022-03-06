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

with open ('timing_log.txt', 'r') as timing_log:
    txt_log = timing_log

def error_tolerant(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception:
        traceback.print_exc()

def get_user_path() -> str:
    # I'm not sure if there's a general way to know what the username is so we'll hardcode Jack's for now
    return f'/g/data/x77/jm0124'

def job(track, ssid) -> None:
    "Saves a binary ndarray for a cyclone given the track JSON and unique ID from the IBTrACS database"
    try:
        print(f"Getting ndarray for cyclone {ssid}")
        s_time = time.perf_counter()
        cyclone = lib.track_to_ndarray(
            track['iso_times'],
            [(lat, long) for [long, lat] in track['coordinates']],
            levels=[200,450,650,750,850],
            degree_window=10
        )
        print(f"Saving ndarray for cyclone {ssid}")
        with open(f'{get_user_path()}/cyclone_binaries/{ssid}', 'wb') as f:
            np.save(f, cyclone)
        e_time = time.perf_counter()
        print(f"Finished processing cyclone {ssid} in {e_time - s_time:.2f} seconds")
    except Exception as e:
        raise Exception(f"Failed to process cyclone {ssid}")

def get_some_cyclones(lower, upper) -> None:
    cyclone_thunks = []

    for ssid in list(tracks.keys())[lower:upper]:
        track = tracks[ssid]
        cyclone_thunks.append(dask.delayed(
            error_tolerant(job, track, ssid)
        ))

    print("Running dask compute")
    dask.compute(*cyclone_thunks)
    print("Finished compute")

if __name__ == '__main__':
    range_under = 1015
    range_upper = 1019
    o_time = time.perf_counter()
    get_some_cyclones(range_under,range_upper)
    n_time = time.perf_counter()
    txt_log.write(f"Finished processing {range_upper - range_under} cyclones in {n_time-o_time:.2f} seconds")
    quit()
