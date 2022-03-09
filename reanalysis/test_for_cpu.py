import dask
import dask.bag as db
from dask.distributed import Client
import traceback
import json
import track_to_ndarray as lib
import numpy as np
import os
import xarray
import time
#import pyinstrument

with open('../tracks/proc_tracks.json', 'r') as f:
    tracks = json.load(f)

def get_user_path() -> str:
    # I'm not sure if there's a general way to know what the username is so we'll hardcode Jack's for now
    # -- probably os.getlogin(), but this wasn't quite working for me!
    return os.getenv("PBS_JOBFS") # just want to run a few core counts at once...
    #return f'/scratch/x77/ahg157'

def job(ssid) -> None:
    "Saves a binary ndarray for a cyclone given the track JSON and unique ID from the IBTrACS database"

    track = tracks[ssid]
    N = None # subset track length
    print(f"Getting ndarray for cyclone {ssid}, length {len(track['iso_times'])}")

    n_workers = len(client.scheduler_info()["workers"])
    #p = pyinstrument.Profiler()

    times = track["iso_times"][:N]
    coords = [(lat, long) for [long, lat] in track['coordinates'][:N]]

    time_bag = db.from_sequence(times, npartitions=n_workers)
    coord_bag = db.from_sequence(coords, npartitions=n_workers)

    def f(times, coords):
        return lib.track_to_ndarray_xr(
            times, coords, levels=[200,450,650,750,850], degree_window=10,
        )

    s_time = time.perf_counter()
    #p.start()
    chunks = db.map_partitions(f, time_bag, coord_bag)
    cyclone = xarray.concat(chunks, "time").to_numpy()
    #p.stop()

    print(f"Saving ndarray for cyclone {ssid}")
    with open(f'{get_user_path()}/cyclone_binaries/{ssid}', 'wb') as f:
        np.save(f, cyclone)
    e_time = time.perf_counter()

    print(f"Finished processing cyclone {ssid} in {e_time - s_time:.2f} seconds")
    #p.print(unicode=True, color=True)

def get_some_cyclones_list(list_of_cyclones) -> None:
    cyclone_thunks = []
    list_of_tracks = list(tracks.keys())

    for n in list_of_cyclones:
        job(list_of_tracks[n])

if __name__ == '__main__':
    client = Client(threads_per_worker=1, local_directory=os.getenv("PBS_JOBFS"))
    o_time = time.perf_counter()
    cyclones = [513, 688, 952, 2021, 3096]
    get_some_cyclones_list(cyclones)
    n_time = time.perf_counter()
    print(f"Finished processing {len(cyclones)} cyclones in {n_time-o_time:.2f} seconds")
