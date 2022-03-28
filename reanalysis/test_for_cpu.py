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
    return '/g/data/x77/ob2720'

def job(ssid) -> None:
    "Saves a binary ndarray for a cyclone given the track JSON and unique ID from the IBTrACS database"

    track = tracks[ssid]
    N = None # subset track length
    print(f"Getting ndarray for cyclone {ssid}, length {len(track['iso_times'])}")

    n_workers = len(client.scheduler_info()["workers"])
    #p = pyinstrument.Profiler()

    times = track["iso_times"][:N]
    coords = [(lat, long) for [long, lat] in track['coordinates'][:N]]

    time_bag = db.from_sequence(times[:10], npartitions=n_workers)
    #coord_bag = db.from_sequence(coords, npartitions=n_workers)

    # In the paper they use 700, 500, 225
    def f(times):
        return lib.track_to_ndarray_xr(#,650,750,850
            times, coords, levels=[225,500], degree_window=40,
        )

    s_time = time.perf_counter()
    #p.start()
    chunks = db.map_partitions(f, time_bag)
    cyclone = xarray.concat(chunks, "time").to_netcdf(f'{get_user_path()}/cyclone_binaries/{ssid}.netcdf',format='NETCDF4',engine='netcdf4')
    #p.stop()

#    print(f"Saving ndarray for cyclone {ssid}")
#    with open(f'{get_user_path()}/cyclone_binaries/{ssid}', 'wb') as f:
#        np.save(f, cyclone)
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
    cyclones = list(range(0,1))
    get_some_cyclones_list(cyclones)
    n_time = time.perf_counter()
    print(f"Finished processing {len(cyclones)} cyclones in {n_time-o_time:.2f} seconds")
