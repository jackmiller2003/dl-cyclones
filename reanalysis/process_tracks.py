
"""
This will process tracks 0 to -1 (ie all of them):
$ python3 process_tracks.py $(whoami) 0 -1
"""

import dask
import dask.bag as db
from dask.distributed import Client
import traceback
import json
import track_lib as lib
import numpy as np
import os
import xarray
import time
import sys
#import pyinstrument

with open('../tracks/proc_tracks.json', 'r') as f:
    tracks = json.load(f)

def get_user_path() -> str:
    # A path that looks like '/g/data/x77/ob2720'
    # We get the username as the first argument, eg `python3 process_tracks.py $(whoami)`
    # user = sys.argv[1]
    # edit: all binaries are stored in Oliver's drive
    user = 'ob2720'
    return f'/g/data/x77/{user}'

def create_netcdf_file_for_track(ssid) -> None:
    "Saves a binary ndarray for a cyclone given the track JSON and unique ID from the IBTrACS database"

    track = tracks[ssid]
    print(f"Getting ndarray for cyclone {ssid}, length {len(track['iso_times'])}")

    #p = pyinstrument.Profiler()

    times = track["iso_times"]
    coords = [(lat, long) for [long, lat] in track['coordinates']]
    levels = [225,500,650,750,850]   # In the tropical cyclone forecasting fusion networks paper they use 700, 500, 225

    s_time = time.perf_counter()
    #p.start()

    cyclone = lib.track_to_xarray_dataset(times, coords, levels, degree_window=40)
    cyclone.to_netcdf(f'{get_user_path()}/cyclone_binaries/{ssid}.nc', format='NETCDF4', engine='netcdf4')

    #p.stop()
    e_time = time.perf_counter()

    print(f"Finished processing cyclone {ssid} in {e_time-s_time:.2f} seconds")
    #p.print(unicode=True, color=True)

def netcdf_file_exists_for_track(ssid) -> bool:
    import os
    return os.path.isfile(f'{get_user_path()}/cyclone_binaries/{ssid}.nc')

if __name__ == '__main__':
    print("Initialising...")
    #client = Client(threads_per_worker=1, local_directory=os.getenv("PBS_JOBFS"))
    #n_workers = len(client.scheduler_info()["workers"])
    s_time = time.perf_counter()

    start = 0 if len(sys.argv) < 4 else int(sys.argv[2])
    end = -1 if len(sys.argv) < 4 else int(sys.argv[3])
    track_ids = list(tracks.keys())[start:end]
    for ssid in track_ids:
        try:
            if not netcdf_file_exists_for_track(ssid):
                create_netcdf_file_for_track(ssid)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to process {ssid}")

    e_time = time.perf_counter()
    print(f"Finished processing {len(track_ids)} cyclones in {e_time-s_time:.2f} seconds")
