 
from typing import List, Tuple
import numpy as np
from datetime import datetime
from calendar import monthrange
import xarray

def netcdf_files(time_interval: Tuple[np.datetime64, np.datetime64], sets: List[str]) -> List[str]:
    """
    Gets a list of absolute file paths to all the files needed for reading specified sets of
    reanalysis data
    """
    # /g/data/rt52/era5/pressure-levels/reanalysis/u/1979/u_era5_oper_pl_19790101-19790131.nc
    basepath = "/g/data/rt52/era5/pressure-levels/reanalysis"
    start = time_interval[0].item() # datetime.datetime
    end = time_interval[1].item()
    s_year, s_month = start.year, start.month
    e_year, e_month = end.year, end.month
    # number of months is 12 * (e_year - s_year) + (e_month - s_month) + 1
    total_months = 12 * (e_year - s_year) + (e_month - s_month) + 1
    year_month_pairs = [(s_year + (s_month + i - 1) // 12, (s_month + i - 1) % 12 + 1) for i in range(total_months)]
    files = []
    for shorthand in sets:
        for (year, month) in year_month_pairs:
            padded_month = f"{month:02d}" # zero padded month number
            last_day = monthrange(year, month)[1]
            files.append(f"{basepath}/{shorthand}/{year}/{shorthand}_era5_oper_pl_{year}{padded_month}01-{year}{padded_month}{last_day}.nc")
    return files

def test_netcdf_files():
    import os
    s = np.datetime64('2009-01-05')
    e = np.datetime64('2010-09-30')
    files = lib.netcdf_files((s, e), ['u', 'v'])
    print(files)
    for file in files:
        assert os.path.isfile(file)
    print('all files!')

def sample_latlong_window(array: xarray.DataArray, degrees: float, lat: float, long: float, time: str, levels: List[int]) -> np.ndarray:
    # note that latitudes are -90 to 90 so we invert the slice
    # the return will (probably) still have a backwards latitude dimension, as long as this is consistent it's no problem
    # TODO: need a levels parameter like track_to_ndarray
    # also note that longitude wraps -180 to 180 and our region might cross this wrapping line
    # depending on where we're sampling around we may get degrees/4 or degrees/4+1 points along an axis. We trim this to degrees/4
    main = array.sel(time=time, longitude=slice(long-degrees/2,long+degrees/2), latitude=slice(lat+degrees/2,lat-degrees/2), level=levels).to_numpy().swapaxes(1,2) # swap (levels,lat,long) for (levels,long,lat)
    points = int(round(degrees / 0.25))
    main = main[:,:points,:points]
    needs_secondary = False # do we need another slice to get data on the other side due to longitude wrapping?
    if long-degrees/2 < -180:
        long += 360
        needs_secondary = True
    elif long+degrees/2 > 180:
        long -= 360
        needs_secondary = True
    if needs_secondary:
        secondary = array.sel(time=time, longitude=slice(long-degrees/2,long+degrees/2), latitude=slice(lat+degrees/2,lat-degrees/2), level=levels).to_numpy().swapaxes(1,2)
        secondary = secondary[:,:points,:points]
        main = np.hstack((main, secondary))

    main = main[:,:points,:points]

    shape = (37, points, points) # (levels, long, lat)
    assert main.shape == shape, f"{main.shape} must be {shape}"
    return main

def track_to_ndarray(iso_times: List[str], coordinates: List[Tuple[float, float]], levels: List[int], degree_window = 35) -> np.ndarray:
    """
    Takes lat/long coordinates and ISO-8601 formatted time strings
    Returns an ndarray of blocks at each time where each block is a 3D pressure map around the coordinate

    Returns (time, set, level, long, lat) shape
    eg [0][1] means 'v' at the first timestep
    [0][1][3][-1][-1] means the bottom right v component on the 4th level at the first timestep
    
    TODO: take a `levels: List[int]` keyword argument and chunk 1 level at a time and edit the shape assertion with len(levels)
    """
    times = [np.datetime64(iso_time) for iso_time in iso_times] # List[np.datetime64]
    time_interval = (min(times), max(times))
    sets = ['u', 'v', 'z', 'pv']

    # time chunks are 2 mins, latitude/longitude are 0.25 degrees
    # and 1 time only because we only sample every 6 hours on average
    ds_dict = { shorthand: xarray.open_mfdataset(netcdf_files(time_interval, [shorthand]),
        chunks={ "time": 1, "latitude": 300, "longitude": 300 },
        combine='nested', concat_dim='time') for shorthand in sets }

    time_series = []
    for time, (lat, long) in zip(iso_times, coordinates):
        chunk = []
        for shorthand in sets:
            ds = ds_dict[shorthand]
            x = sample_latlong_window(ds[shorthand], degree_window, lat, long, time, levels)
            chunk.append(x)
        time_series.append(chunk)
    y = np.array(time_series)

    points = int(round(degree_window / 0.25))
    shape = (len(iso_times), len(sets), len(levels), points, points)
    assert y.shape == shape, f"{y.shape} must be {shape}"
    
    return y

