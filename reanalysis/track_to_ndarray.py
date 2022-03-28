from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime, timezone
from calendar import monthrange
import xarray
import dask
from traceback import print_exc

def netcdf_files(time_interval: Tuple[np.datetime64, np.datetime64], shorthand: str) -> List[Tuple[datetime, str]]:
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
    for (year, month) in year_month_pairs:
        padded_month = f"{month:02d}" # zero padded month number
        # don't use monthrange, find the single filename that matches
        last_day = monthrange(year, month)[1]
        # (start datetime, path)
        files.append((
            datetime(year, month, 1, tzinfo=timezone.utc),
            f"{basepath}/{shorthand}/{year}/{shorthand}_era5_oper_pl_{year}{padded_month}01-{year}{padded_month}{last_day}.nc"
        ))
    return files

def coord_slice(coord: Tuple[float, float],
                degrees: float) -> Dict[str, slice]:
    """
    Return a dictionary of lat/lon slices for the given window size.
    """

    lat, lon = coord
    return {
        "latitude": slice(lat + degrees/2, lat - degrees/2),
        "longitude": slice(lon - degrees/2, lon + degrees/2)
    }


def sample_window(ds: xarray.DataArray, degrees: float, lat: float, lon: float) -> xarray.DataArray:
    """
    Sample ds with the specified window around a lat/lon point. Handles wrapping around the
    periodic X boundary.
    """

    points = int(round(degrees / 0.25)) # hardcode 0.25 deg input

    chunks = []
    for offset in [-360, 0, 360]:
        chunks.append(
            ds
            .sel(coord_slice((lat, lon + offset), degrees))
            .isel(longitude=slice(-points, None), latitude=slice(None, points))
            .reset_index(["longitude", "latitude"], drop=True)
        )

    full = xarray.concat(chunks, "longitude")

    return (
        full
        .isel(longitude=slice(None, points), latitude=slice(None, points))
    )


def track_to_ndarray_xr(iso_times: List[str],
                        coordinates: List[Tuple[float, float]],
                        levels: List[int],
                        degree_window = 35) -> np.ndarray:
    """
    Sample a track given as times and coordinate locations, at the specified levels, with a given window
    around each point.
    """

    times = [np.datetime64(iso_time) for iso_time in iso_times]
    time_interval = (min(times), max(times))
    sets = ["u", "v", "z", "pv"]

    out = []
    for shorthand in sets:
        files = netcdf_files(time_interval, shorthand)

        ds_array = [
            xarray.open_mfdataset(
                filename,
                chunks={"time": 1, "latitude": 324, "longitude": 360},
                combine="nested", concat_dim="time"
            ).sel(level=levels)
            for start_time, filename in files
        ]

        res = []
        for time, (lat, lon) in zip(iso_times, coordinates):
            # get the last dataset with a start time before `time`
            ds = None
            start_time = None
            filename = None
            for _ds, (_start_time, _filename) in zip(ds_array[::-1], files[::-1]): # loop backwards
                if np.datetime64(_start_time) <= np.datetime64(time):
                    ds = _ds
                    start_time = _start_time
                    filename = _filename
            try:
                res.append(sample_window(ds.sel(time=time), degree_window, lat, lon))
            except Exception as e:
                print_exc()
                print(start_time, filename, time, lat, lon, degree_window)

        out.append(xarray.concat(res, "time")[shorthand])

    # concat the different variables along a new "sets" dimension, and set this
    # to be the second dimension (matching the original behaviour)
    mapped = xarray.concat(out, "sets").transpose("time", "sets", "level", "longitude", "latitude")
    return mapped.load(scheduler="synchronous")
