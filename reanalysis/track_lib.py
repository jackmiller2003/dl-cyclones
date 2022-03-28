from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime, timezone
import xarray
import dask
import traceback
from glob import glob
import dask.bag as db

def netcdf_files(start: datetime, end: datetime, shorthand: str) -> List[Tuple[datetime, str]]:
    """
    Gets a list of absolute file paths to all the files needed for reading specified sets of
    reanalysis data
    """
    # Example path: /g/data/rt52/era5/pressure-levels/reanalysis/u/1979/u_era5_oper_pl_19790101-19790131.nc
    basepath = "/g/data/rt52/era5/pressure-levels/reanalysis"
    total_months = 12 * (end.year - start.year) + (end.month - start.month) + 1
    # note that months are 1-indexed
    year_month_pairs = [(start.year + (start.month + i - 1) // 12, (start.month + i - 1) % 12 + 1) for i in range(total_months)]
    files = []
    for (year, month) in year_month_pairs:
        filename = glob(f"{basepath}/{shorthand}/{year}/{shorthand}_era5_oper_pl_{year}{month:02d}01-*.nc")[0]
        files.append((
            datetime(year, month, 1, tzinfo=timezone.utc),
            filename
        ))
    return files

def coord_slice(lat: float, long: float, degrees: float) -> Dict[str, slice]:
    return {
        # ERA5 longitude goes from high to low so we slice in opposite directions
        "latitude": slice(lat + degrees/2, lat - degrees/2),
        "longitude": slice(long - degrees/2, long + degrees/2)
    }

def sample_window(ds: xarray.DataArray, degrees: float, lat: float, long: float) -> xarray.DataArray:
    """
    Sample ds with the specified window around a lat/long point. Handles wrapping around the
    periodic X boundary.
    """
    points = int(round(degrees / 0.25)) # ERA5 uses a 0.25 deg interval everywhere
    chunks = []
    for offset in [-360, 0, 360]:
        chunks.append(
            ds.sel(coord_slice(lat, long + offset, degrees))
              .isel(longitude=slice(None, points), latitude=slice(None, points))
              .reset_index(["longitude", "latitude"], drop=True)
        )
    full = xarray.concat(chunks, "longitude")

    return full.isel(longitude=slice(None, points), latitude=slice(None, points))

def track_single_set_to_xarray(times: List[datetime],
                               coordinates: List[Tuple[float, float]],
                               dataset: str,
                               levels: List[int],
                               degree_window: int) -> xarray.DataArray:
    """
    Sample a track given as times and coordinate locations, at the specified levels, with a given window
    around each point.
    """
    files = netcdf_files(min(times), max(times), dataset)

    ds_array = [
        xarray.open_mfdataset(
            filename,
            chunks={"time": 1, "latitude": 324, "longitude": 360},
            combine="nested", concat_dim="time"
        ).sel(level=levels)
        for start_time, filename in files
    ]

    res = []
    for time, (lat, long) in zip(times, coordinates):
        # get the last dataset with a start time before `time`
        ds = None
        start_time = None
        filename = None
        for _ds, (_start_time, _filename) in zip(ds_array[::-1], files[::-1]): # loop backwards
            if np.datetime64(_start_time) <= np.datetime64(time):
                ds = _ds
                start_time = _start_time
                filename = _filename
        # try to get the window around the coordinate
        try:
            res.append(sample_window(ds.sel(time=time), degree_window, lat, long))
        except Exception as e:
            traceback.print_exc()
            print(start_time, filename, time, lat, long, degree_window)

    return xarray.concat(res, "time")

def track_to_xarray_dataset(times: List[str], coordinates: List[Tuple[float, float]],
                            levels: List[int], degree_window: int) -> xarray.Dataset:
    times: list[datetime] = [np.datetime64(iso_time).item() for iso_time in times]
    sets = ["u", "v", "z", "pv"]

    def f(dataset):
        return track_single_set_to_xarray(times, coordinates, dataset, levels, degree_window)

    # get each dataarray in parallel
    set_dataarrays: List[xarray.DataArray] = dask.compute(*[dask.delayed(f)(dataset) for dataset in sets])

    print(set_dataarrays)
    return xarray.merge(set_dataarrays)
