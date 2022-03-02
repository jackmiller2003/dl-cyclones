# dl-cyclones

Repository for applying deep learning to cyclones on NCI.

## Choice of Atmopsheric Parameters

Here we describe our choice of atmospheric parameters used in the model

### SHIPS

SHIPS is an intensity prediction scheme which uses the following statistical parameters:

* Maximum possibility intesnity - initial intensity (POT)
* Mangitude of the 850-250 mb vertical shear (SHR)
* Intensity change over the past 12h (DVMX)
* 200-mb relative eddy angular momentum flux convergence (REFC)
* Absolute value of Julian day (JDAY)
* POT^2 (POT2)
* Average 200-mb temperature within 1000 km of storm center (T200)
* Average 200-mb zonal wind within 1000 km of storm center (U200)
* Average 850-mb vorticity within 1000 km of storm center (Z850)
* SHR times the sine of the initial storm latitude (LSHR)
* Average 200-mb divergence within 1000 km of storm center (D200)
* Zonal component of initial storm motion vector (SPDX)
* Initial storm maximum wind (VMX)

Considering what pressures to include we reason:

* Pressures for u and v could be 200,250,450,650,850 since these cover SHR, SHR^2 and U200.
* Should pressures for z be the same? I think so if we are considering our data like an image. I'm also guessing that they are using potential vorticity hence the geopotential height. Hence 200,250,450,650,850 for z.
