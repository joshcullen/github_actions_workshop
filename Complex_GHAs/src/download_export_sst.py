
# Download MUR SST 0.01° daily data from ERDDAP

import datetime as dt
import xarray as xr

# Define yesterday's date
get_date = dt.date.today() - dt.timedelta(days=1)
get_date = get_date.strftime("%Y-%m-%d")

# Read in data from coastal Mid-Atlantic Bight
url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41"
ds = xr.open_dataset(url, decode_times=True)

# Subset data over time and space
mab = ds['analysed_sst'].sel(
  latitude=slice(35, 40),  #ymax needs to be listed first
  longitude=slice(-78, -74), 
  time=slice(get_date, get_date))

# Export raster as netCDF to root dir
mab.to_netcdf("mab_sst.nc")