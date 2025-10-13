
## Calculate mean SST for day and region of interest

import xarray as xr


# Open raster file
ds = xr.open_dataset('mab_sst.nc')

# Calc mean from 'Band1' layer storing SST data
mean_sst = ds['Band1'].mean().item()
print(mean_sst)