
## Calculate mean SST for day and region of interest

import xarray as xr


# Open raster file
ds = xr.open_dataset('mab_sst.nc')

# Calc mean from 'analysed_sst' layer
ds['analysed_sst'].mean().item()
