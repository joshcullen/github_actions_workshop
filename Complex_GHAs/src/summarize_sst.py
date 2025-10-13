
## Calculate mean SST for day and region of interest

import xarray as xr


# Open raster file
ds = xr.open_dataset('mab_sst.nc')

# Summarize SST values stored in band named 'Band1'
summ_sst = ds['Band1'].to_series().describe()
print(summ_sst)