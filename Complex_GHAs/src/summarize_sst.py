
## Calculate mean SST for day and region of interest

import xarray as xr


# Open raster file
ds = xr.open_dataset('mab_sst.nc')

# Summarize SST values stored in band named 'analysed_sst'
summ_sst = ds['analysed_sst'].to_series().describe()
print(summ_sst)