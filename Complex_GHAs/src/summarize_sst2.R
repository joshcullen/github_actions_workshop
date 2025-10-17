
# Summarize SST

library(terra)


sst <- rast("Complex_GHAs/ncdf/mab_sst.nc")

# Summarize all values and print to log
values(sst) |> 
  summary() |> 
  print()