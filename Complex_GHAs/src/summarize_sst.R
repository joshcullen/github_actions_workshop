
# Summarize SST

library(terra)


sst <- rast("mab_sst.nc")

# Summarize all values and print to log
values(sst) |> 
  summary() |> 
  print()