
# Download MUR SST 0.01° daily data from ERDDAP

library(glue)
library(terra)

# Define yesterday's date
get_date <- Sys.Date() - 1

# Read in data from coastal Mid-Atlantic Bight
url <- glue("https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.nc?analysed_sst%5B({get_date}T09:00:00Z):1:({get_date}T09:00:00Z)%5D%5B(35.0):1:(40.0)%5D%5B(-78.0):1:(-74.0)%5D")
mab <- rast(url)


# Export raster as GeoTIFF to root dir
writeRaster(mab, "mab_sst.nc")