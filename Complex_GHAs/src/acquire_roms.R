
# Script for downloading ROMS data using GitHub Actions workflow

library(glue)
library(lubridate)
library(ncdf4)
library(terra)

# Source utility function to download ROMS
source("Complex_GHAs/src/acquire_utils.R")


# Define date of interest
get_date <- "2025-08-01"


# Download daily isothermal layer depth (ILD)
# (varname = 'ild_05')
download_roms(ncdir_roms = ".",
              variable_roms = 'ild_05',
              savename_roms = glue("ILD_{get_date}"),
              get_date = get_date)

msg <- glue("ILD acquired from ROMS for {get_date}")

print(msg)
