
# Script for downloading CMEMS data using GitHub Actions workflow

library(glue)

# Source utility function to download CMEMS
source("Complex_GHAs/src/acquire_utils.R")


# Define date of interest
get_date <- as.character(Sys.Date() - 1)


# Download daily mixed layer depth
# (varname = 'mlotst', product = 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m')
download_cmems(path_copernicus_marine_toolbox = "/usr/share/miniconda/envs/test/bin/copernicusmarine",
               ncdir_cmems = ".",
               product_cmems = 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m',
               variable_cmems = 'mlotst',
               savename_cmems = glue("MLD_{get_date}"),
               get_date = get_date,
               var_depth_min = 0,
               var_depth_max = 1)

msg <- glue("MLD acquired from CMEMS for {get_date}")

print(msg)
