
# Script for downloading CMEMS data using GitHub Actions workflow

library(glue)

#-------------------------------------
# Function to download CMEMS data
#' @param path_copernicus_marine_toolbox File path to locally stored version of Copernicus Marine Toolbox. In copernicusmarine v2.0.0 on personal machine, only need to call `copernicusmarine` instead of full path to binary file.
#' @param ncdir_cmems Directory to which netCDF files are saved.
#' @param product_cmems Product name of interest from CMEMS.
#' @param variable_cmems Variable name of interest from relevant CMEMS product.
#' @param savename_cmems The file name to save the downloaded netCDF.
#' @param get_date Date of interest in YYYY-MM-DD format.
#' @param var_depth_min Minimum depth for which to extract values.
#' @param var_depth_max Maximum depth for which to extract values.
#'
#' @return A netCDF file for relevant data is downloaded locally to the directories specified in the function.

download_cmems = function(path_copernicus_marine_toolbox = "copernicusmarine",
                          ncdir_cmems, product_cmems,
                          variable_cmems, savename_cmems, get_date, var_depth_min,
                          var_depth_max) {
  
  # Write code from copernicusmarine via CLI
  command <- glue("{path_copernicus_marine_toolbox} subset -i {product_cmems} \\
                  -x 280.0 -X 295.0 -y 20.0 -Y 60.0 \\
                  -t {get_date} -T {get_date} \\
                  -z {var_depth_min}. -Z {var_depth_max}. \\
                  --variable {variable_cmems} \\
                  -o {ncdir_cmems} -f {savename_cmems}")
  
  # Run command
  system(command)
}
#-------------------------------------


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
