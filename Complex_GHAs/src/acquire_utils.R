
### Utility functions for other scripts in dir ###


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
#' @export
download_cmems = function(path_copernicus_marine_toolbox = "copernicusmarine",
                          ncdir_cmems, product_cmems,
                          variable_cmems, savename_cmems, get_date, var_depth_min,
                          var_depth_max) {
  
  # Write code from copernicusmarine via CLI
  command <- glue::glue("{path_copernicus_marine_toolbox} subset -i {product_cmems} \\
                  -x 280.0 -X 295.0 -y 20.0 -Y 60.0 \\
                  -t {get_date} -T {get_date} \\
                  -z {var_depth_min}. -Z {var_depth_max}. \\
                  --variable {variable_cmems} \\
                  -o {ncdir_cmems} -f {savename_cmems}")
  
  # Run command
  system(command)
}

#-------------------------------------

#' Functions to acquire environmental data from ROMS THREDDS server
#'
#' Download data as netCDF file from the ROMS THREDDS server given url and variable name.
#'
#' @param ncdir_roms Directory to which netCDF files are saved.
#' @param variable_roms The name for the variable of interest.
#' @param savename_roms The file name to save the downloaded netCDF.
#' @param get_date Date of interest in YYYY-MM-DD format.
#'
#' @return A netCDF file for relevant data is downloaded locally to the directories specified in the function.
#'
#' @export
download_roms = function(ncdir_roms, variable_roms, savename_roms, get_date) {
  
  # Define number of days since ref date (2011-01-02) for url index
  ref_date <- lubridate::dmy('02-01-2011')
  new_date <- as.Date(get_date)
  days <- as.numeric(difftime(new_date, ref_date))
  
  # Define url for data download
  my_url <- glue::glue("https://oceanmodeling.ucsc.edu/thredds/dodsC/ccsra_2016a_phys_agg_derived_vars/fmrc/CCSRA_2016a_Phys_ROMS_Derived_Variables_Aggregation_best.ncd?{variable_roms}%5B{days}:1:{days}%5D%5B0:1:180%5D%5B0:1:185%5D,lat_rho%5B0:1:180%5D%5B0:1:185%5D,lon_rho%5B0:1:180%5D%5B0:1:185%5D,time%5B0:1:1%5D")
  
  # Download data and open as R object
  nc.data <- ncdf4::nc_open(my_url)
  
  lat <- ncdf4::ncvar_get(nc.data, 'lat_rho') |>
    as.numeric()
  lon <- ncdf4::ncvar_get(nc.data, 'lon_rho') |>
    as.numeric()
  var <- ncdf4::ncvar_get(nc.data, variable_roms) |>
    as.numeric()
  
  # Transform into {terra} SpatRaster object
  roms_df <- data.frame(x = lon,
                        y = lat,
                        z = var)
  roms_ras <- terra::rast(roms_df, type = "xyz", crs = "+proj=longlat +ellips=WGS84")
  
  # Export as netCDF file
  terra::writeCDF(roms_ras, glue::glue("{ncdir_roms}/{savename_roms}.nc"), varname = variable_roms, overwrite = TRUE)
  
}
