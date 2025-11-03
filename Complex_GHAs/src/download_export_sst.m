
% Download MUR SST 0.01° daily data from ERDDAP

% Define yesterday's date
[y,m,d] = datevec(now-1);

% Read in data from coastal Mid-Atlantic Bight and save as netcdf
url = sprintf('https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.nc?analysed_sst%%5B(%d-%02d-%02dT09:00:00Z):1:(%d-%02d-%02dT09:00:00Z)%%5D%%5B(35.0):1:(40.0)%%5D%%5B(-78.0):1:(-74.0)%%5D',y,m,d,y,m,d);
websave('mab_sst.nc',url);