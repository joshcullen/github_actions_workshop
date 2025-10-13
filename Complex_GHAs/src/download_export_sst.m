
% Download MUR SST 0.01° daily data from ERDDAP


% Define yesterday's date
today_date = datetime('now', 'Format', 'yyyy-MM-dd');
yesterday_date = today_date - days(1);

% Convert to string for use in the URL
get_date = char(yesterday_date); % 'YYYY-MM-DD' format


%%% Read in data from coastal Mid-Atlantic Bight

% Define url
url_template = 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.nc?analysed_sst[(%sT09:00:00Z)][(35.0):(40.0)][(-78.0):(-74.0)]';
final_url = sprintf(url_template, get_date);


% Read the data from the downloaded NetCDF file
% MATLAB ncread is OPeNDAP-compatible and will handle the download and slicing
sst_data = ncread(final_url, 'analysed_sst');
lat = ncread(final_url, 'latitude');
lon = ncread(final_url, 'longitude');

% Check that code is working (also fixed the fprintf syntax)
sst_vector = sst_data(:);
fprintf('Mean is %.4f\n', mean(sst_vector));