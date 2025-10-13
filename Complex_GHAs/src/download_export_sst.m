
% Download MUR SST 0.01° daily data from ERDDAP


% Define yesterday's date
today_date = datetime('now', 'Format', 'yyyy-MM-dd');
yesterday_date = today_date - days(1);

% Convert to string for use in the URL
get_date = char(yesterday_date); % 'YYYY-MM-DD' format


%%% Read in data from coastal Mid-Atlantic Bight

% Define url
url = 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.nc;



% # Subset data over time and space
sst_data = ncread(url, 'analysed_sst');
lat = ncread(final_url, 'latitude');
lon = ncread(final_url, 'longitude');

% Check that code is working (also fixed the fprintf syntax)
sst_vector = sst_data(:);
fprintf('Mean is %.4f\n', mean(sst_vector));