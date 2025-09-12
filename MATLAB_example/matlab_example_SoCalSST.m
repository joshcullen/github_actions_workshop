% Automate a process to:
%   1. Access latest SST output from UCSC opendap/thredds server
%   2. Calculate mean SST in the Southern California Bight
%   3. Append the new data to the existing file

% ============
% User input
% ============

% Thredds file
f_thredds = 'https://oceanmodeling.ucsc.edu/thredds/dodsC/ccsra_2016a_phys_agg_derived_vars/fmrc/CCSRA_2016a_Phys_ROMS_Derived_Variables_Aggregation_best.ncd';

% SST data file
f_sst = 'github_actions_workshop/MATLAB_example/sst_test.csv';

% Spatial bounds for averaging
lonmin = -120.5;
latmin = 32;

% ============
% Get new data
% ============

% Get time dimensions
time = ncread(f_thredds,'time');
nt = numel(time);

% Get latest data
sst = ncread(f_thredds,'sst',[1 1 nt],[Inf Inf 1]);
lon = ncread(f_thredds,'lon_rho');
lat = ncread(f_thredds,'lat_rho');
[y,m,d] = datevec(datenum([2011 1 2])+time(end)/24);
new_dates = datetime([y m d]);

% Get spatial average
sst_reg = mean(sst(lon>=lonmin & lat>=latmin),'omitnan');
new_data = table(new_dates, sst_reg, 'VariableNames', {'date','sst'});

% ============
% Update file
% ============

% Load data from existing file
old_data = readtable(f_sst);

% Combine
combined_data = [old_data;new_data];

% Rewrite file
writetable(combined_data,f_sst)

