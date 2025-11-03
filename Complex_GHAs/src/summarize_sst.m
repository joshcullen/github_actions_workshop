% Summarize SST

sst = ncread('mab_sst.nc','analysed_sst');

% Summarize all values and print to log
summary(sst(:))