
### Download ERDDAP data, summarize by mgmt region over time, export values ###
# Based on request from Dale Robinson

# pip install datetime xarray netcdf4 geopandas regionmask pandas numpy matplotlib
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask
import math
from pathlib import Path


# -----------------------------------------
# Load management regions (CA RAMP zones)
# -----------------------------------------

# RAMP vector layer obtained from https://data-cdfw.opendata.arcgis.com/datasets/CDFW::risk-assessment-and-mitigation-program-ramp-fishing-zones-r7-cdfw-ds3120/about
ramp = gpd.read_file("Risk_Assessment_and_Mitigation_Program_(RAMP)_Fishing_Zones_-_R7_-_CDFW_[ds3120].geojson")
ramp.plot(alpha=0.4, edgecolor='black')
plt.close()

ramp.boundary.plot()
plt.close()

# ----------------------------
# Build ERDDAP griddap URL
# ----------------------------
base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
dataset_id = "erdQCwindproducts3day"   # 3-day composite (daily time steps)
url = f"{base}/{dataset_id}"


# --------------------------------------------
# Open with xarray (remote NetCDF) and subset
# --------------------------------------------
# decode_times=True converts ERDDAP's numeric time to datetime64[ns]
ds = xr.open_dataset(url, decode_times=True)

# Define time period of interest
t0 = "2024-01-01"
t1 = dt.date.today().strftime("%Y-%m-%d")  #today's date

# Subset data over time and space
dc = ds['ekman_upwelling'].sel(
  latitude=slice(50, 30),  #ymax needs to be listed first
  longitude=slice(-130, -115), 
  time=slice(t0, t1))


# ----------------------------
# Plot the most recent map
# ----------------------------
dc.sel(time=dc.time.max()).plot()
plt.close()



# -----------------------------------------------
# Create region masks over grid for RAMP zones
# -----------------------------------------------
regions = regionmask.Regions(
    outlines=list(ramp.geometry),
    names=list(ramp['FishingZone'].astype(str)),
    numbers=list(range(len(ramp)))
)

# regionmask needs to know which coords are lon/lat
mask_3D = regions.mask_3D(dc["longitude"], dc["latitude"])  # dims: latitude, longitude, region


# -------------------------------------------------
# Area-weighted monthly means within each polygon
# -------------------------------------------------
# (Weight by cos(latitude) so higher-lat, smaller cells don’t under/overcount)

lat_weights = np.cos(np.deg2rad(dc["latitude"]))

# Broadcast happens automatically; we just need (latitude, longitude) weights
# Apply masks: result has extra "region" dimension
ramp_masked = dc.where(mask_3D)  # dims -> (time, latitude, longitude, region)

# Compute spatial weighted mean for each polygon/region
# (xarray will broadcast lat_weights across time/region automatically)
spatial_mean = ramp_masked.weighted(lat_weights).mean(dim=("latitude", "longitude"))
# dims now: time, region


# Monthly means (first of each month). If you prefer month labels, use .to_period('M')
monthly = spatial_mean.resample(time="MS").mean()

# Add human-readable region names as a coordinate
monthly = monthly.assign_coords(region=("region", regions.names)).sortby("region")



# ---------------------------------------
# Merge summarized data with RAMP zones
# ---------------------------------------

# tidy table: time, region, ekman_upwelling
df = (monthly
      .to_dataframe(name="ekman_upwelling")
      .reset_index())  # -> columns: time, region, ekman_upwelling

# pick/ensure your geo name column
name_col = "name" if "name" in ramp.columns else ramp.columns[1]  # adjust if needed

# merge on region name (rename for a clean key)
df = df.rename(columns={"region": name_col})
ramp_small = ramp[[name_col, "geometry"]].copy()
ramp_merge = ramp_small.merge(df, on=name_col, how="right")  # keeps all time×region rows



# ----------------
# Viz results
# ----------------

# Heatmap
monthly.plot()
plt.close()

# Line plot
ax = (monthly
      .to_series()               # index: time, region
      .unstack("region")         # columns are regions (in your chosen order)
      .plot(figsize=(10,5), legend=True))
ax.set_xlabel("Month")
ax.set_ylabel("Ekman upwelling (m s$^{-1}$)")
ax.figure.tight_layout()

# Faceted plot of maps
ramp_merge[0:7].plot(col='time', col_wrap=3)
plt.close()



### Facet plots across all zones over time

# choose which months to show (e.g., the last 6)
months = sorted(ramp_merge["time"].unique())
sel_months = months[-6:] if len(months) > 6 else months

# consistent color scale across facets
vmin = ramp_merge["ekman_upwelling"].min()
vmax = ramp_merge["ekman_upwelling"].max()

n = len(sel_months)
ncols = min(3, n)
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), constrained_layout=True)

axes = np.atleast_1d(axes).ravel()
for ax, t in zip(axes, sel_months):
    ramp_t = ramp_merge[ramp_merge["time"] == t]
    ramp_t.plot(column="ekman_upwelling", ax=ax, vmin=vmin, vmax=vmax, legend=False, alpha=0.4, edgecolor='black')
    ax.set_title(pd.to_datetime(t).strftime("%Y-%m"))
    ax.axis("off")

# shared colorbar
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, ax=axes.tolist(), label="Ekman upwelling (m s$^{-1}$)")
# plt.show()



# ---------------
# Export results
# ---------------

# Reduce DF to only necessary columns
df2 = df[['time','FishingZone','ekman_upwelling']]

# Write CSV file
df2.to_csv("Monthly RAMP upwelling.csv", index=False)


# Also export spatial results as GeoJSON
gpd.GeoDataFrame.to_file(ramp_merge, filename="Monthly_RAMP_upwelling.geojson", driver="GeoJSON")


# ------------------------------------
# Append new data to file and export
# ------------------------------------

# old_df = pd.read_csv("Monthly RAMP upwelling.csv")
# new_df = pd.concat([df2, old_df])
# new_df.to_csv("Monthly RAMP upwelling.csv", index=False)