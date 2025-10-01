
# ------------------------------------------------------------
# Daily Chlorophyll-a (VIIRS, gap-filled) → State offshore means + validation
# Gulf of Mexico bbox: xmin=-97.79166, xmax=-80.625, ymin=33.54166, ymax=16.375
# Requires: terra, sf, dplyr, lubridate, pointblank, stringr
# ------------------------------------------------------------

library(tidyverse)
library(terra)
library(sf)
library(pointblank)
library(rnaturalearth)
library(blastula)
library(glue)

source("Data_validation_example/validation_ex_utils.R")


# -----------------------
# Config
# -----------------------

# Spatial extent (lon/lat in EPSG:4326)
xmin <- -97.79166
xmax <- -80.625
ymin <-  16.375
ymax <-  31

# gulf_ext <- ext(xmin, xmax, ymin, ymax)

# Date(s) to pull (ISO yyyy-mm-dd). Example: latest day by default.
# You can set a vector like seq(ymd("2025-09-01"), ymd("2025-09-07"), by = "1 day")
get_date <- Sys.Date() - 14  # 2 weeks ago (this ERDDAP product is heavily lagged)


# -----------------------
# Load U.S. state boundaries and approx terr. waters
# -----------------------

# Load spatial layers for US states
us_states <- ne_states(country = "United States of America",
                                         returnclass = "sf") |>
  st_transform(3395)  #convert to World Mercator/WGS84 (units = meters)

# Keep only GoM states
gulf_names <- c("Texas","Louisiana","Mississippi","Alabama","Florida")

gulf_states <- us_states |>
  filter(name %in% gulf_names) |>
  select(name, abbr = postal)


# Create buffer to approximate territorial waters (12 nm = 22224 m)
gulf_states_buff <- gulf_states |> 
  st_buffer(dist = 22224) |> 
  st_transform(4326)  #convert back to WGS84

ggplot() +
  geom_sf(data = gulf_states_buff, fill = "lightblue") +
  geom_sf(data = gulf_states |> 
            st_transform(4326)) +
  # geom_vline(xintercept = -80.5) +
  theme_bw()



# -----------------------
# Read in ERDDAP data for CHL
# -----------------------

# ERDDAP dataset info
erddap_base <- "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaDaily"
varname     <- "chlor_a"  # mg m^-3, S-NPP VIIRS, L3, 4 km, Global, daily

# ERDDAP dimension order for this dataset: time, altitude, latitude, longitude
url <- build_erddap_url(get_date, xmin, xmax, ymin, ymax)

# Read in data as SpatRaster
r <- rast(url)


# -----------------------
# Extract CHL values per state for specified date
# -----------------------

# Extract CHL values within Gulf state waters (per pixel)
raw_vals <- extract(r, gulf_states_buff, xy = TRUE) |> 
  rename(state = ID,
         chla = `chlor_a_altitude=0`) |> 
  mutate(state = map_chr(state, \(x) gulf_states_buff$name[x]))  #change ID to name

# Extract CHL values within Gulf state waters (and calculate avg per state)
state_means <- extract(r, gulf_states_buff, fun = mean, na.rm = TRUE) |> 
  rename(state = ID,
         chla = `chlor_a_altitude=0`) |> 
  mutate(state = map_chr(state, \(x) gulf_states_buff$name[x]))  #change ID to name


# -----------------------
# Perform data validation with {pointblank}
# -----------------------

# Validation thresholds (from dataset metadata)
# valid_min / valid_max from ERDDAP: ~0.001 to 100 mg m^-3
valid_min <- 0.001
valid_max <- 100

# Warn when there are some high values (may be indicative of HABs)
soft_max  <- 30

# We check:
# 1) no missing means
# 2) values in hard-valid range [valid_min, valid_max]
# 3) values in soft (typical display) range [soft_min, soft_max] -> warning if outside
al_raw <- action_levels(
  warn_at   = 0.01,  # warn if ≥1% fail
  notify_at = 0.05,  # trigger email if ≥5% fail (default email_blast condition uses 'notify')
  stop_at   = 0.25   # mark 'stop' in report if ≥25% fail
)


# Validate raw (per pixel) values
raw_val_agent <-
  create_agent(
    tbl = raw_vals,
    tbl_name = "Gulf state daily Chl-a values",
    actions = al_raw,
    embed_report = TRUE,
    # end_fns = list(
    #   ~ email_blast(
    #     x,
    #     to           = c("recipient1@org.org"),
    #     from         = "you@yourdomain.com",
    #     credentials  = blastula::creds_key(id = "gom_smtp"),
    #     msg_subject  = glue("Gulf Chl-a validation — {format(Sys.Date(), '%Y-%m-%d')}"),
    #     # default is ~ TRUE %in% x$notify; shown here explicitly for clarity
    #     send_condition = ~ TRUE %in% x$notify
    #     # If you want to email on *any* failure level, use:
    #     # send_condition = ~ any(x$warn) || any(x$stop) || any(x$notify)
    #   )
    # )
  ) |>
  # Check for missing values
  col_vals_not_null(vars(chla)) |>
  # Check that values fall between accepted range (based on metadata for product)
  col_vals_between(vars(chla),
                   left = valid_min,
                   right = valid_max,
                   inclusive = c(TRUE, TRUE),
                   na_pass = TRUE) |>
  # Check that values are mostly low (i.e., < 30)
  col_vals_lt(vars(chla),
                   value = soft_max,
              na_pass = TRUE,
                   step_id = "soft_range_check") |>
  interrogate()

# Optional: preview what the email would look like (doesn't send)
pointblank::email_create(raw_val_agent)

# Print a compact report
print(raw_val_agent)
pointblank::get_agent_report(agent)




# Validate raw (per pixel) values
al_mean <- action_levels(
  warn_at   = 0.2,  # warn if ≥20% fail (i.e., 1 state)
  notify_at = 0.2,  # trigger email if ≥20% fail (default email_blast condition uses 'notify')
  stop_at   = 0.25   # mark 'stop' in report if ≥25% fail
)

state_means_agent <-
  create_agent(
    tbl = state_means,
    tbl_name = "Gulf state means for daily Chl-a data",
    actions = al_mean,
    embed_report = TRUE,
    # end_fns = list(
    #   ~ email_blast(
    #     x,
    #     to           = c("recipient1@org.org"),
    #     from         = "you@yourdomain.com",
    #     credentials  = blastula::creds_key(id = "gom_smtp"),
    #     msg_subject  = glue("Gulf Chl-a validation — {format(Sys.Date(), '%Y-%m-%d')}"),
    #     # default is ~ TRUE %in% x$notify; shown here explicitly for clarity
    #     send_condition = ~ TRUE %in% x$notify
    #     # If you want to email on *any* failure level, use:
    #     # send_condition = ~ any(x$warn) || any(x$stop) || any(x$notify)
    #   )
    # )
  ) |>
  # Check for missing values
  col_vals_not_null(vars(chla)) |>
  # Check that values fall between accepted range (especially when averaged for each state)
  col_vals_between(vars(chla),
                   left = 2,
                   right = 20,
                   inclusive = c(TRUE, TRUE),
                   na_pass = TRUE) |>
  # Check that values are mostly low (i.e., < 20)
  col_vals_lt(vars(chla),
              value = 20,
              na_pass = TRUE,
              step_id = "soft_range_check") |>
  interrogate()

# Optional: preview what the email would look like (doesn't send)
pointblank::email_create(state_means_agent)

# Print a compact report
print(state_means_agent)
pointblank::get_agent_report(state_means_agent)






# Save results to CSV for convenience
out_csv <- file.path(tempdir(), paste0("gom_state_chla_means_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
readr::write_csv(results, out_csv)
message("Saved: ", out_csv)

# If you want a quick plot:
# library(ggplot2)
ggplot(results, aes(date, chla_mean, color = state)) +
  geom_line() + geom_point() +
  labs(x = "Date", y = "Chl-a (mg m^-3)", title = "Daily Chlorophyll-a (gap-filled VIIRS) — Gulf State Offshore Means")
