
# Function to build ERDDAP URL for single time slice, altitude=0, and given lat/lon bounds
#the url string is likely to vary across products, so will likely need to modify outside of this example
build_erddap_url <- function(date_iso, xmin, xmax, ymin, ymax) {
  
  # Times must be ISO8601 Z
  tstr <- paste0(date_iso, "T12:00:00Z")
  
  # Build full URL (with encoding for brackets; '%5B' = '[' and '%5D' = ']')
  paste0(
    erddap_base, ".nc?",
    varname, "%5B(", tstr, "):1:(", tstr, ")%5D%5B(0.0):1:(0.0)%5D",
    "%5B(", ymax, "):1:(", ymin, ")%5D",
    "%5B(", xmin, "):1:(", xmax, ")%5D"
  )
}