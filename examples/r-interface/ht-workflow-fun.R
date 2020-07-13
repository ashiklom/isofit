library(here)
library(fs)

# Configuration file
source(here("examples", "r-interface", "config.R"))
source(here("examples", "r-interface", "functions.R"))

# Arguments
wavelengths <- seq(400, 2500, 5)
true_refl <- drop(rrtm::pro4sail_4(1.4, 40, 0.01, 0.01, 3, 0.5)$bdr)
reflectance <- approx(400:2500, true_refl, wavelengths,
                      yleft = 0, yright = 0)$y
template_file <- here("examples", "r-interface", "lrt_template.inp")
libradtran_template <- read_libradtran_template(template_file)
# This object is a list that can be modified. For example:
#     libradtran_template$source <- "solar /path/to/solar_flux..."
#     libradtran_template$time <- c(2017, 7, 1, 18, 0, 0)
#
# Currently, this is a bit clunky (e.g. `mol_modify` corresponds to two
# sections) and will be refined in future versions. The important thing is that
# the way this code figures out whether or not to regenerate LUTs is by creating
# a MD5 digest of this object. Therefore, we skip creating LUTs for identical
# templates, but make sure to create new ones when the template changes.

# Outputs stored in this directory.
outdir <- dir_create(here("examples", "r-interface", "output"))

# If you want to purge all existing outputs, run the following:
# unlink(outdir, recursive = TRUE)

# Run a single reflectance spectrum
r1 <- ht_workflow(
  reflectance, 1.5, 0.2, wavelengths,
  libradtran_template, outdir
)
plot(wavelengths, r1$reflectance, type = 'l',
     ylim = range(r1$reflectance, reflectance, na.rm = TRUE))
lines(wavelengths, reflectance, col = "red")
legend("topright", c("estimated", "true"), lty = 1,
       col = c("black", "red"))

# See ht_workflow documentation in the `functions.R` file for more details.
