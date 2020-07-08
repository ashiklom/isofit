library(reticulate)
library(here)
library(fs)
library(digest)

source(here("examples", "r-interface", "functions.R"))

# Conda environment where Isofit is installed
use_condaenv("new-isofit")
# LibRadTran source code directory
libradtran_basedir <- "~/projects/models/libradtran/libRadtran-2.0.3/"
template_file <- here("examples", "r-interface", "lrt_template.inp")

# Arguments
wavelengths <- seq(400, 2500, 5)
true_refl <- drop(rrtm::pro4sail_4(1.4, 40, 0.01, 0.01, 3, 0.5)$bdr)
reflectance <- approx(400:2500, true_refl, wavelengths,
                      yleft = 0, yright = 0)$y
libradtran_template <- read_libradtran_template(template_file)

# Outputs stored in this directory.
outdir <- dir_create(here("examples", "r-interface", "output"))
