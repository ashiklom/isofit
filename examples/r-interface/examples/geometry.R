library(here)

source(here("examples", "r-interface", "config.R"))
source(here("examples", "r-interface", "functions.R"))

wavelengths <- seq(400, 2500, 5)
true_refl <- drop(rrtm::pro4sail_4(1.4, 40, 0.01, 0.01, 3, 0.5)$bdr)
reflectance <- approx(400:2500, true_refl, wavelengths,
                      yleft = 0, yright = 0)$y
template_file <- here("examples", "r-interface", "lrt_template.inp")
libradtran_template <- read_libradtran_template(template_file)

outdir <- here("examples", "r-interface", "output")
dir.create(outdir, showWarnings = FALSE)

# Default zenith angle
r1 <- ht_workflow(
    reflectance, 1.5, 0.2, wavelengths,
    libradtran_template,
    LIBRADTRAN_DIR,
    libradtran_environment = LIBRADTRAN_ENV,
    outdir = outdir
)

# Modify observer zenith angle
r2 <- ht_workflow(
    reflectance, 1.5, 0.2, wavelengths,
    libradtran_template,
    geom = list(observer_zenith = 20),
    LIBRADTRAN_DIR,
    libradtran_environment = LIBRADTRAN_ENV,
    outdir = outdir
)

if (interactive()) {
    matplot(wavelengths, cbind(r1$reflectance, r2$reflectance), type = "l")
}
