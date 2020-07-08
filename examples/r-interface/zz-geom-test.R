library(here)
source(here("examples/r-interface/zz-common.R"))
source(here("examples/r-interface/functions.R"))
lut <- list(
  AOT550 = c(0.001, 0.6),
  H2OSTR = c(1.0, 3.25),
  OBSZEN = c(0, 20)
)
r <- ht_workflow(
    reflectance, 1.5, 0.2, wavelengths,
    libradtran_template,
    libradtran_basedir,
    lut = lut,
    outdir = outdir
)

plot(wavelengths, r$reflectance, type = "l")
