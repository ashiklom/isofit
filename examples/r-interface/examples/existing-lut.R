library(here)
source(here("examples", "r-interface", "config.R"))
source(here("examples", "r-interface", "functions.R"))

outdir <- here("examples", "r-interface", "output")

lut_path <- here("examples", "r-interface", "examples", "existing-modtran-lut")
modtran_template <- file.path(lut_path, "modtran-config.json")

# LUT wavelengths here need to match the input wavelengths. Therefore, we read
# the wavelengths from one of the existing LUTs as a workaround.
chn <- read.table(file.path(lut_path, "AOT550-0.0100_H2OSTR-1.5000.chn"),
                  skip = 5, header = FALSE)
wl <- chn[,1]
refl_full <- rrtm::pro4sail_4(1.4, 40, 0.01, 0.01, 3, 0.5)$bdr[,1]
refl <- approx(400:2500, refl_full, xout = wl, rule = 2)$y

vconf <- list(
  engine_name = "modtran",
  engine_base_dir = NULL,
  lut_path = lut_path,
  template_file = modtran_template
)

result <- ht_workflow(
  reflectance = refl,
  true_h2o = 1.7,
  true_aot = 0.05,
  wavelengths = wl,
  libradtran_template = NULL,
  outdir = outdir,
  # NOTE: These have to match the existing LUTs. Otherwise, the code will try to
  # rebuild them.
  aot_lut = c(0.01, 0.1),
  h2o_lut = c(1.5, 2.0),
  vswir_configs = vconf
)

if (interactive()) {
  matplot(wl, cbind(refl, result$reflectance), type = "l",
          lty = "solid")
}
