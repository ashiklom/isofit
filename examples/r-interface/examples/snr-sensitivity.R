# Perform a sensitivity analysis against signal-noise ratio
library(here)
library(reticulate)

source(here("examples", "r-interface", "config.R"))
source(here("examples", "r-interface", "functions.R"))

template_file <- here("examples", "r-interface", "lrt_template.inp")

# Arguments
wavelengths <- seq(400, 2500, 5)
true_refl <- drop(rrtm::pro4sail_4(1.4, 40, 0.01, 0.01, 3, 0.5)$bdr)
reflectance <- approx(400:2500, true_refl, wavelengths,
                      yleft = 0, yright = 0)$y
libradtran_template <- read_libradtran_template(template_file)

# Outputs stored in this directory.
outdir <- here("examples", "r-interface", "output")
dir.create(outdir, showWarnings = FALSE)

run_snr <- function(snr) {
  ht_workflow(
    reflectance, 1.5, 0.2, wavelengths,
    libradtran_template,
    LIBRADTRAN_DIR,
    instrument_configs = list(SNR = snr),
    outdir = outdir
  )
}

# Run a single reflectance spectrum
snr <- c(1, 10, 100, 200)
ray <- import("ray")
ray_init <- ray$init()
outlist <- lapply(snr, run_snr)
ray$shutdown()
out_refl <- do.call(cbind, lapply(outlist, "[[", "reflectance"))

figdir <- here("examples", "r-interface", "figures")
dir.create(figdir, showWarnings = FALSE)
png(file.path(figdir, "snr-simple.png"))
matplot(wavelengths, out_refl, type = "l")
legend("topright", as.character(snr), title = "SNR",
       lty = seq_along(snr), col = seq_along(snr))
dev.off()
