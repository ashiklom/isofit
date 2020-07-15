library(here)
library(dplyr)
library(tidyr)
library(purrr)
library(fst)  # For efficient output storage

source(here("examples", "r-interface", "config.R"))
source(here("examples", "r-interface", "functions.R"))

cwd <- here("examples", "r-interface", "examples")
atmosphere_df <- read.csv(file.path(cwd, "data", "atmospheres.csv"))
outdir <- here("examples", "r-interface", "output")
lrt <- read_libradtran_template(here(
  "examples", "r-interface", "lrt_template.inp"
))

endmembers <- read.csv(file.path(cwd, "data", "endmembers.csv"))
wavelengths <- endmembers$wavelength
endmember_mat <- as.matrix(endmembers[, -1])

ht_matrix <- crossing(
  nesting(true_h2o = atmosphere_df$H2O, true_aot = atmosphere_df$AOD),
  noisefile = list.files(file.path(cwd, "data", "noisefiles"), full.names = TRUE)
)

run_ht <- function(true_h2o, true_aot, noisefile) {
  ht_workflow(
    endmember_mat, true_h2o, true_aot, wavelengths, lrt, outdir,
    instrument_configs = list(parametric_noise_file = noisefile,
                              integrations = 1)
  )
}

ray <- reticulate::import("ray")
ray_init <- ray$init()
ht_matrix$results <- pmap(ht_matrix, run_ht)
ray$shutdown()

htm2 <- ht_matrix %>%
  mutate(
    true_refl = list(as_tibble(endmember_mat)),
    est_refl = map(results, "reflectance") %>%
      map(`colnames<-`, colnames(endmember_mat)) %>%
      map(tibble::as_tibble),
    wavelength = list(wavelengths)
  ) %>%
  dplyr::select(-results) %>%
  unnest(c(est_refl, true_refl, wavelength), names_sep = "|") %>%
  rename(wavelength = "wavelength|wavelength")

results_long <- htm2 %>%
  pivot_longer(
    -c(true_h2o, true_aot, noisefile, wavelength),
    names_to = c("source", "endmember"),
    names_sep = "\\|"
  ) %>%
  separate(endmember, c("type", "id"), sep = "\\.")

results2 <- results_long %>%
  pivot_wider(
    names_from = "source",
    values_from = "value"
  )

write_fst(results2, file.path(cwd, "data", "endmember-results.fst"))

# Some basic summary stats
summaries <- results2 %>%
  group_by(true_h2o, true_aot, noise = basename(noisefile), type) %>%
  summarize(rmse = sqrt(mean((true_refl - est_refl)^2, na.rm = TRUE)))

if (interactive()) {
  library(ggplot2)
  ggplot(summaries) +
    aes(x = noise, y = rmse, fill = type) +
    geom_boxplot()
}
