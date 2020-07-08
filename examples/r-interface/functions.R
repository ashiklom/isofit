expcov <- function(ndim, phi = 1, alpha = 1) {
  dmat <- as.matrix(dist(seq_len(ndim)))
  # TODO: Allow alpha to be a vector
  phi * exp(-dmat * alpha)
}

maybe_numeric <- function(x) {
  tryCatch(
    as.numeric(x),
    warning = function(e) x
  )
}

read_libradtran_template <- function(template_file) {
  rawstring <- readLines(template_file)
  nocomments <- grep("^ *#", rawstring, value = TRUE, invert = TRUE)
  noblanks <- grep("^ *$", nocomments, value = TRUE, invert = TRUE)
  splits <- strsplit(noblanks, " ")
  fnames <- vapply(splits, "[[", character(1), 1)
  fvals <- lapply(splits, "[", -1)
  ## fvals <- lapply(fvals, maybe_numeric)
  fvals[fvals == ""] <- NA
  fvals[lengths(fvals) == 0] <- NA
  # TODO: Handle special cases
  # - time
  # - latitude
  # - longitude
  # Concatenate characters
  chars <- vapply(fvals, is.character, logical(1))
  fvals[chars] <- lapply(fvals[chars], paste, collapse = " ")
  names(fvals) <- fnames
  fvals
}

write_libradtran_template <- function(template_list, con) {
  template_list[is.na(template_list)] <- ""
  lines <- paste(names(template_list), template_list)
  writeLines(trimws(lines), con)
}

#' Hypertrace workflow for a single spectrum
#'
#' @param reflectance Known surface reflectance vector
#' @param true_h20 Known atmospheric H2O
#' @param true_aot Known atospheric AOT
#' @param wavelengths Vector of reflectance wavelengths (nm)
#' @param libradtran_template Libradtran template list
#' @param libradtran_basedir Source directory of LibRadTran
#' @param libradtran_environment String of environment declarations. Should be
#'   a single string, with multiple declarations separated by a newline (`\n`).
#' @param instrument_configs Instrument configuration list (default = `list(SNR = 300)`)
#' @param aot_state Modifications to AOT statevector
#' @param h2o_state Modifications to H2O statevector
#' @param lut Look-up table named list (e.g. `list(AOT550 = c(...), H2OSTR = c(...))`)
#' @param outdir Output directory (default = "output")
#' @param inversion_windows List of inversion windows (start, end)
#' @param prior_mean Prior mean (matrix, components x wavelengths)
#' @param prior_cov Prior covariance matrix (array, components x wavelengths x wavelengths)
#' @return Forward uncertainty list
#' @author Alexey Shiklomanov
ht_workflow <- function(reflectance,
                        true_h20,
                        true_aot,
                        wavelengths,
                        libradtran_template,
                        libradtran_basedir,
                        libradtran_environment = "",
                        instrument_configs = list(SNR = 300),
                        aot_state = list(),
                        h2o_state = list(),
                        lut = list(
                          AOT550 = c(0.001, 0.123, 0.6),
                          H2OSTR = c(1.0, 2.5, 3.25)
                        ),
                        outdir = "output",
                        inversion_windows = list(
                          c(400, 1300),
                          c(1450, 1780),
                          c(1950, 2450)
                        ),
                        prior_mean = t(rep(0.5, length(wavelengths))),
                        prior_cov = array(
                          diag(1, length(wavelengths)),
                          c(1, length(wavelengths), length(wavelengths))
                        )) {

  stopifnot(
    length(reflectance) == length(wavelengths),
    length(dim(prior_mean)) == 2
  )
  if (mean(wavelengths) < 300) stop("Wavelengths should be in nm, not um")

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  outdir <- normalizePath(outdir, mustWork = TRUE)
  lrt_digest <- digest::digest(libradtran_template)
  lut_outdir <- file.path(outdir, "lut", lrt_digest)
  dir.create(lut_outdir, showWarnings = FALSE, recursive = TRUE)
  lrt_file <- file.path(lut_outdir, "00-libradtran-template.inp")
  write_libradtran_template(libradtran_template, lrt_file)

  # Create wavelength file
  wavelength_df <- data.frame(wl = wavelengths, fwhm = 10)
  wavelength_file <- file.path(outdir, "wavelengths.txt")
  write.table(wavelength_df, wavelength_file,
              row.names = FALSE, col.names = FALSE)

  # Create prior file
  prior_ref_wl <- numeric()
  for (w in inversion_windows) {
    prior_ref_wl <- c(prior_ref_wl, wavelengths[
      wavelengths >= w[1] & wavelengths <= w[2]
    ])
  }

  prior_file <- file.path(outdir, "prior.mat")
  R.matlab::writeMat(
    con = prior_file,
    normalize = "Euclidean",
    # HACK: Should match number of components
    wl = t(wavelengths),
    means = prior_mean,
    covs = prior_cov,
    # HACK: Should match number of components
    refwl = t(prior_ref_wl)
  )

  # Create Python config dicts
  instrument_configs2 <- modifyList(list(
    wavelength_file = wavelength_file
  ), instrument_configs)
  instrument_settings <- do.call(dict, instrument_configs2)

  state_names <- c("H2OSTR", "AOT550")
  aot_state_default <- list(
    bounds = c(0.001, 0.6),
    init = 0.123,
    prior_mean = 0.123,
    prior_sigma = 0.3,
    scale = 0.01
  )
  aot_state2 <- modifyList(aot_state_default, aot_state)
  h2o_state_default <- list(
    bounds = c(1.0, 3.216),
    init = 2.716,
    prior_mean = 2.716,
    prior_sigma = 1.0,
    scale = 0.01
  )
  h2o_state2 <- modifyList(h2o_state_default, h2o_state)

  lrt_wavelengths_str <- libradtran_template[["wavelength"]]
  lrt_wavelengths <- as.numeric(strsplit(lrt_wavelengths_str, " ")[[1]])

  lut_grid2 <- do.call(dict, lut)
  lut_names <- names(lut)

  rtm_settings <- dict(
    lut_grid = lut_grid2,
    radiative_transfer_engines = dict(
      vswir = dict(
        engine_name = "libradtran",
        engine_base_dir = libradtran_basedir,
        environment = libradtran_environment,
        lut_names = lut_names,
        lut_path = lut_outdir,
        statevector_names = state_names,
        template_file = lrt_file,
        wavelength_range = lrt_wavelengths
      )
    ),
    statevector = dict(
      AOT550 = do.call(dict, aot_state2),
      H2OSTR = do.call(dict, h2o_state2)
    ),
    unknowns = dict(H2O_ABSCO = 0.01)
  )

  surface_settings <- dict(
    surface_category = "multicomponent_surface",
    wavelength_file = wavelength_file,
    surface_file = prior_file
  )
 
  inversion_settings <- dict(implementation = dict(
    mode = "inversion",
    inversion = dict(
      windows = do.call(rbind, inversion_windows)
    )
  ))

  # Load Python modules
  isofit <- import("isofit")
  ray <- import("ray")
  isofit_forward <- import("isofit.core.forward")
  isofit_configs <- import("isofit.configs.configs")
  isofit_inverse <- import("isofit.inversion.inverse")
  isofit_geometry <- import("isofit.core.geometry")

  fm_config <- isofit_configs$Config(dict(
    forward_model = dict(instrument = instrument_settings,
                         surface = surface_settings,
                         radiative_transfer = rtm_settings)
  ))

  if (!ray$is_initialized()) ray_server <- ray$init()
  message("Building forward model...")
  fm <- isofit_forward$ForwardModel(fm_config)

  geom <- isofit_geometry$Geometry()

  radiance <- fm$calc_rdn(np_array(c(reflectance, true_aot, true_h20)), geom)

  inverse_config <- isofit_configs$Config(inversion_settings)
  iv <- isofit_inverse$Inversion(inverse_config, fm)
  message("Performing inversion...")
  state_trajectory <- iv$invert(radiance, geom)

  message("Post-processing...")
  state_est <- np_array(drop(tail(state_trajectory, 1)))
  unc <- iv$forward_uncertainty(state_est, radiance, geom)
  names(unc) <- c("reflectance_full", "radiance", "path", "S_hat", "K", "G")
  unc$A <- unc$G %*% unc$K
  unc$trajectory <- state_trajectory

  # Remove inversion windows
  iwindows_l <- lapply(
    inversion_windows,
    function(x) wavelengths > x[1] & wavelengths < x[2]
  )
  iwindows <- Reduce(`|`, iwindows_l)
  unc$reflectance <- unc$reflectance_full
  unc$reflectance[!iwindows] <- NA
  message("Done!")
  unc
}
