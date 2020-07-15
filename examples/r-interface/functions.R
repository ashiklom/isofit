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
#' @param true_h2o Known atmospheric H2O
#' @param true_aot Known atospheric AOT
#' @param wavelengths Vector of reflectance wavelengths (nm)
#' @param libradtran_template Libradtran template list
#' @param outdir Output directory
#' @param libradtran_basedir Source directory of LibRadTran (default = `LIBRADTRAN_DIR`)
#' @param libradtran_environment String of environment declarations. Should be a
#'   single string, with multiple declarations separated by a newline (`\n`).
#'   Default is value of `LIBRADTRAN_ENV`.
#' @param instrument_configs Instrument configuration list (default = `list(SNR = 300)`)
#' @param geom Named list of geometry parameters. Names must be in:
#'   `observer_azmiuth`, `observer_zenith`, `solar_azimuth`, `solar_zenith`. All
#'   are in degrees.
#' @param aot_state Modifications to AOT statevector
#' @param h2o_state Modifications to H2O statevector
#' @param aot_lut AOT look-up table grid (default = `c(0.001, 0.123, 0.6)`)
#' @param h2o_lut H2O look-up table grid (default = `c(1.0, 2.5, 3.25)`)
#' @param inversion_windows List of inversion windows (start, end)
#' @param prior_mean Prior mean (matrix, components x wavelengths)
#' @param prior_cov Prior covariance matrix (array, components x wavelengths x wavelengths)
#' @param vswir_configs Modifications to VSWIR RTM configuration, as a named list (default = `list()`).
#' @return Forward uncertainty list
#' @author Alexey Shiklomanov
ht_workflow <- function(reflectance,
                        true_h2o,
                        true_aot,
                        wavelengths,
                        libradtran_template,
                        outdir,
                        libradtran_basedir = LIBRADTRAN_DIR,
                        libradtran_environment = LIBRADTRAN_ENV,
                        instrument_configs = list(SNR = 300),
                        geom = list(),
                        aot_state = list(),
                        h2o_state = list(),
                        aot_lut = c(0.001, 0.123, 0.6),
                        h2o_lut = c(1.0, 2.5, 3.25),
                        inversion_windows = list(
                          c(400, 1300),
                          c(1450, 1780),
                          c(1950, 2450)
                        ),
                        prior_mean = t(rep(0.5, length(wavelengths))),
                        prior_cov = array(
                          diag(1, length(wavelengths)),
                          c(1, length(wavelengths), length(wavelengths))
                        ),
                        vswir_configs = list()) {

  stopifnot(
    NROW(reflectance) == length(wavelengths),
    length(dim(prior_mean)) == 2
  )
  if (mean(wavelengths) < 300) stop("Wavelengths should be in nm, not um")

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  outdir <- normalizePath(outdir, mustWork = TRUE)

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

  nwl <- length(wavelengths)
  nwl_ref <- length(prior_ref_wl)
  ncomponents <- nrow(prior_mean)

  prior_file <- file.path(outdir, "prior.mat")
  R.matlab::writeMat(
    con = prior_file,
    normalize = "Euclidean",
    wl = t(matrix(wavelengths, nwl, ncomponents)),
    means = prior_mean,
    covs = prior_cov,
    refwl = t(matrix(prior_ref_wl, nwl_ref, ncomponents))
  )

  # Create Python config dicts
  instrument_configs2 <- modifyList(list(
    wavelength_file = wavelength_file
  ), instrument_configs)
  instrument_settings <- do.call(reticulate::dict, instrument_configs2)

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

  # Geometry configuration
  geom_default <- list(
    path_length = -999, # not used
    observer_azimuth = 0, # Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
    observer_zenith = 0, # Degrees 0-90; 0 = directly overhead, 90 = horizon
    solar_azimuth = 0, # Degrees 0-360; 0 = N, 90 = W, 180 = S, 270 = E
    solar_zenith = 0   # not used (determined from time)
  )
  stopifnot(all(names(geom) %in% names(geom_default)))
  if ("path_length" %in% names(geom)) {
    warning("`path_length` geometry argument is not used.")
  }
  if ("solar_zenith" %in% names(geom)) {
    warning("`solar_zenith` in geometry is not used. ",
            "Set the `time` in the libradtran template instead.")
  }
  geom_list <- modifyList(geom_default, geom)
  geomvec <- unname(unlist(geom_list))

  if (is.null(vswir_configs[["lut_path"]])) {
    # Get the libradtran wavelength range from its template
    lrt_wavelengths_str <- libradtran_template[["wavelength"]]
    lrt_wavelengths <- as.numeric(strsplit(lrt_wavelengths_str, " ")[[1]])

    # Modify libradtran template with geometry information
    phi0 <- geom_list$solar_azimuth + 180
    if (phi0 >= 360) phi0 <- phi0 - 360
    lrt <- modifyList(libradtran_template, list(
      umu = cos(geom_list$observer_zenith * pi / 180),
      phi = geom_list$observer_azimuth,
      phi0 = phi0
    ))
    lrt_digest <- digest::digest(lrt)
    lut_outdir <- file.path(outdir, "lut", lrt_digest)
    dir.create(lut_outdir, showWarnings = FALSE, recursive = TRUE)
    lrt_file <- file.path(lut_outdir, "00-libradtran-template.inp")
    write_libradtran_template(lrt, lrt_file)
  } else {
    message(
      "Detected custom LUT path. ",
      "Assuming pre-existing LUTs and ignoring in-memory Libradtran template. ",
      "If you need to create your own LUTs, you will need to include ",
      "`template_file = '/path/to/template/file'` in the ",
      "`vswir_configs` list."
    )
    lut_outdir <- NULL
    lrt_file <- NULL
    lrt_wavelengths <- NULL
  }

  ## vswir_configs <- vconf

  vswir_config_default <- list(
    engine_name = "libradtran",
    engine_base_dir = libradtran_basedir,
    environment = libradtran_environment,
    lut_names = state_names,
    lut_path = lut_outdir,
    statevector_names = state_names,
    template_file = lrt_file,
    wavelength_range = lrt_wavelengths
  )
  vswir_config2 <- modifyList(vswir_config_default, vswir_configs)
  vswir_dict <- do.call(reticulate::dict, vswir_config2)

  rtm_settings <- reticulate::dict(
    lut_grid = reticulate::dict(AOT550 = aot_lut, H2OSTR = h2o_lut),
    radiative_transfer_engines = reticulate::dict(vswir = vswir_dict),
    statevector = reticulate::dict(
      AOT550 = do.call(reticulate::dict, aot_state2),
      H2OSTR = do.call(reticulate::dict, h2o_state2)
    ),
    unknowns = reticulate::dict(H2O_ABSCO = 0.01)
  )

  surface_settings <- reticulate::dict(
    surface_category = "multicomponent_surface",
    wavelength_file = wavelength_file,
    surface_file = prior_file
  )

  inversion_settings <- reticulate::dict(implementation = reticulate::dict(
    mode = "inversion",
    inversion = reticulate::dict(
      windows = do.call(rbind, inversion_windows)
    )
  ))

  # Load Python modules
  isofit <- reticulate::import("isofit")
  ray <- reticulate::import("ray")
  isofit_forward <- reticulate::import("isofit.core.forward")
  isofit_configs <- reticulate::import("isofit.configs.configs")
  isofit_inverse <- reticulate::import("isofit.inversion.inverse")
  isofit_geometry <- reticulate::import("isofit.core.geometry")

  fm_config <- isofit_configs$Config(reticulate::dict(
    forward_model = reticulate::dict(instrument = instrument_settings,
                         surface = surface_settings,
                         radiative_transfer = rtm_settings)
  ))
  inverse_config <- isofit_configs$Config(inversion_settings)

  if (!ray$is_initialized()) {
    init_ray <- ray$init()
    on.exit(ray$shutdown(), add = TRUE)
  }
  message("Building forward model...")
  fm <- isofit_forward$ForwardModel(fm_config)
  iv <- isofit_inverse$Inversion(inverse_config, fm)

  igeom <- isofit_geometry$Geometry(obs = geomvec)

  # Runs the meat of the workflow, in parallel!
  message("Performing inversions...")
  if (!is.matrix(reflectance)) reflectance <- as.matrix(reflectance)
  reticulate::py_run_string(paste(
    "import numpy as np",
    "import ray",
    "def htworkflow(refl, aot, h2o, fm, iv, igeom):",
    "  statevec = np.concatenate((refl, aot, h2o), axis=None)",
    "  radiance = fm.calc_rdn(statevec, igeom)",
    "  state_trajectory = iv.invert(radiance, igeom)",
    "  state_est = state_trajectory[-1]",
    "  unc = iv.forward_uncertainty(state_est, radiance, igeom)",
    "  return radiance, state_trajectory, unc",
    "htworkflow_r = ray.remote(htworkflow)",
    "def htworkflow_multi(reflmat, *args):",
    "  return ray.get([htworkflow_r.remote(refl, *args) for refl in reflmat.T])",
    sep = "\n"
  ))

  result <- reticulate::py$htworkflow_multi(
    reflectance, true_aot, true_h2o,
    fm, iv, igeom
  )

  message("Post-processing...")

  # Matrix, n_wl x n_reflectance
  radiance <- do.call(cbind, lapply(result, "[[", 1))

  # Trajectory, list(n_reflectance)
  state_trajectory <- lapply(result, "[[", 2)

  # Forward uncertainty
  unc <- lapply(result, "[[", 3)
  unc <- lapply(
    unc, setNames,
    c("reflectance_full", "radiance", "path", "S_hat", "K", "G")
  )

  refl_est <- do.call(cbind, lapply(unc, "[[", "reflectance_full"))

  # Remove inversion windows
  iwindows_l <- lapply(
    inversion_windows,
    function(x) wavelengths > x[1] & wavelengths < x[2]
  )
  iwindows <- Reduce(`|`, iwindows_l)
  refl_est[!iwindows, ] <- NA
  message("Done!")
  list(
    reflectance = refl_est,
    radiance = radiance,
    fw_uncertainty = unc,
    state_trajectory = state_trajectory
  )
}
