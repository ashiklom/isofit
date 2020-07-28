#!/usr/bin/env python3

import copy
import numpy as np
import ray
import pathlib
import spectral
import pickle
import multiprocessing

from isofit.configs.configs import Config
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.core.geometry import Geometry
from isofit.core.fileio import IO


def do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                  libradtran_template_file,
                  lutdir, outdir,
                  surface_file="./data/prior.mat",
                  noisefile=None, snr=300,
                  aod=0.1, h2o=1.0, lrt_atmosphere_type="midlatitude_winter",
                  atm_aod_h2o=None,
                  solar_zenith=0, observer_zenith=0,
                  solar_azimuth=0, observer_azimuth=0,
                  create_lut=True):
    """One iteration of the hypertrace workflow.

    Required arguments:
        isofit_config: dict of isofit configuration options

        `wavelength_file`: Path to ASCII space delimited table containing two
        columns, wavelength and full width half max (FWHM); both in nanometers.

        `reflectance_file`: Path to input reflectance file. Note that this has
        to be an ENVI-formatted binary reflectance file, and this path is to the
        associated header file (`.hdr`), not the image file itself (following
        the convention of the `spectral` Python library, which will be used to
        read this file).

        libradtran_template_file: Path to the LibRadtran template. Note that
        this is slightly different from the Isofit template in that the Isofit
        fields are surrounded by two sets of `{{` while a few additional options
        related to geometry are surrounded by just `{` (this is because
        Hypertrace does an initial pass at formatting the files).

        `lutdir`: Directory where look-up tables will be stored. Will be created
        if missing.

        `outdir`: Directory where outputs will be stored. Will be created if
        missing.

    Keyword arguments:
      surface_file: Matlab (`.mat`) file containing a multicomponent surface
      prior. See Isofit documentation for details.

      noisefile: Parametric instrument noise file. See Isofit documentation for
      details. Default = `None`

      snr: Instrument signal-to-noise ratio. Ignored if `noisefile` is present.
      Default = 300

      aod: True aerosol optical depth. Default = 0.1

      h2o: True water vapor content. Default = 1.0

      lrt_atmosphere_type: LibRadtran atmosphere type. See LibRadtran manual for
      details. Default = `midlatitude_winter`

      atm_aod_h2o: A list containing three elements: The atmosphere type, AOD,
      and H2O. This provides a way to iterate over specific known atmospheres
      that are combinations of the three previous variables. If this is set, it
      overrides the three previous arguments. Default = `None`

      solar_zenith, `observer_zenith`: Solar and observer zenith angles,
      respectively (0 = directly overhead, 90 = horizon). These are in degrees
      off nadir. Default = 0 for both. (Note that off-nadir angles make
      LibRadtran run _much_ more slowly, so be prepared if you need to generate
      those LUTs).

      solar_azimuth, `observer_azimuth`: Solar and observer azimuth angles,
      respectively, in degrees. Observer azimuth is the sensor _position_ (so
      180 degrees off from view direction) relative to N, rotating
      counterclockwise; i.e., 0 = Sensor in N, looking S; 90 = Sensor in W,
      looking E (this follows the LibRadtran convention). Default = 0 for both.
    """

    outdir = mkabs(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reflectance = spectral.open_image(reflectance_file)
    spatial_dim = reflectance.shape[0:2]
    nwl = np.loadtxt(wavelength_file).shape[0]
    output_dim = np.concatenate((spatial_dim, [nwl]))

    isofit_config2 = copy.copy(isofit_config)
    # NOTE: All of these settings are *not* copied, but referenced. So these
    # changes propagate to the `forward_settings` object below.
    forward_settings = isofit_config2["forward_model"]
    instrument_settings = forward_settings["instrument"]
    instrument_settings["wavelength_file"] = wavelength_file
    surface_settings = forward_settings["surface"]
    surface_settings["surface_file"] = surface_file
    surface_settings["wavelength_file"] = wavelength_file
    if noisefile is not None:
        noisetag = f"noise_{pathlib.Path(noisefile).name}"
        if "SNR" in instrument_settings:
            instrument_settings.pop("SNR")
        instrument_settings["parametric_noise_file"] = noisefile
        if "integrations" not in instrument_settings:
            instrument_settings["integrations"] = 1
    elif snr is not None:
        noisetag = f"snr_{snr}"
        instrument_settings["SNR"] = snr

    if atm_aod_h2o is not None:
        lrt_atmosphere_type = atm_aod_h2o[0]
        aod = atm_aod_h2o[1]
        h2o = atm_aod_h2o[2]

    lrttag = f"atm_{lrt_atmosphere_type}__" +\
        f"szen_{solar_zenith}__" +\
        f"ozen_{observer_zenith}__" +\
        f"saz_{solar_azimuth}__" +\
        f"oaz_{observer_azimuth}"
    atmtag = f"aod_{aod}__h2o_{h2o}"

    if create_lut:
        lutdir = mkabs(lutdir)
        lutdir.mkdir(parents=True, exist_ok=True)
        lutdir2 = lutdir / lrttag
        lutdir2.mkdir(parents=True, exist_ok=True)
        lrtfile = lutdir2 / "lrt-template.inp"
        vswir_conf = forward_settings["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
        with open(libradtran_template_file, "r") as f:
            fs = f.read()
            open(lrtfile, "w").write(fs.format(
                atmosphere=lrt_atmosphere_type, solar_azimuth=solar_azimuth,
                solar_zenith=solar_zenith,
                cos_observer_zenith=np.cos(observer_zenith * np.pi / 180.0),
                observer_azimuth=observer_azimuth
            ))
        open(lutdir2 / "prescribed_geom", "w").write(f"99:99:99   {solar_zenith}  {solar_azimuth}")
        vswir_conf["lut_path"] = lutdir2
        vswir_conf["template_file"] = lrtfile

    outdir2 = outdir / lrttag / noisetag / atmtag
    outdir2.mkdir(parents=True, exist_ok=True)

    # Create Isofit objects
    fm = ForwardModel(Config({"forward_model": forward_settings}))
    geomvec = [
        -999,              # path length; not used
        observer_azimuth,  # Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
        observer_zenith,   # Degrees 0-90; 0 = directly overhead, 90 = horizon
        solar_azimuth,     # Degrees 0-360; 0 = N, 90 = W, 180 = S, 270 = E
        solar_zenith       # Same units as observer zenith
    ]
    igeom = Geometry(obs=geomvec)
    inverse_settings = isofit_config2["implementation"]
    iv = Inversion(Config({"implementation": inverse_settings}), fm)

    # Create output files and associated memmaps
    radiance_file = outdir2 / "toa-radiance.hdr"
    spectral.envi.create_image(radiance_file,
                               shape=output_dim,
                               dtype=np.float32,
                               force=True)
    est_refl_file = outdir2 / "estimated-reflectance.hdr"
    spectral.envi.create_image(est_refl_file,
                               shape=output_dim,
                               dtype=np.float32,
                               force=True)

    # Set up indices for parallelization
    irows, icols = np.nonzero(np.ones(spatial_dim))
    n_iter = len(irows)
    if "ncores" in inverse_settings:
        n_workers = min(inverse_settings["ncores"], n_iter)
    else:
        n_workers = min(multiprocessing.cpu_count(), n_iter)
    print(f"Running on {n_workers} workers")
    index_sets = np.linspace(0, n_iter, n_workers + 1, dtype=int)

    # Cache some immutable objects
    irows_id = ray.put(irows)
    icols_id = ray.put(icols)
    fm_id = ray.put(fm)
    iv_id = ray.put(iv)
    igeom_id = ray.put(igeom)

    # Run the workflow (in parallel)
    _ = ray.get([
        ht_run.remote(reflectance_file, irows_id, icols_id,
                      radiance_file, est_refl_file,
                      aod, h2o, fm_id, iv_id, igeom_id,
                      index_sets[i], index_sets[i + 1])
        for i in range(len(index_sets)-1)
    ])
    return outdir2


def ht_radiance(refl, aot, h2o, fm, igeom):
    """Calculate TOA radiance."""
    statevec = np.concatenate((refl, aot, h2o), axis=None)
    radiance = fm.calc_rdn(statevec, igeom)
    return radiance


def ht_invert(rad, iv, igeom):
    """Invert TOA radiance to get reflectance."""
    state_trajectory = iv.invert(rad, igeom)
    state_est = state_trajectory[-1]
    unc = iv.forward_uncertainty(state_est, rad, igeom)
    return unc


@ray.remote
def ht_run(reflectance_file, rows, cols,
           radiance_file, est_refl_file,
           aod, h2o, fm, iv, igeom,
           index_start, index_stop):
    reflectance = spectral.open_image(reflectance_file)
    radiance = spectral.open_image(radiance_file)
    radiance_m = radiance.open_memmap(writable=True)
    est_refl = spectral.open_image(est_refl_file)
    est_refl_m = est_refl.open_memmap(writable=True)
    for index in range(index_start, index_stop):
        ix = rows[index]
        iy = cols[index]
        refl = reflectance[ix, iy]
        if np.all(refl <= 0.0):
            continue
        rad = ht_radiance(refl, aod, h2o, fm, igeom)
        radiance_m[ix, iy] = rad
        unc = ht_invert(rad, iv, igeom)
        # TODO: Store the rest of this uncertainty information
        est_refl = unc[0]
        est_refl_m[ix, iy] = est_refl


def mkabs(path):
    """Make a path absolute."""
    path2 = pathlib.Path(path)
    return path2.expanduser().resolve()
