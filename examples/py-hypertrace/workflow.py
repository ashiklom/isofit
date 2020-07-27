#!/usr/bin/env python3

from pathlib import Path
from atexit import register
import json
import copy
import pickle
import itertools

import numpy as np
import ray
import spectral

from isofit.configs.configs import Config
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.core.geometry import Geometry

from hypertrace import ht_radiance, ht_invert, mkabs

with open("./config.json") as f:
    config = json.load(f)

wavelength_file = mkabs(config["wavelength_file"])
surface_file = mkabs(config["surface_file"])
reflectance_file = mkabs(config["reflectance_file"])
libradtran_template_file = mkabs(config["libradtran_template_file"])

isofit_config = config["isofit"]
hypertrace_config = config["hypertrace"]

# Make RTM paths absolute
vswir_settings = isofit_config["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
for key in ["lut_path", "template_file", "engine_base_dir"]:
    if key in vswir_settings:
        vswir_settings[key] = mkabs(vswir_settings[key])

if not ray.is_initialized():
    ray.init()
    register(ray.shutdown)


def do_hypertrace(isofit_config,
                  wavelength_file, surface_file, reflectance_file,
                  libradtran_template_file,
                  noisefile=None, snr=300,
                  aod=0.1, h2o=1.0, lrt_atmosphere_type="midlatitude_winter",
                  atm_aod_h2o=None,
                  solar_zenith=0, observer_zenith=0,
                  solar_azimuth=0, observer_azimuth=0,
                  create_lut=True,
                  lutdir="./luts",
                  outdir="./output"):
    """One iteration of the hypertrace workflow."""
    outdir = mkabs(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reflectance = spectral.open_image(reflectance_file)
    spatial_dim = reflectance.shape[0:2]
    reflectance_it = spectral.algorithms.iterator(reflectance)

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
        noisetag = f"noise_{Path(noisefile).name}"
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
    fm = ForwardModel(Config({"forward_model": forward_settings}))
    geomvec = [
        -999,              # path length; not used
        observer_azimuth,  # Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
        observer_zenith,   # Degrees 0-90; 0 = directly overhead, 90 = horizon
        solar_azimuth,     # Degrees 0-360; 0 = N, 90 = W, 180 = S, 270 = E
        solar_zenith       # Same units as observer zenith
    ]
    igeom = Geometry(obs=geomvec)
    # Forward simulation
    radiance_l = ray.get([ht_radiance.remote(refl, aod, h2o, fm, igeom)
                          for refl in reflectance_it])
    nwl = len(radiance_l[0])
    output_dim = np.concatenate((spatial_dim, [nwl]))
    radiance = np.reshape(np.asarray(radiance_l), output_dim)
    radiance_img = spectral.envi.create_image(outdir2 / "toa-radiance.hdr",
                                              shape=output_dim,
                                              dtype=np.float64,
                                              force=True)
    radiance_img_m = radiance_img.open_memmap(writable=True)
    radiance_img_m[:, :, :] = radiance
    # Delete memmap to write to disk and free up memory
    del radiance_img_m
    inverse_settings = config["isofit"]["implementation"]
    inverse_settings["mode"] = "inversion"
    iv = Inversion(Config({"implementation": inverse_settings}), fm)
    unc_l = ray.get([ht_invert.remote(rad, iv, igeom) for rad in radiance_l])
    est_refl_l = [item[0] for item in unc_l]
    est_refl = np.reshape(np.asarray(est_refl_l), output_dim)
    est_refl_img = spectral.envi.create_image(outdir2 / "estimated-reflectance.hdr",
                                              shape=output_dim,
                                              dtype=np.float64,
                                              force=True)
    est_refl_img_m = est_refl_img.open_memmap(writable=True)
    est_refl_img_m[:, :, :] = est_refl
    del est_refl_img_m
    pickle.dump(unc_l, open(outdir2 / "uncertainty.pkl", "wb"))
    return outdir2


# Create iterable config permutation object
ht_iter = itertools.product(*hypertrace_config.values())
for ht in ht_iter:
    argd = dict()
    for key, value in zip(hypertrace_config.keys(), ht):
        argd[key] = value
    print(argd)
    do_hypertrace(isofit_config, wavelength_file, surface_file,
                  reflectance_file, libradtran_template_file,
                  **argd)
