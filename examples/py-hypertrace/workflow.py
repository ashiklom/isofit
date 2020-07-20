#!/usr/bin/env python3

from pathlib import Path
from atexit import register
import json
import ray
import spectral
import pickle

import numpy as np

from isofit.configs.configs import Config
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.core.geometry import Geometry

from hypertrace import *

# Inputs
# wavelengths (file)
# reflectance (file)

with open("./config.json") as f:
    config = json.load(f)

wavelength_file = Path(config["wavelength_file"]).expanduser().resolve()
surface_file = Path(config["surface_file"]).expanduser().resolve()
reflectance_file = Path(config["reflectance_file"]).expanduser().resolve()

# Read reflectance image and create iterator
reflectance_img = spectral.open_image(reflectance_file)
reflectance = spectral.algorithms.iterator(reflectance_img)

forward_settings = config["isofit"]["forward_model"]
instrument_settings = forward_settings["instrument"]
instrument_settings["wavelength_file"] = wavelength_file
surface_settings = forward_settings["surface"]
surface_settings["surface_file"] = surface_file
surface_settings["wavelength_file"] = wavelength_file

# Make RTM paths absolute
vswir_settings = config["isofit"]["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
for key in ["lut_path", "template_file", "engine_base_dir"]:
    vswir_settings[key] = Path(vswir_settings[key]).expanduser().resolve()

if not ray.is_initialized():
    ray.init()
    register(ray.shutdown)

hypertrace_config = config["hypertrace"]

for noisefile in hypertrace_config.get("instrument_noise"):
    if noisefile is not None:
        if "SNR" in instrument_settings:
            instrument_settings.pop("SNR")
        instrument_settings["parametric_noise_file"] = noisefile
        instrument_settings["integrations"] = hypertrace_config["instrument_integrations"]
    for aot in config["hypertrace"]["true_AOD"]:
        for h2o in config["hypertrace"]["true_H2O"]:
            fm = ForwardModel(Config({"forward_model": forward_settings}))
            geomvec = [
                -999, # path length; not used
                0, # Observer azimuth; Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
                0, # Observer zenith; Degrees 0-90; 0 = directly overhead, 90 = horizon
                0, # Solar azimuth; Degrees 0-360; 0 = N, 90 = W, 180 = S, 270 = E
                0   # Solar zenith; not used (determined from time)
            ]
            igeom = Geometry(obs=geomvec)
            # Forward simulation
            radiance_l = ray.get([ht_radiance.remote(refl, aot, h2o, fm, igeom) for refl in reflectance])
            nwl = len(radiance_l[0])
            radiance = np.reshape(np.asarray(radiance_l),
                                  np.concatenate((reflectance_img.shape[0:2], [nwl])))
            inverse_settings = config["isofit"]["implementation"]
            inverse_settings["mode"] = "inversion"
            iv = Inversion(Config({"implementation": inverse_settings}), fm)
            unc_l = ray.get([ht_invert.remote(rad, iv, igeom) for rad in radiance_l])
