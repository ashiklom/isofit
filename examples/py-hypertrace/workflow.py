#!/usr/bin/env python3

import pathlib
import json
import ray

import numpy as np

from isofit.configs.configs import Config
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.core.geometry import Geometry

# Inputs
# wavelengths (file)
# reflectance (file)

with open("./config.json") as f:
    config = json.load(f)
wavelength_file = pathlib.Path(config["wavelength_file"]).resolve()
surface_file = pathlib.Path(config["surface_file"]).resolve()
forward_settings = config["isofit"]["forward_model"]
forward_settings["instrument"]["wavelength_file"] = wavelength_file
surface_settings = forward_settings["surface"]
surface_settings["surface_file"] = surface_file
surface_settings["wavelength_file"] = wavelength_file

if not ray.is_initialized():
    ray.init()
fm = ForwardModel(Config({"forward_model": forward_settings}))

inverse_settings = config["isofit"]["implementation"]
inverse_settings["mode"] = "inversion"

iv = Inversion(inverse_config, fm)
igeom = Geometry(obs=geomvec)

def htworkflow(refl, aot, h2o, fm, iv, igeom):
    statevec = np.concatenate((refl, aot, h2o), axis=None)
    radiance = fm.calc_rdn(statevec, igeom)
    state_trajectory = iv.invert(radiance, igeom)
    state_est = state_trajectory[-1]
    unc = iv.forward_uncertainty(state_est, radiance, igeom)
    return radiance, state_trajectory, unc

htworkflow_r = ray.remote(htworkflow)
results = ray.get([htworkflow_r.remote(refl, *args) for refl in reflectance])

import gdal
