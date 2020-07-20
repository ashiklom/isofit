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

# Inputs
# wavelengths (file)
# reflectance (file)

with open("./config.json") as f:
    config = json.load(f)

wavelength_file = Path(config["wavelength_file"]).expanduser().resolve()
surface_file = Path(config["surface_file"]).expanduser().resolve()
reflectance_file = Path(config["reflectance_file"]).expanduser().resolve()
forward_settings = config["isofit"]["forward_model"]
forward_settings["instrument"]["wavelength_file"] = wavelength_file
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
fm = ForwardModel(Config({"forward_model": forward_settings}))

inverse_settings = config["isofit"]["implementation"]
inverse_settings["mode"] = "inversion"

geomvec = [
    -999, # path length; not used
     0, # Observer azimuth; Degrees 0-360; 0 = Sensor in N, looking S; 90 = Sensor in W, looking E
     0, # Observer zenith; Degrees 0-90; 0 = directly overhead, 90 = horizon
     0, # Solar azimuth; Degrees 0-360; 0 = N, 90 = W, 180 = S, 270 = E
     0   # Solar zenith; not used (determined from time)
 ]

iv = Inversion(Config({"implementation": inverse_settings}), fm)
igeom = Geometry(obs=geomvec)

def htworkflow(refl, aot, h2o, fm, iv, igeom):
    statevec = np.concatenate((refl, aot, h2o), axis=None)
    radiance = fm.calc_rdn(statevec, igeom)
    state_trajectory = iv.invert(radiance, igeom)
    state_est = state_trajectory[-1]
    unc = iv.forward_uncertainty(state_est, radiance, igeom)
    return radiance, state_trajectory, unc

htworkflow_r = ray.remote(htworkflow)

reflectance_img = spectral.open_image(reflectance_file)
reflectance = spectral.algorithms.iterator(reflectance_img)

for aot in config["hypertrace"]["true_AOD"]:
    for h2o in config["hypertrace"]["true_H2O"]:
        results = ray.get([htworkflow_r.remote(refl, aot, h2o, fm, iv, igeom) for refl in reflectance])

pickle.dump(results, open("results.pkl", "wb"))
