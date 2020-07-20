#!/usr/bin/env python3

import isofit
import spectral

import numpy as np

from pathlib import Path

from isofit.configs.configs import Config
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.core.geometry import Geometry

noisefiles = ["../r-interface/examples/data/noisefiles/noise_coeff_sbg_cbe0.txt",
              "../r-interface/examples/data/noisefiles/noise_coeff_sbg_cbe1.txt",
              "../r-interface/examples/data/noisefiles/noise_coeff_sbg_cbe2.txt"]

input_img = "/Users/ashiklom/projects/sbg-uncertainty/hypertrace/minimal/data/reference_reflectance.hdr"

img = spectral.open_image(Path(input_img).resolve())

refl = img[0,0]
noisefile = Path(noisefiles[0]).resolve()
wavelengths = np.arange(400., 2500., 5.)
