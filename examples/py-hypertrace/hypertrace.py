#!/usr/bin/env python3

import numpy as np
import ray
import pathlib


@ray.remote
def ht_radiance(refl, aot, h2o, fm, igeom):
    """Calculate TOA radiance."""
    statevec = np.concatenate((refl, aot, h2o), axis=None)
    radiance = fm.calc_rdn(statevec, igeom)
    return radiance


@ray.remote
def ht_invert(rad, iv, igeom):
    """Invert TOA radiance to get reflectance."""
    state_trajectory = iv.invert(rad, igeom)
    state_est = state_trajectory[-1]
    unc = iv.forward_uncertainty(state_est, rad, igeom)
    return unc


def mkabs(path):
    """Make a path absolute."""
    path2 = pathlib.Path(path)
    return path2.expanduser().resolve()
