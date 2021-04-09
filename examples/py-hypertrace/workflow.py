#!/usr/bin/env python3
#
# Authors: Alexey Shiklomanov

import sys
import json
import itertools
import logging
import shutil

from hypertrace import do_hypertrace, mkabs

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

clean = False
use_zarr = False
if len(sys.argv) > 1:
    configfile = sys.argv[1]
    if "--clean" in sys.argv:
        clean = True
    if "--zarr" in sys.argv:
        use_zarr = True
else:
    configfile = "./config.json"

configfile = mkabs(configfile)
logger.info("Using config file `%s`", configfile)
with open(configfile) as f:
    config = json.load(f)

if use_zarr:
    import spectral as sp
    import numpy as np
    import zarr
    import datetime
    import xarray as xr
    zarrfile = mkabs(config["zarrfile"])

wavelength_file = mkabs(config["wavelength_file"])
reflectance_file = mkabs(config["reflectance_file"])
if "libradtran_template_file" in config:
    raise Exception("`libradtran_template_file` is deprecated. Use `rtm_template_file` instead.")
rtm_template_file = mkabs(config["rtm_template_file"])
lutdir = mkabs(config["lutdir"])
outdir = mkabs(config["outdir"])

if clean and outdir.exists():
    shutil.rmtree(outdir)
    if use_zarr and zarrfile.exists():
        shutil.rmtree(zarrfile)

isofit_config = config["isofit"]
hypertrace_config = config["hypertrace"]

# Make RTM paths absolute
vswir_settings = isofit_config["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
for key in ["lut_path", "template_file", "engine_base_dir"]:
    if key in vswir_settings:
        vswir_settings[key] = str(mkabs(vswir_settings[key]))

# Create iterable config permutation object
ht_iter = list(itertools.product(*hypertrace_config.values()))

if use_zarr:
    nrow, ncol, *_ = sp.open_image(str(reflectance_file) + ".hdr").shape
    waves = np.loadtxt(wavelength_file)[:,1]
    if np.mean(waves) < 100:
        waves = waves * 1000
    nwaves = waves.shape[0]
    nht = len(ht_iter)
    # Preallocate "blank" dataset full of zeros, but with the right dimensions
    xr.Dataset(
        {
            "toa_radiance": (["sample", "line", "band", "hypertrace"], zarr.zeros((nrow, ncol, nwaves, nht), dtype='f')),
            "estimated_reflectance": (["sample", "line", "band", "hypertrace"], zarr.zeros((nrow, ncol, nwaves, nht), dtype='f')),
            "posterior_uncertainty": (["sample", "line", "statevec", "hypertrace"], zarr.zeros((nrow, ncol, nwaves+2, nht), dtype='f'))
        },
        coords={
            "band": waves,
            "statevec": np.append(waves, [9200, 9550]),
            "hypertrace": [json.dumps(ht) for ht in ht_iter]
        },
        attr={"hypertrace_config": config}
    ).to_zarr(zarrfile, mode='w')
    # Now, open it for IO
    dsz = xr.open_zarr(zarrfile)

logger.info("Starting Hypertrace workflow.")
for ht, iht in zip(ht_iter, range(len(ht_iter))):
    argd = dict()
    for key, value in zip(hypertrace_config.keys(), ht):
        argd[key] = value
    logger.info("Running config %d of %d: %s", iht+1, len(ht_iter), argd)
    ht_outdir = do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                              rtm_template_file, lutdir, outdir,
                              **argd)
    # Post process files here
    if use_zarr:
        dsz["toa_radiance"][:,:,:,iht] = sp.open_image(str(ht_outdir / "toa-radiance.hdr"))[:,:,:]
        dsz["estimated_reflectance"][:,:,:,iht] = sp.open_image(str(ht_outdir / "estimated-reflectance.hdr"))[:,:,:]
        dsz["posterior_uncertainty"][:,:,:,iht] = sp.open_image(str(ht_outdir / "posterior-uncertainty.hdr"))[:,:,:]
        # ...and purge outputs
        shutil.rmtree(ht_outdir)
logging.info("Workflow completed successfully.")
