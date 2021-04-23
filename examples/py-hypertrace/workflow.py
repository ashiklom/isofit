#!/usr/bin/env python3
#
# Authors: Alexey Shiklomanov

import sys
import json
import itertools
import logging
import shutil
import ray
import uuid

from hypertrace import do_hypertrace, mkabs

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

restart = False
clean = False
consolidate_output = False
cluster = False
if len(sys.argv) > 1:
    configfile = sys.argv[1]
    if "--restart" in sys.argv:
        logger.info("Purging existing output to force workflow restart")
        restart = True
    if "--clean" in sys.argv:
        logger.info("Raw output will be deleted.")
        clean = True
    if "--cluster" in sys.argv:
        logger.info("Connecting to existing Ray cluster")
        cluster = True
else:
    configfile = "./config.json"

configfile = mkabs(configfile)
logger.info("Using config file `%s`", configfile)
with open(configfile) as f:
    config = json.load(f)

consolidate_output = "outfile" in config
if consolidate_output:
    import spectral as sp
    import numpy as np
    import xarray as xr
    import dask
    import h5netcdf
    outfile = mkabs(config["outfile"])
    logger.info("Consolidating output in `%s`", outfile)

wavelength_file = mkabs(config["wavelength_file"])
reflectance_file = mkabs(config["reflectance_file"])
if "libradtran_template_file" in config:
    raise Exception("`libradtran_template_file` is deprecated. Use `rtm_template_file` instead.")
rtm_template_file = mkabs(config["rtm_template_file"])
lutdir = mkabs(config["lutdir"])
outdir = mkabs(config["outdir"])

if restart and outdir.exists():
    shutil.rmtree(outdir)
    if consolidate_output and outfile.exists():
        outfile.unlink()

isofit_config = config["isofit"]
hypertrace_config = config["hypertrace"]

# Make RTM paths absolute
vswir_settings = isofit_config["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
for key in ["lut_path", "template_file", "engine_base_dir"]:
    if key in vswir_settings:
        vswir_settings[key] = str(mkabs(vswir_settings[key]))

# Create iterable config permutation object
ht_iter = list(itertools.product(*hypertrace_config.values()))

if consolidate_output:
    nrow, ncol, *_ = sp.open_image(str(reflectance_file) + ".hdr").shape
    waves = np.loadtxt(wavelength_file)[:,1]
    if np.mean(waves) < 100:
        waves = waves * 1000
    nwaves = waves.shape[0]
    nht = len(ht_iter)
    # Preallocate "blank" dataset full of zeros, but with the right dimensions
    xr.Dataset(
        {
            "toa_radiance": (["sample", "line", "band", "hypertrace"],
                             dask.array.empty((nrow, ncol, nwaves, nht), dtype='f')),
            "estimated_reflectance": (["sample", "line", "band", "hypertrace"],
                                      dask.array.empty((nrow, ncol, nwaves, nht), dtype='f')),
            "estimated_state": (["sample", "line", "statevec", "hypertrace"],
                                dask.array.empty((nrow, ncol, nwaves+2, nht), dtype='f')),
            "posterior_uncertainty": (["sample", "line", "statevec", "hypertrace"],
                                      dask.array.empty((nrow, ncol, nwaves+2, nht), dtype='f')),
            "completed": (["hypertrace"], dask.array.zeros((nht), dtype='?'))
        },
        coords={
            "band": waves,
            "statevec": np.append([f"RFL_{w:.2f}" for w in waves], ["AOT550", "H2OSTR"]),
            "hypertrace": [json.dumps(ht) for ht in ht_iter]
        },
        attrs={"hypertrace_config": json.dumps(config)}
    ).to_netcdf(outfile, mode='w', engine='h5netcdf')

# Start Ray once
rayconfig = None
if cluster:
    implementation = isofit_config["implementation"]
    redis_password = '5241590000000000'
    rayinit = ray.init(address="auto", _redis_password=redis_password)
    rayconfig = {"ip_head": rayinit["redis_address"],
                "redis_password": redis_password}

logger.info("Starting Hypertrace workflow.")
for ht, iht in zip(ht_iter, range(len(ht_iter))):
    argd = dict()
    for key, value in zip(hypertrace_config.keys(), ht):
        argd[key] = value
    logger.info("Running config %d of %d: %s", iht+1, len(ht_iter), argd)
    ht_outdir = do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                              rtm_template_file, lutdir, outdir,
                              rayconfig=rayconfig,
                              **argd)
    # Post process files here
    if consolidate_output and ht_outdir is not None:
        logger.info("Consolidating output from `%s`", str(ht_outdir))
        with h5netcdf.File(outfile, 'r+') as dsz:
            dsz["completed"][iht] = True
            dsz["toa_radiance"][:,:,:,iht] = sp.open_image(str(ht_outdir / "toa-radiance.hdr"))[:,:,:]
            dsz["estimated_reflectance"][:,:,:,iht] = sp.open_image(str(ht_outdir / "estimated-reflectance.hdr"))[:,:,:]
            dsz["estimated_state"][:,:,:,iht] = sp.open_image(str(ht_outdir / "estimated-state.hdr"))[:,:,:]
            dsz["posterior_uncertainty"][:,:,:,iht] = sp.open_image(str(ht_outdir / "posterior-uncertainty.hdr"))[:,:,:]
        if clean:
            logger.info("Deleting output from `%s`", str(ht_outdir))
            shutil.rmtree(ht_outdir)

logger.info("Workflow completed successfully.")
