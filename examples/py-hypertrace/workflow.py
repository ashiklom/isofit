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

restart = False
clean = False
consolidate_output = False
cluster_p = False

if len(sys.argv) > 1:
    configfile = sys.argv[1]
    if "--restart" in sys.argv:
        logger.info("Purging existing output to force workflow restart")
        restart = True
    if "--clean" in sys.argv:
        logger.info("Raw output will be deleted.")
        clean = True
    if "--cluster" in sys.argv:
        logger.info("Running in cluster mode")
        cluster_p = True
else:
    configfile = "./configs/example-srtmnet.json"

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

if cluster_p:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, Lock, wait
    # TODO: Customize 
    cluster = SLURMCluster(cores=30, memory='192GB', header_skip=['--mem'])
    cluster.scale(jobs=2)
    client = Client(cluster)
    if consolidate_output:
        if cluster_p:
            lock = Lock(name="NoConcurrentWrites")
        else:
            # Placeholder
            lock = open("/dev/null", "r")

wavelength_file = mkabs(config["wavelength_file"])
reflectance_file = mkabs(config["reflectance_file"])
if "libradtran_template_file" in config:
    raise Exception("`libradtran_template_file` is deprecated. Use `rtm_template_file` instead.")
rtm_template_file = mkabs(config["rtm_template_file"])
lutdir = mkabs(config["lutdir"])
outdir = mkabs(config["outdir"])
outdir.mkdir(parents=True, exist_ok=True)

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

# Consolidate output
if consolidate_output:
    if outfile.exists():
        logger.info("Using to existing output file: %s", outfile)
    else:
        logger.info("Creating new output file: %s", outfile)
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

def htfun(ht):
    argd = dict()
    for key, value in zip(hypertrace_config.keys(), ht):
        argd[key] = value
    logger.info("Running config: %s", argd)
    ht_outdir = do_hypertrace(isofit_config, wavelength_file, reflectance_file,
            rtm_template_file, lutdir, outdir,
            **argd)
    if consolidate_output and ht_outdir is not None:
        logger.info("Consolidating output from `%s`", str(ht_outdir))
        # Prevent concurrent writes.
        with lock:
            with h5netcdf.File(outfile, 'r+') as dsz:
                curr_ht = json.dumps(ht)
                all_ht = dsz["hypertrace"][:]
                iii = np.where([curr_ht == htstr for htstr in all_ht])
                assert len(iii) < 2, f"Found {len(iii)} matching HT configs in outfile"
                assert len(iii) > 0, "Found no matching HT configs in outfile"
                iii = iii[0]   # Convert to just an integer
                dsz["completed"][iii] = True
                keys_files = {
                        "toa_radiance": "toa-radiance.hdr",
                        "estimated_reflectance": "estimated-reflectance.hdr",
                        "estimated_state": "estimated-state.hdr",
                        "posterior_uncertainty": "posterior-uncertainty.hdr"
                        }
                for key, fname in keys_files.items():
                    vals = sp.open_image(str(ht_outdir / fname))[:,:,:]
                    vals2 = np.expand_dims(vals, axis=3)  # Force dimensions to match
                    dsz[key][:,:,:,iii] = vals2
        if clean:
            logger.info("Deleting output from `%s`", str(ht_outdir))
            shutil.rmtree(ht_outdir)

if cluster_p:
    logger.info("Starting distributed Hypertrace workflow.")
    futures = client.map(htfun, ht_iter)
    wait(futures)
else:
    logger.info("Starting sequential Hypertrace workflow.")
    [htfun(ht) for ht in ht_iter]

logger.info("Workflow completed successfully.")
