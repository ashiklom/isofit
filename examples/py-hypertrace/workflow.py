#!/usr/bin/env python3
#
# Authors: Alexey Shiklomanov

import sys
import copy
import json
import itertools
import logging
import shutil

import ray

from isofit.configs.configs import Config
from isofit.radiative_transfer.radiative_transfer import RadiativeTransfer

from hypertrace import do_hypertrace, mkabs, setup_lut

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

restart = False
newlut = False
clean = False
consolidate_output = False

if len(sys.argv) > 1:
    configfile = sys.argv[1]
    if "--restart" in sys.argv:
        logger.info("Purging existing output to force workflow restart")
        restart = True
    if "--newlut" in sys.argv:
        logger.info("Purging existing luts to force rebuild")
        newlut = True
    if "--clean" in sys.argv:
        logger.info("Raw output will be deleted.")
        clean = True
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

cluster_p = "cluster" in config
if cluster_p:
    import dask_jobqueue
    from dask.distributed import Client, Lock, wait
    cluster_config = config["cluster"]
    cluster_type = cluster_config["type"].lower()
    logger.info("Configuring cluster of type '%s'", cluster_type)
    cluster_func = {
            "slurm": dask_jobqueue.SLURMCluster,
            "sge": dask_jobqueue.SGECluster,
            "pbs": dask_jobqueue.PBSCluster
            }[cluster_type]
    cluster = cluster_func(**cluster_config["args"])
    cluster.scale(jobs=cluster_config["jobs"])
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

if newlut and lutdir.exists():
    shutil.rmtree(lutdir)

if restart and outdir.exists():
    shutil.rmtree(outdir)
    if consolidate_output and outfile.exists():
        outfile.unlink()

lutdir.mkdir(parents=True, exist_ok=True)
outdir.mkdir(parents=True, exist_ok=True)

isofit_config = config["isofit"]
hypertrace_config = config["hypertrace"]

# Make RTM paths absolute
vswir_conf = isofit_config["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
for key in ["lut_path", "template_file", "engine_base_dir"]:
    if key in vswir_conf:
        vswir_conf[key] = str(mkabs(vswir_conf[key]))

# Create iterable config permutation object
ht_iter = list(itertools.product(*hypertrace_config.values()))

# ...and convert it to a dict, for easier processing
ht_keys = hypertrace_config.keys()
ht_dict = list()
for ht in ht_iter:
    htd = {key:value for key, value in zip(ht_keys, ht)}
    if 'atm_aod_h2o' in htd:
        htd_aah = htd['atm_aod_h2o']
        htd['atmosphere_type'] = htd_aah[0]
        htd['aod'] = htd_aah[1]
        htd['h2o'] = htd_aah[2]
        del htd['atm_aod_h2o']
    ht_dict.append(htd)

# If running in cluster mode, eagerly create the inverse look-up tables. This
# prevents collisions between
if cluster_p:
    # 1) Identify the unique LUT combinations that need to be created, based on the HT config
    logger.info("Building RTM look-up tables (LUTs) first")
    ht_atm_keys = ["atmosphere_type", 
            "solar_zenith", "observer_zenith",
            "solar_azimuth", "observer_azimuth",
            "observer_altitude_km",
            "dayofyear", "latitude", "longitude",
            "localtime", "elevation_km"]
    ht_dict_atm = list()
    for htd in ht_dict:
        dd = {key:htd[key] for key in htd if key in ht_atm_keys}
        if dd not in ht_dict_atm:
            ht_dict_atm.append(dd)
    logger.info("Identified %d unique LUTs to build", len(ht_dict_atm))
    # 2) For every unique combination, build the LUT
    def do_rtm(hta):
        lut_conf, *_ = setup_lut(lutdir, copy.deepcopy(vswir_conf), *hta)
        lut_conf_full = copy.deepcopy(isofit_config)
        lut_conf_full["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"] = lut_conf
        lut_conf_full["forward_model"]["instrument"]["wavelength_file"] = str(wavelength_file)
        ray.init()
        RadiativeTransfer(Config(lut_conf_full))
        ray.shutdown()
    lut_futures = client.map(do_rtm, ht_dict_atm)
    wait(lut_futures)

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
