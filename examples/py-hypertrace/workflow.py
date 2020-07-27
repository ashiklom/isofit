#!/usr/bin/env python3

from atexit import register
import sys
import json
import itertools

import ray

from hypertrace import do_hypertrace, mkabs

if len(sys.argv) > 1:
    configfile = sys.argv[1]
else:
    configfile = "./config.json"
configfile = mkabs(configfile)
print(f"Using config file `{configfile}`")

with open(configfile) as f:
    config = json.load(f)

wavelength_file = mkabs(config["wavelength_file"])
reflectance_file = mkabs(config["reflectance_file"])
libradtran_template_file = mkabs(config["libradtran_template_file"])
lutdir = mkabs(config["lutdir"])
outdir = mkabs(config["outdir"])

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

# Create iterable config permutation object
ht_iter = itertools.product(*hypertrace_config.values())
print("Starting Hypertrace workflow.")
for ht in ht_iter:
    argd = dict()
    for key, value in zip(hypertrace_config.keys(), ht):
        argd[key] = value
    print(f"Running config: {argd}")
    do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                  libradtran_template_file, lutdir, outdir,
                  **argd)
print("Workflow completed successfully.")
