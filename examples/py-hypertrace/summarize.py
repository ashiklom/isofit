#!/usr/bin/env python3

from pathlib import Path
import re
import json
import copy

import numpy as np
import pandas as pd
import spectral as sp
import matplotlib.pyplot as plt

configfile = "./configs/quick.json"
with open(configfile, "r") as f:
    config = json.load(f)

outdir = Path(config["outdir"])

reflfiles = list(outdir.glob("**/estimated-reflectance"))

true_refl_file = Path(config["reflectance_file"]).expanduser()
true_reflectance = sp.open_image(str(true_refl_file) + ".hdr")
true_waves = np.array(true_reflectance.metadata["wavelength"], dtype=float)
true_refl_m = true_reflectance.open_memmap()

windows = config["isofit"]["implementation"]["inversion"]["windows"]


def parse_dir(ddir, parent):
    dir_pattern = re.compile(
        str(parent) + "/" +
        r"atm_(?P<atm>.*)__" +
        r"szen_(?P<szen>[0-9.]+)__" +
        r"ozen_(?P<ozen>[0-9.]+)__" +
        r"saz_(?P<saz>[0-9.]+)__" +
        r"oaz_(?P<oaz>[0-9.]+)/" +
        r"(?:noise_(?P<noise>.*)|snr_(?P<snr>[0-9.]+))/" +
        r"prior_(?P<prior>.*)__" +
        r"inversion_(?P<inversion>.*)/" +
        r"aod_(?P<aod>[0-9.]+)__" +
        r"h2o_(?P<h2o>[0-9.]+)"
    )
    grps = dir_pattern.match(str(ddir)).groupdict()
    for key in ["szen", "ozen", "saz", "oaz", "aod", "h2o"]:
        if key in grps:
            grps[key] = float(grps[key])
    return pd.DataFrame(grps, index=[0])


info = pd.concat([parse_dir(x.parent, outdir) for x in reflfiles])\
         .reset_index(drop=True)
info["reflectance"] = reflfiles


def mask_windows(data, waves, windows):
    inside_l = []
    for w in windows:
        inside_l.append(np.logical_and(waves >= w[0],
                                       waves <= w[1]))
    inside = np.logical_or.reduce(inside_l)
    d2 = copy.copy(data)
    d2[:, :, np.logical_not(inside)] = np.nan
    return d2


info["rmse"] = np.nan
info["bias"] = np.nan
info["rel_bias"] = np.nan
for i in range(info.shape[0]):
    f = info["reflectance"][i]
    ddir = f.parent
    est_refl = sp.open_image(str(f) + ".hdr")
    est_refl_waves = np.array(est_refl.metadata["wavelength"], dtype=float)
    est_refl_m = est_refl.open_memmap()
    if est_refl_m.shape != true_refl_m.shape:
        true_resample = np.zeros_like(est_refl_m)
        for r in range(true_resample.shape[0]):
            for c in range(true_resample.shape[1]):
                true_resample[r, c, :] = np.interp(
                    est_refl_waves,
                    true_waves,
                    true_refl_m[r, c, :]
                )
    else:
        true_resample = true_refl_m
    est_refl_m2 = mask_windows(est_refl_m, est_refl_waves, windows)
    bias = est_refl_m2 - true_resample
    rmse = np.sqrt(np.nanmean(bias**2))
    mean_bias = np.nanmean(bias)
    rel_bias = bias / true_resample
    mean_rel_bias = np.nanmean(rel_bias)
    info.loc[i, "rmse"] = rmse
    info.loc[i, "bias"] = mean_bias
    info.loc[i, "rel_bias"] = mean_rel_bias
    # Bias by wavelength
    bias_wl = np.nanmean(bias, axis=(0, 1))
    bias_wl_q = np.nanquantile(bias, (0.05, 0.95), axis=(0, 1))
    plt.axhline(y=0, color="gray")
    plt.plot(est_refl_waves, bias_wl, "k-")
    plt.plot(est_refl_waves, np.transpose(bias_wl_q), "k--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Bias (Estimate - True; 90% CI)")
    plt.savefig(ddir / "bias.png")

print("Simulations sorted by RMSE (lowest first)")
print(info.sort_values("rmse"))

info.to_csv(outdir / "summary.csv")

# plt.plot(
#     est_refl_waves,
#     np.moveaxis(bias, 2, 0).reshape((len(est_refl_waves), -1)),
# )
# plt.show()
