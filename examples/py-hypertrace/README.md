# Hypertrace

Objective: Starting from known surface reflectance and atmospheric conditions, simulate top-of-atmosphere radiance and then perform atmospheric correction to estimate surface reflectance from this simulated radiance.

## Lightning introduction

First, make sure you have Isofit installed.
In addition, you will need to have a working compiled version of the LibRadtran atmospheric radiative transfer model (see Installation section below).

Make a local copy of the `config-example.json` and modify to fit your system (most important is radiative transfer engine base directly and environment.

``` sh
cp config-example.json myconfig.json
```

Then, run the `workflow.py` script with your config file as an argument:

```sh
python workflow.py myconfig.json
```

## Configuration file

Like Isofit, the configuration file is a `json` file.
Top level settings are as follows:

- `wavelength_file` -- Path to ASCII space delimited table containing two columns, wavelength and full width half max (FWHM); both in nanometers.
- `reflectance_file` -- Path to input reflectance file. Note that this has to be an ENVI-formatted binary reflectance file, and this path is to the **associated header file** (`.hdr`), not the image file itself (following the convention of the `spectral` Python library, which will be used to read this file).
- `libradtran_template_file` -- Path to the LibRadtran template. Note that this is slightly different from the Isofit template in that the Isofit fields are surrounded by two sets of `{{` while a few additional options related to geometry are surrounded by just `{` (this is because Hypertrace does an initial pass at formatting the files).
- `lutdir` -- Directory where look-up tables will be stored. Will be created if missing.
- `outdir` -- Directory where outputs will be stored. Will be created if missing.
- `isofit` -- Isofit configuration options (`forward_model`, `implementation`, etc.). This is included to allow you maximum flexiblity in modifying the behavior of Isofit. See the Isofit documentation for more details. Note that some of these will be overwritten by the Hypertrace workflow.
- `hypertrace` -- Each of these is a list of variables that will be iterated as part of Hypertrace. Specifically, Hypertrace will generate the factorial combination of every one of these lists and perform the workflow for each element of that list. Every keyword argument to the `do_hypertrace` function is supported (indeed, that's how they are passed in, via the `**kwargs` mechanism), and include the following:
    - `surface_file` -- Matlab (`.mat`) file containing a multicomponent surface prior. See Isofit documentation for details.
    - `noisefile` -- Parametric instrument noise file. See Isofit documentation for details. Default = `None`
    - `snr` -- Instrument signal-to-noise ratio. Ignored if `noisefile` is present. Default = 300
    - `aod` -- True aerosol optical depth. Default = 0.1
    - `h2o` -- True water vapor content. Default = 1.0
    - `lrt_atmosphere_type` -- LibRadtran atmosphere type. See LibRadtran manual for details. Default = `midlatitude_winter`
    - `atm_aod_h2o` -- A list containing three elements: The atmosphere type, AOD, and H2O. This provides a way to iterate over specific known atmospheres that are combinations of the three previous variables. If this is set, it overrides the three previous arguments. Default = `None`
    - `solar_zenith`, `observer_zenith` -- Solar and observer zenith angles, respectively (0 = directly overhead, 90 = horizon). These are in degrees off nadir. Default = 0 for both. (Note that off-nadir angles make LibRadtran run _much_ more slowly, so be prepared if you need to generate those LUTs).
    - `solar_azimuth`, `observer_azimuth` -- Solar and observer azimuth angles, respectively, in degrees. Observer azimuth is the sensor _position_ (so 180 degrees off from view direction) relative to N, rotating counterclockwise; i.e., 0 = Sensor in N, looking S; 90 = Sensor in W, looking E (this follows the LibRadtran convention). Default = 0 for both.
    
## Installation

### Isofit

We recommend installing Isofit into a new conda environment.
Therefore, make sure you have anaconda or miniconda installed.
In this document, we'll call the conda environment `r-isofit`, but you can call it whatever you'd like (just replace `r-isofit` with your own environment name everywhere in these instructions).

First, download Isofit and checkout this branch.

``` sh
git clone https://github.com/ashiklom/isofit
cd isofit
git checkout r-hypertrace
```

Create a new conda environment and activate it.

``` sh
conda create -n r-isofit python
# Go through the interactive prompts...
conda activate r-isofit
```

Install Isofit and its dependencies.

``` sh
# From inside the isofit root directory (same directory as README.rst, LICENSE, etc.)
pip install .
```

### Libradtran

To generate your own atmospheric look-up tables, you'll need a working installation of LibRadTran.
Follow the instructions in the [Isofit `README`](https://github.com/ashiklom/isofit/tree/r-geom-2#quick-start-with-libradtran-20x) to install.
Note that you must install LibRadTran into the source code directory for it to work properly; i.e.

``` sh
# From the LibRadTran source code directory:
./configure --prefix=$(pwd)
make
```