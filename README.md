# spectralradar-mcmc

This repository contains the code for the retrieval of snowfall microphysics from radar Doppler spectra using a Markov Chain Monte Carlo (MCMC) algorithm.

## Requirements
All the code is written in Python 3.7. In addition to usual packages, the code relies on the forward model PAMTRA and the corresponding pyPamtra library (pamtra.readthedocs.io).

## Running
The main script is `run_mcmc.py`. It reads a configuration file (many examples in the configs/ directory) and runs the MCMC algorithm. The configuration file contains the paths to the input data, the radar settings, the forward model settings, and the MCMC settings.

To run the code, simply execute `python run_mcmc.py --path_to_config path/to/config_file.json`.

