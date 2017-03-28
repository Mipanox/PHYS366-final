# PHYS366: [Special Topics in Astrophysics: Statistical Methods](https://github.com/KIPAC/StatisticalMethods) Final Project

_Note: This README is only meant to help the author document the project. It does not intend to fully describe or elaborate the details._

This project aims at quantifying 2D vector field correlation by modeling vector fields with a composite curl-noise field, generated from a Perlin scalar noise (being a stream function). The model field consists of random fields of which structures are characterized by three distinct scales: ~ 1/2 map size; 1/5 map size; 2x pixel size.

_Details are in the report_

This repository is organized as follows:
## Preparation
### Tests - checks for functionality of the codes
1. [test_pyqg.ipynb](https://github.com/Mipanox/PHYS366-final/blob/master/test_pyqg.ipynb): Using the existing `pyqg` package to generate simulated 2D turbulences' evolution. Not used in the subsequent analysis.<br><br>

2. [test_EB.ipynb](https://github.com/Mipanox/PHYS366-final/blob/master/test_EB.ipynb): Codes for computing the E/B decomposition in the flat-sky limit, neglicting boundary issue, modes mixings, etc. Also included are some simple tests for verification.<br><br>

3. [test_perlin.ipynb](https://github.com/Mipanox/PHYS366-final/blob/master/test_perlin.ipynb): Perlin noise generation and the associated curl-noise/gradient-noise. Numerical proof of the equivalence of the two types of field under E/B decomposition (when 2D, directionless).<br><br>

4. [test_data.ipynb](https://github.com/Mipanox/PHYS366-final/blob/master/test_data.ipynb): Loading and examining the data.<br><br>

### Codes
In the [folder](https://github.com/Mipanox/PHYS366-final/tree/master/codes). The comments should be explanatory.<br><br>

### PGM model - logic for this project
Follows the convention in Bayesian scheme: double circles denote observed distributions, single circles are distributions (i.e. with non-delta function PDF), and dots are deterministic (fixed) values. Generated [here](https://github.com/Mipanox/PHYS366-final/blob/master/pgm.ipynb). The parameters are the relative weights of the composite curl-noises: `Oct.-2`, `Oct.-8`, and the dependent `Oct.-32`. Models and data are compared in the Fourier domain: the radially averaged power spectra (`RAPS`). _RHS_ is the model (free field; `FF`); _LHS_ is the data.<br><br>

## [MCMC and Evidences](https://github.com/Mipanox/PHYS366-final/blob/master/run_MCMC.ipynb)
### Markov-Chain Monte Carlo
Starts [here](http://nbviewer.jupyter.org/github/Mipanox/PHYS366-final/blob/master/run_MCMC.ipynb#MCMC). Several setups are run: (1) Noisy (both model and data have uncertainties), (2) Noiseless (eliminating data's uncertainties), (3) Noisefree (fixing the random seed of the free field, but keep data's uncertainties).

### Model Comparison
Treat the B-field as data, and compare two "models": the free field, and the velocity field. The "similarity" between the data and the models are calculated based on either Bayesian or Frequentist "[evidences](http://nbviewer.jupyter.org/github/Mipanox/PHYS366-final/blob/master/run_MCMC.ipynb#Evidence---B-field-described-by-model-or-velocity-field)".

### Best-fit visualization
[best-fit.ipynb](https://github.com/Mipanox/PHYS366-final/blob/master/best-fit.ipynb) illustrates the RAPS of the free field according to the median and the maximum probable posterior of the parameters.