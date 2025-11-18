# anDiffReg
Regression methods for analysing TA-MSD described in the paper J. Ślęzak, J. Janczura, D. Krapf and R. Metzler "Improved mean squared displacement analysis for
anomalous single particle trajectories" implemented in Julia and Python. Julia version is the basis, Python is a translation.

Main functions are in **AnDiffReg.jl** (Julia) or **anDiffReg.py** (Python) files, the most important ones are:
- `tamsd` for calculating TA-MSD,
- `fit_ols`, `fit_gls` for obtaining diffusivity and anomalous diffusion index estimates together with their predicted covariance matrices,
- `cov_ols`, `cov_gls` for calculating expected covariance of the estimates given their true values,
- `deconvolve_ols`, `deconvolve_gls` for deconvolving the pdf of the estimates which removes the influence of statistical errors.

See `?` in Julia or `help` in Python for more details. These methods work for TA-MSD of FBM trajectories in any dimension. The effects of additive noise can be included if its standard deviation is known. Scripts **exemplaryFitting** show how the basic TA-MSD fitting and analysis could look like. Scripts **exemplaryDeconvolution** demonstrate how the deconvolution procedure can be performed.
