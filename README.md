# anDiffReg
Regression methods for analysing TA-MSD described in the paper J. Ślęzak, J. Janczura, D. Krapf and R. Metzler "Improved mean squared displacement analysis for
anomalous single particle trajectories" implemented in Julia and Python. Julia version is the basis, Python is a translation.

Main functions are in **AnDiffReg.jl** or **anDiffReg.ipynb** files, the most important ones are:
- `tamsd` for calculating TA-MSD,
- `fit_ols`, `fit_gls` for obtaining diffusivity and anomalous diffusion index estimates together with their prediced covariance matrices. 

See `?` in Julia or `help` in Python for more details. These methods work for TA-MSD of FBM trajectories in any dimension. The effect of additive noise can be included if its standard deviation is known. Scripts **exemplaryTAMSDAnalysis** show how the basic TA-MSD analysis could look like.

Functions `deconvolve_ols` or `deconvolve_gls` can be used to deconvolve the pdf of estimates obtained from `fit_ols` or `fit_gls` methods. The obtained deconvolved density has influence of statistical errors removed and should be significantly closer to the original distribution characterising the studied system. Scripts **exemplaryDeconvolution** show how such procedure could look like.
