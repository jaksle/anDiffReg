# anDiffReg
Regression methods for analysing TA-MSD described in the paper J. Ślęzak, J. Janczura, D. Krapf and R. Metzler "Improved mean squared displacement analysis for
anomalous single particle trajectories" implemented in Julia and Python. Julia version is the basis, Python is a translation.

Functions for analysing TA-MSD are in `AnDiffReg.jl` or `anDiffReg.ipynb` files, the most important ones are `tamsd` for calculating TA-MSD and `fit_ols`, `fit_gls` for obtaining diffusivity and anomalous diffusion index estimates together with their prediced covariance matrices. It also contains separate functions for calculating predicted TA-MSD covariance. Check `exemplaryTAMSDAnalysis` scripts to see how TAMSD analysis could look like.
