
using Statistics, LinearAlgebra

include("AnDiffReg.jl")
using .AnDiffReg

##---------------------------------------------------------------
# exemplary data: generate simulated fractional Brownian motion

α, D = 0.8, 1 # FBM parameters: Hurst index H and diffusivity D
n = 10^4 # number of trajectories
ln = 100 # trajectory length
dt = 1. # time interval
ts = dt*(1:ln)

K = (s,t) -> D*(t^α+s^α-abs(s-t)^α) # FBM covariance
S = [K(s,t) for s in ts, t in ts]
A = cholesky(Symmetric(S)).U

ξ = randn(length(ts), n)
X = A'*ξ
ξ = randn(length(ts), n)
Y = A'*ξ

##---------------------------------------------------------------
# alternatively, load data from the file

using CSV

XY = CSV.read("exemplaryData/2D FBM.csv",  CSV.Tables.matrix)
X, Y = XY[:,1:2:end], XY[:,2:2:end]

ln, n = size(X)
dt = 1.
ts = dt*(1:ln)
α = 0.8 # used only optionally for initial estimate

##---------------------------------------------------------------
# TA-MSD analysis

msd = tamsd([X ;;; Y]) # TA-MSD of 2D traj

ols = fit_ols(msd, 2, dt)

# mean and variance of the OLS estimates
mean(ols, dims=2)
cov(ols')

# here init α is true 2H, ols estimate also can be used
gls, covGLS = fit_gls(msd, 2, dt, fill(α, n)) 


mean(gls, dims=2)
cov(gls')

# covariance is computed for each trajectory, could be slow if there are many
glsP, covGLSP = fit_gls(msd[:,1:100], 2, dt, fill(α, n), precompute = false)


##---------------------------------------------------------------
# trajectories with noise

D, σ = 5, 1

X2 = √D*X .+ σ*randn(ln,n)

msd2 = tamsd(X2)

# initial estimates of both α and D and must be used, σ must be known 
gls2, covGLS2 = fit_gls(msd2, 1, dt, fill(α, n), fill(D, n), σ)


mean(gls2, dims=2)

cov(gls2')
