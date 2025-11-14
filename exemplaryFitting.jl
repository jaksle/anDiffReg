
using Statistics, LinearAlgebra

include("AnDiffReg.jl")
using .AnDiffReg

##---------------------------------------------------------------
# exemplary data: generate simulated fractional Brownian motion

α, D = 0.8, 1 # FBM parameters: anomalous index α and diffusivity D
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

##---------------------------------------------------------------
# calculating TA-MSD

msd = tamsd([X ;;; Y]) # TA-MSD of 2D traj


##---------------------------------------------------------------
# TA-MSD plot with pointwise 95% error bars
using CairoMakie

k = 42 # choose TA-MSD to plot
w = 50 # plot window size
err, logErr = AnDiffReg.errCov(ts, 2, α)
fig = Figure()
ax = Axis(fig[1,1],
    xlabel = "time",
    ylabel = "TA-MSD",
    limits = (0,w,0,nothing)
)

qs = [quantile(Normal(0,sqrt(D)*s), 0.975) for s in sqrt.(diag(err))]
scatter!(ax,ts[1:w], msd[1:w,k],
    label = "TA-MSD with 95% pointwise error bars",
)
errorbars!(ax,ts[1:w], msd[1:w,k], qs[1:w],
    color = :black,
)
axislegend(merge=true)
fig

##---------------------------------------------------------------
# parameter estimates

dim = 2
ols, covOLS = fit_ols(msd, dim, dt)


mean(ols, dims=2) # average estimates
cov(ols') # covariance of the estimates, compare with predicted covOLS



# here init α is the true value from the simulation, ols estimate also can be used
α_init = 0.8 # used only optionally for initial estimate
gls, covGLS = fit_gls(msd, dim, dt, fill(α_init, n)) 


mean(gls, dims=2) 
cov(gls') 

# covariance is computed for each trajectory, could be slow if there are many
glsP, covGLSP = fit_gls(msd[:,1:100], dim, dt, fill(α_init, n), precompute = false)


##---------------------------------------------------------------
# scatter plot of the estimated parameters with a 95% confidence area

fig = Figure()
ax = Axis(fig[1,1],
    ylabel = "α",
    xlabel = "log₁₀D",
)
scatter!(ax,gls[1,1:1000], gls[2,1:1000],
    label = "GLS estimated parameters",
)

mlogD, mA = mean(gls, dims = 2) 
C = sqrt(dropdims(mean(covGLS, dims = 3), dims=3)) # note: GLS covariance of the mean parameters can be used instead

# these are formulas for the confidence elipsis of 2D Gaussian
fx(t) = sqrt(5.99) * (C[1,1]*cos(t) + C[1,2]*sin(t) ) + mlogD
fy(t) = sqrt(5.99) * (C[2,1]*cos(t)+ C[2,2]*sin(t) ) + mA

ϕs = LinRange(0,2pi,200)
lines!(ax, fx.(ϕs), fy.(ϕs),
    label = "95% confidence area",
    linestyle = :dash,
    color = :black
)
axislegend()
fig

##---------------------------------------------------------------
# trajectories with noise

D, σ = 5, 1

X2 = √D*X .+ σ*randn(ln,n)

msd2 = tamsd(X2)

# initial estimates of both α and D and must be used, σ must be known 
gls2, covGLS2 = fit_gls(msd2, 1, dt, fill(α, n), fill(D, n), σ)


mean(gls2, dims=2)

cov(gls2')
