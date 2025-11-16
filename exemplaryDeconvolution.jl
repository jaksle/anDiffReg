
using Statistics, Distributions, LinearAlgebra
using KernelDensity

using CairoMakie, ProgressMeter

include("AnDiffReg.jl")
using .AnDiffReg

##---------------------------------------------------------------
# exemplary data: mixture of FBMs
# this is the same system as considered in the "Non-parametric deconvolution" section of the article

ln = 100
n = 10^4
dt =  1
ts = dt*(1:ln)

msd = Matrix{Float64}(undef,ln-1,n)

for k in 1:n
    # (logD, α) distribution is 2 rectangles 
    α = (rand() < 1/2) ? rand(Uniform(0.4,0.6)) : rand(Uniform(0.8,1.0))
    logD = rand(Uniform(-1,1))

    S = [10^logD*(t^α+s^α-abs(s-t)^α)  for s in ts, t in ts]
    A = cholesky(Symmetric(S)).U
    ξ = randn(ln, 1)
    X = A'*ξ
    msd[:,k] .= tamsd(X) 
end

# fitting

ols, covOLS = fit_ols(msd, 1, dt)
gls, covGLS = fit_gls(msd, 1, dt, ols[2,:]) 


##---------------------------------------------------------------
# getting kde

# this is for the gls estimate 
den = kde((gls[1,:],gls[2,:]), boundary = ((-1.5,1.5),(0.2,1.3)), npoints=(512,512))

# for the OLS estimate switch to this line 
# den = kde((ols[1,:],ols[2,:]), boundary = ((-1.5,1.5),(0.2,1.3)), npoints=(512,512))


# plotting initial estimate
xlab = L"$D$ [L$^2$/T$^\alpha$]"
ylab = L"{\alpha}\ [1]"
xtickL = [L"10^{-1}",L"10^{-0.5}","1",L"10^{0.5}",L"10^{1}"]
xtick = (-1:0.5:1,xtickL)
fig = Figure()
ax = Axis(fig[1,1:2],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Initial pdf estimate"
)
heatmap!(ax,den.x,den.y,den.density)
denMarg0 = vec(sum(den.density,dims=1))
denMarg0 .*= 1/(sum(denMarg0)*step(den.y))
ax = Axis(fig[1,3],
    title = "Marginal distribution",
    limits = (0,nothing,0.2,1.2)
)
lines!(denMarg0, den.y)

fig

##---------------------------------------------------------------
# simple deconvolution

d = 1 # dimension
α = 0.7 # α for which to deconvolve

# calculation
deconvolvedPDF1 = deconvolve_gls(den.x, den.y, den.density, dt, ln, d, α)

# for the OLS switch to this lines
# w = 10 # OLS window size
# deconvolvedPDF1 = deconvolve_ols(den.x, den.y, den.density, dt, ln, d, α, w)

# plot
fig = Figure() 
ax = Axis(fig[1,1:2],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Deconvolved pdf estimate, simple method"
)
heatmap!(ax, den.x, den.y, deconvolvedPDF1)

denMarg1 = vec(sum(deconvolvedPDF1,dims=1))
denMarg1 .*= 1/(sum(denMarg1)*step(den.y))
ax = Axis(fig[1,3],
    title = "Marginal distribution",
    limits = (0,nothing,0.2,1.2)
)
lines!(denMarg1, den.y)

fig


##---------------------------------------------------------------
# full deconvolution 

d = 1
α_min, α_max = 0.3, 1.1 # range of α for which to deconvolve

# calculation, note it can take significantly longer time
deconvolvedPDF2 = deconvolve_gls(den.x, den.y, den.density, dt, ln, d, (α_min,α_max))

# for the OLS switch to this lines
# w = 10 # OLS window size
# deconvolvedPDF2 = deconvolve_ols(den.x, den.y, den.density, dt, ln, d, (α_min,α_max), w)

# plot
fig = Figure()
ax = Axis(fig[1,1:2],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Deconvolved pdf estimate, full method"
)
heatmap!(ax,den.x,den.y,deconvolvedPDF2)

denMarg2 = vec(sum(deconvolvedPDF2,dims=1))
denMarg2 .*= 1/(sum(denMarg2)*step(den.y))
ax = Axis(fig[1,3],
    title = "Marginal distribution",
    limits = (0,nothing,0.2,1.2)
)
lines!(denMarg2, den.y)

fig

