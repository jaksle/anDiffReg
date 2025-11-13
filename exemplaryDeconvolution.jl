
using Statistics, Distributions, LinearAlgebra
using KernelDensity

using CairoMakie, ProgressMeter

include("AnDiffReg.jl")
using .AnDiffReg

##---------------------------------------------------------------
# exemplary data: mixture of FBMs
# this is the same system as considered in the "Non-parametric deconvolution" section of the article.

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

# for OLS estimate switch to this line 
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


## simple deconvolution

# calculation

deconvolvedPDF1 = deconvolve_gls(den.x, den.y, den.density, dt, ln, 1, 0.7)

# for ols switch to this line
# deconvolvedPDF1 = deconvolve_ols(den.x, den.y, den.density, dt, ln, 1, 0.7, 10)

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

## full deconvolution 

# calculation
deconvolvedPDF2 = deconvolve_gls(den.x, den.y, den.density, dt, ln, 1, (0.3,1.1))

# for ols switch to this line
# deconvolvedPDF2 = deconvolve_ols(den.x, den.y, den.density, dt, ln, 1, (0.3,1.1), 10)

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

##





##

###############



thDen = [( -1 <= x <= 1 && ( 0.4 <= y <= 0.6 || 0.8 <= y <= 1.0)) ? 1/(0.4*2) : 0. for x in den.x, y in den.y ]

sum((thDen .- den.density) .^2)*step(den.x)*step(den.y)
sum((thDen .- resO) .^2)*step(den.x)*step(den.y)
sum((thDen .- resI) .^2)*step(den.x)*step(den.y)

scatter(bB[1,:],bB[2,:],markersize=3)

heatmap(den.x,den.y,thDen)
heatmap(den.x,den.y,den.density)
heatmap(den.x,den.y,resO)
heatmap(den.x,den.y,resI)

surface(den.x,den.y,resI')

denMarg = vec(sum(den.density,dims=1))
denMarg .*= 1/(sum(denMarg)*step(den.y))

denMarg2 = vec(sum(resO,dims=1))
denMarg2 .*= 1/(sum(denMarg2)*step(den.y))

denMarg3 = vec(sum(resI,dims=1))
denMarg3 .*= 1/(sum(denMarg3)*step(den.y))


denMarg = vec(sum(den.density,dims=2))
denMarg .*= 1/(sum(denMarg)*step(den.x))

denMarg2 = vec(sum(res,dims=2))
denMarg2 .*= 1/(sum(denMarg2)*step(den.x))

denMarg3 = vec(sum(resI,dims=2))
denMarg3 .*= 1/(sum(denMarg3)*step(den.x))

## top row
xlab = L"$D$ [L$^2$/T$^\alpha$]"
ylab = L"{\alpha}\ [1]"
xtickL = [L"10^{-1}",L"10^{-0.5}","1",L"10^{0.5}",L"10^{1}"]
xtick = (-1:0.5:1,xtickL)

with_theme(theme_latexfonts()) do
fig = Figure(size=(800,400))
ga = fig[1, 1] = GridLayout()
gb = fig[1, 2] = GridLayout()

ax = Axis(ga[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Estimation results"
)

th = poly!(ax, Rect(-1,0.4,2,0.2),
    color = :silver,
    strokewidth = 1,
    strokecolor = :black,
)
poly!(ax, Rect(-1,0.8,2,0.2),
color = :silver,
    strokewidth = 1,
    strokecolor = :black,
)

gls = scatter!(ax,bB[1,:],bB[2,:],
    color = :red,
    alpha = 0.7,
    markersize = 2.5,
)
axislegend(ax,[th, MarkerElement(color = :red,alpha=0.7, marker=:circle, markersize = 8)],
    ["original distribution","GLS estimation"]
)
ax = Axis(ga[1,2],
    xlabel = L"histogram of $p_\alpha$",
    yticklabelsvisible = false,
    limits = (0,3.1,0.2,1.2)
)
hist!(ax,bB[2,:],normalization=:pdf,direction=:x,
    color = (:red,0.5),
    strokewidth = 1,
    strokecolor = :firebrick,
    #fillalpha = 0.5
)
lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
colsize!(ga, 1, Relative(4/5))
colgap!(ga,10)

ax2 = Axis(gb[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    title = "Density estimate"
    #ylabel = ylab,
)
heatmap!(ax2,den.x,den.y,den.density,
    colormap = :thermal,
)

ax = Axis(gb[1,2],
    limits = (0,3.1,0.2,1.2),
    yticklabelsvisible = false,
    xlabel = L"density $p_\alpha$",
)

lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax,denMarg,den.y,
    color = :orange,
)
colsize!(gb, 1, Relative(4/5))
colgap!(gb,10)

#fig
save("deconI1.pdf",fig)
end

## bottom row


xlab = L"$D$ [L$^2$/T$^\alpha$]"
xtickL = [L"10^{-1}",L"10^{-0.5}","1",L"10^{0.5}",L"10^{1}"]
xtick = (-1:0.5:1,xtickL)
ylab = L"{\alpha}\ [1]"
with_theme(theme_latexfonts()) do
fig = Figure(size=(800,400))
ga = fig[1, 1] = GridLayout()
gb = fig[1, 2] = GridLayout()


ax = Axis(ga[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    title = "Simple deconvolution",
    ylabel = ylab,
)
heatmap!(ax,den.x,den.y,resO,
    #colormap = :thermal,
)

ax = Axis(ga[1,2],
    limits = (0,3.1,0.2,1.2),
    yticklabelsvisible = false,
    xlabel = L"density $p_\alpha$",
)

lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax,denMarg2,den.y,
    color = :turquoise4,
)
colsize!(ga, 1, Relative(4/5))
colgap!(ga,10)


ax = Axis(gb[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    title = "Interpolated deconvolution",
    #ylabel = ylab,
)
heatmap!(ax,den.x,den.y,resI,
    #colormap = :thermal,
)

ax = Axis(gb[1,2],
    limits = (0,3.1,0.2,1.2),
    yticklabelsvisible = false,
    xlabel = L"density $p_\alpha$",
)

lines!(ax, [0,2.5,2.5,0],[0.4,0.4,0.6,0.6],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax, [0,2.5,2.5,0],[0.8,0.8,1.0,1.0],
    color = :black,
    linewidth = 1.5,
    linestyle = :dash
)
lines!(ax,denMarg3,den.y,
    color = :turquoise4,
)
colsize!(gb, 1, Relative(4/5))
colgap!(gb,10)

#fig
save("deconI2.pdf",fig)
end