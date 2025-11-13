
using Statistics, Distributions, LinearAlgebra
using KernelDensity, FFTW

using CairoMakie, ProgressMeter

include("AnDiffReg.jl")
using .AnDiffReg

##---------------------------------------------------------------
# exemplary data: mixture of FBMs
# this is the same system as considered in the "Non-parametric deconvolution" section

ln = 100
n = 10^4
dt =  1
ts = dt*(1:ln)
lts = log10.(ts[1:ln-1])
Ts = [ones(ln-1) lts]

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

## fitting

ols = fit_ols(msd, 1, dt)
gls, covGLS = fit_gls(msd, 1, dt, ols[2,:]) 



## getting kde

den = kde((gls[1,:],gls[2,:]),boundary = ((-1.5,1.5),(0.2,1.3)),npoints=(512,512))

# plotting initial estimate
xlab = L"$D$ [L$^2$/T$^\alpha$]"
ylab = L"{\alpha}\ [1]"
xtickL = [L"10^{-1}",L"10^{-0.5}","1",L"10^{0.5}",L"10^{1}"]
xtick = (-1:0.5:1,xtickL)
fig = Figure()
ax = Axis(fig[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Initial pdf estimate"
)
heatmap!(ax,den.x,den.y,den.density)
fig


## simple deconvolution

function deconvolve_gls(logDs::AbstractVector, αs::AbstractVector, den::AbstractMatrix, ts::AbstractVector, dim::Integer, α::Real, nIter::Integer = 30)  
    Ts = [ones(ln-1) log10.(ts[1:end-1])]
    _, Σ = AnDiffReg.errCov(ts, dim, α)

    return deconvolve_internal(logDs, αs, den, (Ts'*Σ^-1*Ts)^-1, nInter)
end

function deconvolve_ols(logDs::AbstractVector, αs::AbstractVector, den::AbstractMatrix, ts::AbstractVector, dim::Integer, α::Real, w::Integer, nIter::Integer = 30)
    Ts = [ones(ln-1) log10.(ts[1:end-1])]
    S = (Ts'*Ts)^-1 * Ts'
    _, Σ = AnDiffReg.errCov(ts, dim, α, w)

    return deconvolve_internal(logDs, αs, den, S*Σ*S', nInter)
end


deconvolvedPDF = deconvolve_gls(den.x, den.y, den.density, ts, 1, 0.7)


fig = Figure() # plot
ax = Axis(fig[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Deconvolved pdf estimate"
)
heatmap!(ax, den.x, den.y, deconvolvedPDF)
fig


denMarg = vec(sum(res,dims=1))
denMarg .*= 1/(sum(denMarg3)*step(den.y))
lines(den.y,denMarg3)

## full deconvolution 

function deconvolve_internal(logDs, αs, den, C, nIter)
    res = copy(den)
    nn = MvNormal([logDs[end÷2], αs[end÷2]], Symmetric(C))
    ns = [pdf(nn,[x,y]) for x in logDs, y in αs]
    ns = circshift(ns, (length(logDs)÷2, length(αs)÷2))
    ins = reverse(ns)
    res = copy(den)

    for _ in 1:nIter
        d = real.(ifft( fft(res) .* fft(ns)))
        d[abs.(d) .< 10^-12] .= 10^-12
        res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
    end
    return res
end


function deconvolve_gls(logDs::AbstractVector, αs::AbstractVector, den::AbstractMatrix, ts::AbstractVector, dim::Integer, (α_min,α_max)::Tuple{<:Real,<:Real}, nIter::Integer = 30)
    _, Σ = AnDiffReg.errCov(ts, dim, α_min)
    Ts = [ones(ln-1) log10.(ts[1:end-1])]
    res = deconvolve_internal(logDs, αs, den, (Ts'*Σ^-1*Ts)^-1, nInter)

    j1 = findfirst(αs .> α_min)
    j2 = findlast(αs .< α_max)

    @showprogress for k in j1+1:j2-1
        _, Σ = AnDiffReg.errCov(ts, 1, αs[k] )
        zs = deconvolve_internal(logDs, αs, den, (Ts'*Σ^-1*Ts)^-1, nInter)
        res[:,k] .= zs[:,k]
    end

    _, Σ = AnDiffReg.errCov(ts, 1, α_max )
    zs = deconvolve_internal(logDs, αs, den, (Ts'*Σ^-1*Ts)^-1, nInter)

    res[:,j2:end] .= zs[:,j2:end]

    res ./= (sum(res)*step(αs)*step(logDs))
    return res
end


function deconvolve_ols(logDs::AbstractVector, αs::AbstractVector, den::AbstractMatrix, ts::AbstractVector, dim::Integer, (α_min,α_max)::Tuple{<:Real,<:Real}, w::Integer, nIter::Integer = 30)
    _, Σ = AnDiffReg.errCov(ts, dim, α_min, w)
    Ts = [ones(ln-1) log10.(ts[1:w])]
    S = (Ts'*Ts)^-1 * Ts'
    res = deconvolve_internal(logDs, αs, den, S*Σ*S', nInter)

    j1 = findfirst(αs .> α_min)
    j2 = findlast(αs .< α_max)

    @showprogress for k in j1+1:j2-1
        _, Σ = AnDiffReg.errCov(ts, 1, αs[k], w)
        zs = deconvolve_internal(logDs, αs, den, S*Σ*S', nInter)
        res[:,k] .= zs[:,k]
    end

    _, Σ = AnDiffReg.errCov(ts, 1, α_max, w)
    zs = deconvolve_internal(logDs, αs, den, S*Σ*S', nInter)

    res[:,j2:end] .= zs[:,j2:end]

    res ./= (sum(res)*step(αs)*step(logDs))
    return res
end


resI = copy(den.density)
nIter = 100
j1 = findfirst(den.y .> 0.2)
j2 = findlast(den.y .< 1.2)




fig = Figure()
ax = Axis(fig[1,1],
    yticks = 0.2:0.2:1.4,
    xticks = xtick,
    limits = (-1.2,1.2,0.2,1.2),
    xlabel = xlab,
    ylabel = ylab,
    title = "Deconvolved pdf estimate"
)
heatmap!(ax,den.x,den.y,den.density)
fig







###############

D = 1 # mean(bB[1,:])
mA = 0.7 #mean(bB[2,:])
K = (s,t) -> D*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:100 #nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

heatmap(den.x,den.y,res)
#surface(den.x,den.y,res)
#marg = vec(sum(res,dims=1))
#plot(marg)

#resO = copy(res)
## interpolation
nIter = 30

mA = 0.2 #mean(bB[2,:])
K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1] #1D
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))


zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end

#heatmap(den.x,den.y,res')


mA = 1.2 #mean(bB[2,:])
K = (s,t) -> 1*(t^(mA)+s^(mA)-abs(s-t)^(mA))
Σ = [theorCovEff(i,i2,ln,K)/(K(ts[i],ts[i])*K(ts[i2],ts[i2]))* 1/(log(10)^2) for i in 1:ln-1,i2 in 1:ln-1]
eM = (Ts'*Σ^-1*Ts)^-1
nn= MvNormal([den.x[end÷2], den.y[end÷2]], Symmetric(eM))

ns = [pdf(nn,[x,y]) for x in den.x, y in den.y]
ns = circshift(ns,(length(den.x)÷2,length(den.y)÷2))
#heatmap(den.x,den.y,ns')

#dec = deconv(den.density,ns,-1)
zs = den.density
ins = reverse(ns)
res = copy(zs)
for _ in 1:nIter
    d = real.(ifft( fft(res) .* fft(ns)))
    d[abs.(d) .< 10^-12] .= 10^-12
    res .*= real.(ifft( fft(zs ./ d) .* fft(ins)))
end
resI[:,j2:end] .= res[:,j2:end]

resI ./= (sum(resI)*step(den.x)*step(den.y))

#using JLD2
#@save "deConT.jld2" den resI
##

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