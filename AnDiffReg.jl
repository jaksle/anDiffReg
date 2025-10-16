module AnDiffReg

using Statistics, LinearAlgebra, ProgressMeter

export tamsd, fit_ols, fit_gls

"""
    TA-MSD of trajectories. Time should go along first axis, subsequent trajectories along second axis, x, y, z coordinates along third axis.
"""
function tamsd(X::AbstractArray{T,N}) where {T <: Real, N}
    ln, n, w =  size(X)
    msd = Matrix{T}(undef, ln-1, n)
    for j in 1:n, i in 1:ln-1
        msd[i, j] = mean(sum((X[k,j,l] - X[k+i,j,l])^2 for l in 1:w) for k in 1:ln-i)
    end
    return msd
end

function tamsd(X::AbstractMatrix{T}) where {T <: Real}
    ln, n =  size(X)
    msd = Matrix{T}(undef, ln-1, n)
    for j in 1:n, i in 1:ln-1
        msd[i, j] = mean((X[k,j] - X[k+i,j])^2 for k in 1:ln-i)
    end
    return msd
end

function tamsd(X::AbstractVector{T}) where {T <: Real}
    ln =  length(X)
    msd = Vector{T}(undef, ln-1)
    for i in 1:ln-1
        msd[i] = mean((X[k,j] - X[k+i,j])^2 for k in 1:ln-i)
    end
    return msd
end

"""
    fit_ols(tamsd::AbstractMatrix, dim::Integer, Δt::Real, w::Integer)

Fitting TA-MSD with the OLS method.
Input:
- tamsd: ln-1×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
- dim: original trajectory dimension (usually 1, 2 or 3)
- Δt: sampling interval
- w = max(5,size(tamsd)[1]÷10): integer fitting width size
"""
function fit_ols(tamsd::AbstractMatrix, dim::Integer, Δt::Real, w::Integer = max(5,size(tamsd)[1]÷10))
    Ts = [ones(w) log10.(Δt*(1:w))]
    estPar = (Ts'*Ts)^-1*Ts' * log10.(tamsd[1:w,:])
    estPar[1,:] .-= log10(2dim)
    return estPar
end

"""
    fit_gls(tamsd::AbstractMatrix, dim::Integer, Δt::Real, init_α::AbstractVector; ...)
    fit_gls(tamsd::AbstractMatrix, dim::Integer, Δt::Real, init_α::AbstractVector, init_D::AbstractVector, σ = 0.; ...)

Fitting TA-MSD with the GLS method.
Input:
- tamsd: ln-1×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
- dim: original trajectory dimension (usually 1, 2 or 3)
- Δt: sampling interval
- init_α: vector with initial approximate values of anomalous exponent
Keyword input:
- precompute = true: if true first tabularise error covariances, if false calculate it for each trajectory
- precompute_αs = 0.1:0.02:1.6: points at which precompute
Output:
- gls: 2×n matrix values of (log10 D, α) estimates
- errCov: 2×2×n matrix with estimated parameter error covariances 

For estimation with experimental noise provide also:
- init_D: vector with initial approximate values of diffusivity
- σ: noise amplitude, X_obs = X_true + σξ
"""
function fit_gls(tamsd::AbstractMatrix, dim::Integer, Δt::Real, init_α::AbstractVector;
     precompute::Bool = true,
     precompute_alphas::AbstractVector = 0.1:0.02:1.6
     )
    ln, n = size(tamsd)[1]+1, size(tamsd)[2] # ln is of orignal trajectory (!)
    ts = Δt*(1:ln)
    Ts = [ones(ln-1) log10.(ts[1:ln-1])]
    gls = Matrix{Float64}(undef, 2, n)
    fitCov = Array{Float64}(undef, 2, 2, n)
    lmsd = log10.(tamsd)

    if precompute
        # precompute covariances
        na = length(precompute_alphas)
        errC = Array{Float64}(undef,ln-1,ln-1,na)
        iC = Array{Float64}(undef,ln-1,ln-1,na)
        bias = Array{Float64}(undef,ln-1,na)

        @showprogress for k in 1:na
            c = errCov(ts, dim, precompute_alphas[k])[2]
            errC[:,:,k] .= c
            bias[:,k] .=  -log(10) .* diag(c) ./2
            iC[:,:,k] = inv(c)
        end

        # estimate
        @showprogress for i in 1:n
            j0 = argmin(abs.(precompute_alphas .- init_α[i]))
            gR = (Ts'*iC[:,:,j0]*Ts)^-1*Ts'*iC[:,:,j0]
            gls[:,i] .= gR*(lmsd[:,i] .- bias[:,j0])

            j1 = argmin(abs.(precompute_alphas .- gls[2,i]))
            fitCov[:,:,i] .= (Ts'*iC[:,:,j1]*Ts)^-1
        end
    else # separate calculation for each trajectory
        @showprogress for i in 1:n
            errC = errCov(ts, dim, init_α[i])[2]
            bias = -log(10) .* diag(errC) ./2
            iC = inv(errC)
            gR = (Ts'*iC*Ts)^-1*Ts'*iC
            gls[:,i] .= gR*(lmsd[:,i] .- bias)
            errC2 = errCov(ts, dim, gls[2,i])[2]
            fitCov[:,:,i] .= (Ts'*inv(errC2)*Ts)^-1
        end
    end

    gls[1,:] .-= log10(2dim)
    return gls, fitCov
end

# with noise
function fit_gls(tamsd::AbstractMatrix, dim::Integer, Δt::Real, init_α::AbstractVector, init_D::AbstractVector, σ::Real;
     precompute::Bool = true,
     precompute_alphas::AbstractVector = 0.1:0.02:1.6
     )
    ln, n = size(tamsd)[1]+1, size(tamsd)[2]
    ts = Δt*(1:ln)
    Ts = [ones(ln-1) log10.(ts[1:ln-1])]
    gls = Matrix{Float64}(undef, 2, n)
    fitCov = Array{Float64}(undef, 2, 2, n)
    lg(x) = x >= 0 ? log10(x) : NaN
    lmsd = lg.(tamsd .- 2dim*σ^2)

    noiseC = noiseCov.(ln, 1:ln-1, (1:ln-1)')

    if precompute
        # precompute covariances
        na = length(precompute_alphas)
        orgC = Array{Float64}(undef, ln-1, ln-1, na) # pure FBM, no noise, no log scale
        crossC = Array{Float64}(undef, ln-1, ln-1, na) # cross term in cov

        @showprogress for k in 1:na
            orgC[:,:,k] .= errCov(ts, dim, precompute_alphas[k])[1]
            crossC[:,:,k] .= crossCov(ts, dim, precompute_alphas[k])
        end

        # estimate
        @showprogress for i in 1:n
            α0, D0 = init_α[i], init_D[i]
            j0 = argmin(abs.(precompute_alphas .- α0))
            errC0 = @. 1/(log(10)^2) * (D0^2*orgC[:,:,j0] + σ^2*D0*crossC[:,:,j0] + σ^4*dim*noiseC)/((2D0*dim*ts[1:ln-1]^(α0)) *(2D0*dim*ts[1:ln-1]'^(α0)))
            
            mask = .!isnan.(lmsd[:,i])
            iC0 = inv(errC0[mask,mask])
            bias = -log(10) .* diag(errC0) ./2
            gR = (Ts[mask,:]'*iC0*Ts[mask,:])^-1*Ts[mask,:]'*iC0
            gls[:,i] .= gR*(lmsd[mask,i] .- bias[mask])
            gls[1,i] -= log10(2dim)

            α1, D1 = gls[2,i], 10^gls[1,i]
            j1 = argmin(abs.(precompute_alphas .- α1))
            errC1 = @. 1/(log(10)^2) * (D1^2*orgC[:,:,j1] + σ^2*D1*crossC[:,:,j1] + σ^4*dim*noiseC)/((2D1*dim*ts[1:ln-1]^(α1))*(2D1*dim*ts[1:ln-1]'^(α1))) 
            iC1 = inv(errC1)
            fitCov[:,:,i] .= (Ts'*iC1*Ts)^-1
        end
    else # separate calculation for each trajectory
        @showprogress for i in 1:n
            α0, D0 = init_α[i], init_D[i]
            orgC = errCov(ts, dim, α0)[1]
            crossC = crossCov(ts, dim, α0)
            errC0 = @. 1/(log(10)^2) * (D0^2*orgC + σ^2*D0*crossC + σ^4*dim*noiseC)/((2D0*dim*ts[1:ln-1]^(α0))*(2D0*dim*ts[1:ln-1]'^(α0)))
            
            bias = -log(10) .* diag(errC0) ./2

            mask = .!isnan.(lmsd[:,i])
            iC0 = inv(errC0[mask,mask])
            gR = (Ts[mask,:]'*iC0*Ts[mask,:])^-1*Ts[mask,:]'*iC0
            gls[:,i] .= gR*(lmsd[mask,i] .- bias[mask])
            gls[1,i] -= log10(2dim)

            α1, D1 = gls[2,i], 10^gls[1,i]
            orgC = errCov(ts, dim, α1)[1]
            crossC = crossCov(ts, dim, α1)
            errC1 = @. 1/(log(10)^2) * (D1^2*orgC + σ^2*D1*crossC + σ^4*dim*noiseC)/((2D1*dim*ts[1:ln-1]^(α1))*(2D1*dim*ts[1:ln-1]'^(α1)))
            fitCov[:,:,i] .= (Ts'*inv(errC1)*Ts)^-1
        end
    end

    return gls, fitCov
end


## utility functions

function incrCov(ts,i,j,k,l,K) 
    a, b, c, d = ts[i], ts[j], ts[k], ts[l]
    K(a,b) + K(a+c,b+d) - K(a,b+d) - K(a+c,b)
end

"""
Covariance betweeen points ts[k] and ts[l] of TA-MSD calculated from trajectory with covariance function K = K(s,t).
""" 
function theorCovEff(ts,k,l,ln,K)
    if k > l
        k, l = l, k
    end
    N1 = h -> ln-l-h+1
    N2 = h -> (h <= l-k+1) ? ( ln-l ) : ( ln-k-h+1 )

    return 2/((ln-k)*(ln-l)) * ( 
          sum(N1(h)*incrCov(ts,1,h,k,l,K)^2 for h in 2:ln-l; init=0) 
        + sum( N2(h)*incrCov(ts,h,1,k,l,K)^2 for h in 1:ln-k ) )
end

"""
Specialised TA-MSD covariance for FBM
"""
function theorCovEffFBM(ts,k,l,ln,α) 
    K(s,t) = (α ≈ 1.0) ? 2min(s,t) : (s^α + t^α - abs(s-t)^α) 
    k, l = minmax(k, l)

    S1 = 0.
    @simd for h in 2:ln-l
        S1 += (ln-l-h+1) * incrCov(ts,1,h,k,l,K)^2
    end
    S2 = 0.
    @simd for h in 1:ln-k 
        S2 += ((h <= l-k+1) ? ( ln-l ) : ( ln-k-h+1 )) * incrCov(ts,h,1,k,l,K)^2
    end
    return  2/((ln-k)*(ln-l)) * (S1 + S2)
end

"""
Covariance matrix of errors of TA-MSD and log TA-MSD. Data is assumed to come from FBM.
"""
function errCov(ts::AbstractVector, dim::Integer, α::Real,  logBase::Integer = 10)

    ln = length(ts)
    S = float(eltype(ts))
    errCov = Matrix{S}(undef, ln-1, ln-1)
    logErrCov = Matrix{S}(undef, ln-1, ln-1)

    for i in 1:ln-1, j in i:ln-1
        c = theorCovEffFBM(ts,i,j,ln,α)
        errCov[i,j] = dim*c
        logErrCov[i,j] = c / ( dim * 2ts[i]^α * 2ts[j]^α * log(logBase)^2 ) 
    end

    return Symmetric(errCov), Symmetric(logErrCov)
end

function crossCov(ts::AbstractVector, dim::Integer, α::Real)

    function crossCovEffFBM(ts, k, l, ln, α)
        K(s,t) = (α ≈ 1.0) ? 2min(s,t) : (s^α + t^α - abs(s-t)^α) 
        k, l = minmax(k, l)

        S1 = 0.
        @simd for h in 2:ln-l
            S1 += (ln-l-h+1) * incrCov(ts,1,h,k,l,K)*(==(1,h) + ==(1+k,h+l) - ==(1,h+l) - ==(1+k,h))
        end
        S2 = 0.
        @simd for h in 1:ln-k 
            S2 += ((h <= l-k+1) ? ( ln-l ) : ( ln-k-h+1 )) * incrCov(ts,h,1,k,l,K)*(==(h,1) + ==(h+k,1+l) - ==(h,1+l) - ==(h+k,1))
        end
        return  4/((ln-k)*(ln-l)) * (S1 + S2)
    end

    ln = length(ts)
    S = float(eltype(ts))
    cov = Matrix{S}(undef, ln-1, ln-1)
    for i in 1:ln-1, j in i:ln-1
        cov[i,j] = dim*crossCovEffFBM(ts,i,j,ln,α)
    end

    return Symmetric(cov)
end

"""
Covariance of 1D iid noise TA-MSD
"""
function noiseCov(ln,k,l)
    if k > l
        l, k = k, l
    end
    if k == l 
        return 4/(ln-k)^2 * ( (ln >= 2k) ? (3ln-4k) : (2ln-2k) )
    else
        return 4/((ln-k)*(ln-l)) * ( (ln >= k+l) ? ( 2ln-k-2l) : (ln-l) )
    end
end


end