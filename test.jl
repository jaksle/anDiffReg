using Revise, BenchmarkTools
using .AnDiffReg

ts = 1:100
ln = 100
alpha = 0.8

C1,L1 = AnDiffReg.errCovNonAlloc(ts, 2, alpha, 50)
C2, L2 = AnDiffReg.errCov(ts, 2, alpha, 50)

D1 = AnDiffReg.crossCovNonAlloc(ts, 2, 0.4, 50)
D2 = AnDiffReg.crossCov(ts, 2, 0.4, 50)

##

@benchmark AnDiffReg.crossCovNonAlloc(ts, 2, 0.4, 50)

##

@benchmark AnDiffReg.crossCov(ts, 2, 0.4, 50)

##

α = 0.4
K(s,t) = (α ≈ 1.0) ? 2min(s,t) : (s^α + t^α - abs(s-t)^α)
S = Float64
cFBM = Matrix{S}(undef, ln, ln)
for i in 1:ln, j in i:ln # tabularise cov matrix of the trajectory
    cFBM[i,j] = K(ts[i],ts[j])
    cFBM[j,i] = cFBM[i,j]
end

AnDiffReg.theorCovEff(1,10,ln,cFBM) 