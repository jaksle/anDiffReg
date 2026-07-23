
# %%
ln = 100 # trajectory length
dt = 1 # time interval
alpha, D = 0.4, 1 
ts = dt * np.arange(1, ln)
K = lambda s, t: 2 * np.minimum(s, t) if np.abs(alpha - 1.0) < 1e-8 else (s**alpha + t**alpha - np.abs(s - t)**alpha)

cFBM = np.empty((ln-1, ln-1), dtype=np.float64)
for i in range(ln-1):
    for j in range(i,ln-1):
        cFBM[i,j] = K(ts[i],ts[j])
        cFBM[j,i] = cFBM[i,j]

k, l, h = 2, 2, 3

a, b, c, d  = K(ts[0], ts[h-1]), K(ts[0] + ts[k-1], ts[h-1] + ts[l-1]), K(ts[0], ts[h-1] + ts[l-1]), K(ts[0] + ts[k-1], ts[h-1])

a2, b2, c2, d2 = cFBM[0,h-1], cFBM[k, h+l-1], cFBM[0, h+l-1], cFBM[k, h-1]

# %%

a, b, c, d = K(ts[h-1], ts[0]), K(ts[h-1] + ts[k-1], ts[0] + ts[l-1]), K(ts[h-1], ts[0] + ts[l-1]), K(ts[h-1] + ts[k-1], ts[0])
a2, b2, c2, d2 = cFBM[h-1, 0], cFBM[h+k-1, l], cFBM[h-1, l], cFBM[h+k-1, 0]
# %%
i, j = 10, 10
c = theorCovEff(ts, i + 1, j + 1, ln, alpha) 
c2 = theorCovEffMtx(i + 1, j + 1, ln, cFBM) 


# %%
ln = 100 # trajectory length
dt = 1 # time interval
ts = dt * np.arange(1, ln+1)
alpha, D = 0.8, 1 

C1, L1 = an.errCovNonAlloc(ts, 2, alpha, 50)

# %%

C2, L2 = an.errCov(ts, 2, alpha, 50)

#
# %%
import anDiffReg as an
import numpy as np

ln = 100 # trajectory length
dt = 1 # time interval
ts = dt * np.arange(1, ln+1)
alpha, D = 0.8, 1 

C1 = an.crossCovNonAlloc(ts, 2, alpha, 50)

# %%

C2 = an.crossCov(ts, 2, alpha, 50)

# %%

@njit
def crossCov(ts, dim, alpha, w = None):
    K = lambda s, t: 2 * np.minimum(s, t) if np.abs(alpha - 1.0) < 1e-8 else (s**alpha + t**alpha - np.abs(s - t)**alpha)
    ln = len(ts)
    if w == None:
        w = ln-1
        
    cov = np.empty((w, w), dtype=np.float64)
    cFBM = np.empty((ln, ln), dtype=np.float64)
    for i in range(ln):
        for j in range(i,ln):
            cFBM[i,j] = K(ts[i],ts[j])
            cFBM[j,i] = cFBM[i,j]

    for i in range(1, w+1):
        for j in range(i, w+1):
            cov[i - 1, j - 1] = dim * crossCovEffMtx(i, j, ln, cFBM)
            cov[j - 1, i - 1] = cov[i - 1, j - 1]

    return cov
# %%
