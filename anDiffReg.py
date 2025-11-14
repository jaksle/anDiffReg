import numpy as np
from numba import njit

# main functions

def tamsd(X):
    """
    TA-MSD of trajectories. Time should go along first axis, subsequent trajectories along second axis, x, y, z coordinates along third axis.
    """
    ln, n = X.shape[:2]
    msd = np.empty((ln - 1, n), dtype=X.dtype)

    if X.ndim == 3:
        for j in range(n):
            for i in range(1, ln):
                msd[i - 1, j] = np.mean(
                    np.sum((X[:ln - i, j, :] - X[i:, j, :]) ** 2, axis = 1), axis=0
                )
    else:
        for j in range(n):
            for i in range(1, ln):
                msd[i - 1, j] = np.mean((X[:ln - i, j] - X[i:, j]) ** 2)

    
    return msd

@njit
def fit_ols(tamsd, dim, dt, w=None):
    """
    fit_ols(tamsd::AbstractMatrix, dim::Integer, dt::Real, w::Integer)

    Fitting TA-MSD with the OLS method.
    Input:
    - tamsd: (ln-1)×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
    - dim: original trajectory dimension (usually 1, 2 or 3)
    - dt: sampling interval
    - w = max(5, tamsd.shape[0] // 10): window size
    Output:
    - ols: 2×n matrix values of (log10 D, α) estimates
    - fitCov: 2×2×n matrix with estimated parameter error covariances 
    """
    ln, n = tamsd.shape
    if w is None:
        w = max(5, ln // 10)
    
    ts = dt * np.arange(1, ln)
    Ts = np.column_stack((np.ones(w), np.log10(ts[:w])))
    S = np.linalg.inv(Ts.T @ Ts)
    ols = S @ Ts.T @ np.log10(tamsd[:w, :])
    ols[0, :] -= np.log10(2 * dim)

    fitCov = np.empty((2, 2, n), dtype=np.float64)
    for i in range(n):
        _, Sigma = errCov(ts, dim, ols[1,i], w)
        fitCov[:,:,i] = S @ Ts.T @ Sigma @ Ts @ S

    return ols, fitCov

def fit_gls(tamsd, dim, dt, init_alpha, init_D = None, sigma = None, precompute=True, precompute_alphas=np.arange(0.1, 1.62, 0.02)):
    """
        fit_gls(tamsd, dim, dt, init_α, ...)
        fit_gls(tamsd, dim, dt, init_α,, init_D, sigma, ...)

    Fitting TA-MSD with the GLS method.
    Input:
    - tamsd: ln-1×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
    - dim: original trajectory dimension (usually 1, 2 or 3)
    - dt: sampling interval
    - init_alpha: vector with initial approximate values of anomalous exponent
    Optional input:
    - precompute = True: if true first tabularise error covariances, if false calculate it for each trajectory
    - precompute_alphas = range(0.1,1.62,0.02): points at which precompute
    Output:
    - gls: 2×n matrix values of (log10 D, α) estimates
    - errCov: 2×2×n matrix with estimated parameter error covariances 

    For estimation with experimental noise provide also:
    - init_D: initial approximate values of diffusivity
    - sigma: noise amplitude, X_obs = X_true + σξ
    """
    if init_D is None or sigma is None:
        return fit_gls_base(tamsd, dim, dt, init_alpha, precompute, precompute_alphas)
    else:
        return fit_gls_noise(tamsd, dim, dt, init_alpha, init_D, sigma, precompute, precompute_alphas)


# utility functions

@njit
def fit_gls_base(tamsd, dim, dt, init_alpha, precompute=True, precompute_alphas=np.arange(0.1, 1.62, 0.02)):
    ln, n = tamsd.shape[0] + 1, tamsd.shape[1]  # ln is of original trajectory
    ts = dt * np.arange(1, ln + 1)
    Ts = np.vstack((np.ones(ln - 1), np.log10(ts[:ln - 1]))).T
    gls = np.empty((2, n), dtype=np.float64)
    fitCov = np.empty((2, 2, n), dtype=np.float64)
    lmsd = np.log10(tamsd)

    if precompute:
        # precompute covariances
        na = len(precompute_alphas)
        errC = np.empty((ln - 1, ln - 1, na), dtype=np.float64)
        iC = np.empty((ln - 1, ln - 1, na), dtype=np.float64)
        bias = np.empty((ln - 1, na), dtype=np.float64)

        for k in range(na):
            c = errCov(ts, dim, precompute_alphas[k])[1]
            errC[:, :, k] = c
            bias[:, k] = -np.log(10) * np.diag(c) / 2
            iC[:, :, k] = np.linalg.inv(c)

        # estimate
        for i in range(n):
            j0 = np.argmin(np.abs(precompute_alphas - init_alpha[i]))
            gR = np.linalg.inv(Ts.T @ iC[:, :, j0] @ Ts) @ Ts.T @ iC[:, :, j0]
            gls[:, i] = gR @ (lmsd[:, i] - bias[:, j0])

            j1 = np.argmin(np.abs(precompute_alphas - gls[1, i]))
            fitCov[:, :, i] = np.linalg.inv(Ts.T @ iC[:, :, j1] @ Ts)
    else: # separate calculation for each trajectory
        for i in range(n):
            errC = errCov(ts, dim, init_alpha[i])[1]
            bias = -np.log(10) * np.diag(errC) / 2
            iC = np.linalg.inv(errC)
            gR = np.linalg.inv(Ts.T @ iC @ Ts) @ Ts.T @ iC
            gls[:, i] = gR @ (lmsd[:, i] - bias)
            errC2 = errCov(ts, dim, gls[1, i])[1]
            fitCov[:, :, i] = np.linalg.inv(Ts.T @ np.linalg.inv(errC2) @ Ts)

    gls[0, :] -= np.log10(2 * dim)
    return gls, fitCov


@njit
def fit_gls_noise(tamsd, dim, dt, init_alpha, init_D, sigma, precompute=True, precompute_alphas=np.arange(0.1, 1.62, 0.02)):
    ln, n = tamsd.shape[0] + 1, tamsd.shape[1]
    na = len(precompute_alphas)
    ts = dt * np.arange(1, ln + 1)
    Ts = np.column_stack((np.ones(ln - 1), np.log10(ts[:ln - 1])))
    gls = np.empty((2, n), dtype=np.float64)
    fitCov = np.empty((2, 2, n), dtype=np.float64)
 
    lmsd = np.log10(tamsd - 2 * dim * sigma**2)

    noiseC = np.empty((ln - 1, ln - 1), dtype=np.float64) 
    for i in range(1,ln-1):
        for j in range(1,ln-1):
            noiseC[i-1,j-1] = noiseCov(ln,i,j)
            noiseC[j-1,i-1] = noiseC[i-1,j-1]

    if precompute:
        orgC = np.empty((ln - 1, ln - 1, na), dtype=np.float64)  # pure FBM, no noise, no log scale
        crossC = np.empty((ln - 1, ln - 1, na), dtype=np.float64)  # cross term in cov

        for k in range(na):
            orgC[:, :, k] = errCov(ts, dim, precompute_alphas[k])[0]
            crossC[:, :, k] = crossCov(ts, dim, precompute_alphas[k])

        for i in range(n):
            α0, D0 = init_alpha[i], init_D[i]
            j0 = np.argmin(np.abs(precompute_alphas - α0))
            errC0 = (1 / (np.log(10)**2)) * (D0**2 * orgC[:, :, j0] + sigma**2 * D0 * crossC[:, :, j0] + sigma**4 * dim * noiseC) / ((2 * D0 * dim * ts[:ln - 1]**(α0)) * (2 * D0 * dim * ts[:ln - 1][:, None]**(α0)))
            mask = ~np.isnan(lmsd[:, i])
            iC0 = np.linalg.inv(errC0[mask][:, mask])
            bias = -np.log(10) * np.diag(errC0) / 2
            gR = np.linalg.inv(Ts[mask, :].T @ iC0 @ Ts[mask, :]) @ Ts[mask, :].T @ iC0
            gls[:, i] = gR @ (lmsd[mask, i] - bias[mask])
            gls[0, i] -= np.log10(2 * dim)

            α1, D1 = gls[1, i], 10**gls[0, i]
            j1 = np.argmin(np.abs(precompute_alphas - α1))
            errC1 = (1 / (np.log(10)**2)) * (D1**2 * orgC[:, :, j1] + sigma**2 * D1 * crossC[:, :, j1] + sigma**4 * dim * noiseC) / ((2 * D1 * dim * ts[:ln - 1]**(α1)) * (2 * D1 * dim * ts[:ln - 1][:, None]**(α1))) 
            iC1 = np.linalg.inv(errC1)
            fitCov[:, :, i] = np.linalg.inv(Ts.T @ iC1 @ Ts)
    else: # separate calculation for each trajectory
        for i in range(n):
            α0, D0 = init_alpha[i], init_D[i]
            orgC = errCov(ts, dim, α0)[0]
            crossC = crossCov(ts, dim, α0)
            errC0 = (1 / (np.log(10)**2)) * (D0**2 * orgC + sigma**2 * D0 * crossC + sigma**4 * dim * noiseC) / ((2 * D0 * dim * ts[:ln - 1]**(α0)) * (2 * D0 * dim * ts[:ln - 1][:, None]**(α0)))
            
            bias = -np.log(10) * np.diag(errC0) / 2

            mask = ~np.isnan(lmsd[:, i])
            iC0 = np.linalg.inv(errC0[mask][:, mask])
            gR = np.linalg.inv(Ts[mask, :].T @ iC0 @ Ts[mask, :]) @ Ts[mask, :].T @ iC0
            gls[:, i] = gR @ (lmsd[mask, i] - bias[mask])
            gls[0, i] -= np.log10(2 * dim)

            α1, D1 = gls[1, i], 10**gls[0, i]
            orgC = errCov(ts, dim, α1)[0]
            crossC = crossCov(ts, dim, α1)
            errC1 = (1 / (np.log(10)**2)) * (D1**2 * orgC + sigma**2 * D1 * crossC + sigma**4 * dim * noiseC) / ((2 * D1 * dim * ts[:ln - 1]**(α1)) * (2 * D1 * dim * ts[:ln - 1][:, None]**(α1)))
            fitCov[:, :, i] = np.linalg.inv(Ts.T @ np.linalg.inv(errC1) @ Ts)

    return gls, fitCov

@njit
def theorCovEff(ts, k, l, ln, alpha):
    K = lambda s, t: 2 * np.minimum(s, t) if np.abs(alpha - 1.0) < 1e-8 else (s**alpha + t**alpha - np.abs(s - t)**alpha)
    if k > l:
        k, l = l, k
    N1 = lambda h,k,l,ln: ln - l - h + 1
    N2 = lambda h,k,l,ln: (ln - l) if h <= l - k + 1 else (ln - k - h + 1)

    S1 = 0.
    for h in range(2, ln - l + 1):
        S1 += N1(h,k,l,ln) * (K(ts[0], ts[h-1]) + K(ts[0] + ts[k-1], ts[h-1] + ts[l-1]) - K(ts[0], ts[h-1] + ts[l-1]) - K(ts[0] + ts[k-1], ts[h-1]))**2
    S2 = 0.
    for  h in range(1, ln - k + 1):
        S2 += N2(h,k,l,ln) * (K(ts[h-1], ts[0]) + K(ts[h-1] + ts[k-1], ts[0] + ts[l-1]) - K(ts[h-1], ts[0] + ts[l-1]) - K(ts[h-1] + ts[k-1], ts[0]))**2
    return 2 / ((ln - k) * (ln - l)) * (S1 + S2)

@njit
def errCov(ts, dim, alpha, w=None, logBase=10):
    """
    Covariance matrix of errors of TA-MSD and log TA-MSD. Data is assumed to come from FBM, D = 1. Labels ts correspond to the original trajectory.
    """
    K = lambda s, t: 2 * np.minimum(s, t) if np.abs(alpha - 1.0) < 1e-8 else (s**alpha + t**alpha - np.abs(s - t)**alpha)
    
    ln = len(ts)
    if w == None:
        w = ln - 1
    errC = np.empty((w, w), dtype=np.float64)
    logErrCov = np.empty((w, w), dtype=np.float64)

    for i in range(w):
        for j in range(i, w):
            c = theorCovEff(ts, i + 1, j + 1, ln, alpha) 
            errC[i, j] = dim * c
            logErrCov[i, j] = c / (dim * K(ts[i], ts[i]) * K(ts[j], ts[j]) * (np.log(logBase) ** 2))
            errC[j, i] = errC[i, j]
            logErrCov[j, i] = logErrCov[i, j]

    return errC, logErrCov

@njit
def crossCovEff(ts, k, l, ln, alpha):
    K = lambda s, t: 2 * np.minimum(s, t) if np.abs(alpha - 1.0) < 1e-8 else (s**alpha + t**alpha - np.abs(s - t)**alpha)
    if k > l:
        k, l = l, k
    N1 = lambda h: ln - l - h + 1
    N2 = lambda h: (ln - l) if h <= l - k + 1 else (ln - k - h + 1)

    S1 = 0.
    for h in range(2, ln - l + 1):
        S1 += N1(h) * ( K(ts[0], ts[h-1]) + K(ts[0] + ts[k-1], ts[h-1] + ts[l-1]) - K(ts[0], ts[h-1] + ts[l-1]) - K(ts[0] + ts[k-1], ts[h-1]) ) * ((h == 1) + (1 + k == h + l) - (1 == h + l) - (1 + k == h) ) 
    
    S2 = 0.
    for h in range(1, ln - k + 1):
        S2 += N2(h) * ( K(ts[h-1], ts[0]) + K(ts[h-1] + ts[k-1], ts[0] + ts[l-1]) - K(ts[h-1], ts[0] + ts[l-1]) - K(ts[h-1] + ts[k-1], ts[0]) ) * ((h == 1) + (h + k == 1 + l) - (h == 1 + l) - (h + k == 1)) 

    return 4 / ((ln - k) * (ln - l)) * (S1 + S2)

@njit
def crossCov(ts, dim, alpha):
    ln = len(ts)
    cov = np.empty((ln - 1, ln - 1), dtype=np.float64)
    for i in range(1, ln):
        for j in range(i, ln):
            cov[i - 1, j - 1] = dim * crossCovEff(ts, i, j, ln, alpha)
            cov[j - 1, i - 1] = cov[i - 1, j - 1]

    return cov

@njit
def noiseCov(ln, k, l):
    """
    Covariance of 1D iid noise TA-MSD
    """
    if k > l:
        l, k = k, l
    if k == l:
        return 4 / (ln - k) ** 2 * ((3 * ln - 4 * k) if ln >= 2 * k else (2 * ln - 2 * k))
    else:
        return 4 / ((ln - k) * (ln - l)) * ((2 * ln - k - 2 * l) if ln >= k + l else (ln - l))