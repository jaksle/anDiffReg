import numpy as np
from numba import njit
from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve
from tqdm.auto import tqdm

# main functions

def tamsd(X):
    """
    TA-MSD of trajectories. Time should go along first axis, subsequent trajectories along second axis, x, y, z coordinates along third axis.
    """
    if X.ndim == 1:
        X = X.reshape(-1,1)
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
        fit_ols(tamsd, dim, dt, w)

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
    S = np.linalg.inv(Ts.T @ Ts) @ Ts.T
    ols = S @ np.log10(tamsd[:w, :])
    ols[0, :] -= np.log10(2 * dim)

    fitCov = np.empty((2, 2, n), dtype=np.float64)
    for i in range(n):
        _, Sigma = errCov(ts, dim, ols[1,i], w)
        fitCov[:,:,i] = S @ Sigma @ S.T

    return ols, fitCov

def fit_gls(tamsd, dim, dt, init_alpha, init_D = None, sigma = None, precompute=True, precompute_alphas=np.arange(0.1, 1.62, 0.02)):
    """
        fit_gls(tamsd, dim, dt, init_alpha, ...)
        fit_gls(tamsd, dim, dt, init_alpha, init_D, sigma, ...)

    Fitting TA-MSD with the GLS method.
    Input:
    - tamsd: ln-1×n  matrix containing the entire TA-MSDs of the n length ln sample trajectories
    - dim: original trajectory dimension (usually 1, 2 or 3)
    - dt: sampling interval
    - init_alpha: vector with initial approximate values of anomalous exponent
    Optional input:
    - precompute = True: if true first tabularise error covariances, if false calculate it for each trajectory
    - precompute_alphas = range(0.1,1.62,0.02): points at which precompute
    For estimation with experimental noise provide also:
    - init_D: initial approximate values of diffusivity
    - sigma: noise amplitude, X_obs = X_true + σξ

    Output:
    - gls: 2×n matrix values of (log10 D, α) estimates
    - errCov: 2×2×n matrix with estimated parameter error covariances 
    """
    if init_D is None or sigma is None:
        return fit_gls_base(tamsd, dim, dt, init_alpha, precompute, precompute_alphas)
    else:
        return fit_gls_noise(tamsd, dim, dt, init_alpha, init_D, sigma, precompute, precompute_alphas)


def deconvolve_gls(logDs, alphas, den, dt, ln, dim, alpha, method = "simple", nIter = 30):
    """
        deconvolve_gls(logDs, alphas, den, dt, ln, dim, alpha, method, nIter = 30)
        deconvolve_gls(logDs, alphas, den, dt, ln, dim, (alpha_min,alpha_max), method, nIter = 30)

    Deconvolving pdf of estimated (log10 D, α) obtained with the GLS method. It removes the blur caused by the estimation errors, reconstructing the original distribution of (log10 D, α). This method assumes the data was FBM.
    Input:
    - logDs: labels of log diffusivity
    - alphas: labels of anomalous index
    - den: density which we want to deconvolve
    - dt: sampling inverval
    - ln: length of the orignal trajectory used
    - dim: trajectory dimension (typically 1, 2 or 3)
    - method: "simple" or "full"
    For simple deconvolution provide:
    - alpha: the anomalous index value for which deconvolve, should be the most representative of the sample
    For full deconvolution provide:
    - (alpha_min,alpha_max): range of α for which deconvolve
    Full deconvolution is much more computationally expensive, but the result better reflects the original distribution.

    Optional input:
    - nIter = 30: number of steps in the Richardson-Lucy deconvolution algorithm

    Output: matrix with the deconvolved pdf. 
    """
    if method == "simple":
        return deconvolve_gls_simple(logDs, alphas, den, dt, ln, dim, alpha, nIter)
    elif method == "full":
        return deconvolve_gls_full(logDs, alphas, den, dt, ln, dim, alpha, nIter)

def deconvolve_ols(logDs, alphas, den, dt, ln, dim, alpha, w, method = "simple", nIter = 30):
    """
        deconvolve_ols(logDs, alphas, den, dt, ln, dim, alpha, w, method, nIter = 30)
        deconvolve_ols(logDs, alphas, den, dt, ln, dim, alpha, w, method, nIter = 30)
    Deconvolving pdf of estimated (log10 D, α) obtained with the OLS method. It removes the blur caused by the estimation errors, reconstructing the original distribution of (log10 D, α). This method assumes the data was FBM.
    Input:
    - logDs: labels of log diffusivity
    - alphas: labels of anomalous index
    - den: density which we want to deconvolve
    - dt: sampling inverval
    - ln: length of the orignal trajectory used
    - dim: trajectory dimension (typically 1, 2 or 3)
    - w: size of window in which the OLS was calculated 
    - method: "simple" or "full"
    For simple deconvolution provide:
    - alpha: the value for which deconvolve, should be the most typical in the sample
    For full deconvolution provide:
    - (alpha_min,alpha_max): range of α for which deconvolve
    Full deconvolution is much more computationally expensive, but the result better reflects the original distribution.

    Optional input:
    - nIter = 30: number of steps in the Richardson-Lucy deconvolution algorithm

    Output: matrix with the deconvolved pdf. 
    """
    if method == "simple":
        return deconvolve_ols_simple(logDs, alphas, den, dt, ln, dim, alpha, w, nIter)
    elif method == "full":
        return deconvolve_ols_full(logDs, alphas, den, dt, ln, dim, alpha, w, nIter)


# numerical functions

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

def deconvolve_internal(logDs, alphas, den, C, nIter):
    """
    Performs Richardson-Lucy deconvolution with Gaussian kernel given covariance C.
    """
    res = np.copy(den)
    
    mean = [logDs[len(logDs) // 2], alphas[len(alphas) // 2]]
    mvn = multivariate_normal(mean=mean, cov=C)
    
    ns = np.array([[mvn.pdf([x, y]) for y in alphas] for x in logDs])
    ins = np.flip(ns)

    for _ in range(nIter):
        d = fftconvolve(res, ns, mode = "same")
        d[np.abs(d) < 1e-12] = 1e-12
        res *= fftconvolve(den / d, ins, mode = "same")

    return res

def deconvolve_gls_simple(logDs, alphas, den, dt, ln, dim, alpha, nIter):
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(ln-1), np.log10(ts[:-1])]
    _, Sigma = errCov(ts, dim, alpha)

    return deconvolve_internal(logDs, alphas, den, np.linalg.inv(Ts.T @ np.linalg.inv(Sigma) @ Ts), nIter)

def deconvolve_ols_simple(logDs, alphas, den, dt, ln, dim, alpha, w, nIter):
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(w), np.log10(ts[:w])]
    _, Sigma = errCov(ts, dim, alpha, w)
    S = np.linalg.inv(Ts.T @ Ts) @ Ts.T
    return deconvolve_internal(logDs, alphas, den, S @ Sigma @ S.T, nIter)

def deconvolve_gls_full(logDs, alphas, den, dt, ln, dim, alpha_range, nIter = 30):
    alpha_min,alpha_max = alpha_range
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(ln-1), np.log10(ts[:-1])]
    _, Sigma = errCov(ts, dim, alpha_min)

    res = deconvolve_internal(logDs, alphas, den, np.linalg.inv(Ts.T @ np.linalg.inv(Sigma) @ Ts), nIter)

    j1 = np.argmax(alphas >= alpha_min)
    j2 = np.argwhere(alphas <= alpha_max).max()

    for k in tqdm(range(j1,j2)):
        _, Sigma = errCov(ts, 1, alphas[k])
        zs = deconvolve_internal(logDs, alphas, den, np.linalg.inv(Ts.T @ np.linalg.inv(Sigma) @ Ts), nIter)
        res[:,k] = zs[:,k]

    _, Sigma = errCov(ts, 1, alpha_max )
    zs = deconvolve_internal(logDs, alphas, den, np.linalg.inv(Ts.T @ np.linalg.inv(Sigma) @ Ts), nIter)

    res[:,j2:] = zs[:,j2:]

    res /= (np.sum(res)*(alphas[1]-alphas[0])*(logDs[1]-logDs[0])) # normalise
    return res

def deconvolve_ols_full(logDs, alphas, den, dt, ln, dim, alpha_range, w, nIter):
    alpha_min,alpha_max = alpha_range
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(w), np.log10(ts[:w])]
    _, Sigma = errCov(ts, dim, alpha_min, w)
    S = np.linalg.inv(Ts.T @ Ts) @ Ts.T

    res = deconvolve_internal(logDs, alphas, den, S @ Sigma @ S.T, nIter)

    j1 = np.argmax(alphas >= alpha_min)
    j2 = np.argwhere(alphas <= alpha_max).max()

    for k in tqdm(range(j1,j2)):
        _, Sigma = errCov(ts, 1, alphas[k], w)
        zs = deconvolve_internal(logDs, alphas, den, S @ Sigma @ S.T, nIter)
        res[:,k] = zs[:,k]

    _, Sigma = errCov(ts, 1, alpha_max, w)
    zs = deconvolve_internal(logDs, alphas, den, S @ Sigma @ S.T, nIter)

    res[:,j2:] = zs[:,j2:]

    res /= (np.sum(res)*(alphas[1]-alphas[0])*(logDs[1]-logDs[0])) # normalise
    return res

# covariance functions

def cov_gls(alpha, dt, ln, dim, D = None, sigma = None, w = None, logBase = 10):
    """
    cov_gls(alpha, dt, ln, dim, w, logBase = 10)
    cov_gls(alpha, dt, ln, dim, D, sigma, w, logBase = 10)

    Calculating the expected covariance of the anomalous diffusion parameters GLS estimates (log D, α) given their true values.
    Input:
    - alpha: true anomalous diffusion index
    - dt: sampling inverval
    - ln: length of the orignal trajectory used
    - dim: trajectory dimension (typically 1, 2 or 3)
    - w = ln-1: size of window in which the GLS was calculated 
    - logBase = 10: logarithm base used for log TA-MSD and log diffusivity
    For the FBM with additive experimental noise provide also:
    - D: true value of the diffusivity
    - sigma: noise amplitude, X_obs = X_true + σξ

    Output: 2×2 matrix with the expected covariance of (log D, α)
    """
    if D == None or sigma == None:
        return cov_gls_base(alpha, dt, ln, dim, w, logBase)
    else:
        return cov_gls_noise(alpha, dt, ln, dim, D, sigma, w, logBase)

def cov_gls_base(alpha, dt, ln, dim, w, logBase):
    if w == None:
        w = ln-1
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(w), np.log(ts[:w])/np.log(logBase)]
    _, Sigma = errCov(ts, dim, alpha, w, logBase)

    return np.linalg.inv(Ts.T @ np.linalg.inv(Sigma) @ Ts)

def cov_gls_noise(alpha, dt, ln, dim, D, sigma, w, logBase):
    if w == None:
        w = ln-1
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(w), np.log(ts[:w])/np.log(logBase)]
    orgC, _ = errCov(ts, dim, alpha, w, logBase)
    crossC = crossCov(ts, dim, alpha)
    noiseC = np.empty((ln - 1, ln - 1), dtype=np.float64) 
    for i in range(1,ln-1):
        for j in range(1,ln-1):
            noiseC[i-1,j-1] = noiseCov(ln,i,j)
            noiseC[j-1,i-1] = noiseC[i-1,j-1]

    Sigma = (1 / (np.log(logBase)**2)) * (D**2 * orgC + sigma**2 * D * crossC + sigma**4 * dim * noiseC) / ((2 * D * dim * ts[:w]**alpha) * (2 * D * dim * ts[:w][:, None]**alpha))
    return np.linalg.inv(Ts.T @ np.linalg.inv(Sigma) @ Ts)

def cov_ols(alpha, dt, ln, dim, w, logBase = 10):
    """
        cov_ols(alpha, dt, ln, dim, w, logBase = 10)
    Calculating the expected covariance of the anomalous diffusion parameters OLS estimates (log D, α) given their true values.
    Input:
    - alpha: true anomalous diffusion index
    - dt: sampling inverval
    - ln: length of the orignal trajectory used
    - dim: trajectory dimension (typically 1, 2 or 3)
    - w: size of window in which the OLS was calculated 
    - logBase = 10: logarithm base used for log TA-MSD and log diffusivity

    Output: 2×2 matrix with the expected covariance of (log D, α)
    """
    ts = dt * np.arange(1,ln+1) 
    Ts = np.c_[np.ones(w), np.log(ts[:w])/np.log(logBase)]
    _, Sigma = errCov(ts, dim, alpha, w, logBase)
    S = np.linalg.inv(Ts.T @ Ts) @ Ts.T
    return S @ Sigma @ S.T


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
def errCov(ts, dim, alpha, w = None, logBase = 10):
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
def crossCov(ts, dim, alpha, w = None):
    ln = len(ts)
    if w == None:
        w = ln-1
    cov = np.empty((w, w), dtype=np.float64)
    for i in range(1, w+1):
        for j in range(i, w+1):
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