from scipy import sparse
from scipy.sparse import linalg as slin
import numpy as np

def denoise_signal(position, alpha, beta, Nit, denoised_signal=None):
    """
    This function implements a nonlinear filtering of eye-movement time-series
    to reduce noise based on sparse time-series properties. In the basence of 
    noise, the first-order and higher-order temporal derivatives of eye-movement
    time-series can be modeled as sparse. Then, from a noisy eye-movement time-series
    y, the time-series x can be estimated by minimizing the objective function:
    f(x) = 1/2 ||y-x||^2 + alpha ||D1 x||1 + beta ||D3x||1.

    Reference: W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson, 
    "Detection of normal and slow saccades using implicit piecewise polynomial 
    approximation," Journal of Vision, vol. 21, no. 6, pp. 1-18, 2021. 

    @param position: time-series saccade position.
    @param alpha: regularization parameter used for denoising.
    @param beta: regularization parameter used for denoising
    @param Nit: number of iterations.
    @denoised_signal: a previously denoised signal
    """

    # Get the length of saccade position signal and create an array of ones
    N = len(position)
    e = np.ones(N)

    # Create the first-order difference and third-order difference operators
    D1 = sparse.spdiags([-e, e], [0, 1], N-1, N)
    D3 = sparse.spdiags([-e, 3*e, -3*e, e], [0, 1, 2, 3], N-3, N)
    I = sparse.spdiags(e, 0, N, N)

    # If previously denoised signal exists, continue denosing on that signal
    if denoised_signal is None:
        x = position
    else:
        x = denoised_signal

    # Implement the nonlinear filter to denoise the signal
    EPS = 1E-10
    x =  np.diff(x)
    # def psi(x): return np.sqrt(x**2 + EPS)
    for i in range(Nit):
        Lam1 = sparse.spdiags(alpha/np.sqrt(x**2 + EPS), 0, N-1, N-1)
        Lam3 = sparse.spdiags(beta/np.sqrt(x**2 + EPS), 0, N-3, N-3)
        temp = I + ((D1.T).dot(Lam1)).dot(D1) + ((D3.T).dot(Lam3)).dot(D3)
        x = slin.spsolve(temp, position)
    return x