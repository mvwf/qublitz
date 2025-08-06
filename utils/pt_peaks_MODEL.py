"""
pt_peaks_mc.py   (Monte-Carlo peak-location utilities)

* _peak_location_core(…)          – unchanged analytic helper
* peak_location(…)                – single-shot wrapper (unchanged API)
* peak_location_mc(…, n_draws=1e3)
        → draws  N(μ,Σ)  and returns mean / min / max for each peak

Assumes parameter errors are 1-σ and **independent**
(use your own full covariance if you have one).
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------
# analytic helpers (unchanged) ----------------------------------------

def _eigenvalues_core(J: float, kappa_c: float, delta_f: float, delta_kappa: float, f_c: float, phi: float):
    lambda_0 = (delta_kappa - kappa_c)/2 + 1j * (delta_f/2 - f_c)
    delta_lambda = np.sqrt(-delta_f**2 + 2 * delta_f * delta_kappa * 1j + delta_kappa**2 - 4 * J * np.exp(1j * phi))
    lambda_plus = lambda_0 + delta_lambda / 2
    lambda_minus = lambda_0 - delta_lambda / 2
    lambdas = np.array([lambda_plus, lambda_minus])
    
    # Sort so lambda_plus has the higher imaginary part
    idx = np.argsort(np.imag(lambdas))[::-1]  # descending order
    return lambdas[idx]

def _p_q_discriminant_core(
        J: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float,
):
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    kappa_bar = kappa_c - delta_kappa
    p = (
            - (delta_f / 2) ** 2
            + (delta_kappa / 2) ** 2
            - cos_p * J ** 2
            + (kappa_bar / 2) ** 2
    )
    q = (kappa_bar / 4) * (delta_f * delta_kappa - 2 * J ** 2 * sin_p)
    discriminant = -4 * p ** 3 - 27 * q ** 2
    return p, q, discriminant


def _peak_location_core(
        J: float,
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float,
) -> np.ndarray:
    offset = f_c - delta_f / 2
    p, q, disc = _p_q_discriminant_core(J, kappa_c, delta_f, delta_kappa, phi)

    # Small value close to zero to avoid gaps in plot
    eps = 1e-20

    if disc > eps:
        theta = (np.pi / 3) + (1 / 3) * np.arccos((3 * q / (2 * p)) * np.sqrt(-3 / p))
        fac = 2 * np.sqrt(-p / 3)
        nu_plus  = fac * np.cos(theta + np.pi / 3) + offset
        nu_minus = fac * np.cos(theta - np.pi / 3) + offset
        return np.array([nu_plus, nu_minus])

    # single-root branches (rare in fitted region)
    if disc < -eps:
        if p < 0:            # cosh branch
            theta = (1 / 3) * np.arccosh((-3 * abs(q) / (2 * p)) * np.sqrt(-3 / p))
            root = -2 * (abs(q) / q) * np.sqrt(-p / 3) * np.cosh(theta) + offset
        else:                # sinh branch
            theta = (1 / 3) * np.arcsinh((3 * q / (2 * p)) * np.sqrt(3 / p))
            root = -2 * np.sqrt(p / 3) * np.sinh(theta) + offset
        return np.array([root])

    # disc == 0 → repeated roots (tangent); treat the same as single
    root = 2 * np.cbrt(-q / 2) + offset
    return np.array([root])


# public “single-shot” wrapper (unchanged)
def peak_location(
        J: float,
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float,
):
    return _peak_location_core(J, f_c, kappa_c, delta_f, delta_kappa, phi)

def eigenvalues(
        J: float,
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float,
):
    return _eigenvalues_core(J, kappa_c, delta_f, delta_kappa, f_c, phi)


# ---------------------------------------------------------------------
# Monte-Carlo uncertainty propagation ---------------------------------
def peak_location_mc(
        means: dict[str, float] | tuple[float, ...],
        sigmas: dict[str, float] | tuple[float, ...],
        *,
        n_draws: int = 1000,
        rng: np.random.Generator | None = None,
):
    """
    Monte-Carlo propagate parameter uncertainties to peak positions.

    Parameters
    ----------
    means, sigmas
        Either dictionaries with keys
            ['J', 'f_c', 'kappa_c', 'delta_f', 'delta_kappa', 'phi']
        or 6-element tuples in that order.
    n_draws : int
        Number of MC samples (default 1000).
    rng : numpy.random.Generator, optional
        Custom RNG.  If None, a fresh default_rng() is used.

    Returns
    -------
    stats : dict
        {
          'nu_plus' : {'mean': μ, 'min': m, 'max': M},
          'nu_minus': {...}          # absent if single-root case
        }
    """
    if isinstance(means, dict):
        order = ['J', 'f_c', 'kappa_c', 'delta_f', 'delta_kappa', 'phi']
        mu  = np.array([means[k]  for k in order], dtype=float)
        sig = np.array([sigmas[k] for k in order], dtype=float)

        bad = ~np.isfinite(sig) | (sig < 0)
        if bad.any():
            for k, bad_flag in zip(order, bad):
                if bad_flag:
                    print(f"⚠️  invalid sigma for {k}: {sigmas[k]}")
    else:
        print('WHAT TF IS THIS ??')
        mu  = np.asarray(means,  dtype=float)
        sig = np.asarray(sigmas, dtype=float)

    cov = np.diag(sig ** 2)
    rng = np.random.default_rng() if rng is None else rng
    samples = rng.multivariate_normal(mu, cov, size=n_draws)

    # collect peak samples (ragged)
    plus_list, minus_list = [], []
    for J, f_c, k_c, d_f, d_k, phi in samples:
        peaks = _peak_location_core(J, f_c, k_c, d_f, d_k, phi)
        if len(peaks) == 2:
            plus_list.append(peaks[0])
            minus_list.append(peaks[1])
        else:  # single root → add to both
            minus_list.append(peaks[0])
            plus_list.append(peaks[0])

    def _stats(arr):
        if len(arr) == 0:
            return None
        arr = np.asarray(arr)
        return {
                "mean": float(arr.mean()),
                "min": float(np.percentile(arr, 16)),
                "max":  float(np.percentile(arr, 84)),
        }

    result = {"nu_plus": _stats(plus_list)}
    if minus_list:
        result["nu_minus"] = _stats(minus_list)
    return result
