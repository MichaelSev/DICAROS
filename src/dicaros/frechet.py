"""Frechet (Karcher) mean on the LDDMM landmark manifold.

Thin wrapper around ``jaxgeometry`` that minimises the mean squared Riemannian
``Log`` distance from a candidate mean to all input shapes, using BFGS with an
analytic gradient (``grad = -2/n * sum_i Log_x(y_i)``). Adapted from Michael's
``mean_estimator_Frechet.py``.

This module imports ``jax``/``jaxgeometry`` at call time, so projects that only
use the Euclidean mean never pay the JAX import cost.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frechet_mean"]


def frechet_mean(shapes_flat, initial_mean, d, options=None):
    """Frechet mean of ``shapes_flat`` on the landmark manifold.

    Parameters
    ----------
    shapes_flat : ndarray
        ``(n_shapes, n_landmarks*d)`` aligned shapes.
    initial_mean : ndarray
        ``(n_landmarks*d,)`` starting point (typically the Euclidean mean).
    d : int
        Landmark dimension.
    options : dict or None
        Overrides for the BFGS options (defaults: ``gtol=1e-1, maxiter=20``).

    Returns
    -------
    ndarray
        ``(n_landmarks*d,)`` Frechet mean coordinates.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist
    from jaxgeometry.manifolds.landmarks import landmarks
    from jaxgeometry.Riemannian import metric, Log
    from jaxgeometry.dynamics import Hamiltonian

    shapes_flat = np.asarray(shapes_flat, dtype=float)
    initial_mean = np.asarray(initial_mean, dtype=float)
    n_landmarks = shapes_flat.shape[1] // d

    sigma_k = np.mean(pdist(initial_mean.reshape(-1, d), metric="euclidean"))
    sigma_k = jnp.array([sigma_k] * d)

    M = landmarks(n_landmarks, k_sigma=sigma_k * jnp.eye(d), m=d)
    metric.initialize(M)
    Hamiltonian.initialize(M)
    Log.initialize(M, f=M.Exp_Hamiltonian)

    @jax.jit
    def single_log(x, y):
        return M.Log((x, [0]), (y, [0]))[0]

    @jax.jit
    def all_logs(x, ys):
        return jax.vmap(lambda y: single_log(x, y))(ys)

    ys = jnp.array(shapes_flat)

    def objective(x):
        logs = all_logs(jnp.array(x), ys)
        n = logs.shape[0]
        res = jnp.sum(logs * logs) / n
        grad = -2.0 * jnp.sum(logs, axis=0) / n
        return float(res), np.asarray(grad)

    opts = {"gtol": 1e-1, "maxiter": 20}
    if options:
        opts.update(options)
    result = minimize(objective, initial_mean, method="BFGS", jac=True, options=opts)
    return np.asarray(result.x)
