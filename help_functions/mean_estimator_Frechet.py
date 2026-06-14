# Step 1
from help_functions.align_functions import *
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from jaxgeometry.manifolds.landmarks import *
from jaxgeometry.Riemannian import metric
from jaxgeometry.dynamics import Hamiltonian
from jaxgeometry.Riemannian import Log


def _jax_optimize_frechet_mean(input_shapes, initial_mean,d,options=None):
    if options is None:
        options = {}

    chart = jnp.array([0])
    n_landmarks = input_shapes.shape[1] // d

    sigma_k = np.mean(pdist(np.array(initial_mean).reshape(-1, d), metric="euclidean"))
    sigma_k = jnp.array([sigma_k] * d)

    M = landmarks(n_landmarks, k_sigma=sigma_k * jnp.eye(d), m=d)
    metric.initialize(M)
    Hamiltonian.initialize(M)
    Log.initialize(M, f=M.Exp_Hamiltonian)

    @jax.jit
    def jit_single_log(x, y):
        return M.Log((x, [0]), (y, [0]))[0]

    @jax.jit
    def compute_all_logs(x, ys_array):
        return jax.vmap(lambda y: jit_single_log(x, y))(ys_array)

    @jax.jit
    def final_computation(logs_array):
        n = logs_array.shape[0]
        sum_logs_sq = jnp.sum(logs_array * logs_array)
        sum_logs = jnp.sum(logs_array, axis=0)
        res = sum_logs_sq / n
        grad = -2.0 * sum_logs / n
        return res, grad

    ys_jax = jnp.array(input_shapes)

    def objective_function(x):
        x_jax = jnp.array(x)
        logs = compute_all_logs(x_jax, ys_jax)
        res, grad = final_computation(logs)
        return float(res), np.array(grad)

    optimization_options = {"gtol": 1e-1, "maxiter": 20}
    optimization_options.update(options)
    res = minimize(objective_function, np.array(initial_mean), method="BFGS", jac=True, options=optimization_options)

    return (res.x, chart), res.fun


# Function
def make_species_frechet_mean_common(landmarks, species_all, idxs, d,mean_shape_save=None, output_dir=None, filename_mean=None):
    reference_list = []
    species_list = []
    count_list = []

    for species in species_all.unique():
        species_idx = np.where(species_all == species)[0]
        landmarks_subset = landmarks.iloc[species_idx]

        if idxs is not None:
            shapes, _ = align_species_landmarks_with_idx(landmarks_subset, idxs)
        else:
            shapes, _ = align_species_landmarks(landmarks_subset)

        n_specimens = shapes.shape[0]
        n_landmarks = shapes.shape[1]
        input_shapes_flat = shapes.reshape(n_specimens, d * n_landmarks)
        initial_mean = np.mean(input_shapes_flat, axis=0)

        (frechet_mean, _), _ = _jax_optimize_frechet_mean(input_shapes_flat, initial_mean,d)

        species_list.append(species)
        reference_list.append(frechet_mean)
        count_list.append(n_specimens)

    if idxs is not None:
        reference_list_output, _ = align_species_landmarks_with_idx(reference_list, idxs)
    else:
        reference_list_output, _ = align_species_landmarks(reference_list)

    reference_list_output = reference_list_output.reshape(-1, d * n_landmarks)
    reference_matrix = np.column_stack((species_list, count_list, reference_list_output))
    reference_df = pd.DataFrame(reference_matrix)

    if mean_shape_save is not None:
        reference_df.to_csv(f"{output_dir}/{filename_mean}", index=False)

    return reference_df
