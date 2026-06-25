"""Per-species mean shapes.

For every species (= tip of the phylogeny) the specimens are Procrustes-aligned
and reduced to a single mean shape, which becomes the observed tip value for the
DICAROS reconstruction. Two estimators are available:

* ``"euclidean"`` -- the ordinary (arithmetic) mean of the aligned shapes.
* ``"frechet"``  -- the Frechet (Karcher) mean on the LDDMM landmark manifold,
  computed with ``jaxgeometry``. Slower, but consistent with the Riemannian
  geometry used in the reconstruction step.

**Single-specimen tips.** When a species is represented by only one specimen,
there is nothing to average: the lone specimen *is* the mean. The original code
would still run the alignment machinery on a 1-element set; here we short-circuit
that, use the specimen directly, and emit a clear ``UserWarning`` naming the tip
so the user knows the tip rests on a single observation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .align import align_shapes, to_flat

__all__ = ["species_means", "SingleSpecimenWarning"]


class SingleSpecimenWarning(UserWarning):
    """Raised when a species/tip is represented by exactly one specimen."""


def _per_species_mean_euclidean(shapes_flat, d):
    """Within-species Euclidean mean shape (flat coords) after GPA."""
    aligned, _ = align_shapes(shapes_flat, d, idxs=None)
    flat = to_flat(aligned)
    return np.mean(flat, axis=0)


def _per_species_mean_frechet(shapes_flat, d, options=None):
    """Within-species Frechet mean on the landmark manifold (flat coords)."""
    # Imported lazily: only needed for the Frechet path, and pulls in jax.
    from .frechet import frechet_mean

    aligned, _ = align_shapes(shapes_flat, d, idxs=None)
    flat = to_flat(aligned)
    initial = np.mean(flat, axis=0)
    return frechet_mean(flat, initial, d, options=options)


def species_means(
    landmarks,
    species,
    d,
    idxs=None,
    method="euclidean",
    frechet_options=None,
    output_csv=None,
):
    """Compute one mean shape per species and align them into a common frame.

    Parameters
    ----------
    landmarks : pandas.DataFrame or ndarray
        ``(n_specimens, n_landmarks*d)`` landmark coordinates (no metadata cols).
    species : sequence
        Length-``n_specimens`` species label per row.
    d : int
        Landmark dimension (2 or 3).
    idxs : sequence of int or None
        Anchor landmarks for the *cross-species* alignment of the mean shapes.
        ``None`` => full GPA. (Within-species alignment always uses full GPA.)
    method : {"euclidean", "frechet"}
        Mean estimator.
    frechet_options : dict or None
        Extra options forwarded to the Frechet optimiser (e.g. ``maxiter``).
    output_csv : str or None
        If given, write the resulting table to this path.

    Returns
    -------
    pandas.DataFrame
        Columns ``["species", "count", c0, c1, ...]`` -- one row per species,
        coordinates aligned into a single common frame.
    """
    method = method.lower()
    if method not in ("euclidean", "frechet"):
        raise ValueError(f"method must be 'euclidean' or 'frechet', got {method!r}")

    landmarks = (
        landmarks.to_numpy(dtype=float)
        if isinstance(landmarks, pd.DataFrame)
        else np.asarray(landmarks, dtype=float)
    )
    species = pd.Series(np.asarray(species)).reset_index(drop=True)

    species_list, mean_list, count_list = [], [], []
    singletons = []

    for sp in pd.unique(species):
        rows = np.where(species.values == sp)[0]
        subset = landmarks[rows]
        n = subset.shape[0]

        if n == 1:
            # One observation: it IS the mean. Standardise it on its own so it
            # lands in the same Procrustes frame as multi-specimen means.
            singletons.append(str(sp))
            mean_flat = to_flat(align_shapes(subset, d, idxs=None)[0])[0]
        elif method == "euclidean":
            mean_flat = _per_species_mean_euclidean(subset, d)
        else:
            mean_flat = _per_species_mean_frechet(subset, d, options=frechet_options)

        species_list.append(sp)
        mean_list.append(mean_flat)
        count_list.append(n)

    if singletons:
        warnings.warn(
            f"{len(singletons)} species are represented by a single specimen; "
            f"the lone specimen is used as the tip mean (no within-species "
            f"averaging possible): {', '.join(singletons)}",
            SingleSpecimenWarning,
            stacklevel=2,
        )

    # Cross-species alignment: put every species mean into ONE common frame.
    aligned, _ = align_shapes(np.vstack(mean_list), d, idxs=idxs)
    aligned_flat = to_flat(aligned)

    out = pd.DataFrame(aligned_flat)
    out.insert(0, "count", count_list)
    out.insert(0, "species", species_list)

    if output_csv is not None:
        out.to_csv(output_csv, index=False)
    return out
