"""High-level orchestration: CSV + tree in, reconstructed shapes + labelled tree out.

This module wires together the steps the original notebooks performed by hand:

1. Load the landmark table and pick the coordinate columns (:mod:`dicaros.io`).
2. Collapse specimens to one mean shape per species, Euclidean or Frechet,
   warning on single-specimen tips (:mod:`dicaros.mean`).
3. Load + prune the phylogeny to the species present, resolving NEXUS translate
   tables and polytomies (:mod:`dicaros.trees`).
4. Run the DICAROS bottom-up reconstruction (:mod:`dicaros.recon`).
5. Return / write the reconstructed shapes for every node and the tree with
   internal nodes labelled.

JAX runs on CPU by default (``JAX_PLATFORMS=cpu``): the DICAROS Hamiltonian +
flow-differential graph is large and is what the original pipeline ran on CPU.
Set ``device="gpu"`` to override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import io as _io
from . import mean as _mean
from . import trees as _trees

__all__ = ["ReconstructionResult", "reconstruct"]


@dataclass
class ReconstructionResult:
    node_shapes: object  # pandas.DataFrame: [node_names, edges, c0, c1, ...]
    labelled_newick: str
    mean_shapes: object  # pandas.DataFrame from species_means
    kept_taxa: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)


def reconstruct(
    landmarks_csv,
    tree_path,
    d,
    species_col,
    *,
    landmark_cols=None,
    landmark_start=None,
    landmark_regex=None,
    drop_cols=None,
    idxs=None,
    mean_method="euclidean",
    frechet_options=None,
    scale=1.0,
    internal_prefix="xx_",
    output_dir=None,
    output_prefix="reconstructed",
    device="cpu",
    threads=None,
    verbose=True,
):
    """Reconstruct ancestral shapes on a phylogeny.

    Parameters
    ----------
    landmarks_csv : str
        Path to the landmark CSV.
    tree_path : str
        Path to a Newick or NEXUS tree.
    d : int
        Landmark dimension (2 or 3).
    species_col : str
        Name of the species/taxon column in the CSV.
    landmark_cols, landmark_start, landmark_regex : see :func:`dicaros.io.load_landmarks`
        Exactly one selects the coordinate columns.
    drop_cols : list[int] or None
        Coordinate columns (0-based, post-selection) to drop, e.g. repeated landmarks.
    idxs : list[int] or None
        Anchor landmark indices for cross-species alignment (``None`` => full GPA).
    mean_method : {"euclidean", "frechet"}
        Per-species mean estimator.
    frechet_options : dict or None
        Options for the Frechet optimiser.
    scale : float
        Multiply output coordinates (branch lengths unaffected). Default 1.0.
    internal_prefix : str
        Prefix for internal-node labels.
    output_dir : str or None
        If given, write ``<prefix>_shapes.csv`` and ``<prefix>_tree.nwk`` there.
    device : {"cpu", "gpu"}
        JAX platform. CPU is the default and recommended.
    threads : int or None
        Cap the number of CPU threads (sets OMP/BLAS thread env vars and the
        XLA CPU intra-op pool before JAX is imported). ``None`` (default) lets
        JAX/XLA use all logical cores. On a shared machine, set this -- or, for
        a hard OS-level guarantee, launch under ``taskset -c 0-N``.
    verbose : bool
        Print one-line progress messages to stderr (default True).

    Returns
    -------
    ReconstructionResult
    """
    import sys

    def _log(msg):
        if verbose:
            print(f"[dicaros] {msg}", file=sys.stderr, flush=True)

    # Cap CPU threads *before* importing jax/numpy-heavy backends. The XLA CPU
    # pool is the dominant cost and is set here in time; BLAS env vars are
    # best-effort (numpy may already be imported) -- use taskset for a hard cap.
    if threads is not None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, str(threads))
        xla = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{xla} --xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={threads}"
        ).strip()

    # Pin the JAX platform before any jaxgeometry/hyperiax import happens.
    # JAX has no platform literally named "gpu" (only cpu/cuda/rocm/tpu); the
    # empty string is JAX's documented "auto-pick whatever accelerator is
    # present", which is correct for CUDA/ROCm/TPU. "cpu" pins to CPU.
    os.environ.setdefault("JAX_PLATFORMS", "" if device == "gpu" else "cpu")

    # 1. Load landmarks.
    _log(f"Loading landmarks from {os.path.basename(landmarks_csv)} ...")
    landmarks, species = _io.load_landmarks(
        landmarks_csv,
        species_col=species_col,
        d=d,
        landmark_cols=landmark_cols,
        landmark_start=landmark_start,
        landmark_regex=landmark_regex,
        drop_cols=drop_cols,
        verbose=verbose,
    )
    n_species = species.nunique()
    _log(f"{len(landmarks)} specimens, {n_species} species, "
         f"{landmarks.shape[1] // d} landmarks ({d}D).")

    # 2. Per-species mean shapes (warns on single-specimen tips).
    _log(f"Computing per-species mean shapes ({mean_method}) for {n_species} species ...")
    mean_df = _mean.species_means(
        landmarks,
        species,
        d=d,
        idxs=idxs,
        method=mean_method,
        frechet_options=frechet_options,
    )

    # 3. Load + prune the tree to the species we actually have means for.
    _log("Loading and pruning the tree ...")
    species_with_means = set(mean_df["species"].astype(str))
    tree_dp, kept = _trees.load_tree(tree_path, taxa=species_with_means)
    mean_df = mean_df[mean_df["species"].astype(str).isin(set(kept))].reset_index(drop=True)
    newick = _trees.newick_for_hyperiax(tree_dp)
    _log(f"Tree: {len(kept)} tips kept; reconstructing {len(kept) - 1} ancestral nodes.")

    # 4. Run DICAROS on a hyperiax 3.0 tree (heavy imports happen inside).
    _log("Running DICAROS reconstruction on "
         f"{'GPU' if device == 'gpu' else 'CPU'} -- JAX is JIT-compiling the "
         "geodesic integration; the first call can take from under a minute to "
         "several minutes with no further output. Please wait ...")
    from .recon import reconstruct_tree

    node_df, labelled_newick = reconstruct_tree(
        newick, mean_df, d=d, scale=scale, internal_prefix=internal_prefix
    )

    # 5. Write outputs.
    paths = {}
    if output_dir is not None:
        paths = _io.write_outputs(
            node_df, labelled_newick, output_dir, prefix=output_prefix
        )
        _log(f"Done. Wrote {paths['shapes_csv']} and {paths['tree_nwk']}.")
    else:
        _log("Done.")

    return ReconstructionResult(
        node_shapes=node_df,
        labelled_newick=labelled_newick,
        mean_shapes=mean_df,
        kept_taxa=kept,
        paths=paths,
    )
