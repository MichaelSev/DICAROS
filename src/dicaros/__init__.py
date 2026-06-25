"""dicaros -- ancestral shape reconstruction on phylogenies.

DICAROS reconstructs ancestral landmark configurations at every internal node of
a phylogeny from per-species mean shapes, using LDDMM landmark dynamics on the
tree (a bottom-up geodesic fusion of sibling shapes).

Typical use::

    import dicaros

    result = dicaros.reconstruct(
        landmarks_csv="leaves.csv",
        tree_path="tree.nwk",
        d=2,
        species_col="species",
        landmark_regex=r"^[xy]\\d+$",
        mean_method="euclidean",     # or "frechet"
        idxs=None,                    # or a list of anchor landmark indices
        output_dir="outputs/leaf",
    )
    result.node_shapes      # reconstructed shapes for every node (tips + internal)
    result.labelled_newick  # tree with internal nodes labelled

See :func:`dicaros.reconstruct` for all options.
"""

from .pipeline import reconstruct, ReconstructionResult
from .mean import species_means, SingleSpecimenWarning
from .align import align_shapes, procrustes_with_transform

__version__ = "0.1.0"

__all__ = [
    "reconstruct",
    "ReconstructionResult",
    "species_means",
    "SingleSpecimenWarning",
    "align_shapes",
    "procrustes_with_transform",
    "__version__",
]
