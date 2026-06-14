# DICAROS pipeline

Ancestral landmark-shape reconstruction along a phylogenetic tree using a
Riemannian (LDDMM-style) shape-fusion algorithm. The pipeline takes raw
specimen landmarks + a species tree, computes species mean shapes via GPA,
and uses Hamiltonian-flow landmark dynamics to recursively fuse children
back to their ancestor.

## Contents

```
DICAROS_pipline.ipynb       2D pipeline (this is the primary entry point)
DICAROS_pipline_3d.ipynb    3D variant
help_functions/
  align_functions.py        full GPA + anchor-based Procrustes alignment
  mean_estimator.py         Euclidean (non-Frechet) per-species mean shape
  mean_estimator_Frechet.py Riemannian Frechet mean (slower, geodesic-correct)
  shape_reconstruction.py   the core DICAROS fuse + tree initialisation
start_data/                 sample input: male butterfly landmarks + tree
reconstructed/              sample output of running the notebook on start_data
```

## Quick start

```bash
pip install plotly hyperiax jaxdifferentialgeometry  # or as a one-liner; see notebook cell 1
jupyter notebook DICAROS_pipline.ipynb
```

The notebook is configured to read from `./start_data` and write to
`./reconstructed`, so it works out of the box on the bundled sample.

## Pipeline outline

1. **Per-species mean shape** — `make_species_mean_common` (Euclidean) or
   `make_species_frechet_mean_common` (Riemannian); GPA aligns specimens per
   species, then aligns species means to a common frame.
2. **Tree initialisation** — `tree_initialization` attaches each species'
   mean shape to its tip in the `hyperiax` tree.
3. **Edge-length correction** — `fuse_edgelength` adjusts internal-node
   branch lengths.
4. **DICAROS reconstruction** — `fuse_DICAROS` computes the ancestral shape
   at each internal node from its two children's shapes using
   Hamiltonian-flow dynamics on the landmark manifold (jaxgeometry).
5. **Output** — reconstructed shapes for every node (tips + internals) plus
   the renamed-node tree.

## Anchor alignment vs full GPA

Set `idxs = None` in cell 4 to use full GPA on all landmarks (recommended for
most use cases). Set `idxs = [list of landmark indices]` to use the
anchor-based alignment via `align_species_landmarks_with_idx`, which fits the
Procrustes transform from the listed anchors and applies it to the full shape.
