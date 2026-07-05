"""Python-API example: reconstruct both bundled datasets.

Mirrors the two shell scripts but through the importable API, showing how to
read the returned tables in-process instead of from disk.

    python examples/reconstruct_python.py
"""

import dicaros

# --- 2D leaves -------------------------------------------------------------
leaf = dicaros.reconstruct(
    landmarks_csv="data/leaf_2d/raw_landmarks.csv",
    tree_path="data/leaf_2d/tree_581_pruned.nwk",
    d=2,
    species_col="species",
    # coordinate columns auto-detected (override with landmark_regex=... if needed)
    mean_method="euclidean",   # or "frechet"
    idxs=None,                  # or a list of anchor landmark indices
    output_dir="outputs/leaf_2d",
    output_prefix="leaf",
)
print("leaf :", leaf.node_shapes.shape, "nodes;", len(leaf.kept_taxa), "tips")

# --- 3D guenon skulls ------------------------------------------------------
guenon = dicaros.reconstruct(
    landmarks_csv="data/guenon_3d/justLandmarks.csv",
    tree_path="data/guenon_3d/guenon_tree.nex",
    d=3,
    species_col="genus_species",
    # coordinate columns auto-detected
    mean_method="euclidean",
    output_dir="outputs/guenon_3d",
    output_prefix="guenon",
)
print("guenon:", guenon.node_shapes.shape, "nodes;", len(guenon.kept_taxa), "tips")
