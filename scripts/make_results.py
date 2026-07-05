"""Reconstruct both bundled datasets, then summarise and visualise the results.

Outputs (under ``paper/figures/`` and ``outputs/``):

* ``<name>_phylomorphospace.png`` -- PCA of every reconstructed node shape
  (tips + ancestors) with the tree edges drawn in shape space.
* ``<name>_shapes.png``           -- the reconstructed root shape overlaid on
  the tip mean shapes (2D directly; 3D projected onto its first two PCs).
* ``results_summary.csv`` / ``results_table.tex`` -- one row per dataset.

Usage
-----
    python scripts/make_results.py            # run reconstruction + figures
    python scripts/make_results.py --load     # reuse outputs/*/*_shapes.csv + tree

Run from the repository root. JAX stays on CPU; pin cores with taskset if shared:
    OMP_NUM_THREADS=8 taskset -c 0-31 python scripts/make_results.py
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dicaros  # noqa: E402
from dicaros.mean import SingleSpecimenWarning  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "paper", "figures")
OUT_DIR = os.path.join(ROOT, "outputs")

DATASETS = [
    dict(
        name="leaf_2d", label="Leaves (2D)", d=2,
        landmarks_csv="data/leaf_2d/raw_landmarks.csv",
        tree_path="data/leaf_2d/tree_581_pruned.nwk",
        species_col="species", landmark_regex=r"^[xy]\d+$",
    ),
    dict(
        name="guenon_3d", label="Guenon skulls (3D)", d=3,
        landmarks_csv="data/guenon_3d/justLandmarks.csv",
        tree_path="data/guenon_3d/guenon_tree.nex",
        species_col="genus_species", landmark_regex=r"^lm\d+[xyz]$",
    ),
]


def run_or_load(cfg, load):
    """Return (node_df, labelled_newick, n_singletons, runtime_s)."""
    out_sub = os.path.join(OUT_DIR, cfg["name"])
    shapes_csv = os.path.join(out_sub, f"{cfg['name'].split('_')[0]}_shapes.csv")
    tree_nwk = os.path.join(out_sub, f"{cfg['name'].split('_')[0]}_tree.nwk")

    if load and os.path.exists(shapes_csv) and os.path.exists(tree_nwk):
        node_df = pd.read_csv(shapes_csv)
        newick = open(tree_nwk).read().strip()
        return node_df, newick, None, None

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dicaros.reconstruct(
            landmarks_csv=os.path.join(ROOT, cfg["landmarks_csv"]),
            tree_path=os.path.join(ROOT, cfg["tree_path"]),
            d=cfg["d"],
            species_col=cfg["species_col"],
            landmark_regex=cfg["landmark_regex"],
            mean_method="euclidean",
            output_dir=out_sub,
            output_prefix=cfg["name"].split("_")[0],
        )
    runtime = time.time() - t0
    n_singletons = sum(
        len(str(w.message).split(": ")[-1].split(", "))
        for w in caught if isinstance(w.message, SingleSpecimenWarning)
    )
    return result.node_shapes, result.labelled_newick, n_singletons, runtime


def parent_child_pairs(newick):
    """List of (parent_label, child_label) from a labelled Newick string."""
    import dendropy

    tree = dendropy.Tree.get(data=newick, schema="newick", preserve_underscores=True)
    pairs = []
    for node in tree.preorder_node_iter():
        plabel = node.taxon.label if node.taxon else node.label
        for child in node.child_nodes():
            clabel = child.taxon.label if child.taxon else child.label
            if plabel and clabel:
                pairs.append((plabel, clabel))
    return pairs


def phylomorphospace(cfg, node_df, newick, ax):
    """PCA of all node shapes + tree edges drawn in PC1-PC2 space."""
    names = node_df["node_names"].astype(str).to_numpy()
    coords = node_df.iloc[:, 2:].to_numpy(dtype=float)
    X = coords - coords.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = X @ Vt[:2].T
    var = (S**2 / np.sum(S**2) * 100)[:2]
    pos = {n: pcs[i] for i, n in enumerate(names)}

    is_internal = np.array([n.startswith("xx_") for n in names])
    for p, c in parent_child_pairs(newick):
        if p in pos and c in pos:
            ax.plot([pos[p][0], pos[c][0]], [pos[p][1], pos[c][1]],
                    "-", color="0.7", lw=0.5, zorder=1)
    ax.scatter(pcs[~is_internal, 0], pcs[~is_internal, 1], s=14, c="#1f77b4",
               label="tips (species means)", zorder=3)
    ax.scatter(pcs[is_internal, 0], pcs[is_internal, 1], s=14, c="#d62728",
               marker="^", label="reconstructed ancestors", zorder=3)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title(f"{cfg['label']} — phylomorphospace")
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", "datalim")


def shape_overlay(cfg, node_df, ax):
    """Overlay reconstructed root shape on tip mean shapes."""
    d = cfg["d"]
    names = node_df["node_names"].astype(str).to_numpy()
    coords = node_df.iloc[:, 2:].to_numpy(dtype=float)

    # Project to 2D for plotting (3D -> first 2 PCs of the pooled landmark cloud).
    def to_xy(flat):
        pts = flat.reshape(-1, d)
        if d == 2:
            return pts
        return pts @ proj

    if d == 3:
        allpts = coords.reshape(-1, 3)
        allpts = allpts - allpts.mean(0)
        proj = np.linalg.svd(allpts, full_matrices=False)[2][:2].T

    is_internal = np.array([n.startswith("xx_") for n in names])
    for i in np.where(~is_internal)[0]:
        xy = to_xy(coords[i])
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color="0.7", alpha=0.5, zorder=1)
    # Root = first internal node in BFS order (xx_0).
    root_idx = np.where(names == "xx_0")[0]
    if len(root_idx):
        xy = to_xy(coords[root_idx[0]])
        ax.scatter(xy[:, 0], xy[:, 1], s=20, color="#d62728", zorder=3,
                   label="reconstructed root")
    ax.set_title(f"{cfg['label']} — landmark shapes")
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x" if d == 2 else "shape PC1")
    ax.set_ylabel("y" if d == 2 else "shape PC2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true",
                    help="reuse existing outputs instead of reconstructing")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []

    # One combined figure for the application note (Bioinformatics allows ~one
    # figure): a row per dataset, each with its phylomorphospace + shape overlay.
    n = len(DATASETS)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.4 * n))
    if n == 1:
        axes = axes[None, :]
    panel = iter("abcdefgh")

    for i, cfg in enumerate(DATASETS):
        print(f"== {cfg['label']} ==")
        node_df, newick, n_singletons, runtime = run_or_load(cfg, args.load)
        names = node_df["node_names"].astype(str)
        n_internal = int(names.str.startswith("xx_").sum())
        n_tips = len(names) - n_internal
        n_landmarks = (node_df.shape[1] - 2) // cfg["d"]

        phylomorphospace(cfg, node_df, newick, axes[i, 0])
        shape_overlay(cfg, node_df, axes[i, 1])
        for ax in axes[i]:
            ax.set_title(f"({next(panel)}) " + ax.get_title(), fontsize=10)

        rows.append(dict(
            dataset=cfg["label"], dim=cfg["d"], tips=n_tips,
            ancestors=n_internal, landmarks=n_landmarks,
            singletons=("" if n_singletons is None else n_singletons),
            runtime_s=("" if runtime is None else round(runtime, 1)),
        ))

    fig.tight_layout()
    combined_png = os.path.join(FIG_DIR, "dicaros_results.png")
    fig.savefig(combined_png, dpi=200)
    plt.close(fig)
    print(f"   wrote {combined_png}")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(FIG_DIR, "results_summary.csv"), index=False)
    print("\n", summary.to_string(index=False))

    # LaTeX table fragment for the application note.
    tex = summary.rename(columns={
        "dataset": "Dataset", "dim": "Dim.", "tips": "Tips",
        "ancestors": "Ancestors", "landmarks": "Landmarks",
        "singletons": "Singleton tips", "runtime_s": "Runtime (s)",
    }).to_latex(index=False, escape=True)
    with open(os.path.join(FIG_DIR, "results_table.tex"), "w") as fh:
        fh.write(tex)
    print("wrote results_table.tex")


if __name__ == "__main__":
    main()
