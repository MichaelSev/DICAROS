"""Command-line interface for dicaros.

Example
-------
::

    dicaros \\
        --landmarks leaves.csv --tree tree.nwk --dim 2 \\
        --species-col species --landmark-regex '^[xy]\\d+$' \\
        --mean euclidean --output-dir outputs/leaf
"""

from __future__ import annotations

import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="dicaros",
        description="Ancestral shape reconstruction on a phylogeny via DICAROS.",
    )
    p.add_argument("--landmarks", required=True, help="landmark CSV path")
    p.add_argument("--tree", required=True, help="Newick or NEXUS tree path")
    p.add_argument("--dim", type=int, required=True, choices=(2, 3),
                   help="landmark dimension (2 or 3)")
    p.add_argument("--species-col", required=True, help="species/taxon column name")

    # Coordinate columns are auto-detected by default; override with at most one.
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--landmark-cols", nargs="+",
                     help="explicit coordinate column names (default: auto-detect)")
    grp.add_argument("--landmark-start", type=int,
                     help="index of the first coordinate column (default: auto-detect)")
    grp.add_argument("--landmark-regex",
                     help=r"regex for coordinate columns, e.g. '^lm\d+[xyz]$' "
                          "(default: auto-detect)")

    p.add_argument("--drop-cols", type=int, nargs="*", default=None,
                   help="coordinate columns (0-based, post-selection) to drop")
    p.add_argument("--idxs", type=int, nargs="*", default=None,
                   help="anchor landmark indices for alignment (omit for full GPA)")
    p.add_argument("--mean", choices=("euclidean", "frechet"), default="euclidean",
                   help="per-species mean estimator (default: euclidean)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply output coordinates (default: 1.0)")
    p.add_argument("--internal-prefix", default="xx_",
                   help="prefix for internal node labels (default: xx_)")
    p.add_argument("--output-dir", required=True, help="directory for outputs")
    p.add_argument("--output-prefix", default="reconstructed",
                   help="output file name prefix (default: reconstructed)")
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                   help="JAX platform (default: cpu). 'gpu' needs a CUDA build "
                        "of JAX: pip install 'dicaros[gpu]'")
    p.add_argument("--threads", type=int, default=None,
                   help="cap CPU threads (default: JAX uses all logical cores). "
                        "For a hard cap on shared machines, prefer taskset.")
    p.add_argument("--quiet", action="store_true",
                   help="suppress progress messages")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Import after arg parsing so --help is instant and JAX_PLATFORMS is set in time.
    from .pipeline import reconstruct

    result = reconstruct(
        landmarks_csv=args.landmarks,
        tree_path=args.tree,
        d=args.dim,
        species_col=args.species_col,
        landmark_cols=args.landmark_cols,
        landmark_start=args.landmark_start,
        landmark_regex=args.landmark_regex,
        drop_cols=args.drop_cols,
        idxs=args.idxs,
        mean_method=args.mean,
        scale=args.scale,
        internal_prefix=args.internal_prefix,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        device=args.device,
        threads=args.threads,
        verbose=not args.quiet,
    )

    n_nodes = len(result.node_shapes)
    n_tips = len(result.kept_taxa)
    print(f"Reconstructed {n_nodes} nodes ({n_tips} tips, {n_nodes - n_tips} internal).")
    print(f"  shapes : {result.paths.get('shapes_csv')}")
    print(f"  tree   : {result.paths.get('tree_nwk')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
