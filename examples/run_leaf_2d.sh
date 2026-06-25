#!/usr/bin/env bash
# Reconstruct ancestral leaf shapes (2D, 102 landmarks, 217 species).
#
# The leaf tree contains all 217 species; 17 of them have a single specimen,
# so dicaros will print a SingleSpecimenWarning for those tips and use the lone
# specimen as the tip mean.
#
# Run from the repository root.  Pin CPU cores to stay polite on shared machines.
set -euo pipefail
cd "$(dirname "$0")/.."

OMP_NUM_THREADS=8 taskset -c 0-31 dicaros \
    --landmarks    data/leaf_2d/raw_landmarks.csv \
    --tree         data/leaf_2d/tree_581_pruned.nwk \
    --dim          2 \
    --species-col  species \
    --mean         euclidean \
    --output-dir   outputs/leaf_2d \
    --output-prefix leaf

# Options you can flip:
#   --mean frechet            use the Frechet (manifold) mean instead of Euclidean
#   --idxs 0 50 101           anchor cross-species alignment on landmarks 0,50,101
#   --device gpu              run JAX on GPU (CPU is default and recommended)
