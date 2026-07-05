#!/usr/bin/env bash
# Reconstruct ancestral guenon skull shapes (3D, 155 landmarks, 22 species).
#
# The tree is a NEXUS file with a translate table (10kTrees); dicaros resolves
# it automatically.  One species (Cercopithecus_solatus) has a single
# specimen and triggers a SingleSpecimenWarning.
#
# Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."

OMP_NUM_THREADS=8 taskset -c 0-31 dicaros \
    --landmarks    data/guenon_3d/justLandmarks.csv \
    --tree         data/guenon_3d/guenon_tree.nex \
    --dim          3 \
    --species-col  genus_species \
    --mean         euclidean \
    --output-dir   outputs/guenon_3d \
    --output-prefix guenon
