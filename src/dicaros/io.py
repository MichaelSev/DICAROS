"""Input/output: reading landmark tables and writing reconstruction outputs.

Expected input
--------------
A landmark table is a CSV with **one row per specimen** and:

* one **species/taxon column** (any name; given by ``species_col``), and
* ``n_landmarks * d`` numeric **coordinate columns**.

By default the coordinate columns are **detected automatically** -- you do not
need to describe them. Detection recognises the common geometric-morphometrics
naming schemes (``x1,y1,x2,y2,...``; ``lm1x,lm1y,lm1z,...``; ``1.X,1.Y,1.Z,...``;
``X1,Y1,...``) and otherwise falls back to "every numeric column other than the
species column". Leading metadata columns (ids, sex, museum, ...) are ignored as
long as they are non-numeric or excluded by the chosen naming scheme.

You only need to point at the columns explicitly when auto-detection cannot
(unusual naming mixed with stray numeric metadata). In that case give exactly
one of:

* ``landmark_cols``  -- explicit list of coordinate column names;
* ``landmark_start`` -- integer index of the first coordinate column;
* ``landmark_regex`` -- regex matching coordinate column names.

``drop_cols`` removes specific coordinate columns *after* selection (e.g.
repeated landmarks), by 0-based index into the selected coordinate block.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

__all__ = ["load_landmarks", "write_outputs"]

# Common landmark-column naming schemes, tried in order during auto-detection.
# Each must capture a per-landmark axis so the count is a multiple of d.
_LANDMARK_PATTERNS = (
    r"^[xyz]\d+$",        # x1,y1,z1, ...      (e.g. 2D leaf data)
    r"^lm\d+[xyz]$",      # lm1x,lm1y,lm1z, ... (e.g. 3D guenon data)
    r"^\d+\.[XYZ]$",      # 1.X,1.Y,1.Z, ...
    r"^[XYZ]\d+$",        # X1,Y1,Z1, ...
    r"^[xyz]_?\d+$",      # x_1,y_1, ...
)


def _autodetect_landmark_cols(df, species_col, d):
    """Return the coordinate column names, inferred from the table.

    Tries the known naming schemes (picking one whose match count is a positive
    multiple of ``d``); otherwise uses every numeric column except the species
    column. Raises with a clear message if nothing usable is found.
    """
    candidates = [c for c in df.columns if c != species_col]

    for pat in _LANDMARK_PATTERNS:
        rx = re.compile(pat)
        cols = [c for c in candidates if rx.match(str(c))]
        if cols and len(cols) % d == 0:
            return cols, f"naming scheme /{pat}/"

    # Fallback: numeric columns only (drops non-numeric metadata automatically).
    numeric = [
        c for c in candidates
        if pd.to_numeric(df[c], errors="coerce").notna().all()
    ]
    if numeric and len(numeric) % d == 0:
        return numeric, "all numeric columns"

    raise ValueError(
        "Could not auto-detect landmark coordinate columns "
        f"(d={d}). Found {len(numeric)} numeric non-species columns, "
        f"which is not a multiple of {d}. Specify the columns explicitly with "
        "landmark_cols=, landmark_start=, or landmark_regex= "
        "(CLI: --landmark-cols / --landmark-start / --landmark-regex)."
    )


def load_landmarks(
    csv_path,
    species_col,
    d,
    landmark_cols=None,
    landmark_start=None,
    landmark_regex=None,
    drop_cols=None,
    verbose=False,
):
    """Read a landmark table.

    With no explicit selector, the coordinate columns are auto-detected (see the
    module docstring). Pass at most one of ``landmark_cols`` / ``landmark_start``
    / ``landmark_regex`` to override.

    Returns
    -------
    landmarks : pandas.DataFrame
        Coordinate columns only, ``(n_specimens, n_landmarks*d)``.
    species : pandas.Series
        Species label per row.
    """
    df = pd.read_csv(csv_path)

    if species_col not in df.columns:
        raise KeyError(
            f"species column {species_col!r} not found. Available: {list(df.columns)[:15]}..."
        )
    species = df[species_col].astype(str)

    provided = [x is not None for x in (landmark_cols, landmark_start, landmark_regex)]
    if sum(provided) > 1:
        raise ValueError(
            "Provide at most one of landmark_cols / landmark_start / landmark_regex"
        )

    if landmark_cols is not None:
        cols = list(landmark_cols)
    elif landmark_start is not None:
        cols = list(df.columns[landmark_start:])
    elif landmark_regex is not None:
        pat = re.compile(landmark_regex)
        cols = [c for c in df.columns if pat.match(str(c))]
        if not cols:
            raise ValueError(f"landmark_regex {landmark_regex!r} matched no columns")
    else:
        cols, how = _autodetect_landmark_cols(df, species_col, d)
        if verbose:
            print(
                f"[dicaros] auto-detected {len(cols)} coordinate columns "
                f"({len(cols) // d} landmarks, {d}D) via {how}",
                flush=True,
            )

    landmarks = df[cols].apply(pd.to_numeric, errors="coerce")

    if drop_cols:
        keep = [c for i, c in enumerate(landmarks.columns) if i not in set(drop_cols)]
        landmarks = landmarks[keep]

    if landmarks.shape[1] % d != 0:
        raise ValueError(
            f"selected {landmarks.shape[1]} coordinate columns, not divisible by d={d}. "
            "Check landmark column selection / drop_cols."
        )
    if landmarks.isna().any().any():
        n_bad = int(landmarks.isna().any(axis=1).sum())
        raise ValueError(
            f"{n_bad} specimens have non-numeric/missing landmark values; "
            "clean or impute them before reconstruction."
        )

    return landmarks.reset_index(drop=True), species.reset_index(drop=True)


def write_outputs(node_df, newick_str, output_dir, prefix="reconstructed"):
    """Write the reconstructed-shapes CSV and the labelled tree.

    Returns a dict of the two paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    shapes_path = os.path.join(output_dir, f"{prefix}_shapes.csv")
    tree_path = os.path.join(output_dir, f"{prefix}_tree.nwk")

    node_df.to_csv(shapes_path, index=False)
    with open(tree_path, "w") as fh:
        fh.write(newick_str.strip() + "\n")

    return {"shapes_csv": shapes_path, "tree_nwk": tree_path}
