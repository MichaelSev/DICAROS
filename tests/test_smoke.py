"""End-to-end smoke test on tiny synthetic data (CPU, seconds).

Builds a 4-tip tree and a small 2D landmark table -- including one
single-specimen species to exercise the SingleSpecimenWarning path -- then runs
the full DICAROS reconstruction and checks the outputs are well-formed.
"""

import os
import warnings

import numpy as np
import pandas as pd

import dicaros


def _make_dataset(tmpdir, d=2, n_landmarks=6):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n_landmarks, d))
    rows, species = [], []
    # 3 multi-specimen species + 1 singleton
    counts = {"sp_a": 4, "sp_b": 3, "sp_c": 2, "sp_d": 1}
    for sp, k in counts.items():
        center = base + rng.normal(scale=0.3, size=base.shape)
        for _ in range(k):
            shape = center + rng.normal(scale=0.05, size=base.shape)
            rows.append(shape.flatten())
            species.append(sp)
    cols = [f"{ax}{i+1}" for i in range(n_landmarks) for ax in ("x", "y")]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "species", species)
    csv = os.path.join(tmpdir, "landmarks.csv")
    df.to_csv(csv, index=False)

    tree = "((sp_a:1.0,sp_b:1.0):1.0,(sp_c:1.0,sp_d:1.0):1.0);"
    tree_path = os.path.join(tmpdir, "tree.nwk")
    with open(tree_path, "w") as fh:
        fh.write(tree)
    return csv, tree_path, len(cols) // d


def test_reconstruct_smoke(tmp_path):
    csv, tree_path, n_lm = _make_dataset(str(tmp_path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dicaros.reconstruct(
            landmarks_csv=csv,
            tree_path=tree_path,
            d=2,
            species_col="species",
            landmark_regex=r"^[xy]\d+$",
            mean_method="euclidean",
            idxs=None,
            output_dir=str(tmp_path / "out"),
        )

    # Singleton tip should have warned.
    assert any(isinstance(w.message, dicaros.SingleSpecimenWarning) for w in caught)

    df = result.node_shapes
    # 4 tips + 3 internal nodes = 7 nodes; 6 landmarks * 2 = 12 coord columns.
    assert len(df) == 7
    assert df.shape[1] == 2 + n_lm * 2
    assert set(["sp_a", "sp_b", "sp_c", "sp_d"]).issubset(set(df["node_names"]))
    assert any(str(n).startswith("xx_") for n in df["node_names"])  # internal labelled
    assert np.isfinite(df.iloc[:, 2:].to_numpy()).all()

    # Files written.
    assert os.path.exists(result.paths["shapes_csv"])
    assert os.path.exists(result.paths["tree_nwk"])


if __name__ == "__main__":
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_reconstruct_smoke(pathlib.Path(d))
    print("SMOKE TEST PASSED")
