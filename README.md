# dicaros

**Ancestral shape reconstruction on phylogenies via DICAROS landmark dynamics.**

`dicaros` is a documented, installable implementation of the **DICAROS** method
(*Diffeomorphic Independent Contrasts for Ancestral Reconstruction of Shapes*)
introduced by Severinsen, Akhøj, Nielsen, Sommer & Hipsley, *Systematic Biology*
(2026), [doi:10.1093/sysbio/syag019](https://doi.org/10.1093/sysbio/syag019). It
makes that paper's pipeline easy to apply to any landmark dataset.

It reconstructs the landmark configuration at *every internal node* of a
phylogeny from per-species mean shapes. Sibling shapes are fused bottom-up along
the tree using LDDMM landmark dynamics (a diffeomorphic independent contrast
between shapes) combined with Felsenstein's independent contrasts, rather than
coordinate-wise Brownian interpolation, so the reconstructed ancestors are
themselves valid shapes.

It works for **2D and 3D** landmark data, accepts **Newick or NEXUS** trees
(including 10kTrees-style `translate` tables), and outputs the reconstructed
shapes for all nodes plus the tree with internal nodes labelled.

---

## Installation

Requires Python ≥ 3.10. From the repository root, run (note the **trailing dot**
— it means "install the package in this directory"):

```bash
git clone https://github.com/MichaelSev/DICAROS.git
cd DICAROS
pip install -e .            # the "." is required!
```

That one command does everything: it installs all dependencies
([JAX](https://github.com/google/jax) [CPU build],
[hyperiax](https://github.com/ComputationalEvolutionaryMorphometry/hyperiax) ≥ 3,
[jaxgeometry](https://bitbucket.org/stefansommer/jaxgeometry), `numpy`, `pandas`,
`scipy`, `dendropy`) **and** registers the package, after which:

* the **`dicaros` command** is available in your shell (there is no standalone
  executable to find — it's created by the install; just type `dicaros --help`), and
* **`import dicaros`** works in Python.

Until you run `pip install -e .`, both `import dicaros` and the `dicaros`
command will fail — that is the cause of the usual "ModuleNotFoundError" /
"command not found" on first try. Use a fresh virtual environment (conda or
`venv`) to keep it isolated.

### Running on a GPU (optional)

Unlike PyTorch, **`pip install jax` gives a CPU-only build** — it does *not*
bundle CUDA. So a plain install runs on CPU even on a GPU box. For an NVIDIA
GPU, install the CUDA build of JAX via the `gpu` extra, then pass `--device gpu`:

```bash
pip install -e ".[gpu]"     # installs jax[cuda12]
dicaros ... --device gpu
```

CPU is the default and is recommended for exact reproducibility; GPU is roughly
6–8× faster on the bundled datasets (see `paper/` Table S1) and matches the CPU
result to floating-point tolerance. DICAROS runs comfortably on a 49 GB GPU;
only very large landmark configurations risk exceeding GPU memory, in which case
use CPU.

## Input expectations

- **Landmark CSV** — one row per specimen, containing:
  - a **species/taxon column** (any name; pass it as `--species-col`), and
  - the **landmark coordinate columns** (`n_landmarks × d` of them).

  Coordinate columns are **auto-detected** — you normally don't describe them.
  Detection recognises the usual schemes (`x1,y1,x2,y2,…`; `lm1x,lm1y,lm1z,…`;
  `1.X,1.Y,1.Z,…`; `X1,Y1,…`) and otherwise uses *all numeric columns except the
  species column*. Leading metadata columns (ids, sex, museum, …) are ignored.
- **Tree** — Newick (`.nwk`, `.tre`, `.txt`) or NEXUS (`.nex`), including
  10kTrees-style `translate` tables. It is pruned to the species present in the
  CSV, and any polytomies are resolved.
- **Dimension** — `--dim 2` or `--dim 3`.

## Quick start (command line)

No need to describe the coordinate columns — they are detected automatically:

```bash
# 2D leaves (102 landmarks, 217 species)
dicaros --landmarks data/leaf_2d/raw_landmarks.csv \
        --tree      data/leaf_2d/tree_581_pruned.nwk \
        --dim 2 --species-col species \
        --output-dir outputs/leaf_2d --output-prefix leaf

# 3D guenon skulls (155 landmarks, 22 species; NEXUS tree)
dicaros --landmarks data/guenon_3d/justLandmarks.csv \
        --tree      data/guenon_3d/guenon_tree.nex \
        --dim 3 --species-col genus_species \
        --output-dir outputs/guenon_3d --output-prefix guenon
```

The run prints one-line progress messages (loading, mean shapes, tree,
reconstruction). The reconstruction step JIT-compiles and can run for a while
with no further output — that is expected, not a hang. It is also
memory-intensive (~10 GB RAM for these datasets) — see
[Performance, memory, and shared machines](#performance-memory-and-shared-machines).

Ready-to-run wrappers for both bundled datasets live in `examples/`.

## Quick start (Python)

```python
import dicaros

result = dicaros.reconstruct(
    landmarks_csv="data/leaf_2d/raw_landmarks.csv",
    tree_path="data/leaf_2d/tree_581_pruned.nwk",
    d=2,
    species_col="species",
    # coordinate columns auto-detected; override with landmark_regex=... if needed
    mean_method="euclidean",   # or "frechet"
    idxs=None,                  # or [0, 50, 101] to anchor alignment
    output_dir="outputs/leaf_2d",
)

result.node_shapes      # DataFrame: reconstructed shapes for tips + ancestors
result.labelled_newick  # tree with internal nodes labelled xx_0, xx_1, ...
result.mean_shapes      # the per-species mean shapes that seeded the tips
```

## Options

| Option | Choices | Meaning |
|---|---|---|
| `mean_method` / `--mean` | `euclidean` (default), `frechet` | Per-species mean: ordinary Euclidean mean, or the Fréchet (Karcher) mean on the landmark manifold. |
| `idxs` / `--idxs` | `None` (default) or list of ints | Landmarks that anchor the cross-species Procrustes alignment. Omit for full GPA on all landmarks; supply indices to align on a chosen subset (e.g. homologous reference points). |
| `scale` / `--scale` | float (default `1.0`) | Multiply output coordinates (branch lengths untouched). |
| `device` / `--device` | `cpu` (default), `gpu` | JAX platform. `gpu` needs `pip install -e ".[gpu]"`. |
| `threads` / `--threads` | int (default: all cores) | Cap CPU threads (see *Performance* below). |
| `verbose` / `--quiet` | on by default / `--quiet` to silence | Progress messages. |

### Landmark-column selection (overrides)

Auto-detection (above) covers the common cases. If your columns use an unusual
scheme *mixed with* stray numeric metadata, point at them explicitly with one of:

- `--landmark-regex '^lm[0-9]+[xyz]$'` — regex on column names;
- `--landmark-start 13` — index of the first coordinate column;
- `--landmark-cols x1 y1 x2 y2 ...` — explicit list.

Use `--drop-cols` to remove specific coordinate columns after selection (e.g.
repeated landmarks).

## Performance, memory, and shared machines

> **⚠️ Resource intensity.** DICAROS reconstruction is compute- and
> memory-heavy: the LDDMM geodesic integration with the flow-differential lift
> builds a large JAX/XLA graph. Expect on the order of **~10 GB of RAM** for the
> bundled datasets (more for larger landmark sets / deeper trees, and similar on
> GPU vRAM), **all CPU cores by default**, and runtimes from minutes (small/3D)
> to ~1–2 hours (the 217-tip 2D leaf set on CPU). Make sure the machine has
> enough free memory before launching, especially on a shared server.

By default JAX/XLA sizes its CPU thread pool to **all logical cores**, so on a
shared server `dicaros` will use the whole machine. To be polite:

- `--threads N` (or `threads=N`) caps the XLA CPU pool and BLAS threads; or
- for a hard, OS-level guarantee, launch under `taskset`:
  ```bash
  OMP_NUM_THREADS=8 taskset -c 0-7 dicaros ...   # 8 cores only
  ```
  (the bundled `examples/*.sh` pin cores this way).

The DICAROS step is one-shot: when the process exits, all CPU/GPU memory is
released. JAX preallocates much of the GPU's memory up front and holds it for
the process lifetime, so on GPU it is *not* freed mid-run even after an error —
this is JAX/XLA behaviour, not something `dicaros` controls. For long-lived or
interactive GPU sessions, set `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (frees on
demand) or `XLA_PYTHON_CLIENT_PREALLOCATE=false` to reduce the footprint.

## Handling tips with a single specimen

When a species has only **one** specimen there is nothing to average, so
`dicaros` uses that lone specimen as the tip mean and emits a
`SingleSpecimenWarning` naming the affected tips — it never errors out. (The
bundled leaf dataset has 17 such tips; the guenon dataset has 1.)

## Outputs

Written to `--output-dir`:

- `<prefix>_shapes.csv` — `node_names, edges, c0, c1, …`; one row per node (tips
  **and** reconstructed ancestors), in breadth-first order. Internal nodes are
  labelled `xx_0, xx_1, …`.
- `<prefix>_tree.nwk` — the input tree with those internal-node labels, so the
  CSV rows map back onto the phylogeny.

## Reproducing the application-note results

After `pip install -e .`, run the script **from the repository root** (it uses
the bundled `data/` and writes into `outputs/` and `paper/figures/`):

```bash
cd DICAROS                                   # repo root, where data/ lives
OMP_NUM_THREADS=8 python scripts/make_results.py
```

Don't `cd scripts && python make_results.py` — it resolves paths relative to the
repo root, and it needs the package installed (`import dicaros`). This runs the
**full reconstruction of both datasets**, so it is the heaviest entry point:
~10 GB RAM and up to ~1–2 hours on CPU for the leaf set (see *Performance*
above); add `--load` to reuse existing `outputs/` and only redraw the figures.
It produces `paper/figures/<name>_results.png` (phylomorphospace + shape
overlays), `results_summary.csv` and `results_table.tex`.

## Bundled datasets

| Dataset | Dim | Landmarks | Species | Tree | Source |
|---|---|---|---|---|---|
| `data/leaf_2d`   | 2D | 102 | 217 | Newick | grass-family leaf outlines |
| `data/guenon_3d` | 3D | 155 | 22  | NEXUS  | Cardini & Elton (2017); tree from 10kTrees (Arnold et al. 2010) |

See `data/guenon_3d/source_README.md` for the guenon data provenance and
citations.

## Citation

If you use `dicaros`, please cite the method paper:

> Severinsen M.L., Akhøj M., Nielsen R., Sommer S. & Hipsley C.A. (2026).
> Diffeomorphic Independent Contrasts for Ancestral Reconstruction of Shapes.
> *Systematic Biology*. doi:10.1093/sysbio/syag019

and the application note describing this package (see `paper/`). The
reconstruction relies on `hyperiax` and `jaxgeometry`.

## License

MIT — see [LICENSE](LICENSE).
