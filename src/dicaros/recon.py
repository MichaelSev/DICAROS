"""DICAROS ancestral-shape reconstruction on a phylogeny (hyperiax >= 3.0).

The reconstruction is a single bottom-up (leaves -> root) pass over the tree.
At each internal node the two child shapes are fused on the LDDMM landmark
manifold: the shorter child branch defines a base point, the geodesic ``Log``
toward the other child gives a momentum (contrast), and the Hamiltonian
geodesic is integrated to a point along the branch proportional to the relative
branch lengths. The fused estimate becomes the node's shape and the pass
continues toward the root.

Ported from Michael Lind Severinsen's ``help_functions/shape_reconstruction.py``;
the numerical logic is unchanged. The traversal now uses the hyperiax 3.0 sweep
API (``hyperiax.up`` + segment-based ``Children`` views) instead of the pre-3.0
``UpLambda``/``OrderedExecutor``. Because DICAROS fuses *both* siblings together
(it is not a per-child reduction), the sweep reshapes the flat children block
into per-parent pairs ``(P, 2, *trailing)`` and ``vmap``s the original
pairwise fuse over parents. Trees are required to be strictly bifurcating, so
every parent contributes exactly two children to that block.

Everything imports ``jax`` / ``jaxgeometry`` / ``hyperiax`` lazily inside
functions so the rest of the package (IO, alignment, Euclidean means) works
without them.

DICAROS = Diffeomorphic Independent Contrasts for Ancestral Reconstruction of
Shapes (Severinsen et al., Systematic Biology 2026; doi:10.1093/sysbio/syag019).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "find_sigma",
    "build_tree",
    "run_reconstruction",
    "reconstruct_tree",
    "extract_node_table",
    "relabel_internal_names",
]


def find_sigma(landmark_row, d):
    """Mean nearest-neighbour distance among the landmarks of one shape.

    Used to set the LDDMM kernel width so the kernel matches the data scale.
    """
    arr = np.asarray(landmark_row, dtype=float).reshape(-1, d)
    distances = np.linalg.norm(arr[:, None] - arr, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.mean(np.min(distances, axis=1)))


# --- hyperiax 3.0 sweeps ------------------------------------------------------

def _build_sweeps(d):
    """Construct the edge-length and DICAROS up-sweeps for dimension ``d``.

    Heavy deps (jax, jaxgeometry, hyperiax) are imported here so importing
    :mod:`dicaros.recon` stays cheap.
    """
    import jax
    import jax.numpy as jnp
    from hyperiax import up
    from jaxgeometry.manifolds.landmarks import landmarks
    from jaxgeometry.Riemannian import metric, Log
    from jaxgeometry.dynamics import Hamiltonian, flow_differential
    from jaxgeometry.utils import dts

    # --- per-parent pairwise fuse (identical math to the pre-3.0 version) ---
    def _dicaros(child_coords1, child_coords2, kernel_sigma, parent_index, dim):
        M = landmarks(
            jnp.shape(child_coords1)[0] // dim,
            k_sigma=kernel_sigma * jnp.eye(dim),
            m=dim,
        )
        metric.initialize(M)
        q = M.coords(jnp.array(child_coords1))
        v = (jnp.array(child_coords2), [0])
        Hamiltonian.initialize(M)
        Log.initialize(M, f=M.Exp_Hamiltonian)
        p = M.Log(q, v)[0]
        (_, qps, _) = M.Hamiltonian_dynamics(q, p, dts(n_steps=100))
        flow_differential.initialize(M)
        _, dphis, _ = M.flow_differential(qps, dts())
        return qps[:, 0][parent_index], p, dphis[parent_index]

    def _pair_fuse(child_coords_pair, child_edge_length_pair, kernel_sigma_pair):
        """Fuse one parent's two children. Shapes per parent:
        child_coords_pair (2, l_dim), child_edge_length_pair (2, 1),
        kernel_sigma_pair (2, d). Returns (value, p, phi)."""
        dim = kernel_sigma_pair.shape[1]
        edge_sum = child_edge_length_pair.sum()

        def from_left(_):
            pi = jnp.floor(
                child_edge_length_pair[0] / edge_sum * 100 - 1
            ).astype(int)[0]
            return _dicaros(child_coords_pair[0, :], child_coords_pair[1, :],
                            kernel_sigma_pair[0, :], pi, dim)

        def from_right(_):
            pi = jnp.floor(
                child_edge_length_pair[1] / edge_sum * 100 - 1
            ).astype(int)[0]
            return _dicaros(child_coords_pair[1, :], child_coords_pair[0, :],
                            kernel_sigma_pair[0, :], pi, dim)

        return jax.lax.cond(
            (child_edge_length_pair[0] < child_edge_length_pair[1])[0],
            from_left, from_right, operand=None,
        )

    @up(reads_children=("value", "sigma", "edge_length"),
        writes=("value", "p_adj", "phi"))
    def dicaros_sweep(node, children, params):
        # Flat children block for this level: (2P, *trailing), grouped by parent
        # (BFS keeps siblings contiguous; the tree is binary so each parent has
        # exactly two children). Reshape to per-parent pairs and vmap the fuse.
        cv = children.value.flat                       # (2P, l_dim)
        ce = children.edge_length.flat                 # (2P,)
        cs = children.sigma.flat                       # (2P, d)
        P = cv.shape[0] // 2
        cv = cv.reshape(P, 2, cv.shape[-1])
        ce = ce.reshape(P, 2, 1)
        cs = cs.reshape(P, 2, cs.shape[-1])
        qps, p, dphis = jax.vmap(_pair_fuse, in_axes=(0, 0, 0))(cv, ce, cs)
        p_adj = p / ce.sum(axis=1)                      # divide by sum of child edges
        return {"value": qps, "p_adj": p_adj, "phi": dphis}

    @up(reads=("edge_length",), reads_children=("edge_length",),
        writes=("edge_length",))
    def edge_sweep(node, children, params):
        # parent_edge += prod(child_edges) / sum(child_edges)
        prod = children.edge_length.prod(0)
        s = children.edge_length.sum(0)
        return {"edge_length": node.edge_length + prod / s}

    return edge_sweep, dicaros_sweep


# --- tree setup ---------------------------------------------------------------

def build_tree(newick, mean_df, d):
    """Read the Newick string into a hyperiax Tree, attach DICAROS fields, and
    seed the leaves with the per-species mean shapes.

    Parameters
    ----------
    newick : str
        Newick string (one line); branch lengths become ``edge_length``.
    mean_df : pandas.DataFrame
        Columns ``[species, count, c0, c1, ...]`` (output of
        :func:`dicaros.mean.species_means`).
    d : int
        Landmark dimension.

    Returns
    -------
    tree : hyperiax Tree
    """
    import jax.numpy as jnp
    from hyperiax import from_newick

    mean_species = mean_df.iloc[:, 0].to_numpy()
    mean_shapes = mean_df.iloc[:, 2:]
    l_dim = int(mean_shapes.shape[1])
    n_lm = l_dim // d

    sigma = float(np.mean([find_sigma(row, d) for _, row in mean_shapes.iterrows()]))

    tree = from_newick(
        newick,
        schema={
            "value": (l_dim,),
            "sigma": (d,),
            "p_adj": (l_dim,),
            "phi": (n_lm, d, d),
        },
    )
    topo = tree.topology

    if not (topo.max_degree == 2 and topo.equal_degree):
        raise ValueError(
            "DICAROS requires a strictly bifurcating tree; got max_degree="
            f"{topo.max_degree}, equal_degree={topo.equal_degree}. Resolve "
            "polytomies before reconstruction (dicaros.trees.load_tree does this)."
        )

    n = topo.size
    tree = tree.set(sigma=jnp.tile(jnp.array([sigma] * d), (n, 1)))

    # Seed leaves (in ascending node order) with the matching species means.
    leaf_names = [topo.names[i] for i in range(n) if bool(topo.is_leaf[i])]
    idx = np.array([np.where(mean_species == name)[0][0] for name in leaf_names])
    reordered = jnp.asarray(mean_shapes.iloc[idx, :].to_numpy(), dtype=float)
    tree = tree.at[np.asarray(topo.is_leaf)].set(value=reordered)
    return tree


def run_reconstruction(tree, d):
    """Run the two bottom-up sweeps (edge-length fusion, then DICAROS)."""
    edge_sweep, dicaros_sweep = _build_sweeps(d)
    tree = edge_sweep(tree)
    tree = dicaros_sweep(tree)
    return tree


def relabel_internal_names(topo, prefix="xx_"):
    """Return a names tuple with every internal node renamed ``prefix + i``
    (in node/BFS order), leaf names kept."""
    names = list(topo.names) if topo.names is not None else [""] * topo.size
    i = 0
    for k in range(topo.size):
        if not bool(topo.is_leaf[k]):
            names[k] = f"{prefix}{i}"
            i += 1
    return tuple(names)


def extract_node_table(tree, d, scale=1.0):
    """Collect per-node reconstructed shapes into a DataFrame.

    Returns columns ``[node_names, edges, c0, c1, ...]`` for *all* nodes
    (tips + internal), in node (BFS) order. Internal nodes are labelled
    ``xx_0, xx_1, ...``. ``scale`` multiplies coordinates only.
    """
    import pandas as pd

    topo = tree.topology
    names = relabel_internal_names(topo)
    shapes = np.asarray(tree["value"]) * float(scale)
    edges = np.asarray(tree["edge_length"])

    df = pd.DataFrame(shapes)
    df.insert(0, "node_names", list(names))
    df.insert(1, "edges", edges)
    return df, names


def reconstruct_tree(newick, mean_df, d, scale=1.0, internal_prefix="xx_"):
    """End-to-end reconstruction from a Newick string + species means.

    Returns
    -------
    node_df : pandas.DataFrame
        Reconstructed shapes for all nodes (tips + internal).
    labelled_newick : str
        The tree with internal nodes labelled ``xx_i``.
    """
    from hyperiax import Topology, Tree, to_newick

    tree = build_tree(newick, mean_df, d)
    tree = run_reconstruction(tree, d)

    node_df, names = extract_node_table(tree, d, scale=scale)
    if internal_prefix != "xx_":
        names = relabel_internal_names(tree.topology, prefix=internal_prefix)
        node_df["node_names"] = list(names)

    # Rebuild the topology with the new (labelled) names for Newick output.
    labelled_topo = Topology.from_parents(np.asarray(tree.topology.parents), names=names)
    labelled_tree = Tree(topology=labelled_topo, schema=tree.schema, data=tree.data)
    labelled_newick = to_newick(labelled_tree)
    return node_df, labelled_newick
