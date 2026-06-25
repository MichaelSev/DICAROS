"""Phylogenetic tree loading, normalisation and pruning.

Accepts Newick (``.nwk``, ``.tre``, ``.txt``) and NEXUS (``.nex``, ``.nexus``)
trees. NEXUS ``translate`` tables (integer leaf codes -> taxon names, as in the
10kTrees guenon tree) are resolved so the returned leaf labels are the taxon
names that match the landmark species column.

``dendropy`` does the parsing and pruning; the result is handed to
``hyperiax.from_newick`` as a one-line Newick string. The reconstruction
requires a rooted, fully bifurcating tree, so :func:`load_tree` resolves
polytomies and ensures a root.
"""

from __future__ import annotations

import os
import warnings

import dendropy

__all__ = ["load_tree", "newick_for_hyperiax"]


def _schema_for(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nex", ".nexus"):
        return "nexus"
    return "newick"


def load_tree(path, taxa=None, schema=None):
    """Load a tree and (optionally) prune it to ``taxa``.

    Parameters
    ----------
    path : str
        Path to a Newick or NEXUS tree file.
    taxa : set/sequence of str or None
        If given, prune the tree to the intersection of its leaves and ``taxa``,
        and report leaves/taxa that do not overlap.
    schema : {"newick", "nexus"} or None
        Force a parser; inferred from the extension when ``None``.

    Returns
    -------
    tree : dendropy.Tree
        The (possibly pruned) rooted, bifurcating tree.
    kept : list of str
        Leaf labels retained, in the tree's order.
    """
    schema = schema or _schema_for(path)
    tree = dendropy.Tree.get(
        path=path,
        schema=schema,
        preserve_underscores=True,  # keep "Genus_species" intact
    )

    leaf_labels = {lf.taxon.label for lf in tree.leaf_node_iter() if lf.taxon}

    if taxa is not None:
        taxa = set(map(str, taxa))
        common = leaf_labels & taxa
        if not common:
            raise ValueError(
                "No overlap between tree leaves and dataset species. "
                f"Example tree leaves: {sorted(leaf_labels)[:5]}; "
                f"example species: {sorted(taxa)[:5]}"
            )
        missing_from_tree = taxa - leaf_labels
        missing_from_data = leaf_labels - taxa
        if missing_from_tree:
            warnings.warn(
                f"{len(missing_from_tree)} dataset species are absent from the "
                f"tree and will be dropped: {', '.join(sorted(missing_from_tree))}",
                stacklevel=2,
            )
        if missing_from_data:
            warnings.warn(
                f"{len(missing_from_data)} tree leaves are absent from the "
                f"dataset and will be pruned: {', '.join(sorted(missing_from_data))}",
                stacklevel=2,
            )
        tree.retain_taxa_with_labels(sorted(common))

    # DICAROS fuses children pairwise -> tree must be rooted and bifurcating.
    # dendropy resolves polytomies with 0-length edges; the DICAROS branch-length
    # arithmetic divides by sums of sibling branch lengths, so we replace those
    # new zero-length edges with a very small positive length (a tiny fraction of
    # the smallest real branch length) to keep the reconstruction well defined.
    existing = {id(nd) for nd in tree.preorder_node_iter()}
    tree.resolve_polytomies(rng=None)
    if tree.seed_node is not None and len(tree.seed_node.child_nodes()) > 2:
        tree.resolve_polytomies(rng=None)

    positive = [e.length for e in tree.preorder_edge_iter()
                if e.length is not None and e.length > 0]
    eps = (min(positive) * 1e-6) if positive else 1e-6
    for nd in tree.preorder_node_iter():
        if id(nd) in existing or nd.edge is None:
            continue
        if nd.edge.length is None or nd.edge.length == 0:
            nd.edge.length = eps  # newly inserted branch from polytomy resolution

    tree.is_rooted = True

    kept = [lf.taxon.label for lf in tree.leaf_node_iter() if lf.taxon]
    return tree, kept


def newick_for_hyperiax(tree):
    """Serialise a dendropy tree to a single-line Newick string suitable for
    ``hyperiax.from_newick``."""
    nwk = tree.as_string(
        schema="newick",
        suppress_rooting=True,
        unquoted_underscores=True,
        suppress_annotations=True,
    )
    return nwk.strip().replace("\n", "")
