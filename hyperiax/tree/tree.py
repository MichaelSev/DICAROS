from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Type, Dict, List, Iterator
from .childrenlist import ChildList


@dataclass(repr=False)
class TreeNode:
    """
    The data nodes within a tree

    """
    parent : TreeNode = None
    data : Dict = field(default_factory=dict)
    children : List[TreeNode] = None
    name : str = None
   

    def __repr__(self):
        return f'TreeNode({self.data}) with {len(self.children)} children' if self.children else f'TreeNode({self.data}) with no children'

    def __getitem__(self, arg):
        return self.data.__getitem__(arg)

    def __setitem__(self, key: str, arg):
        self.data.__setitem__(key, arg)

    def __delitem__(self, key: str):
        self.data.__delitem__(key)

    def add_child(self, child: Type[TreeNode]) -> Type[TreeNode]:
        """ 
        Add an individual child to the node

        :param child: child node to be added
        :return: the child node

        """
        child.parent = self
        self.children._add_child(child)
        return child



class HypTree:
    """
    The tree class that wraps behavious around a set of nodes.

    The set of nodes is given via the `root` node, and can be iterated conveniently using the utility in this class.

    """
    def __init__(self, root : TreeNode) -> None:
        self.root = root
        self.order = None
        
    def __repr__(self) -> str:
        return f'HypTree with {len(list(self.iter_levels()))} levels and {len(self)} nodes'

    def __len__(self) -> int:
        return len(list(self.iter_bfs()))
    
    def __getitem__(self, arg):
        for node in self.iter_bfs():
            yield node.data.__getitem__(arg)

    def __setitem__(self, key, arg):
        if '__iter__' in getattr(arg, '__dict__', dict()):
            for node, value in zip(self.iter_bfs(), arg):
                node.data.__setitem__(key, value)
        else:
            for node in self.iter_bfs():
                node.data.__setitem__(key, arg)
    
    def copy(self) -> Type[HypTree]:
        """
        Returns a copy of the tree

        :return: a copy of the tree

        """
        return copy.deepcopy(self)

    def iter_leaves(self) -> Iterator[TreeNode]:
        """
        Iterate over all of the leaves in the tree

        """

        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            else:
                yield current

    def iter_bfs(self) -> Iterator[TreeNode]:
        """
        Iterate over all of the nodes in a breadth first manner

        """
        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            if current.children:
                queue.extend(current.children)
            yield current

    def iter_dfs(self) -> Iterator[TreeNode]:
        """
        Iterate over all of the nodes in a depth-first manner.

        """
        stack = deque([self.root])

        while stack:
            current = stack.pop()
            if current.children:
                stack.extend(current.children)
            yield current

    def iter_levels(self) -> Iterator[List[TreeNode]]:
        """
        Iterate over each level in the tree

        """
        queue = deque()
        buffer_queue = deque([self.root])
        while queue or buffer_queue:
            if not queue: # if queue is empty, flush the buffer and yield a level
                queue = buffer_queue
                yield list(buffer_queue) # to not pass the reference
                buffer_queue = deque()

            if children := queue.popleft().children:
                buffer_queue.extend(children)

    def iter_leaves_dfs(self) -> Iterator[TreeNode]:
        """
        Iterates over the leaves in the tree using depth-first search.

        ??? Duplicate of iter_leaves
        """
        stack = deque([self.root])

        while stack:
            current = stack.pop()
            if current.children:
                stack.extend(reversed(current.children))
            else:
                yield current

    def plot_tree_2d(self, ax=None, selector=None):
        from .plot_utils import plot_tree_2d_
        plot_tree_2d_(self, ax, selector)

    def plot_tree(self, ax=None,inc_names=False):
        tree = self.copy()
        from .plot_utils import plot_tree_
        plot_tree_(tree, ax,inc_names)

    def plot_tree_shape(self, ax=None,inc_names=False,shape="landmarks"):
        tree = self.copy()
        from .plot_utils import plot_tree_shape_
        plot_tree_shape_(tree, ax,inc_names,shape)

    def plot_tree_text(self):
        from .plot_utils import HypTreeFormatter
        formatter = HypTreeFormatter(self)
        formatter.print_tree()


    def to_newick(self):
        # Recursive function to convert tree to Newick string
        def to_newick(node:TreeNode):
            """Recursively generates a Newick string from a JaxNode tree."""
            if node is None:
                return ''
            
            parts = []
            if node.children is not None:
                for child in node.children:

                    part = to_newick(child)
                    parts.append(part)
                
            # If the current node has children, enclose the children's string representation in parentheses
            if parts:
                children_str = '(' + ','.join(parts) + ')'
            else:
                children_str = ''

            # Node name and distance formatting
            node_info = node.name if node.name else ''
            if 'edge_length' in node.data:
                node_info += ':' + str(node.data['edge_length'])

            # For nodes that have both children and their own information (name or distance)
            if children_str or node_info:
                return children_str + node_info
            else:
                # For the very rare case where a node might not have a name or children (unlikely in valid Newick)
                return ''

        """Converts a JaxTree to a Newick string."""
        return  to_newick(self.root) + ';'
