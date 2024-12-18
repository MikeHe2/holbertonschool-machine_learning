#!/usr/bin/env python3
"""Supervised learning"""
import numpy as np


class Node:
    """Node class represents a single node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a node"""

        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculate the max depth"""

        if self.is_leaf:
            return self.depth
        else:
            if self.left_child:
                left_depth = self.left_child.max_depth_below()
            else:
                left_depth = 0
            if self.right_child:
                right_depth = self.right_child.max_depth_below()
            else:
                right_depth = 0
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Calculate the nodes below"""

        if self.is_leaf:
            return 1
        else:
            left_count = self.left_child.count_nodes_below(only_leaves)
            right_count = self.right_child.count_nodes_below(only_leaves)

            if only_leaves:
                return left_count + right_count
            else:
                return 1 + left_count + right_count


class Leaf(Node):
    """Leaf node class for a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a leaf"""

        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Calculate the max depth"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Calculate the nodes below"""
        return 1


class Decision_Tree():
    """Decision Tree class"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a decision tree"""

        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Calculate the depth of a decision tree"""

        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in a decision tree"""

        return self.root.count_nodes_below(only_leaves=only_leaves)
