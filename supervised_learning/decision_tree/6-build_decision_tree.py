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
            return self.is_leaf
        else:
            left_count = self.left_child.count_nodes_below(only_leaves)
            right_count = self.right_child.count_nodes_below(only_leaves)

            if only_leaves:
                return left_count + right_count
            else:
                return 1 + left_count + right_count

    def get_leaves_below(self):
        """Get the leaves that are below"""

        if self.is_leaf:
            return self
        else:
            left_leaves = self.left_child.get_leaves_below()
            right_leaves = self.right_child.get_leaves_below()
            return left_leaves + right_leaves

    def __str__(self):
        """Generate a string representation of the node and its subtree"""

        # Generate the string for the current node (root or non-root)
        if self.is_root:
            node_str = f"root [feature={self.feature},\
                        threshold={self.threshold}]\n"
        else:
            node_str = f"-> node [feature={self.feature},\
                        threshold={self.threshold}]\n"

        # If the node is a leaf, return its string representation
        if self.is_leaf:
            return node_str

        # Generate strings for the left and right children, if they exist
        left_str = ""
        if self.left_child:
            left_str = self.left_child_add_prefix(
                self.left_child.__str__())

        right_str = ""
        if self.right_child:
            right_str = self.right_child_add_prefix(
                self.right_child.__str__())

        # Combine the current node's string with its children
        return node_str + str(left_str) + str(right_str)

    def left_child_add_prefix(self, text):
        """Add prefix to left child"""

        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix to right child"""

        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("      "+x) + "\n"
        return new_text

    def update_bounds_below(self):
        """Update the bounds below"""

        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:

            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Function to update indicator"""

        def is_large_enough(x):
            """Check if the values are larger"""

            return np.all([x[:, key] > self.lower[key]
                           for key in self.lower.keys()], axis=0)

        def is_small_enough(x):
            """Check if the values are smaller"""

            return np.all([x[:, key] <= self.upper[key]
                           for key in self.upper.keys()], axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf node class for a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a leaf"""

        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Print a leaf"""

        return (f"-> leaf [value={self.value}]")

    def max_depth_below(self):
        """Calculate the max depth"""

        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Calculate the nodes below"""

        return 1

    def get_leaves_below(self):
        """Get the leaves that are below"""

        return [self]

    def update_bounds_below(self):
        """Update the bounds below"""
        pass

    def pred(self, x):
        return self.value


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

    def __str__(self):
        """Print the decision tree"""

        return self.root.__str__()

    def depth(self):
        """Calculate the depth of a decision tree"""

        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in a decision tree"""

        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Get the leaves of a decision tree"""

        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds of a decision tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        return self.root.pred(x)
