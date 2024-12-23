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
        """Update the predict"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        """Pred"""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Compute min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly split a node using a random feature and threshold."""

        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Function """

        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)     # <--- to be defined later

        self.update_predict()     # <--- defined in the previous task

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(
                                    self.explanatory, self.target)}""")

    def fit_node(self, node):
        """Get the split feature and threshold from the criterion"""
        node.feature, node.threshold = self.split_criterion(node)

        # Split the population based on the feature and threshold
        # Use boolean indexing for efficiency
        feature_values = self.explanatory[:, node.feature][node.sub_population]
        left_mask = feature_values > node.threshold
        right_mask = ~left_mask

        # Create sub_population arrays for children
        left_population = node.sub_population.copy()
        right_population = node.sub_population.copy()

        # Update the sub_populations using the masks
        left_population[node.sub_population] = left_mask
        right_population[node.sub_population] = right_mask

        # Check if left node should be a leaf
        left_target = self.target[left_population]
        is_left_leaf = (
            np.sum(left_population) < self.min_pop or  # Too few samples
            node.depth + 1 == self.max_depth or        # Max depth reached
            len(np.unique(left_target)) == 1         # Pure node (single class)
        )

        # Create and recursively fit left child
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Check if right node should be a leaf
        right_target = self.target[right_population]
        is_right_leaf = (
            np.sum(right_population) < self.min_pop or  # Too few samples
            node.depth + 1 == self.max_depth or         # Max depth reached
            len(np.unique(right_target)) == 1        # Pure node (single class)
        )

        # Create and recursively fit right child
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Get the most common class (mode) for the leaf's value"""
        target_values = self.target[sub_population]
        value = np.bincount(target_values).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Function to get the node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Function"""
        return np.sum(np.equal(self.predict(
            test_explanatory), test_target)) / test_target.size
