"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    
    distances = np.array(distances, dtype=float, copy=True)  # never modify input
    n_leaves = distances.shape[0]
    if distances.shape != (n_leaves, n_leaves) or len(labels) != n_leaves:
        raise ValueError("distances must be square and match the labels")
    if n_leaves < 2:
        raise ValueError("Need at least two taxa")
 
    # active nodes, new nodes are appended to the end
    nodes = [Node(str(label), None, 0.0, None, 0.0) for label in labels]
    taken_names = {node.name for node in nodes}
    internal_counter = 0
 
    while len(nodes) > 2:
        n = len(nodes)
        row_sums = distances.sum(axis=1)
 
        # Q matrix: Q(i, j) = (n - 2) d(i, j) - r_i - r_j
        q_matrix = (n - 2) * distances - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        i, j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        if i > j:
            i, j = j, i
 
        d_ij = distances[i, j]
        dist_i = 0.5 * d_ij + (row_sums[i] - row_sums[j]) / (2.0 * (n - 2))
        dist_j = d_ij - dist_i
 
        internal_counter += 1
        new_name = f"node_{internal_counter}"
        while new_name in taken_names:
            internal_counter += 1
            new_name = f"node_{internal_counter}"
        taken_names.add(new_name)
        new_node = Node(new_name, nodes[i], float(dist_i), nodes[j], float(dist_j))
 
        # distances from the new node to every other remaining node
        keep = [k for k in range(n) if k not in (i, j)]
        d_new = 0.5 * (distances[i, keep] + distances[j, keep] - d_ij)
 
        new_distances = np.zeros((len(keep) + 1, len(keep) + 1))
        new_distances[: len(keep), : len(keep)] = distances[np.ix_(keep, keep)]
        new_distances[-1, : len(keep)] = d_new
        new_distances[: len(keep), -1] = d_new
 
        distances = new_distances
        nodes = [nodes[k] for k in keep] + [new_node]
 
    # last two nodes are joined by the root, each at half of the distance
    half = float(distances[0, 1]) / 2.0
    root_name = "root"
    while root_name in taken_names:
        root_name += "_"
    return Node(root_name, nodes[0], half, nodes[1], half)
 
 
def _count_leaves(node: Node) -> int:
    """Number of leaves below (and including) a node."""
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)
 
 
def _is_leaf(node: Node) -> bool:
    return node.left is None and node.right is None
 
 
def _layout_tree(tree: Node):
    """Compute (x, y) for every node. x = distance from root, y = leaf order
    for leaves and the midpoint of the children for internal nodes."""
    positions = {}
    counter = [0]
 
    def recurse(node, depth):
        if _is_leaf(node):
            y = float(counter[0])
            counter[0] += 1
        else:
            y_left = recurse(node.left, depth + node.left_distance)
            y_right = recurse(node.right, depth + node.right_distance)
            y = (y_left + y_right) / 2.0
        positions[id(node)] = (depth, y)
        return y
 
    recurse(tree, 0.0)
    return positions


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    import matplotlib.pyplot as plt
 
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
 
    label_colors = kwargs.get("label_colors", {}) or {}
    label_map = kwargs.get("label_map", {}) or {}
    show_confidence = kwargs.get("show_confidence", False)
    line_color = kwargs.get("line_color", "k")
    line_width = kwargs.get("line_width", 1.5)
    font_size = kwargs.get("font_size", 12)
    show_scale = kwargs.get("show_scale", False)
 
    positions = _layout_tree(tree)
    root_x, root_y = positions[id(tree)]
 
    # small stub to the left of the root
    max_x = max(p[0] for p in positions.values())
    stub = kwargs.get("root_stub", 0.02 * max_x if max_x > 0 else 0.2)
    ax.plot([root_x - stub, root_x], [root_y, root_y], color=line_color, lw=line_width)
 
    stack = [tree]
    while stack:
        node = stack.pop()
        if _is_leaf(node):
            continue
        x, y = positions[id(node)]
        child_ys = []
        for child in (node.left, node.right):
            cx, cy = positions[id(child)]
            child_ys.append(cy)
            # horizontal line = genetic distance
            ax.plot([x, cx], [cy, cy], color=line_color, lw=line_width,
                    solid_capstyle="butt")
            if _is_leaf(child):
                ax.text(cx, cy, f"  {label_map.get(child.name, child.name)}",
                        color=label_colors.get(child.name, "k"),
                        fontsize=font_size, verticalalignment="center")
            else:
                stack.append(child)
        # vertical line = divergence from the common ancestor
        ax.plot([x, x], child_ys, color=line_color, lw=line_width,
                solid_capstyle="butt")
        if show_confidence and node.confidence is not None:
            ax.text(x, y, f"{node.confidence:.2f} ", fontsize=max(font_size - 4, 5),
                    color="tab:red", ha="right", va="bottom")
 
    if show_scale:
        ax.set_xlabel("Genetic distance (mismatches)")
    ax.set_yticks([])
    for loc in ["top", "right", "left"]:
        ax.spines[loc].set_visible(False)
    if not show_scale:
        ax.spines["bottom"].set_visible(True)
    ax.margins(x=0.05)
    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree is None or _is_leaf(tree):
        return
 
    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)
 
    # the child with fewer leaves becomes the left child (ties keep the order)
    if _count_leaves(tree.left) > _count_leaves(tree.right):
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = (
            tree.right_distance,
            tree.left_distance,
        )


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    import matplotlib.pyplot as plt
 
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    label_colors = kwargs.get("label_colors", {}) or {}
 
    n_leaves = _count_leaves(tree)
    positions = _layout_tree(tree)
 
    def to_xy(depth, leaf_y):
        angle = 2 * np.pi * (leaf_y + 0.5) / n_leaves
        return depth * np.cos(angle), depth * np.sin(angle), angle
 
    stack = [tree]
    while stack:
        node = stack.pop()
        if _is_leaf(node):
            continue
        depth, y = positions[id(node)]
        ys = []
        for child in (node.left, node.right):
            cdepth, cy = positions[id(child)]
            ys.append(cy)
            # arc at the parent's radius up to the child's angle, then radial line
            x0, y0, _ = to_xy(depth, cy)
            x1, y1, angle = to_xy(cdepth, cy)
            ax.plot([x0, x1], [y0, y1], color="k", lw=1.2)
            if _is_leaf(child):
                ha = "left" if np.cos(angle) >= 0 else "right"
                ax.text(x1, y1, f" {child.name} " , ha=ha, va="center", fontsize=8,
                        color=label_colors.get(child.name, "k"))
            else:
                stack.append(child)
        thetas = np.linspace(2 * np.pi * (ys[0] + 0.5) / n_leaves,
                             2 * np.pi * (ys[1] + 0.5) / n_leaves, 50)
        ax.plot(depth * np.cos(thetas), depth * np.sin(thetas), color="k", lw=1.2)
 
    ax.set_aspect("equal")
    ax.axis("off")
    return ax