"""
Fiedler-BFS Tree Layout Visualization
====================================

A Python implementation for visualizing trees using BFS layout based on Fiedler vector.
The algorithm partitions nodes by the sign of Fiedler vector components and arranges
them in a radial BFS layout with V_- nodes below and V_>=0 nodes above.

Author: Your Name
Date: 2024
License: MIT
"""

import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import numpy as np
from collections import defaultdict


def get_subtree_by_sign(G, root, include_set):
    """
    Extract a subtree rooted at 'root' containing only nodes in 'include_set'.
    
    Args:
        G: Original graph
        root: Root node of the subtree
        include_set: Set of nodes allowed to be included
        
    Returns:
        nx.Graph: Independent subtree
    """
    subtree = nx.Graph()
    visited = set()
    queue = [root]

    while queue:
        node = queue.pop(0)
        if node in visited or node not in include_set:
            continue
        visited.add(node)

        for neighbor in G.neighbors(node):
            if neighbor in include_set and neighbor not in visited:
                subtree.add_edge(node, neighbor)
                queue.append(neighbor)

    return subtree


def compute_tree_layout(tree, root, direction):
    """
    Compute tree coordinates with strict direction control.
    
    Args:
        direction: 'up' (positive y) or 'down' (negative y)
    """
    if root not in tree:
        print(f"Warning: root {root} not in tree, returning empty layout")
        return {root: (0, 0)}
    
    depths = {root: 0}
    queue = [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        if node not in tree:
            continue
        for neighbor in tree.neighbors(node):
            if neighbor not in depths:
                depths[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    pos = {}
    max_depth = max(depths.values()) if depths else 0

    for node, depth in depths.items():
        siblings = [n for n in depths if depths[n] == depth]
        x = (siblings.index(node) - (len(siblings) - 1) / 2) * 2

        if direction == 'up':
            y = depth + 1
        else:
            y = -(depth + 1)

        pos[node] = (x, y)

    return pos



def find_tree_center(G, fiedler_vector, node_index):
    """
    Find the tree center: node with smallest absolute Fiedler component.
    
    This function finds the node with the smallest absolute value in the
    Fiedler vector, which is the node "closest" to the graph cut.
    
    Args:
        G: The input graph
        fiedler_vector: Fiedler vector (second smallest eigenvector of Laplacian)
        node_index: Mapping from node to index in fiedler_vector
        
    Returns:
        center: Node with smallest absolute Fiedler component
        v_neg: Set of nodes with negative Fiedler components
    """
    v_neg = set()
    for n in G.nodes():
        idx = node_index[n]
        if fiedler_vector[idx] < 0:
            v_neg.add(n)

    center = None
    min_abs_val = float('inf')
    
    for n in G.nodes():
        val = fiedler_vector[node_index[n]]
        abs_val = abs(val)
        if abs_val < min_abs_val:
            min_abs_val = abs_val
            center = n

    return center, v_neg


def bfs_tree_layout(G, fiedler_vector, nodes):
    """
    Main BFS-TREE layout function.
    
    1. Partition by Fiedler sign: V_- and V_>=0
    2. Find tree center (maximum in V_-)
    3. Compute BFS layout (radiating from center)
    
    Returns:
        pos: Node positions dictionary
        center: Center node
        v_neg: V_- set
        v_nonneg: V_>=0 set
        layers: BFS layers dictionary {k: [node list]}
    """
    node_index = {n: i for i, n in enumerate(nodes)}

    v_neg = set()
    v_nonneg = set()
    for n in G.nodes():
        idx = node_index[n]
        if fiedler_vector[idx] < 0:
            v_neg.add(n)
        else:
            v_nonneg.add(n)

    center, _ = find_tree_center(G, fiedler_vector, node_index)
    if center is None:
        print("Warning: No center found, using first node")
        center = list(G.nodes())[0]

    pos = compute_radial_layout(G, center, v_neg, v_nonneg)
    layers = compute_bfs_layers(G, center)

    return pos, center, v_neg, v_nonneg, layers


def compute_radial_layout(G, center, v_neg, v_nonneg):
    """
    Compute symmetric radial layout: V_- downward, V_>=0 upward.
    """
    pos = {center: (0, 0)}
    layers = compute_bfs_layers(G, center)
    
    for depth, nodes in layers.items():
        if depth == 0:
            continue
            
        neg_nodes = [n for n in nodes if n in v_neg]
        pos_nodes = [n for n in nodes if n in v_nonneg]
        
        for i, node in enumerate(neg_nodes):
            if len(neg_nodes) == 1:
                x = 0
            else:
                x = (i - (len(neg_nodes) - 1) / 2) * 2.5
            y = -depth * 1.5
            pos[node] = (x, y)
        
        for i, node in enumerate(pos_nodes):
            if len(pos_nodes) == 1:
                x = 0
            else:
                x = (i - (len(pos_nodes) - 1) / 2) * 2.5
            y = depth * 1.5
            pos[node] = (x, y)
    
    return pos


def compute_bfs_layers(G, root):
    """
    Compute BFS layers from root node.
    
    Returns:
        dict: {layer_number: [node_list]}
    """
    layers = defaultdict(list)
    visited = {root}
    queue = [(root, 0)]

    while queue:
        node, depth = queue.pop(0)
        layers[depth].append(node)

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return dict(layers)


def compute_fiedler_vector(G):
    """
    Compute the Fiedler vector (second smallest eigenvector of Laplacian).
    """
    nodes = sorted(G.nodes())
    try:
        L = nx.laplacian_matrix(G, nodelist=nodes).astype(float)
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
        if len(eigenvalues) >= 2:
            return eigenvectors[:, 1], nodes
    except:
        pass
    return np.zeros(len(nodes)), nodes


def visualize_bfs_tree(G, fiedler_vector, nodes, center, v_neg, v_nonneg, pos, layers):
    """
    Visualize the BFS-TREE layout with Fiedler vector components.
    """
    n_nodes = G.number_of_nodes()
    node_index = {n: i for i, n in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=(13, 9))

    node_colors = []
    for n in G.nodes():
        if n == center:
            node_colors.append('#FF1493')
        elif n in v_neg:
            node_colors.append('#4169E1')
        else:
            node_colors.append('#DC143C')

    node_sizes = [1200 if n == center else 700 for n in G.nodes()]

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u == center or v == center:
            edge_colors.append('#FF1493')
            edge_widths.append(2.5)
        else:
            edge_colors.append('#666666')
            edge_widths.append(1.5)

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=False,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1,
        ax=ax
    )

    node_labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=node_labels,
        font_size=18, font_color='white', font_weight='bold',
        font_family='Times New Roman', ax=ax
    )

    fiedler_labels = {n: f"{fiedler_vector[node_index[n]]:.3f}" for n in G.nodes()}
    label_pos = {n: (x + 0.5, y) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos, labels=fiedler_labels,
        font_size=18, font_color='#8B0000',
        font_family='Times New Roman', ax=ax
    )

    layer_colors = ['#FF1493', '#4169E1', '#32CD32', '#9370DB', '#FFA500']
    
    x_max = max(x for x, y in pos.values())
    right_margin = 1.5
    
    for k, layer_nodes in layers.items():
        if k == 0 or not layer_nodes:
            continue
            
        neg_nodes = [n for n in layer_nodes if n in v_neg]
        pos_nodes = [n for n in layer_nodes if n in v_nonneg]
        
        color = layer_colors[k % len(layer_colors)]
        
        if neg_nodes:
            neg_y_coords = [pos[n][1] for n in neg_nodes]
            neg_ref_x = x_max + right_margin
            neg_ref_y = sum(neg_y_coords) / len(neg_y_coords)
            ax.text(neg_ref_x, neg_ref_y, f'$L_{k}^-$',
                   fontsize=24, color=color, weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.15", facecolor='white', 
                            edgecolor=color, alpha=0.9))
        
        if pos_nodes:
            pos_y_coords = [pos[n][1] for n in pos_nodes]
            pos_ref_x = x_max + right_margin
            pos_ref_y = sum(pos_y_coords) / len(pos_nodes)
            
            ax.text(pos_ref_x, pos_ref_y, f'$L_{k}^+$',
                   fontsize=24, color=color, weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.15", facecolor='white', 
                            edgecolor=color, alpha=0.9))

    max_depth = max(layers.keys())
    outermost_leaves = set()
    for k, layer_nodes in layers.items():
        if k == max_depth:
            for n in layer_nodes:
                if G.degree(n) == 1:
                    outermost_leaves.add(n)

    for leaf in outermost_leaves:
        x, y = pos[leaf]
        square = plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5,
                              fill=False, edgecolor='black', linewidth=2.5)
        ax.add_patch(square)

    min_val = min(fiedler_vector)
    min_nodes = [nodes[i] for i, val in enumerate(fiedler_vector) if abs(val - min_val) < 1e-10]
    for min_node in min_nodes:
        if min_node not in outermost_leaves and min_node in pos:
            x, y = pos[min_node]
            square = plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5,
                                  fill=False, edgecolor='black', linewidth=2.5)
            ax.add_patch(square)

    legend_elements = [
        plt.scatter([], [], c='#FF1493', s=200, label='Spectral Center c', edgecolors='black', linewidths=1),
        plt.scatter([], [], c='#4169E1', s=150, label='V_- (negative)', edgecolors='black', linewidths=1),
        plt.scatter([], [], c='#DC143C', s=150, label='V_>=0 (non-negative)', edgecolors='black', linewidths=1),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2.5, 
                     label='Fiedler extrema (max/min)', linestyle='-'),
    ]
    ax.legend(handles=legend_elements, loc='center left', fontsize=18)

    center_val = fiedler_vector[node_index[center]]
    ax.set_title(
        f"Fiedler-BFS-tree Visualization ({n_nodes} vertices)\n"
        f"Spectral Center: Node {center} (Fiedler ={center_val:.3f})",
        fontsize=18, pad=20, weight='bold'
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("output_figure.pdf", format='pdf', bbox_inches='tight')


def generate_random_tree(n_nodes, seed=None):
    """
    Generate a random labeled tree.
    """
    if seed is not None:
        np.random.seed(seed)
    G = nx.random_labeled_tree(n_nodes, seed=seed)
    return G


def main():
    """
    Main function to demonstrate the BFS-TREE layout algorithm.
    """
    print("=" * 60)
    print("BFS-TREE Visualization Tool")
    print("=" * 60)

    n_nodes = 17
    seed = 42
    G = generate_random_tree(n_nodes, seed=seed)

    print(f"\nRandom Tree Information:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Leaves: {len([n for n in G.nodes() if G.degree(n) == 1])}")

    fiedler_vector, nodes = compute_fiedler_vector(G)
    node_index = {n: i for i, n in enumerate(nodes)}

    print(f"\nFiedler vector computed")
    print(f"Algebraic connectivity lambda_2 = {2 * np.sum(fiedler_vector ** 2):.6f}")

    center, v_neg = find_tree_center(G, fiedler_vector, node_index)
    v_nonneg = set(G.nodes()) - v_neg

    print(f"\nNode classification:")
    print(f"  V_- (f < 0): {len(v_neg)} nodes")
    print(f"  V_>=0 (f >= 0): {len(v_nonneg)} nodes")
    print(f"  Tree center (maximum in V_-): Node {center} (f = {fiedler_vector[node_index[center]]:.4f})")

    print(f"\nFiedler components per node:")
    for n in sorted(nodes):
        val = fiedler_vector[node_index[n]]
        marker = " ← tree center" if n == center else ""
        marker += " ← minimum" if val == min(fiedler_vector) else ""
        print(f"  Node {n}: {val:+.4f}{marker}")

    print(f"\nComputing BFS-TREE layout...")
    pos, center, v_neg, v_nonneg, layers = bfs_tree_layout(G, fiedler_vector, nodes)

    print(f"\nBFS-TREE layout coordinates:")
    for n in sorted(pos.keys()):
        x, y = pos[n]
        print(f"  Node {n}: ({x:+.2f}, {y:+.2f})")

    print(f"\nBFS layer information:")
    for k in sorted(layers.keys()):
        print(f"  Layer {k}: {sorted(layers[k])}")

    print(f"\nGenerating visualization...")
    visualize_bfs_tree(G, fiedler_vector, nodes, center, v_neg, v_nonneg, pos, layers)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
