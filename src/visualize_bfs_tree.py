"""
Fiedler-BFS-Tree Layout Visualization
======================================

Visualizes a tree using the **Fiedler-BFS-Tree layout** introduced in the paper:

  "Outer-Leaf Extremal Property of Fiedler Vectors on Trees"
  (submitted to Linear Algebra and Its Applications)

Layout algorithm:
  1. Compute the Fiedler vector f (eigenvector for λ₂ of the Laplacian).
  2. Partition vertices: V₋ = {f(v) < 0} drawn downward,
                         V₊ = {f(v) ≥ 0} drawn upward.
  3. Find spectral center c = argmin|f(v)| as the layout root.
  4. Arrange nodes in BFS layers from c:
       - V₋ nodes at layer k → y = −k·1.5
       - V₊ nodes at layer k → y = +k·1.5
  5. Within each layer, nodes are spread evenly along the x-axis.
  6. BFS layer labels Lₖ⁻ / Lₖ⁺ are shown on the right margin.
  7. Fiedler extrema (global max / min) are highlighted with squares.

Output:
  output_figure.pdf — publication-quality PDF visualization

Usage:
  python src/visualize_bfs_tree.py

Requirements:
  Python >= 3.9, networkx, matplotlib, scipy, numpy
"""

import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import numpy as np
from collections import defaultdict


# ===========================================================================
# Fiedler vector computation
# ===========================================================================

def compute_fiedler_vector(G):
    """
    Compute the Fiedler vector of graph G using a sparse eigensolver.

    Uses scipy's ARPACK-based eigsh (shift-invert) for efficiency on
    larger trees (n > 100).  Falls back to zero vector on failure.

    Parameters
    ----------
    G : networkx.Graph  — connected tree

    Returns
    -------
    fiedler_vector : ndarray, shape (n,)
        Eigenvector for λ₂ (second smallest Laplacian eigenvalue).
        Sign convention: the function does NOT enforce max(f) > 0 here;
        sign is left to the caller.
    nodes : list
        Sorted node labels (index i → nodes[i]).
    """
    nodes = sorted(G.nodes())
    try:
        L = nx.laplacian_matrix(G, nodelist=nodes).astype(float)
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
        if len(eigenvalues) >= 2:
            return eigenvectors[:, 1], nodes
    except Exception:
        pass
    return np.zeros(len(nodes)), nodes


# ===========================================================================
# Spectral center and sign partition
# ===========================================================================

def find_spectral_center(G, fiedler_vector, node_index):
    """
    Find the spectral center c = argmin|f(v)| and the negative set V₋.

    The spectral center is the vertex whose Fiedler component is closest
    to zero — it lies at the boundary between V₊ and V₋ and serves as
    the natural BFS root for the layout.

    Parameters
    ----------
    G              : networkx.Graph
    fiedler_vector : ndarray
    node_index     : dict {node: int}  — maps vertex label to array index

    Returns
    -------
    center : vertex label  — argmin|f(v)|
    v_neg  : set           — {v : f(v) < 0}
    """
    v_neg = {n for n in G.nodes() if fiedler_vector[node_index[n]] < 0}

    center = min(G.nodes(), key=lambda n: abs(fiedler_vector[node_index[n]]))
    return center, v_neg


# ===========================================================================
# BFS layer decomposition
# ===========================================================================

def compute_bfs_layers(G, root):
    """
    Compute BFS layer assignment from root.

    Returns
    -------
    dict : {layer_depth: [node_list]}  — layer 0 = {root}
    """
    layers  = defaultdict(list)
    visited = {root}
    queue   = [(root, 0)]

    while queue:
        node, depth = queue.pop(0)
        layers[depth].append(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return dict(layers)


# ===========================================================================
# Layout computation
# ===========================================================================

def compute_radial_layout(G, center, v_neg, v_nonneg):
    """
    Assign (x, y) coordinates for the Fiedler-BFS-Tree layout.

    V₋ nodes at BFS layer k → y = −k·1.5  (below center)
    V₊ nodes at BFS layer k → y = +k·1.5  (above center)
    Within each (layer, sign-group), nodes are equally spaced horizontally
    with spacing 2.5.

    Parameters
    ----------
    G       : networkx.Graph
    center  : vertex label — BFS root (spectral center)
    v_neg   : set of V₋ vertices
    v_nonneg: set of V₊ vertices

    Returns
    -------
    pos : dict {vertex: (x, y)}
    """
    pos    = {center: (0, 0)}
    layers = compute_bfs_layers(G, center)

    for depth, layer_nodes in layers.items():
        if depth == 0:
            continue

        neg_nodes = [n for n in layer_nodes if n in v_neg]
        pos_nodes = [n for n in layer_nodes if n in v_nonneg]

        # V₋ nodes: spread horizontally, y = −depth·1.5
        for i, node in enumerate(neg_nodes):
            x = (i - (len(neg_nodes) - 1) / 2) * 2.5 if len(neg_nodes) > 1 else 0
            pos[node] = (x, -depth * 1.5)

        # V₊ nodes: spread horizontally, y = +depth·1.5
        for i, node in enumerate(pos_nodes):
            x = (i - (len(pos_nodes) - 1) / 2) * 2.5 if len(pos_nodes) > 1 else 0
            pos[node] = (x, depth * 1.5)

    return pos


def bfs_tree_layout(G, fiedler_vector, nodes):
    """
    Compute the full Fiedler-BFS-Tree layout.

    Parameters
    ----------
    G              : networkx.Graph
    fiedler_vector : ndarray
    nodes          : list — sorted node labels (from compute_fiedler_vector)

    Returns
    -------
    pos     : dict {vertex: (x, y)}
    center  : spectral center vertex
    v_neg   : set V₋
    v_nonneg: set V₊
    layers  : dict {depth: [node_list]}
    """
    node_index = {n: i for i, n in enumerate(nodes)}

    v_neg    = {n for n in G.nodes() if fiedler_vector[node_index[n]] < 0}
    v_nonneg = set(G.nodes()) - v_neg

    center, _ = find_spectral_center(G, fiedler_vector, node_index)
    if center is None:
        center = list(G.nodes())[0]

    pos    = compute_radial_layout(G, center, v_neg, v_nonneg)
    layers = compute_bfs_layers(G, center)

    return pos, center, v_neg, v_nonneg, layers


# ===========================================================================
# Visualization
# ===========================================================================

def visualize_bfs_tree(G, fiedler_vector, nodes, center, v_neg, v_nonneg, pos, layers):
    """
    Render the Fiedler-BFS-Tree layout and save to output_figure.pdf.

    Visual encoding:
      #FF1493 pink     — spectral center c
      #4169E1 blue     — V₋ (negative Fiedler components)
      #DC143C crimson  — V₊ (non-negative Fiedler components)
      Black squares    — Fiedler global max and min vertices
      Lₖ⁻ / Lₖ⁺ labels — BFS layer annotations on the right margin
    """
    node_index = {n: i for i, n in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=(13, 9))

    # Node colors and sizes
    node_colors = ['#FF1493' if n == center
                   else '#4169E1' if n in v_neg
                   else '#DC143C'
                   for n in G.nodes()]
    node_sizes  = [1200 if n == center else 700 for n in G.nodes()]

    # Edge colors: highlight edges incident to spectral center
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u == center or v == center:
            edge_colors.append('#FF1493'); edge_widths.append(2.5)
        else:
            edge_colors.append('#666666'); edge_widths.append(1.5)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           width=edge_widths, arrows=False, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='black', linewidths=1, ax=ax)

    # Node labels: vertex index
    nx.draw_networkx_labels(G, pos,
                            labels={n: str(n) for n in G.nodes()},
                            font_size=18, font_color='white',
                            font_weight='bold',
                            font_family='Times New Roman', ax=ax)

    # Fiedler value labels (offset to the right of each node)
    label_pos = {n: (x + 0.5, y) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos,
                            labels={n: f"{fiedler_vector[node_index[n]]:.3f}"
                                    for n in G.nodes()},
                            font_size=18, font_color='#8B0000',
                            font_family='Times New Roman', ax=ax)

    # BFS layer annotations: Lₖ⁻ / Lₖ⁺ on the right margin
    layer_colors = ['#FF1493', '#4169E1', '#32CD32', '#9370DB', '#FFA500']
    x_max = max(x for x, y in pos.values())

    for k, layer_nodes in layers.items():
        if k == 0 or not layer_nodes:
            continue
        color     = layer_colors[k % len(layer_colors)]
        neg_nodes = [n for n in layer_nodes if n in v_neg]
        pos_nodes = [n for n in layer_nodes if n in v_nonneg]

        if neg_nodes:
            ref_y = np.mean([pos[n][1] for n in neg_nodes])
            ax.text(x_max + 1.5, ref_y, f'$L_{k}^-$',
                    fontsize=24, color=color, weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, alpha=0.9))
        if pos_nodes:
            ref_y = np.mean([pos[n][1] for n in pos_nodes])
            ax.text(x_max + 1.5, ref_y, f'$L_{k}^+$',
                    fontsize=24, color=color, weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, alpha=0.9))

    # Mark Fiedler extrema with black squares
    max_val   = max(fiedler_vector)
    min_val   = min(fiedler_vector)
    max_nodes = [nodes[i] for i, v in enumerate(fiedler_vector)
                 if abs(v - max_val) < 1e-10]
    min_nodes_list = [nodes[i] for i, v in enumerate(fiedler_vector)
                      if abs(v - min_val) < 1e-10]
    for extremum_node in set(max_nodes + min_nodes_list):
        if extremum_node in pos:
            x, y = pos[extremum_node]
            ax.add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5,
                                       fill=False, edgecolor='black',
                                       linewidth=2.5))

    # Legend
    legend_elements = [
        plt.scatter([], [], c='#FF1493', s=200,
                    label='Spectral center $c$',
                    edgecolors='black', linewidths=1),
        plt.scatter([], [], c='#4169E1', s=150,
                    label='$V_-$  (negative Fiedler)',
                    edgecolors='black', linewidths=1),
        plt.scatter([], [], c='#DC143C', s=150,
                    label='$V_+$  (non-negative Fiedler)',
                    edgecolors='black', linewidths=1),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black',
                       linewidth=2.5,
                       label='Fiedler extrema (global max/min)'),
    ]
    ax.legend(handles=legend_elements, loc='center left', fontsize=18)

    center_val = fiedler_vector[node_index[center]]
    ax.set_title(
        f"Fiedler-BFS-Tree Visualization  ({G.number_of_nodes()} vertices)\n"
        f"Spectral center: node {center}  "
        f"($f(c) = {center_val:.3f}$)",
        fontsize=18, pad=20, weight='bold',
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('output_figure.pdf', format='pdf', bbox_inches='tight')
    print("Saved: output_figure.pdf")


# ===========================================================================
# Helper: random tree generator
# ===========================================================================

def generate_random_tree(n_nodes, seed=None):
    """Generate a uniformly random labeled tree on n_nodes vertices."""
    return nx.random_labeled_tree(n_nodes, seed=seed)


# ===========================================================================
# Main demo
# ===========================================================================

def main():
    """
    Demonstrate the Fiedler-BFS-Tree layout on a random tree (n=17, seed=42).
    Edit `n_nodes` and `seed` to visualize different trees.
    """
    print("=" * 60)
    print("Fiedler-BFS-Tree Layout Visualization")
    print("=" * 60)

    n_nodes = 17
    seed    = 42
    G = generate_random_tree(n_nodes, seed=seed)

    print(f"\nTree: n={G.number_of_nodes()}, "
          f"leaves={sum(1 for n in G.nodes() if G.degree(n)==1)}")

    fiedler_vector, nodes = compute_fiedler_vector(G)
    node_index = {n: i for i, n in enumerate(nodes)}

    center, v_neg = find_spectral_center(G, fiedler_vector, node_index)
    v_nonneg      = set(G.nodes()) - v_neg

    print(f"Spectral center c = {center}  "
          f"(f(c) = {fiedler_vector[node_index[center]]:+.4f})")
    print(f"|V₋| = {len(v_neg)},  |V₊| = {len(v_nonneg)}")

    pos, center, v_neg, v_nonneg, layers = bfs_tree_layout(G, fiedler_vector, nodes)

    print("\nBFS layers:")
    for k in sorted(layers.keys()):
        print(f"  L{k}: {sorted(layers[k])}")

    print("\nGenerating layout figure...")
    visualize_bfs_tree(G, fiedler_vector, nodes, center,
                       v_neg, v_nonneg, pos, layers)
    print("Done.")


if __name__ == "__main__":
    main()
