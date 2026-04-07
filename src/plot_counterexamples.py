#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_counterexamples.py — OLEP Counterexample Visualization
============================================================

Generates a publication-quality figure showing the two smallest (n=13)
OLEP counterexamples, T1 and T2.

Each panel displays:
  - The tree in a BFS-layer hierarchical layout rooted at spectral center c
  - Node colors encoding Fiedler sign (V₊ red, V₋ blue)
  - The outermost negative leaf set F₋ (globally deepest V₋ leaves) in orange
  - The actual global minimum vertex in dark red — which lies *outside* F₋,
    demonstrating the OLEP violation
  - Dashed circle: F₋ membership; solid circle: global min ∉ F₋

The OLEP violation mechanism is the **collective pull effect**:
  Hub node 8 at BFS depth 1 has four leaf children {9,10,11,12} in V₋.
  Their combined weight in the Laplacian eigenvector equation pulls
  f(8) to a more negative value than the leaf at depth 2 (node 7),
  causing the global minimum to land at the depth-1 leaves, not the
  globally deepest leaf.

Output:
  olep_counterexamples.pdf  (300 dpi, for publication)
  olep_counterexamples.png  (150 dpi, for preview)

Usage:
  python src/plot_counterexamples.py

Reference:
  Section 4 ("Counterexample analysis") of the companion paper.
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import eigh


# ===========================================================================
# Graph construction — two minimal n=13 counterexamples
# ===========================================================================

def make_T1():
    """
    T1: n=13, diameter=7.

    Structure: 0–1–2–3–4  (chain of length 4, positive side)
               0–5–6–7    (chain of length 3, negative side)
               0–8–{9,10,11,12}  (hub at depth 1 with 4 leaf children, V₋)

    Spectral center c=0.
    OLEP violation: global min f is at {9,10,11,12} (depth 2),
    not at the deepest V₋ leaf 7 (depth 3).
    """
    G = nx.Graph()
    G.add_nodes_from(range(13))
    G.add_edges_from([
        (0, 1), (0, 5), (0, 8),          # arms from root
        (1, 2), (2, 3), (3, 4),           # positive chain arm
        (5, 6), (6, 7),                   # negative chain arm
        (8, 9), (8, 10), (8, 11), (8, 12) # hub with 4 leaf children
    ])
    return G


def make_T2():
    """
    T2: n=13, diameter=6.

    Same as T1 except the positive arm is shortened:
    instead of 0–1–2–3–4, it is 0–1–2–{3,4} (hub at depth 2).

    Spectral center c=0.
    OLEP violation: same collective-pull mechanism as T1.
    """
    G = nx.Graph()
    G.add_nodes_from(range(13))
    G.add_edges_from([
        (0, 1), (0, 5), (0, 8),
        (1, 2), (2, 3), (2, 4),            # positive arm: hub at depth 2
        (5, 6), (6, 7),
        (8, 9), (8, 10), (8, 11), (8, 12)
    ])
    return G


# ===========================================================================
# Fiedler vector computation
# ===========================================================================

def compute_fiedler(G):
    """
    Compute Fiedler vector via dense symmetric eigensolver.

    Returns
    -------
    lambda2 : float  — algebraic connectivity
    f_dict  : dict   — vertex → Fiedler value
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    L = np.zeros((n, n))
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        L[i, i] += 1; L[j, j] += 1
        L[i, j] -= 1; L[j, i] -= 1

    vals, vecs = eigh(L)
    f = vecs[:, 1].copy()
    if np.max(f) < -np.min(f):
        f = -f

    return float(vals[1]), dict(zip(nodes, f))


# ===========================================================================
# Tree layout
# ===========================================================================

def tree_layout(G, root):
    """
    Hierarchical BFS layout for a tree.

    Leaf nodes are evenly spread horizontally; internal nodes are
    centered above their subtree.  Returns a dict {vertex: (x, y)}.
    """
    children = {v: [] for v in G.nodes()}
    bfs_pred = dict(nx.bfs_predecessors(G, root))
    for child, parent in bfs_pred.items():
        children[parent].append(child)
    bfs_depth = nx.single_source_shortest_path_length(G, root)

    def count_leaves(v):
        if not children[v]:
            return 1
        return sum(count_leaves(c) for c in children[v])

    pos = {}

    def assign(v, x_start):
        if not children[v]:
            pos[v] = (x_start, -bfs_depth[v] * 1.4)
            return 1
        total = 0
        for c in children[v]:
            w = count_leaves(c)
            assign(c, x_start + total + w / 2 - 0.5)
            total += w
        pos[v] = (x_start + total / 2 - 0.5, -bfs_depth[v] * 1.4)
        return total

    assign(root, 0)

    # Normalize to [0,7] × [0,5]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xr = max(max(xs) - min(xs), 0.1)
    yr = max(max(ys) - min(ys), 0.1)
    pos = {v: ((pos[v][0] - min(xs)) / xr * 7,
               (pos[v][1] - min(ys)) / yr * 5)
           for v in pos}
    return pos


# ===========================================================================
# Drawing
# ===========================================================================

def draw_panel(ax, G, title, root, f_dict, F_minus, F_plus):
    """
    Draw one tree panel.

    Color scheme:
      #9b59b6  purple  — spectral center c
      #e74c3c  red     — V₊ (positive Fiedler)
      #3498db  blue    — V₋ (negative Fiedler), non-outermost
      #e67e22  orange  — F₋ (outermost negative leaves, globally deepest)
      #c0392b  dark red— actual global minimum vertex (OLEP violator)
    """
    pos = tree_layout(G, root)
    bfs_depth = nx.single_source_shortest_path_length(G, root)

    global_min = min(f_dict.values())
    global_max = max(f_dict.values())

    # Assign node colors
    node_colors = []
    node_sizes  = []
    for v in sorted(G.nodes()):
        fv = f_dict[v]
        if v == root:
            node_colors.append('#9b59b6'); node_sizes.append(420)
        elif v in F_minus:
            node_colors.append('#e67e22'); node_sizes.append(390)
        elif abs(fv - global_min) < 1e-8:
            # Global min outside F₋ — the OLEP violator
            node_colors.append('#c0392b'); node_sizes.append(430)
        elif fv < 0:
            node_colors.append('#3498db'); node_sizes.append(300)
        else:
            node_colors.append('#e74c3c'); node_sizes.append(300)

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5,
                           edge_color='#555555', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           linewidths=1.5, edgecolors='white')
    labels = {v: f'{v}\n{f_dict[v]:.3f}' for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_color='white', font_weight='bold')

    # Dashed circle: F₋ membership
    for v in F_minus:
        px, py = pos[v]
        ax.add_patch(plt.Circle((px, py), 0.36, fill=False,
                                color='#27ae60', linewidth=2.5,
                                linestyle='--', zorder=5))

    # Solid circle: global min ∉ F₋
    for v in G.nodes():
        if abs(f_dict[v] - global_min) < 1e-8 and v not in F_minus:
            px, py = pos[v]
            ax.add_patch(plt.Circle((px, py), 0.40, fill=False,
                                    color='#c0392b', linewidth=2.5,
                                    linestyle='-', zorder=5))

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-0.8, 5.8)
    ax.axis('off')

    # Sub-caption
    ax.text(0.5, -0.08,
            r'OLEP violated: $\min_V f \notin \mathcal{F}_-$'
            '\n(collective pull from 4 leaf children at hub 8)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8, style='italic', color='#7f8c8d')


# ===========================================================================
# Main
# ===========================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':   9,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, (name, G) in zip(axes, [
    (r'$T_1$: $n=13$, $D=7$', make_T1()),
    (r'$T_2$: $n=13$, $D=6$', make_T2()),
]):
    lam2, f = compute_fiedler(G)
    root     = min(G.nodes(), key=lambda v: abs(f[v]))   # spectral center
    depth    = nx.single_source_shortest_path_length(G, root)

    leaves_m = [v for v in G.nodes() if G.degree(v) == 1 and f[v] < 0]
    leaves_p = [v for v in G.nodes() if G.degree(v) == 1 and f[v] > 0]

    K_m  = max(depth[v] for v in leaves_m)
    K_p  = max(depth[v] for v in leaves_p)
    F_minus = [v for v in leaves_m if depth[v] == K_m]
    F_plus  = [v for v in leaves_p if depth[v] == K_p]

    draw_panel(ax, G, name, root, f, F_minus, F_plus)

# Legend
legend_elements = [
    mpatches.Patch(color='#9b59b6',
                   label=r'Spectral center $c$ (root)'),
    mpatches.Patch(color='#3498db',
                   label=r'$V_-$ (negative Fiedler)'),
    mpatches.Patch(color='#e74c3c',
                   label=r'$V_+$ (positive Fiedler)'),
    mpatches.Patch(color='#e67e22',
                   label=r'$\mathcal{F}_-$: outermost $V_-$ leaves (globally deepest)'),
    mpatches.Patch(color='#c0392b',
                   label=r'Global min $\notin \mathcal{F}_-$ (OLEP violated)'),
    mpatches.Patch(facecolor='none', edgecolor='#27ae60',
                   linewidth=2, linestyle='--',
                   label=r'Dashed ring: $\mathcal{F}_-$ member'),
    mpatches.Patch(facecolor='none', edgecolor='#c0392b',
                   linewidth=2, linestyle='-',
                   label=r'Solid ring: global min $\notin \mathcal{F}_-$'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02),
           fancybox=True, shadow=False)

fig.suptitle(
    r'Two Minimal OLEP Counterexamples ($n=13$ vertices each)' + '\n'
    r'Global $\min f$ occurs at depth-2 hub leaves $\{9,10,11,12\}$, '
    r'not at the deepest $V_-$ leaf $\{7\}$ ($\mathcal{F}_-$)',
    fontsize=10.5, y=1.01,
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('olep_counterexamples.pdf', dpi=300, bbox_inches='tight')
plt.savefig('olep_counterexamples.png', dpi=150, bbox_inches='tight')
print("Saved: olep_counterexamples.pdf  and  olep_counterexamples.png")
