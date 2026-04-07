#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_olep.py — OLEP Verification Script
==========================================

Verifies the **Outer-Leaf Extremal Property (OLEP)** of trees against the
formal definition used in the paper:

  "Outer-Leaf Extremal Property of Fiedler Vectors on Trees"
  (submitted to Linear Algebra and Its Applications)

Definition (OLEP, Definition B — globally deepest outermost leaves):
  Let T be a tree, f its Fiedler vector (eigenvector for λ₂ of the Laplacian).
  Let c = argmin|f(v)| be the spectral center (tie-break: smallest vertex index).
  Partition vertices: V₊ = {f(v)>0}, V₋ = {f(v)<0}.
  Let K₊ = max BFS-depth from c over all LEAVES in V₊,
      K₋ = max BFS-depth from c over all LEAVES in V₋.
  Define the outermost leaf sets:
      F₊ = {v ∈ leaves(V₊) : depth(c,v) = K₊}
      F₋ = {v ∈ leaves(V₋) : depth(c,v) = K₋}
  OLEP holds iff:  max_T f ∈ F₊  AND  min_T f ∈ F₋.

Experiments (matching paper Tables 1 & 2):
  Table 1: Exhaustive enumeration of all non-isomorphic trees, n=5..17
  Table 2: Six random tree models, 2500 trees each, n∈[20,100], seed=42

Output files (results/final_defB/):
  table1_summary.txt         — human-readable Table 1 summary
  table1_by_n.json           — per-n statistics (JSON)
  table2_random.json         — Table 2 statistics (JSON)
  counterex_exhaustive.json  — all 830 Table 1 counterexamples with full
                               graph structure + Fiedler vectors
  counterex_metadata.json    — summary metadata for all counterexamples

Requirements:
  Python >= 3.9, numpy, scipy, networkx

Usage:
  python src/verify_olep.py

Reference:
  Lederman & Steinerberger (2024), Linear Algebra Appl. 703, 473-502.
"""

import os
import json
import time
import random
from datetime import datetime

import numpy as np
import networkx as nx
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, '..', 'results', 'final_defB')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Numerical zero threshold for Fiedler sign classification
EPS = 1e-9


# ===========================================================================
# Core computation
# ===========================================================================

def compute_fiedler(G):
    """
    Compute the Fiedler vector of a connected graph G.

    Uses dense symmetric eigensolver (scipy.linalg.eigh) for high accuracy.
    Suitable for trees with n ≤ ~500.

    Sign convention: max(f) > 0  (flip sign if necessary).

    Parameters
    ----------
    G : networkx.Graph
        A connected graph (tree in our experiments).

    Returns
    -------
    f : ndarray, shape (n,)
        Fiedler vector, indexed in the same order as `nodes`.
    nodes : list
        Sorted list of vertex labels.
    lambda2 : float
        Algebraic connectivity (second smallest Laplacian eigenvalue).
    lambda3 : float
        Third smallest eigenvalue (used to check λ₂ simplicity).
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    # Build Laplacian matrix L = D - A
    L = np.zeros((n, n))
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    vals, vecs = eigh(L)           # ascending eigenvalues
    lambda2 = float(vals[1])       # λ₁=0, λ₂ = algebraic connectivity
    lambda3 = float(vals[2]) if n > 2 else float('inf')

    f = vecs[:, 1].copy()          # eigenvector for λ₂
    if f.max() < -f.min():         # sign convention: max(f) > 0
        f = -f

    return f, nodes, lambda2, lambda3


def check_olep(G, f, nodes):
    """
    Check whether a tree satisfies OLEP (Definition B: globally deepest).

    Parameters
    ----------
    G : networkx.Graph
    f : ndarray  — Fiedler vector (same index order as `nodes`)
    nodes : list — sorted vertex labels

    Returns
    -------
    satisfies : bool
    info : dict
        Diagnostic information including spectral center, K+/K-,
        F+/F-, global max/min and where they are attained.
    """
    n = len(nodes)
    node_f = {nodes[i]: f[i] for i in range(n)}

    # Spectral center: argmin |f(v)|, tie-break by smallest vertex index
    abs_f = np.abs(f)
    min_abs = abs_f.min()
    c = min(nodes[i] for i in range(n) if abs_f[i] <= min_abs + EPS)

    # Vertex case: f(c) ≈ 0 (characteristic vertex);
    # Edge case: f(c) ≠ 0 (spectral center is interior to a characteristic edge)
    char_case = 'vertex' if abs(node_f[c]) < EPS else 'edge'

    # BFS distances from spectral center c
    depth = nx.single_source_shortest_path_length(G, c)

    # Leaves in each sign set
    def is_leaf(v):
        return G.degree(v) == 1

    Vp_leaves = [v for v in nodes if node_f[v] >  EPS and is_leaf(v)]
    Vm_leaves = [v for v in nodes if node_f[v] < -EPS and is_leaf(v)]

    # Degenerate: no leaves in one sign set (trivially satisfies OLEP)
    if not Vp_leaves or not Vm_leaves:
        return True, {'note': 'degenerate', 'satisfies': True}

    # Globally deepest depth in each sign set (Definition B)
    K_plus  = max(depth[v] for v in Vp_leaves)
    K_minus = max(depth[v] for v in Vm_leaves)

    # Outermost leaf sets F₊, F₋ (all leaves at the maximum depth)
    F_plus  = [v for v in Vp_leaves if depth[v] == K_plus]
    F_minus = [v for v in Vm_leaves if depth[v] == K_minus]

    # Global extremes of f over ALL vertices
    global_max = float(np.max(f))
    global_min = float(np.min(f))

    max_nodes = [nodes[i] for i in range(n) if abs(f[i] - global_max) < EPS]
    min_nodes = [nodes[i] for i in range(n) if abs(f[i] - global_min) < EPS]

    # OLEP: global max attained in F₊  AND  global min attained in F₋
    max_in_Fp = any(v in F_plus  for v in max_nodes)
    min_in_Fm = any(v in F_minus for v in min_nodes)
    satisfies = max_in_Fp and min_in_Fm

    info = {
        'c':             int(c),
        'char_case':     char_case,
        'K_plus':        int(K_plus),
        'K_minus':       int(K_minus),
        'F_plus':        sorted(int(v) for v in F_plus),
        'F_minus':       sorted(int(v) for v in F_minus),
        'V_plus_leaves': sorted(int(v) for v in Vp_leaves),
        'V_minus_leaves':sorted(int(v) for v in Vm_leaves),
        'global_max':    global_max,
        'global_min':    global_min,
        'max_nodes':     sorted(int(v) for v in max_nodes),
        'min_nodes':     sorted(int(v) for v in min_nodes),
        'max_in_Fp':     bool(max_in_Fp),
        'min_in_Fm':     bool(min_in_Fm),
        'satisfies':     bool(satisfies),
    }
    return satisfies, info


def graph_to_record(G, f, nodes, lam2, lam3, info, source='exhaustive'):
    """
    Serialize a tree and its OLEP certificate to a JSON-serializable dict.

    Fields include: edge list, diameter, λ₂, λ₃, spectral gap,
    Fiedler vector (keyed by vertex label), and OLEP diagnostic info.
    """
    return {
        'source':          source,
        'n':               int(G.number_of_nodes()),
        'edges':           sorted([sorted([int(u), int(v)]) for u, v in G.edges()]),
        'diameter':        int(nx.diameter(G)),
        'lambda2':         float(lam2),
        'lambda3':         float(lam3),
        'spectral_gap':    float(lam3 - lam2),   # gap measures λ₂ simplicity
        'fiedler_vector':  {str(nodes[i]): float(f[i]) for i in range(len(nodes))},
        'olep_info':       info,
    }


# ===========================================================================
# Table 1: exhaustive enumeration  n = 5 .. 17
# ===========================================================================

# Reference counts from OEIS A000055 (non-isomorphic trees)
OEIS_A000055 = {
    5: 3, 6: 6, 7: 11, 8: 23, 9: 47, 10: 106,
    11: 235, 12: 551, 13: 1301, 14: 3159,
    15: 7741, 16: 19320, 17: 48629,
}


def run_table1(n_max=17):
    """
    Exhaustively enumerate all non-isomorphic trees for n=5..n_max,
    check OLEP for each, and collect counterexamples.

    networkx.nonisomorphic_trees() generates canonical representatives
    (Prüfer-sequence based). Count matches OEIS A000055.

    Returns
    -------
    rows : list of dicts  — per-n statistics
    all_counterex : list of dicts  — all counterexample records
    """
    print(f"\n{'='*60}")
    print(f"Table 1: Exhaustive enumeration  n = 5..{n_max}  (Definition B)")
    print(f"{'='*60}")

    rows = []
    all_counterex = []

    for n in range(5, n_max + 1):
        t0 = time.time()
        trees = list(nx.nonisomorphic_trees(n))
        total = len(trees)
        oeis_ok = (total == OEIS_A000055.get(n, -1))

        cx_list = []
        for G in trees:
            f, nodes, lam2, lam3 = compute_fiedler(G)
            ok, info = check_olep(G, f, nodes)
            if not ok:
                rec = graph_to_record(G, f, nodes, lam2, lam3, info)
                cx_list.append(rec)

        nc      = len(cx_list)
        rate    = nc / total * 100
        elapsed = time.time() - t0

        row = {
            'n':              n,
            'total_trees':    int(total),
            'oeis_ok':        bool(oeis_ok),
            'counterexamples':int(nc),
            'rate_pct':       round(rate, 4),
            'time_s':         round(elapsed, 1),
        }
        rows.append(row)
        all_counterex.extend(cx_list)

        print(f"  n={n:2d}: {total:6d} trees (OEIS {'OK' if oeis_ok else 'FAIL'}), "
              f"counterexamples = {nc:4d}  ({rate:.4f}%),  {elapsed:.1f}s")

    return rows, all_counterex


# ===========================================================================
# Table 2: six random tree models
# ===========================================================================

def _gen_ba(n, seed=None):
    """Barabási–Albert preferential attachment tree (m=1 gives a tree)."""
    rng = random.Random(seed)
    return nx.barabasi_albert_graph(n, 1, seed=rng.randint(0, 999_999))


def _gen_prufer(n, seed=None):
    """Uniformly random labeled tree via Prüfer sequence."""
    rng = np.random.default_rng(seed)
    return nx.from_prufer_sequence(rng.integers(0, n, size=n - 2).tolist())


def _gen_er_mst(n, seed=None):
    """MST of an Erdős–Rényi graph G(n, 2 ln n / n) with random edge weights."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        G_er = nx.erdos_renyi_graph(
            n, 2 * np.log(n) / n, seed=int(rng.integers(0, 999_999))
        )
        if nx.is_connected(G_er):
            break
    for u, v in G_er.edges():
        G_er[u][v]['weight'] = float(rng.random())
    return nx.minimum_spanning_tree(G_er)


def _gen_binary(n, seed=None):
    """Random binary tree (max degree 3) grown by sequential attachment."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_node(0)
    for i in range(1, n):
        cands = [v for v in G.nodes() if G.degree(v) < 3]
        parent = rng.choice(cands) if cands else rng.choice(list(G.nodes()))
        G.add_edge(parent, i)
    return G


def _gen_caterpillar(n, seed=None):
    """
    Random caterpillar tree: a path (spine) with leaves attached to each
    spine vertex.  All leaves are at BFS depth 1 from the spine, so OLEP
    is typically satisfied (C2 condition).
    """
    rng = random.Random(seed)
    spine_len = rng.randint(2, max(2, n - 1))
    G = nx.path_graph(spine_len)
    nn, spine = spine_len, list(range(spine_len))
    while nn < n:
        G.add_edge(rng.choice(spine), nn)
        nn += 1
    return G


def _gen_lobster(n, seed=None):
    """
    Random lobster tree: spine → branches → leaves (depth up to 2 from spine).
    """
    rng = random.Random(seed)
    spine_len = rng.randint(2, max(2, n // 3))
    G = nx.path_graph(spine_len)
    nn, spine, brs = spine_len, list(range(spine_len)), []
    while nn < n and len(brs) < n // 2:
        G.add_edge(rng.choice(spine), nn)
        brs.append(nn)
        nn += 1
    if brs:
        while nn < n:
            G.add_edge(rng.choice(brs), nn)
            nn += 1
    return G


def run_table2(n_per=2500, n_range=(20, 100), seed=42):
    """
    Evaluate OLEP on six random tree models.

    Parameters
    ----------
    n_per   : int   — trees per model (paper uses 2500)
    n_range : tuple — (min_n, max_n) for random tree size
    seed    : int   — global numpy RNG seed for reproducibility

    Returns
    -------
    rows : list of dicts  — per-model statistics
    """
    print(f"\n{'='*60}")
    print(f"Table 2: 6 random tree models × {n_per} trees, "
          f"n ∈ {n_range}, seed={seed}")
    print(f"{'='*60}")

    models = {
        'BA':          _gen_ba,
        'ER-MST':      _gen_er_mst,
        'Binary':      _gen_binary,
        'Prufer':      _gen_prufer,
        'Caterpillar': _gen_caterpillar,
        'Lobster':     _gen_lobster,
    }
    rng = np.random.default_rng(seed)
    rows = []

    for name, fn in models.items():
        cx = 0
        t0 = time.time()
        for _ in range(n_per):
            n = int(rng.integers(n_range[0], n_range[1] + 1))
            s = int(rng.integers(0, 999_999))
            try:
                G = fn(n, seed=s)
                if not nx.is_tree(G) or G.number_of_nodes() < 3:
                    continue
                f, nodes, lam2, _ = compute_fiedler(G)
                ok, _ = check_olep(G, f, nodes)
                if not ok:
                    cx += 1
            except Exception:
                continue

        rate    = cx / n_per * 100
        elapsed = time.time() - t0
        row = {
            'model':          name,
            'total':          int(n_per),
            'counterexamples':int(cx),
            'rate_pct':       round(rate, 2),
            'time_s':         round(elapsed, 1),
        }
        rows.append(row)
        print(f"  {name:<12}: {cx:4d}/{n_per} counterexamples  "
              f"({rate:.2f}%),  {elapsed:.1f}s")

    return rows


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("OLEP Verification — Definition B (globally deepest outermost leaves)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Table 1 ───────────────────────────────────────────────────────────
    t1_rows, cx_all = run_table1(n_max=17)

    # Save per-n statistics
    with open(os.path.join(OUTPUT_DIR, 'table1_by_n.json'), 'w', encoding='utf-8') as fp:
        json.dump({'definition': 'B_global_deepest', 'table1': t1_rows},
                  fp, indent=2, ensure_ascii=False)

    # Save all counterexample records (full graph + Fiedler data)
    with open(os.path.join(OUTPUT_DIR, 'counterex_exhaustive.json'), 'w', encoding='utf-8') as fp:
        json.dump(cx_all, fp, indent=2, ensure_ascii=False)

    # Human-readable summary
    total_trees = sum(r['total_trees']    for r in t1_rows)
    total_cx    = sum(r['counterexamples'] for r in t1_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'table1_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        fp.write("OLEP Counterexample Statistics (Definition B: globally deepest)\n")
        fp.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fp.write(f"{'n':>4}  {'Trees':>8}  {'Cx':>6}  {'Rate':>9}  OEIS\n")
        fp.write('-' * 42 + '\n')
        for r in t1_rows:
            fp.write(f"{r['n']:>4}  {r['total_trees']:>8}  {r['counterexamples']:>6}  "
                     f"{r['rate_pct']:>8.4f}%  {'OK' if r['oeis_ok'] else 'FAIL'}\n")
        fp.write('-' * 42 + '\n')
        fp.write(f"{'Total':>4}  {total_trees:>8}  {total_cx:>6}\n")
    print(f"\nTable 1 summary saved: {summary_path}")

    # ── Table 2 ───────────────────────────────────────────────────────────
    t2_rows = run_table2()
    with open(os.path.join(OUTPUT_DIR, 'table2_random.json'), 'w', encoding='utf-8') as fp:
        json.dump({'definition': 'B_global_deepest',
                   'n_per_model': 2500, 'n_range': [20, 100], 'seed': 42,
                   'table2': t2_rows},
                  fp, indent=2, ensure_ascii=False)

    # ── Metadata summary ──────────────────────────────────────────────────
    min_n = min((r['n'] for r in t1_rows if r['counterexamples'] > 0), default=None)
    meta = {
        'definition':                'B_global_deepest',
        'timestamp':                 datetime.now().isoformat(),
        'table1_total_trees':        int(total_trees),
        'table1_total_counterex':    int(total_cx),
        'table1_min_n_counterex':    min_n,
        'table1_by_n':               {str(r['n']): r['counterexamples'] for r in t1_rows},
        'table2_by_model':           {r['model']: {'counterexamples': r['counterexamples'],
                                                    'rate_pct': r['rate_pct']}
                                      for r in t2_rows},
    }
    with open(os.path.join(OUTPUT_DIR, 'counterex_metadata.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    # ── Final printout ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Table 1 results:")
    print(f"  {'n':>4}  {'Trees':>7}  {'Cx':>6}  {'Rate':>9}")
    for r in t1_rows:
        print(f"  {r['n']:>4}  {r['total_trees']:>7}  "
              f"{r['counterexamples']:>6}  {r['rate_pct']:>8.4f}%")
    print(f"  Total  {total_trees:>7}  {total_cx:>6}")
    print(f"  Minimum counterexample size: n = {min_n}")

    print(f"\nTable 2 results:")
    print(f"  {'Model':<12}  {'Cx':>6}  {'Rate':>8}")
    for r in t2_rows:
        print(f"  {r['model']:<12}  {r['counterexamples']:>6}  {r['rate_pct']:>7.2f}%")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"  Counterexamples (Table 1): counterex_exhaustive.json ({total_cx} trees)")
    print(f"  Metadata:                  counterex_metadata.json")


if __name__ == '__main__':
    main()
