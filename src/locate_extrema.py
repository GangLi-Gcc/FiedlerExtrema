#!/usr/bin/env python3
"""
locate_extrema.py
=================

Path-B algorithm for locating the spectral center c and the Fiedler
extremum vertices v+, v- on a tree T, *without* computing the Fiedler
vector of T.

Three stages, all based on scalar Dirichlet eigenvalues of subtree
Laplacians:

  Stage 1 -- Combinatorial candidate set
      K(T) := { spine vertices with deg >= 2 }                 if T is a caterpillar
           := { v in V : dist(v, C(T)) <= 2, deg(v) >= 2 }     otherwise
      where C(T) is the graph-theoretic tree center.

  Stage 2 -- Dirichlet test (Bıyıkoğlu-Leydold-Stadler 2007 Thm 4.5)
      For each v in K(T):
          mu_i(v) = lambda_min^Dirichlet of the i-th component of T - {v}
                    (computed via scipy.sparse.linalg.eigsh, k=1)
          mu(v)   = min_i mu_i(v)
      Pick c = argmax_{v in K(T)} mu(v).

  Stage 3 -- Extremum localisation
      Order components of T - {c} by ascending mu_i.
      For each of the two smallest-mu arms:
          compute the (positive) Dirichlet eigenvector g_i,
          F_i := { leaves at maximum BFS-depth from c in this arm },
          u_i := argmax_{w in F_i} g_i(w).
      Return (c, {u_1, u_2}).

Provable correctness (paper Theorem 8.5): in the characteristic-vertex
case, this algorithm returns the correct (c, {v+, v-}) for all trees of
diameter at most 4 and for all caterpillars.
Empirical accuracy on the full 80,910-tree corpus is 99.965% for c and
99.642% for the extremum pair.

Usage as a script: runs evaluation on (1) the 830 OLEP counterexamples,
(2) random Prüfer trees, (3) Barabási-Albert trees, (4) caterpillars,
(5) random D <= 4 trees, and prints a summary.
"""

import os
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EPS = 1e-9
GAP_TOL = 1e-8


# =====================================================================
# Ground truth (used only for evaluation; not used by the algorithm)
# =====================================================================

def fiedler_truth(T):
    """Compute (c, char_case, lambda_2, f) via full eigendecomposition.
    Returns None if lambda_2 is not simple. Used for ground truth only."""
    L = nx.laplacian_matrix(T).toarray().astype(float)
    w, v = eigh(L)
    if w[2] - w[1] <= GAP_TOL:
        return None
    nodes = list(T.nodes())
    f = dict(zip(nodes, v[:, 1]))
    lam2 = float(w[1])
    zero_vs = [u for u in nodes if abs(f[u]) < EPS and T.degree(u) >= 2]
    if zero_vs:
        c = zero_vs[0]
        char = "vertex"
    else:
        c = None
        char = None
        for u, w_ in T.edges():
            if f[u] * f[w_] < 0:
                c = u if abs(f[u]) <= abs(f[w_]) else w_
                char = "edge"
                break
    return {
        "c": c,
        "char": char,
        "lam2": lam2,
        "f": f,
        "v_max": max(f, key=lambda x: f[x]),
        "v_min": min(f, key=lambda x: f[x]),
    }


# =====================================================================
# Stage 1: combinatorial candidate set
# =====================================================================

def is_caterpillar(T):
    """Return True iff every non-leaf vertex of T lies on a path."""
    non_leaves = [v for v in T.nodes() if T.degree(v) > 1]
    if len(non_leaves) <= 1:
        return True
    H = T.subgraph(non_leaves)
    if not nx.is_connected(H):
        return False
    return all(d <= 2 for _, d in H.degree())


def candidate_set(T):
    """K(T) per the paper's caterpillar-aware definition."""
    if is_caterpillar(T):
        spine = {v for v in T.nodes() if T.degree(v) > 1}
        return {v for v in spine if T.degree(v) >= 2}
    centers = set(nx.center(T))
    cands = set(centers)
    for c in centers:
        cands.update(T.neighbors(c))
    two_hop = set()
    for v in cands:
        two_hop.update(T.neighbors(v))
    cands.update(two_hop)
    return {v for v in cands if T.degree(v) >= 2}


# =====================================================================
# Stage 2: Dirichlet eigenvalues per component
# =====================================================================

def dirichlet_blocks(T, c, L_csr=None):
    """For each component T_i of T - {c}, compute the smallest Dirichlet
    eigenvalue mu_i and its (positive) Perron eigenvector g_i.
    Returns list of (component_set, mu_i, g_i_dict)."""
    if L_csr is None:
        L_csr = nx.laplacian_matrix(T).astype(float).tocsr()
    nodes = list(T.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    ic = idx[c]
    keep_idx = [i for i in range(len(nodes)) if i != ic]
    L_red = L_csr[keep_idx, :][:, keep_idx]
    pos_to_node = [nodes[i] for i in keep_idx]

    Tminusc = T.copy()
    Tminusc.remove_node(c)
    out = []
    for comp in nx.connected_components(Tminusc):
        comp_set = set(comp)
        block_idx = [i for i, nd in enumerate(pos_to_node) if nd in comp_set]
        block_nodes = [pos_to_node[i] for i in block_idx]
        M = L_red[block_idx, :][:, block_idx]
        if M.shape[0] == 1:
            mu = float(M[0, 0])
            ev = {block_nodes[0]: 1.0}
        else:
            try:
                w, V = eigsh(M, k=1, which="SA", maxiter=5000, tol=1e-10)
                mu = float(w[0])
                vec = V[:, 0]
            except Exception:
                Md = M.toarray()
                w, V = eigh(Md, subset_by_index=[0, 0])
                mu = float(w[0])
                vec = V[:, 0]
            if vec.sum() < 0:
                vec = -vec
            ev = dict(zip(block_nodes, vec.tolist()))
        out.append((comp_set, mu, ev))
    return out


def locate_c(T):
    """Stage 1 + Stage 2: pick c = argmax of mu(v) on K(T)."""
    L_csr = nx.laplacian_matrix(T).astype(float).tocsr()
    cands = candidate_set(T)
    if not cands:
        return list(T.nodes())[0]
    best = None
    best_mu = -float("inf")
    for v in cands:
        blocks = dirichlet_blocks(T, v, L_csr=L_csr)
        mu = min(b[1] for b in blocks)
        if mu > best_mu or (mu == best_mu and (best is None or v < best)):
            best_mu = mu
            best = v
    return best


# =====================================================================
# Stage 3: extremum localisation
# =====================================================================

def locate_extrema(T):
    """Full Path-B pipeline.
    Returns (c, v_a, v_b) where {v_a, v_b} is the predicted unordered
    extremum pair (one of them is v+, the other v-)."""
    c = locate_c(T)
    blocks = dirichlet_blocks(T, c)
    if len(blocks) < 2:
        # Pathological case (deg(c) = 1, shouldn't happen since deg(c) >= 2)
        return c, None, None
    blocks.sort(key=lambda b: b[1])
    arm0, arm1 = blocks[0], blocks[1]

    depths = nx.single_source_shortest_path_length(T, c)

    def deepest_leaf_argmax(comp_set, ev_dict):
        leaves = [v for v in comp_set if T.degree(v) == 1]
        if not leaves:
            leaves = list(comp_set)
        max_d = max(depths[v] for v in leaves)
        F = [v for v in leaves if depths[v] == max_d]
        return max(F, key=lambda v: ev_dict.get(v, -float("inf")))

    u0 = deepest_leaf_argmax(arm0[0], arm0[2])
    u1 = deepest_leaf_argmax(arm1[0], arm1[2])
    return c, u0, u1


# =====================================================================
# Evaluation against ground truth
# =====================================================================

def evaluate(T, gt):
    """Compare locate_extrema(T) against ground truth gt = fiedler_truth(T).
    Return a dict of correctness flags."""
    c_pred, va, vb = locate_extrema(T)
    f = gt["f"]
    c_true = gt["c"]
    char = gt["char"]
    vmax_true = gt["v_max"]
    vmin_true = gt["v_min"]

    c_ok = (c_pred == c_true)
    c_ok_or_other = c_ok
    if not c_ok and char == "edge" and T.has_edge(c_pred, c_true):
        if f[c_pred] * f[c_true] < 0:
            c_ok_or_other = True

    fmax_val = max(f.values())
    fmin_val = min(f.values())
    vmax_set = {v for v, fv in f.items() if abs(fv - fmax_val) < 1e-9}
    vmin_set = {v for v, fv in f.items() if abs(fv - fmin_val) < 1e-9}

    pair_pred = {va, vb}
    ext_ok = ((va in vmax_set and vb in vmin_set) or
              (va in vmin_set and vb in vmax_set))
    vmax_hit = bool(pair_pred & vmax_set)
    vmin_hit = bool(pair_pred & vmin_set)

    return {
        "n": T.number_of_nodes(),
        "char": char,
        "c_ok": c_ok,
        "c_ok_or_other": c_ok_or_other,
        "ext_ok": ext_ok,
        "vmax_hit": vmax_hit,
        "vmin_hit": vmin_hit,
        "c_true": c_true,
        "c_pred": c_pred,
    }


# =====================================================================
# Main: standalone evaluation script
# =====================================================================

def main():
    print("=" * 60)
    print("LOCATE-EXTREMA: Path-B algorithm evaluation")
    print("=" * 60)

    rows = []

    # Set 1: 830 OLEP counterexamples
    ce_path = os.path.join(REPO_ROOT, "results", "final_defB",
                           "counterex_exhaustive.json")
    if os.path.exists(ce_path):
        with open(ce_path, encoding="utf-8") as fp:
            ces = json.load(fp)
        print(f"\n[Set 1] {len(ces)} OLEP counterexamples")
        s = defaultdict(int)
        tot = 0
        for ce in ces:
            T = nx.Graph()
            T.add_edges_from(ce["edges"])
            gt = fiedler_truth(T)
            if gt is None:
                continue
            r = evaluate(T, gt)
            r["source"] = "counterexample"
            rows.append(r)
            tot += 1
            for k in ("c_ok", "c_ok_or_other", "ext_ok", "vmax_hit", "vmin_hit"):
                if r[k]:
                    s[k] += 1
        print(f"  c_or_other_endpoint: {s['c_ok_or_other']/tot*100:.1f}%")
        print(f"  vmax found anywhere: {s['vmax_hit']/tot*100:.1f}%")
        print(f"  vmin found anywhere: {s['vmin_hit']/tot*100:.1f}%")
    else:
        print(f"\n[Set 1] skipped (no {ce_path})")

    # Set 2: random uniform Prüfer trees
    rng = np.random.default_rng(2026)
    print("\n[Set 2] Random uniform trees n=10..20 (200 each)")
    for n in range(10, 21):
        s = defaultdict(int)
        tot = 0
        for k in range(200):
            prufer = rng.integers(0, n, size=n - 2).tolist()
            T = nx.from_prufer_sequence(prufer)
            gt = fiedler_truth(T)
            if gt is None:
                continue
            r = evaluate(T, gt)
            r["source"] = "random"
            rows.append(r)
            tot += 1
            for kk in ("c_ok", "c_ok_or_other", "ext_ok", "vmax_hit", "vmin_hit"):
                if r[kk]:
                    s[kk] += 1
        print(f"  n={n:2d}: c-or-oth={s['c_ok_or_other']/tot*100:5.1f}%, "
              f"ext_pair={s['ext_ok']/tot*100:5.1f}%")

    # Set 3: Barabási-Albert trees
    print("\n[Set 3] Barabási-Albert trees")
    for n in (20, 30, 50, 75, 100):
        s = defaultdict(int)
        tot = 0
        for k in range(50):
            T = nx.barabasi_albert_graph(n, 1, seed=int(rng.integers(0, 1 << 30)))
            gt = fiedler_truth(T)
            if gt is None:
                continue
            r = evaluate(T, gt)
            r["source"] = "BA"
            rows.append(r)
            tot += 1
            for kk in ("c_ok", "c_ok_or_other", "ext_ok", "vmax_hit", "vmin_hit"):
                if r[kk]:
                    s[kk] += 1
        print(f"  n={n:3d}: c-or-oth={s['c_ok_or_other']/tot*100:5.1f}%, "
              f"ext_pair={s['ext_ok']/tot*100:5.1f}%")

    # Set 4: random caterpillars
    print("\n[Set 4] Random caterpillars (spine 5..20, pendants 0..3)")
    s = defaultdict(int)
    tot = 0
    for spine_len in range(5, 21):
        for trial in range(15):
            T = nx.path_graph(spine_len)
            nid = spine_len
            for v in range(spine_len):
                k = int(rng.integers(0, 4))
                for _ in range(k):
                    T.add_edge(v, nid)
                    nid += 1
            gt = fiedler_truth(T)
            if gt is None:
                continue
            r = evaluate(T, gt)
            r["source"] = "caterpillar"
            rows.append(r)
            tot += 1
            for kk in ("c_ok", "c_ok_or_other", "ext_ok", "vmax_hit", "vmin_hit"):
                if r[kk]:
                    s[kk] += 1
    print(f"  caterpillars: c-or-oth={s['c_ok_or_other']/tot*100:6.2f}%, "
          f"ext_pair={s['ext_ok']/tot*100:6.2f}%")

    # Set 5: random trees with D <= 4
    print("\n[Set 5] Random trees with D <= 4 (500)")
    s = defaultdict(int)
    tot = 0
    samp = 0
    while samp < 500:
        n = int(rng.integers(5, 30))
        prufer = rng.integers(0, n, size=n - 2).tolist()
        T = nx.from_prufer_sequence(prufer)
        if nx.diameter(T) > 4:
            continue
        gt = fiedler_truth(T)
        if gt is None:
            continue
        r = evaluate(T, gt)
        r["source"] = "D<=4"
        rows.append(r)
        tot += 1
        samp += 1
        for kk in ("c_ok", "c_ok_or_other", "ext_ok", "vmax_hit", "vmin_hit"):
            if r[kk]:
                s[kk] += 1
    print(f"  D<=4: c-or-oth={s['c_ok_or_other']/tot*100:6.2f}%, "
          f"ext_pair={s['ext_ok']/tot*100:6.2f}%")

    # Save report
    out_dir = os.path.join(REPO_ROOT, "results", "locate_extrema")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fp:
        json.dump(rows, fp)

    print("\n" + "=" * 60)
    print("Aggregate by characteristic case")
    print("=" * 60)
    for case in ("vertex", "edge"):
        sub = [r for r in rows if r["char"] == case]
        if not sub:
            continue
        n = len(sub)
        print(f"  {case:8s} n={n:5d}: "
              f"c-or-oth={sum(r['c_ok_or_other'] for r in sub)/n*100:5.2f}%  "
              f"ext_pair={sum(r['ext_ok'] for r in sub)/n*100:5.2f}%  "
              f"vmax={sum(r['vmax_hit'] for r in sub)/n*100:5.2f}%  "
              f"vmin={sum(r['vmin_hit'] for r in sub)/n*100:5.2f}%")
    print(f"\n  Report saved to: {out_dir}/report.json")


if __name__ == "__main__":
    main()
