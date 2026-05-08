# FiedlerExtrema: OLEP Verification for Trees

Code and data accompanying the paper:

> **"The Outer-Leaf Extremal Property of the Fiedler Vector on Trees"**
> Gang Li (Guangzhou College of Commerce)
> Submitted to *, 2026.

---

## What is OLEP?

The **Outer-Leaf Extremal Property (OLEP)** states that the maximum and
minimum entries of the Fiedler vector (the eigenvector for the algebraic
connectivity λ₂ of the graph Laplacian) are attained at the *outermost
leaves* of a BFS-tree rooted at the spectral center.

Formally, under the Working Assumption that λ₂ is simple, let:
- `c` = the *spectral center* (the unique zero entry of `f` if any vertex
  of degree ≥ 2 has `f(v) = 0`; otherwise the smaller-|f| endpoint of the
  unique sign-change edge).
- `V₊ = {v : f(v) > 0}`, `V₋ = {v : f(v) < 0}` — sign supports.
- `K₊ = max BFS-depth(c, v)` over all `v ∈ V₊`.
- `F₊ = {v ∈ V₊ : depth(c,v) = K₊, deg(v) = 1}` — outermost positive leaves.
- `K₋`, `F₋` defined symmetrically.

**OLEP** holds if `max f = max_{F₊} f` AND `min f = min_{F₋} f`.

---

## Repository Structure

```
FiedlerExtrema/
├── src/
│   ├── verify_olep.py          # Main OLEP verification script (Tables 1 & 2)
│   ├── plot_counterexamples.py # Figure 2: minimal counterexamples (n=13)
│   └── visualize_bfs_tree.py   # Figure 1: Fiedler-BFS-tree layout
│
├── results/
│   └── final_defB/
│       ├── table1_summary.txt          # Human-readable Table 1 summary
│       ├── table1_by_n.json            # Per-n statistics (n=5..17)
│       ├── table2_random.json          # Table 2: six random tree models
│       ├── counterex_exhaustive.json   # All 830 counterexamples (full data)
│       └── counterex_metadata.json     # Summary metadata
│
└── figures/
    ├── fiedler_bfs_tree.pdf            # Figure 1: Fiedler-BFS-tree (n=15)
    ├── olep_counterexamples.pdf        # Figure 2: counterexamples CE1, CE2
    └── olep_counterexamples.png        # PNG preview
```

---

## Key Results

### Table 1 — Exhaustive Enumeration (n = 5..17, simple λ₂)

| n | Generated | Excluded | Eligible | Counterex. | Rate |
|---|-----------|----------|----------|------------|------|
| 5–12 | 982 | 7 | 975 | 0 | 0.00 % |
| 13 | 1,301 | 1 | 1,300 | 2 | 0.15 % |
| 14 | 3,159 | 2 | 3,157 | 12 | 0.38 % |
| 15 | 7,741 | 4 | 7,737 | 47 | 0.61 % |
| 16 | 19,320 | 7 | 19,313 | 176 | 0.91 % |
| 17 | 48,629 | 12 | 48,617 | 593 | 1.22 % |
| **Total** | **81,132** | **33** | **81,099** | **830** | **1.02 %** |

- Smallest counterexample size: **n = 13**.
- All 830 counterexamples exhibit the *collective pull effect*.
- Of the 830 counterexamples, 8 have D = 5 and 822 have D ≥ 6 — consistent
  with Theorem 5.7 (D ≤ 4 implies OLEP).

### Table 2 — Random Tree Models (2,500 trees each, n ∈ [20, 100])

| Model | Counterex. | Rate | Guarantee |
|-------|------------|------|-----------|
| Barabási–Albert (BA) | 289 | 11.56 % | None |
| Erdős–Rényi MST | 219 | 8.76 % | None |
| Binary | 226 | 9.04 % | None |
| Prüfer (uniform random) | 189 | 7.56 % | None |
| Caterpillar | **0** | **0.00 %** | Yes (Proposition 5.5) |
| Lobster | 4 | 0.16 % | None |

Caterpillar trees always satisfy OLEP (Proposition 5.5).
Trees with diameter D ≤ 4 always satisfy OLEP (Theorem 5.7).

---

## Counterexample Mechanism: Collective Pull Effect

All 830 enumerated counterexamples share the same structural pattern:
a hub `h ∈ V₋` at BFS depth 1 from the spectral center, with `deg(h) ≥ 5`
and at least 4 leaf children at depth 2, plus one chain arm reaching
depth 3 in `V₋`.

The leaf eigenvalue equation `f(ℓ) = f(h)/(1−λ₂)` makes the depth-2 hub
leaves *more negative* than the deeper chain-arm leaf, violating OLEP.

The eigenvector criterion (Proposition 6.2) and the explicit hub-degree
threshold (Corollary 6.3) characterise all observed failures.

---

## Main Theoretical Results (paper v32)

| Condition | Statement | Status |
|-----------|-----------|--------|
| **Theorem 5.7 (main)** | All trees with diameter D ≤ 4 satisfy OLEP. | Proved (D = 4 sharp) |
| **Proposition 5.5** | All caterpillar trees satisfy OLEP, regardless of diameter. | Proved |
| Theorem 5.1 (C1) | Sign-reversing automorphism + equal-depth leaves ⇒ OLEP. | Proved |
| Remark (C2) | All `V₊`-leaves at depth `K₊` and all `V₋`-leaves at depth `K₋` ⇒ OLEP. | Trivial |
| Conjecture 5.4 (C3) | Layer entropy `H_k ≤ 1` for all k ⇒ OLEP. | Open |
| Proposition 7.1 | OLEP strictly strengthens GP19 Lemma 8(a); independent of LS24. | Proved |

The paper extends the leaf-extremal results of Gernandt–Padé (2019, LAA 570)
and is complementary to the augmented-path / hitting-time framework of
Lederman–Steinerberger (2024, LAA 703); see Proposition 7.1 in the paper.

---

## Installation

```bash
pip install numpy scipy networkx matplotlib
```

Python ≥ 3.9 required.

---

## Usage

### Reproduce Tables 1 & 2

```bash
python src/verify_olep.py
```

Output is saved to `results/final_defB/`. Running time: ~10 minutes
(dominated by n = 17 exhaustive enumeration of 48,629 trees).
Random-model seed is fixed at 42 for reproducibility.

### Regenerate Figure 1 (Fiedler-BFS-tree, n = 15)

```bash
python src/visualize_bfs_tree.py
```

### Regenerate Figure 2 (minimal counterexamples CE1, CE2)

```bash
python src/plot_counterexamples.py
```

Saves `olep_counterexamples.pdf` / `.png` in the current directory.

---

## Counterexample Data Format

Each entry in `results/final_defB/counterex_exhaustive.json` is a JSON object:

```json
{
  "n": 13,
  "edges": [[0, 1], [0, 5], "..."],
  "diameter": 7,
  "lambda2": 0.1338,
  "lambda3": 0.3087,
  "spectral_gap": 0.1750,
  "fiedler_vector": {"0": -0.0481, "1": 0.062, "...": "..."},
  "olep_info": {
    "c": 0,
    "char_case": "edge",
    "K_plus": 4,  "K_minus": 3,
    "F_plus": [4], "F_minus": [7],
    "global_max": 0.5623, "global_min": -0.2236,
    "max_nodes": [4], "min_nodes": [9, 10, 11, 12],
    "max_in_Fp": true, "min_in_Fm": false,
    "satisfies": false
  }
}
```

The flag `satisfies: false` together with `min_in_Fm: false` certifies
the OLEP violation.

---

## Citation

If you use this code or data, please cite:

```
@article{Li2026OLEP,
  author  = {Gang Li},
  title   = {The Outer-Leaf Extremal Property of the Fiedler Vector on Trees},
  journal = {},
  year    = {2026},
  note    = {Submitted}
}
```

(Citation details to be updated upon acceptance.)

---

## License

MIT License. See [LICENSE](LICENSE).
