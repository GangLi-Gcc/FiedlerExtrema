# FiedlerExtrema: OLEP Verification for Trees

Code and data accompanying the paper:

> **"Outer-Leaf Extremal Property of Fiedler Vectors on Trees"**  
> *Submitted to Linear Algebra and Its Applications (LAA)*

---

## What is OLEP?

The **Outer-Leaf Extremal Property (OLEP)** states that the maximum and
minimum entries of the Fiedler vector (eigenvector for the algebraic
connectivity λ₂ of the graph Laplacian) are attained at the *outermost
leaves* of a BFS-tree rooted at the spectral center.

Formally, let:
- `c = argmin |f(v)|` — the *spectral center* (tie-break: smallest vertex index)
- `V₊ = {v : f(v) > 0}`, `V₋ = {v : f(v) < 0}` — sign partition
- `K₊ = max BFS-depth(c, v)` over all leaves in V₊
- `F₊ = {v ∈ leaves(V₊) : depth(c,v) = K₊}` — outermost positive leaves
- `F₋` defined symmetrically

**OLEP** holds if: `max f ∈ F₊`  AND  `min f ∈ F₋`.

---

## Repository Structure

```
FiedlerExtrema-/
├── src/
│   ├── verify_olep.py          # Main OLEP verification script (Tables 1 & 2)
│   └── plot_counterexamples.py # Figure: two minimal counterexamples (n=13)
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
    ├── olep_counterexamples.pdf        # Figure 2 from the paper (publication quality)
    └── olep_counterexamples.png        # PNG preview
```

---

## Key Results

### Table 1 — Exhaustive Enumeration (n = 5..17)

| n | Trees | Counterexamples | Rate |
|---|-------|-----------------|------|
| 5–12 | 4,680 | 0 | 0.00% |
| 13 | 1,301 | 2 | 0.15% |
| 14 | 3,159 | 12 | 0.38% |
| 15 | 7,741 | 47 | 0.61% |
| 16 | 19,320 | 176 | 0.91% |
| 17 | 48,629 | 593 | 1.22% |
| **Total** | **81,132** | **830** | **1.02%** |

- **Minimum counterexample size: n = 13**
- All counterexamples exhibit the *collective pull effect* (see below)

### Table 2 — Random Tree Models (2500 trees each, n ∈ [20,100])

| Model | Counterexamples | Rate |
|-------|-----------------|------|
| BA (Barabási–Albert) | 289 | 11.56% |
| ER-MST | 219 | 8.76% |
| Binary | 226 | 9.04% |
| Prüfer (uniform random) | 189 | 7.56% |
| Caterpillar | 0 | 0.00% |
| Lobster | 4 | 0.16% |

Caterpillar trees always satisfy OLEP (proved as condition C2 in the paper).

---

## Counterexample Mechanism: Collective Pull Effect

All 830 counterexamples share the same structural pattern:

```
       c (spectral center)
      /|\
    arm  arm  hub
               |  \  \  \
              L   L   L   L   ← 4 leaf children at depth 2 (V₋)
```

A hub node at BFS depth 1 has ≥ 4 leaf children in V₋.
Their combined weight in the Laplacian eigenvector equation pulls
`f(hub)` to a more negative value than the leaf at the deepest depth,
so the global minimum lands at depth-2 leaves, violating OLEP.

---

## Main Theoretical Results

| Condition | Statement | Status |
|-----------|-----------|--------|
| **C5** (main) | All trees with diameter D ≤ 4 satisfy OLEP (under λ₂ simple) | Proved |
| C1 | Sign-reversing automorphism + equal-depth leaves | Proved |
| C2 | All leaves are outermost by structure (e.g., caterpillars) | Proved |
| C3 | Pendant trees (leaves only at max depth) | Conjecture |
| C4b | Hub trees (hub + one chain arm) | Conjecture |

The paper extends Lederman & Steinerberger (2024, LAA 703) who proved
Fiedler extremals lie on leaves for D ≤ 3; this work raises the bound
to D ≤ 4 and strengthens the conclusion to *outermost* leaves.

---

## Installation

```bash
pip install numpy scipy networkx matplotlib
```

Python ≥ 3.9 required.

---

## Usage

### Run OLEP verification (reproduces Tables 1 & 2)

```bash
python src/verify_olep.py
```

Output is saved to `results/final_defB/`. Running time: ~10 minutes
(dominated by n=17 exhaustive enumeration of 48,629 trees).

### Generate counterexample figure

```bash
python src/plot_counterexamples.py
```

Saves `olep_counterexamples.pdf` and `olep_counterexamples.png` in the
current directory.

---

## Counterexample Data Format

Each entry in `counterex_exhaustive.json` is a JSON object:

```json
{
  "n": 13,
  "edges": [[0,1], [0,5], "..."],
  "diameter": 7,
  "lambda2": 0.1716,
  "lambda3": 0.3087,
  "spectral_gap": 0.1371,
  "fiedler_vector": {"0": 0.0, "1": 0.123, "...": "..."},
  "olep_info": {
    "c": 0,
    "char_case": "vertex",
    "K_plus": 4,  "K_minus": 3,
    "F_plus": [4], "F_minus": [7],
    "global_max": 0.512, "global_min": -0.491,
    "max_nodes": [4], "min_nodes": [9],
    "max_in_Fp": true, "min_in_Fm": false,
    "satisfies": false
  }
}
```

The field `satisfies: false` with `min_in_Fm: false` certifies the OLEP violation.

---

## Citation

If you use this code or data, please cite the accompanying paper
(citation details to be updated upon acceptance).

---

## License

MIT License. See [LICENSE](LICENSE).
