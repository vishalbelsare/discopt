# Closing the Gap with BARON: Task Plan

**Goal:** Bring discopt's general MINLP solving capability to within 1.5x of BARON on standard benchmarks, while retaining the batch/diff/JIT advantages.

**Current state (Phase 3):** 48/73 solved, 9/27 incorrect (nonconvex local minima), no BARON head-to-head data.

---

## Phase A: Wire Existing Code (2–4 weeks)

Low-hanging fruit — code is implemented and tested but not called from `solver.py`.

### A1. Wire feasibility pump into B&B loop
- **File:** `solver.py` (call from B&B loop), `primal_heuristics.py` (already implemented)
- **What:** After root multi-start, run feasibility pump on the best relaxation solution to find integer-feasible incumbents faster. Also run periodically (every N nodes) when no incumbent exists.
- **Impact:** Better incumbents earlier → more pruning → fewer nodes
- **Tests:** Verify on MINLP instances that were previously timing out

### A2. Enable GNN-driven branching (not just advisory)
- **File:** `branching.rs` — add `set_branch_hint(node_id, var_index)` method
- **File:** `solver.py` lines 1028–1042 — pass GNN scores to Rust
- **What:** The GNN branching model (T29) computes scores but the Rust TreeManager ignores them. Add a mechanism to accept Python-side branching hints. Rust falls back to most-fractional if no hint.
- **Impact:** 10–20% node reduction on problems where GNN was trained
- **Tests:** Verify GNN-branched tree has ≤ node count of most-fractional on training instances

### A3. Enable OA cuts for convex constraints
- **File:** `solver.py` line 736 — `_oa_enabled = False`
- **What:** Currently OA is globally disabled because the loop handles nonconvex problems. But OA cuts *are* valid for individual convex constraints, even in a nonconvex problem. Detect per-constraint convexity (Hessian PSD check on the constraint function) and enable OA only for those.
- **Impact:** Tighter relaxations on mixed convex/nonconvex problems
- **Tests:** Add convex-constraint detection tests; verify OA cuts valid on known-convex constraints

### A4. Wire McCormick relaxation bounds by default
- **File:** `solver.py` — change `mccormick_bounds` default from `"none"` to `"auto"`
- **What:** McCormick midpoint bounds are implemented and tested (batch evaluator, relaxation compiler) but disabled by default. Enabling them gives valid lower bounds at every node, which is critical for pruning nonconvex problems.
- **Impact:** This is the single biggest correctness fix — addresses the 9/27 incorrect results by providing valid lower bounds for nonconvex B&B.
- **Tests:** Re-run correctness suite; expect incorrect count to drop significantly

---

## Phase B: Reliability Branching (4–6 weeks)

BARON's branching is far more sophisticated than most-fractional. Reliability branching is the sweet spot: nearly as good as strong branching, much cheaper.

### B1. Pseudocost tracking in Rust TreeManager
- **File:** `crates/discopt-core/src/bnb/` — new `pseudocosts.rs`
- **What:** After each branching decision, record the bound change per unit of variable change (pseudocost). Maintain running averages per variable. Expose via PyO3.
- **Impact:** Foundation for reliability branching
- **Tests:** Unit tests for pseudocost update, averaging, initialization

### B2. Reliability branching in Rust
- **File:** `branching.rs` — new `select_branch_variable_reliability()`
- **What:** For variables with < `η_rel` (e.g., 8) pseudocost observations, do strong branching (solve child LPs). Otherwise use pseudocost scores. Hybrid: start with strong branching, switch to pseudocost as tree matures.
- **Impact:** 20–40% node reduction vs. most-fractional (well-established in literature)
- **Tests:** Benchmark node counts on 10+ instances vs. most-fractional
- **Depends on:** B1

### B3. Python-side strong branching via LP solves
- **File:** `solver.py` — add `_strong_branch()` helper
- **What:** For the strong branching phase of reliability branching, solve LP relaxations (not full NLPs) for candidate child nodes. Use HiGHS with warm-start. Much cheaper than NLP solves.
- **Impact:** Makes strong branching practical (seconds, not minutes)
- **Tests:** Verify LP bound matches full NLP bound on LP-relaxable problems

---

## Phase C: Incumbent-Based Bound Tightening (3–4 weeks)

BARON aggressively tightens bounds using the incumbent objective value. discopt's OBBT exists but doesn't exploit incumbents.

### C1. Objective-based variable bound tightening
- **File:** `obbt.py` — add `incumbent_cutoff` parameter
- **What:** When we have an incumbent with objective value z*, add the constraint `f(x) ≤ z*` to the OBBT LPs. This often dramatically tightens variable bounds because many variables can't reach their original bounds without violating the objective cutoff.
- **Impact:** 30–50% bound tightening on variables, leading to tighter relaxations and more pruning
- **Tests:** Verify bounds tighten on instances where incumbent is known

### C2. Periodic OBBT in B&B loop
- **File:** `solver.py` — add OBBT call every N nodes when gap improves
- **What:** Currently OBBT only runs at root (if at all). Re-run OBBT periodically when a new incumbent is found, using the incumbent cutoff from C1. Only worth doing when the gap improved significantly (e.g., >10% relative improvement).
- **Impact:** Progressive bound tightening throughout the tree
- **Depends on:** C1
- **Tests:** Verify periodic OBBT reduces total node count

### C3. FBBT with incumbent cutoff in Rust
- **File:** `crates/discopt-core/src/presolve/fbbt.rs`
- **What:** Extend FBBT to accept an objective upper bound. Propagate `f(x) ≤ z*` through the expression DAG to tighten variable intervals. Much cheaper than OBBT (no LP solves), can run at every node.
- **Impact:** Cheap per-node tightening, stacks with OBBT
- **Tests:** Verify tighter bounds than standard FBBT on instances with known incumbents

---

## Phase D: Specialized Envelopes (6–8 weeks)

Close the relaxation tightness gap. BARON has dozens of hand-tuned envelopes; discopt has 3 in `envelopes.py` plus compositional McCormick.

### D1. Exponential/logarithm envelopes
- **File:** `python/discopt/_jax/envelopes.py`
- **What:** Tight convex/concave envelopes for `exp(x)` and `log(x)` (secant + tangent line construction). These are exact (not McCormick compositions) and significantly tighter.
- **Impact:** Major gap reduction on problems with exp/log (common in chemical engineering, economics)
- **Tests:** Verify tightness vs. compositional McCormick on [a,b] intervals

### D2. Trigonometric envelopes (sin, cos)
- **File:** `envelopes.py`
- **What:** Piecewise-linear envelopes for sin/cos on bounded intervals. Use tangent lines at inflection points + secant construction.
- **Impact:** Required for engineering design problems with periodic functions
- **Tests:** Verify valid relaxation on multiple interval widths

### D3. Power/signomial envelopes
- **File:** `envelopes.py`
- **What:** Tight envelopes for `x^p` (convex for p≥1 or p≤0, concave for 0<p<1) and general signomial terms `∏ x_i^{a_i}`. Use known convex/concave envelope constructions from the literature (Liberti & Pantelides, 2003).
- **Impact:** Handles polynomial and signomial optimization (pooling, blending)
- **Tests:** Compare gap vs. McCormick on signomial test cases

### D4. Integrate envelopes into relaxation compiler
- **File:** `relaxation_compiler.py`
- **What:** When compiling relaxations, check if a specialized envelope exists for the operation before falling back to compositional McCormick. Add a `mode="envelope"` or extend `"standard"` to prefer envelopes.
- **Depends on:** D1–D3
- **Tests:** Verify relaxation compiler picks envelopes over McCormick when available

---

## Phase E: Convexity Detection & Adaptive Dispatch (4–6 weeks)

BARON detects convex substructures and avoids unnecessary relaxation overhead.

### E1. Convexity detector for expression DAG
- **File:** new `python/discopt/_jax/convexity.py`
- **What:** Walk the expression DAG and classify each subexpression as convex, concave, affine, or unknown using composition rules (e.g., convex ∘ affine = convex, sum of convex = convex). Cache results.
- **Impact:** Foundation for E2 and E3
- **Tests:** Test on known convex/nonconvex expressions

### E2. Skip relaxation for convex subproblems
- **File:** `solver.py`
- **What:** When all constraints and the objective are convex at a node (from E1), the NLP relaxation *is* the original problem — no McCormick/alphaBB needed. Solve directly with IPM and trust the result as a valid lower bound.
- **Impact:** Huge speedup on convex subproblems (skip relaxation compilation entirely)
- **Depends on:** E1
- **Tests:** Verify convex detection matches known convex problems

### E3. Per-constraint convexity for OA cuts
- **File:** `solver.py`, `cutting_planes.py`
- **What:** Enables OA cuts on a per-constraint basis (replacing the global `_oa_enabled` flag from A3 with a per-constraint flag from E1). More precise than A3's Hessian check.
- **Depends on:** E1, A3
- **Tests:** Verify correct per-constraint classification

---

## Phase F: Benchmarking & Validation (ongoing, 2–3 weeks dedicated)

### F1. Install BARON and run head-to-head benchmarks
- **What:** Get a BARON license (academic). Run the existing benchmark suite with `--solvers discopt,baron`. Establish actual gap measurements.
- **Impact:** Everything else is speculation until we have real numbers

### F2. Profile node-by-node comparison
- **What:** On 10 representative instances, compare discopt vs. BARON: root gap, node count, time per node, final gap. Identify which phase (B/C/D/E) matters most for each problem class.
- **Impact:** Prioritize remaining work based on data

### F3. Regression test suite against BARON solutions
- **What:** For every instance where BARON finds the global optimum, add it as a known-optimal reference. Enforce `incorrect_count = 0` against these.
- **Impact:** Correctness gate that prevents regressions

---

## Dependency Graph

```
A1 (feasibility pump)     ──┐
A2 (GNN branching)        ──┤
A3 (OA for convex)        ──┼── Phase A (wire existing) ── F1 (benchmark)
A4 (McCormick default)    ──┘
                                       │
B1 (pseudocosts)          ──┐          │
B2 (reliability branch)   ──┤── Phase B (branching)
B3 (strong branch LP)     ──┘          │
                                       │
C1 (incumbent OBBT)       ──┐          │
C2 (periodic OBBT)        ──┤── Phase C (tightening)
C3 (FBBT + incumbent)     ──┘          │
                                       ├── F2 (profile)
D1 (exp/log envelopes)    ──┐          │
D2 (trig envelopes)       ──┤── Phase D (envelopes)
D3 (signomial envelopes)  ──┤          │
D4 (compiler integration) ──┘          │
                                       │
E1 (convexity detector)   ──┐          │
E2 (skip relaxation)      ──┤── Phase E (detection)
E3 (per-constraint OA)    ──┘          │
                                       │
                                F3 (regression suite)
```

**Phases A–C are independent and can run in parallel.**
Phase D is independent of B/C.
Phase E depends on A3 (conceptually) but can start in parallel.
Phase F (benchmarking) should start after A is complete and continue throughout.

---

## Timeline Estimate

| Phase | Effort | Cumulative Impact (projected) |
|-------|--------|-------------------------------|
| **A** (wire existing) | 2–4 weeks | Fix 9/27 incorrect; ~30% node reduction |
| **B** (branching) | 4–6 weeks | ~50% node reduction vs. current |
| **C** (tightening) | 3–4 weeks | ~60% node reduction; tighter root gap |
| **D** (envelopes) | 6–8 weeks | Root gap ≤ 1.3x BARON target |
| **E** (convexity) | 4–6 weeks | 2–5x speedup on convex subproblems |
| **F** (benchmarks) | 2–3 weeks | Validates everything; guides priorities |

**Total:** ~20–30 weeks of focused work to reach broad BARON competitiveness.

Phase A alone should move correctness from 18/27 to ~25/27 and is the obvious first priority.

---

## What This Does NOT Cover (and Why)

- **Duality-based domain reduction (TDo/MDo):** Complex theory, marginal gains. Revisit if Phase C is insufficient.
- **RENS/local branching:** Nice-to-have primal heuristics, but feasibility pump covers the core need.
- **Adaptive algorithm switching:** Premature until we have benchmark data showing where it matters.
- **Full factorable programming framework:** BARON's architecture is fundamentally different (factorable decomposition). Replicating it would be a rewrite. The envelope approach (Phase D) captures most of the benefit.
