# T24 Audit: IPM Integration Gaps in solver.py

**Date**: 2026-02-08
**Files reviewed**: `python/discopt/solver.py`, `python/discopt/_jax/ipm.py`, `python/discopt/_jax/nl_evaluator.py`, `python/discopt/_jax/lp_ipm.py`, `python/discopt/_jax/qp_ipm.py`, `python/discopt/_jax/primal_heuristics.py`

---

## Gap 1: Root-node multistart runs serial loops instead of vmap

**Location**: `solver.py:149-187` (`_solve_root_node_multistart`)

**Description**: The root-node multistart generates 5 starting points (midpoint, lower-quarter, upper-quarter, 2 random) and solves each sequentially in a Python `for` loop (line 169). Each call goes through `_solve_node_nlp` -> `_solve_node_nlp_ipm` -> `ipm_solve`, which is a single-instance JAX IPM solve. Since `solve_nlp_batch` already supports vmapping over different `x0` values with shared bounds, all 5 solves could be dispatched as a single vmap call.

**Impact**: **HIGH** -- The root node is the most expensive single node in the B&B tree (no pruning yet, wide bounds). Multi-start with 5 serial IPM solves takes ~5x the wall time of a single solve. Vmapping would reduce this to approximately 1x (assuming sufficient hardware parallelism), giving a ~5x speedup on root-node solve time.

**Current behavior**: Serial Python loop at line 169:
```python
for x0 in starting_points:
    nlp_result = _solve_node_nlp(evaluator, x0, node_lb, node_ub, ...)
```

**Proposed fix**: When `nlp_solver == "ipm"` and `hasattr(evaluator, "_obj_fn")`:
1. Stack all starting points into `x0_batch` of shape `(n_starts, n_vars)`.
2. Broadcast `node_lb`/`node_ub` to `(n_starts, n_vars)`.
3. Call `solve_nlp_batch(obj_fn, con_fn, x0_batch, xl_batch, xu_batch, g_l, g_u, opts)`.
4. Select the best-objective result from the batched output.
5. Fall back to the existing serial loop for non-IPM solvers and `.nl` evaluators.

---

## Gap 2: Batch IPM path skipped when `n_batch == 1`

**Location**: `solver.py:409` (condition `nlp_solver == "ipm" and n_batch > 1 and hasattr(evaluator, "_obj_fn")`)

**Description**: The batch vmap path is only taken when `n_batch > 1`. When `n_batch == 1` (common for the root node, or when most nodes have been pruned), the code falls through to the serial `else` branch (line 421), which on iteration 0 calls `_solve_root_node_multistart` (serial) and on later iterations calls `_solve_node_nlp`. The serial path still works correctly, but:

1. On iteration 0 with `n_batch == 1`, it runs the serial multistart (Gap 1).
2. On later iterations with `n_batch == 1`, the single IPM solve via `_solve_node_nlp_ipm` is correct but misses the benefit of JIT warmup -- the batch path pre-builds JAX arrays differently.

**Impact**: **LOW** -- Single-node batches are infrequent in normal B&B (usually only root node or near termination). The fallback to serial `_solve_node_nlp_ipm` is functionally correct and has no performance penalty for a single solve. The main concern is that the root-node multistart (iteration 0, n_batch == 1) does not use vmap, which is covered by Gap 1.

**Current behavior**: Correct fallback to serial path. No bugs.

**Proposed fix**: Change the condition to `n_batch >= 1` (i.e., always use the batch path when IPM is selected and evaluator supports it). For `n_batch == 1`, vmap over a single element is equivalent to a single solve. However, Gap 1 (root multistart batching) is the higher-priority fix. If Gap 1 is fixed, the `n_batch == 1` path on iteration 0 should also be batched.

---

## Gap 3: `.nl` model evaluators cannot use IPM -- silent fallback to Ipopt

**Location**: `solver.py:645-653` (`_solve_node_nlp`), specifically lines 648-651

**Description**: The `NLPEvaluatorFromNl` class (`nl_evaluator.py`) uses Rust-backed evaluation and finite-difference derivatives. It does NOT have `_obj_fn` or `_cons_fn` attributes (JAX-compiled callables). The check `hasattr(evaluator, "_obj_fn")` at `solver.py:648` correctly detects this and falls back to Ipopt. However:

1. There is no user-facing warning that `.nl` models silently fall back to Ipopt even when `nlp_solver="ipm"` is requested.
2. The `_solve_continuous` path (`solver.py:595`) has the same silent fallback.
3. The batch condition at `solver.py:409` also requires `hasattr(evaluator, "_obj_fn")`, so `.nl` models never enter the batch path.

This is architecturally correct -- the pure-JAX IPM requires JAX-traced functions for autodiff (Hessians, Jacobians), and finite-difference evaluators cannot provide these. However, the gap means `.nl` models miss all IPM/vmap performance benefits.

**Impact**: **MEDIUM** -- Users loading `.nl` files with `from_nl()` always get Ipopt, regardless of the `nlp_solver` parameter. This is functionally fine (Ipopt is a good solver), but:
- No batch parallelism for `.nl` B&B nodes.
- No GPU acceleration path for `.nl` models.
- Silent behavior surprise for users.

**Proposed fix (short-term)**: Add a warning when `nlp_solver="ipm"` but evaluator is `NLPEvaluatorFromNl`:
```python
if nlp_solver == "ipm" and not hasattr(evaluator, "_obj_fn"):
    import warnings
    warnings.warn("IPM backend requires JAX-compiled evaluator; .nl models fall back to Ipopt.")
```

**Proposed fix (long-term, outside T24 scope)**: Implement a JAX-traced expression evaluator for `.nl` models by compiling the Rust expression tree into a JAX computation graph. This would enable IPM + vmap + GPU for `.nl` models.

---

## Gap 4: MILP B&B loop uses serial LP solves, not `lp_ipm_solve_batch`

**Location**: `solver.py:1004-1031` (`_solve_milp_bb`, inner loop)

**Description**: The MILP solver dispatches LP relaxations at each B&B node. `lp_ipm.py` already provides `lp_ipm_solve_batch` (line 518) which vmaps over per-node variable bounds. However, `_solve_milp_bb` uses a serial Python loop:
```python
for i in range(n_batch):
    ...
    state = lp_ipm_solve(lp_data.c, lp_data.A_eq, lp_data.b_eq, x_l_full, x_u_full)
```

This misses the batch parallelism available from `lp_ipm_solve_batch`.

**Impact**: **HIGH** -- LP relaxations are the inner loop of MILP B&B. With `batch_size=16` (default), each B&B iteration runs 16 serial LP solves. Vmapping would reduce this to ~1 equivalent LP solve per iteration, giving up to ~16x speedup on MILP problems. LP solves are typically fast individually, but the Python loop overhead and lack of hardware parallelism compound over thousands of B&B iterations.

**Current behavior**: Serial loop at line 1004. Each LP solve independently creates JAX arrays and runs the IPM.

**Proposed fix**:
1. Stack per-node bounds into `xl_batch` and `xu_batch` of shape `(n_batch, n_full)`.
2. Call `lp_ipm_solve_batch(lp_data.c, lp_data.A_eq, lp_data.b_eq, xl_batch, xu_batch)`.
3. Unpack the batched `LPIPMState` to fill `result_lbs`, `result_sols`, `result_feas`.
4. Handle per-element exceptions via convergence status (batch path returns all results).

---

## Gap 5: MIQP B&B loop uses serial QP solves, not `qp_ipm_solve_batch`

**Location**: `solver.py:1139-1173` (`_solve_miqp_bb`, inner loop)

**Description**: Identical to Gap 4, but for MIQP. `qp_ipm.py` provides `qp_ipm_solve_batch` (line 564), but `_solve_miqp_bb` uses a serial Python loop over QP relaxations.

**Impact**: **HIGH** -- Same reasoning as Gap 4. Default `batch_size=16` means up to 16x speedup potential.

**Current behavior**: Serial loop at line 1139.

**Proposed fix**: Same pattern as Gap 4:
1. Stack per-node bounds into batched arrays.
2. Call `qp_ipm_solve_batch(Q, c, A, b, xl_batch, xu_batch)`.
3. Unpack batched results.

---

## Gap 6: `_solve_batch_ipm` skips root-node multistart

**Location**: `solver.py:409-420` (batch IPM entry) and `solver.py:431-439` (serial root multistart)

**Description**: When the batch IPM path is taken (`nlp_solver == "ipm" and n_batch > 1 and hasattr(evaluator, "_obj_fn")`), it calls `_solve_batch_ipm` which uses the midpoint of bounds as starting points for all nodes. It does NOT perform multistart at the root node. Multistart only happens in the serial `else` branch at line 431-439 (conditioned on `iteration == 0`).

This means:
- Batch path (n_batch > 1 at iteration 0): No multistart. Single midpoint start per node.
- Serial path (n_batch == 1 at iteration 0): Full multistart with 5 starting points.

At iteration 0, `n_batch` is typically 1 (only the root node), so the serial multistart path is usually taken. However, if `batch_size > 1` and the tree starts with multiple root-equivalent nodes (unlikely but possible), multistart would be skipped.

**Impact**: **LOW** -- In practice, iteration 0 almost always has `n_batch == 1`, so the serial multistart is used. This is more of an architectural inconsistency than a performance gap.

**Proposed fix**: After fixing Gap 1 (vmapped multistart), ensure the batch path at iteration 0 also calls the vmapped multistart for all nodes in the batch.

---

## Gap 7: `MultiStartNLP` and `feasibility_pump` hardcoded to Ipopt

**Location**: `python/discopt/_jax/primal_heuristics.py:144` and `primal_heuristics.py:226`

**Description**: Both `MultiStartNLP.solve()` (line 144) and `feasibility_pump()` (line 226) unconditionally call `solve_nlp` (the Ipopt wrapper). They do not accept a `nlp_solver` parameter and cannot use the IPM backend. If these heuristics are called during B&B (e.g., the root multistart at `solver.py:432` calls `_solve_root_node_multistart` which calls `_solve_node_nlp` which respects `nlp_solver`), the main solver path is fine. But standalone use of `MultiStartNLP` or `feasibility_pump` always uses Ipopt.

**Impact**: **LOW** -- These are standalone heuristics, not in the B&B hot path. The solver orchestrator uses its own `_solve_root_node_multistart` which does respect the `nlp_solver` parameter. Standalone callers wanting IPM would need to use `solve_nlp_ipm` directly.

**Proposed fix (optional)**: Add an optional `nlp_solver` parameter to `MultiStartNLP` and `feasibility_pump`, defaulting to `"ipopt"` for backward compatibility.

---

## Gap 8: Continuous model path does not use multistart with IPM

**Location**: `solver.py:561-626` (`_solve_continuous`)

**Description**: The `_solve_continuous` path (for pure NLP with no integer variables) uses a single starting point (midpoint of bounds) regardless of the solver backend. The B&B MINLP path has multistart at the root node, but continuous models skip B&B entirely and solve directly. For nonconvex continuous problems, this means the solution quality depends entirely on the single starting point.

**Impact**: **MEDIUM** -- Nonconvex NLPs would benefit from multistart. With IPM + vmap, multiple starting points could be solved in parallel at near-zero extra cost. Currently, users must manually implement multistart for continuous nonconvex problems.

**Proposed fix**: When `nlp_solver == "ipm"`, generate multiple starting points and use `solve_nlp_batch` to solve them in parallel, then return the best result. This would be a natural extension of Gap 1.

---

## Summary Table

| # | Gap | Location | Impact | Effort |
|---|-----|----------|--------|--------|
| 1 | Root multistart serial, not vmapped | solver.py:149-187 | HIGH | Medium |
| 2 | Batch path skipped for n_batch==1 | solver.py:409 | LOW | Low |
| 3 | .nl evaluators cannot use IPM (silent fallback) | solver.py:648, nl_evaluator.py | MEDIUM | Low (warning), High (JAX .nl compiler) |
| 4 | MILP B&B serial LP solves, not batched | solver.py:1004-1031 | HIGH | Medium |
| 5 | MIQP B&B serial QP solves, not batched | solver.py:1139-1173 | HIGH | Medium |
| 6 | Batch path skips root multistart | solver.py:409-439 | LOW | Low (after Gap 1) |
| 7 | Primal heuristics hardcoded to Ipopt | primal_heuristics.py:144,226 | LOW | Low |
| 8 | Continuous path uses single start | solver.py:561-626 | MEDIUM | Medium |

## Recommended Priority for T24

1. **Gap 1** (root multistart vmap) -- Highest ROI, directly improves B&B root quality + speed.
2. **Gap 4** (MILP batch LP) -- Large speedup for MILP problems.
3. **Gap 5** (MIQP batch QP) -- Large speedup for MIQP problems.
4. **Gap 3** (warning for .nl fallback) -- Quick win for user experience.
5. **Gap 8** (continuous multistart) -- Improves NLP solution quality.
6. Gaps 2, 6, 7 -- Low priority, small impact.
