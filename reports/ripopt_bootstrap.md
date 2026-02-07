# ripopt: A Memory-Safe Interior Point Method in Rust

**Status:** Pre-implementation (bootstrap document)
**License:** Eclipse Public License 2.0 (EPL-2.0)
**Repository:** `github.com/discopt-org/ripopt`
**Parent project:** [discopt](https://github.com/discopt-org/discopt) (MIT/Apache-2.0)

---

## 1. What ripopt Is

ripopt (Rust Interior Point OPTimizer) is a faithful translation of [Ipopt](https://github.com/coin-or/Ipopt)'s core interior point algorithm from C++ to Rust. It produces a standalone, memory-safe nonlinear programming (NLP) solver that:

- Preserves 20+ years of numerical edge-case handling from Ipopt
- Compiles with `cargo build` — no Fortran, no MUMPS, no BLAS/LAPACK system dependency (goal, pending linear solver decision)
- Exposes a clean `NlpProblem` trait + `solve()` interface
- Is useful as a standalone Rust NLP solver, independent of discopt

ripopt is **not** a wrapper around C++ Ipopt. It is a source-level translation that produces native Rust code we own and can evolve — specifically for GPU batching and vmappability in Phase 2 of the discopt project.

---

## 2. Why Translate Rather Than Build from Scratch

1. **De-risking.** Ipopt handles hundreds of numerical edge cases accumulated over two decades: degenerate Jacobians, rank-deficient KKT systems, cycling detection, tiny step recovery. A from-scratch IPM would rediscover these failures one benchmark at a time. Translation preserves this institutional knowledge.

2. **Correctness by construction.** Each translated function can be validated against Ipopt's own output on the same problem instance, providing a natural regression test suite. Bit-for-bit matching on well-conditioned problems, tolerance matching on ill-conditioned ones.

3. **Path to vmappability.** Once the core algorithm is in Rust with explicit state (no global variables, no hidden allocation), refactoring for batch execution (struct-of-arrays layout, parameterizing over problem data) becomes a tractable Phase 2 task rather than a greenfield build.

4. **Community value.** There is no production-quality Rust NLP solver. ripopt would be immediately useful to the growing Rust scientific computing ecosystem (Argmin, Russell, faer), independent of discopt. If the linear solver problem is solved well, ripopt completely supplants the C++ version.

5. **Maintenance.** Rust's ownership model eliminates use-after-free and data races that have caused real Ipopt bugs. `cargo clippy` and `miri` catch entire categories of issues at compile time.

6. **LLM-assisted translation is practical now.** Ipopt's code is structured, algorithmic, and well-documented — ideal for LLM-assisted C++-to-Rust translation with human review.

7. **Critical enabler for Phase 2.** The GPU/batch adaptation work (WS4 in the discopt plan) cannot proceed until we have a Rust IPM codebase to adapt. Starting the translation early means GPU work can begin immediately after Phase 1.

8. **Build simplification.** cyipopt depends on C/Fortran Ipopt + MUMPS + BLAS/LAPACK, creating installation friction. A Rust crate with `cargo build` is the goal.

---

## 3. Licensing: EPL-2.0

Ipopt is licensed under the **Eclipse Public License 2.0**. A line-by-line translation to Rust is almost certainly a derivative work under copyright law, requiring ripopt to also be distributed under EPL-2.0.

**Key points:**
- EPL-2.0 is **not viral** like GPL. Code that *depends on* ripopt (e.g., discopt) does not need to be EPL-2.0.
- ripopt lives in its own repository with its own license. discopt (MIT/Apache-2.0) depends on it as a Cargo crate dependency.
- A `NOTICE` file must attribute the COIN-OR Ipopt project.

**Alternative paths (for future consideration):**
- **(a) Clean-room reimplementation**: Study Ipopt's algorithms from Wächter & Biegler (2006) without reference to source code. Produces MIT-licensable code but loses edge-case hardening. Higher risk. Could replace ripopt later if EPL becomes a barrier.
- **(b) Upstream relicensing**: Contact COIN-OR / Ipopt maintainers about adding MIT/Apache as a secondary license under EPL-2.0 Section 3. Worth attempting but unlikely to succeed quickly for a multi-contributor codebase.
- **(c) Functional API boundary**: If the translation produces sufficiently different internal architecture, it may qualify as an independent implementation. Legal gray area.

**Recommendation**: Start with EPL-2.0. Pursue upstream relicensing conversation in parallel. Consider clean-room only if EPL becomes a real barrier to adoption.

---

## 4. The Linear Solver Question (Critical Early Decision)

**This is the single most important technical decision for ripopt.** It must be resolved through research before committing to an implementation path.

### Why this matters

Ipopt's filter line search method requires the linear solver to report the **inertia** of the KKT matrix — the count of positive, negative, and zero eigenvalues of the diagonal D after LDL^T factorization. This is not optional: it drives Ipopt's inertia correction mechanism (`IpPDPerturbationHandler`), which adds regularization when the KKT system has incorrect inertia. Without inertia information, the standard Ipopt algorithm cannot function.

Specifically:
- At each IPM iteration, the KKT system is symmetric indefinite (n positive, m negative eigenvalues for n variables and m constraints)
- After LDL^T factorization, Ipopt reads the signs of the D diagonal to verify correct inertia
- If inertia is wrong, Ipopt perturbs the diagonal and refactors — this loop may execute multiple times per iteration
- This mechanism is essential for convergence on hard problems (degenerate, near-infeasible, poorly scaled)

**MUMPS and MA27/MA57 provide inertia as a byproduct of factorization.** Not all linear algebra libraries do.

### Performance implications

The linear solver typically consumes **80%+ of total IPM solve time**. Even for mathematically identical algorithms (e.g., sparse LDL^T), implementation details dominate: pivot selection strategy, fill-reducing ordering (AMD, nested dissection), supernodal vs left-looking factorization, cache-aware memory layout, and parallelism. MUMPS and MA57 have decades of tuning in these areas. A naive Rust LDL^T will be dramatically slower.

Getting this right is what separates "a research prototype" from "a solver that supplants C++ Ipopt."

### Research questions to resolve (Task R0)

Before choosing a linear solver strategy, investigate:

1. **faer capabilities**: Does faer's LDL^T (dense or sparse) expose the D diagonal for inertia computation? What is the performance vs MUMPS on representative KKT systems (500×500, 2000×2000, 10000×10000)?

2. **Clarabel's approach**: Clarabel is a Rust conic optimization solver that solves KKT systems. How does it handle inertia? Does it use faer, nalgebra, or a custom factorization? What can we reuse or learn from?

3. **Inertia-free IPM variants**: Ipopt has an inertia-free option (`IpInexactPDSolver`). How mature is it? What are the robustness tradeoffs? Are there other inertia-free IPM formulations in the literature that are well-tested?

4. **Rust sparse LDL^T with inertia**: How hard is it to implement a sparse LDL^T in Rust that tracks inertia? Could we extend faer? What about wrapping `amd` (fill-reducing ordering) in Rust?

5. **FFI as bootstrap**: How clean is an FFI binding to MUMPS from Rust? Could we use MUMPS for initial correctness validation, then replace it with a pure Rust solver later? Does this create an acceptable development path?

6. **Performance benchmarks**: On representative KKT systems from Ipopt's test suite, what is the factorization time for: (a) MUMPS, (b) faer dense, (c) faer sparse, (d) nalgebra, (e) a simple Rust LDL^T? At what problem size does sparse vs dense matter?

### Candidate paths (to be evaluated by R0)

| Path | Inertia? | Pure Rust? | Performance | Risk |
|------|----------|-----------|-------------|------|
| **A. Pure Rust LDL^T with inertia** | Yes | Yes | Unknown — must benchmark | Highest value if it works; significant engineering |
| **B. Inertia-free IPM variant** | Not needed | Yes | Unknown — algorithm change | Avoids LA problem but changes battle-tested algorithm |
| **C. FFI to MUMPS, replace later** | Yes | No (initially) | Known-good | Pragmatic but reintroduces Fortran dependency |
| **D. Extend faer with inertia** | Yes | Yes | Likely good (faer is fast) | Depends on faer maintainer willingness / architecture |
| **E. Hybrid: MUMPS FFI default + pure Rust optional** | Yes | Both | Known-good + experimental | Two code paths to maintain |

**Decision point:** Task R0 must be completed before significant translation work begins on `kkt.rs`. The rest of the translation (problem interface, filter, IPM loop, restoration) can proceed in parallel with the linear solver research — they depend on the linear solver through a well-defined trait interface.

### Linear solver trait (design now, implement after R0)

Regardless of which path is chosen, the linear solver should be behind a trait:

```rust
pub trait LinearSolver {
    /// Factor the symmetric matrix. Returns inertia if available.
    fn factor(&mut self, matrix: &SymmetricMatrix) -> Result<Inertia, SolverError>;

    /// Solve the factored system for the given RHS.
    fn solve(&self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError>;

    /// Whether this solver provides inertia information.
    fn provides_inertia(&self) -> bool;
}

pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}
```

This lets us swap implementations (faer, MUMPS FFI, custom) without changing the IPM code, and makes the inertia-free path explicit (`provides_inertia() → false` triggers alternative regularization logic).

---

## 5. Scope

### In scope (translate from C++ Ipopt)

| Component | Ipopt C++ Source | ripopt Target | Description |
|-----------|-----------------|---------------|-------------|
| IPM main loop | `IpIpoptAlg.cpp` | `src/ipm.rs` | Filter line search interior point iteration |
| KKT system | `IpPDFullSpaceSolver.cpp` | `src/kkt.rs` | KKT system assembly, normal equation reduction |
| Linear solver interface | (various) | `src/linear_solver.rs` | `LinearSolver` trait with inertia support |
| Inertia correction | `IpPDPerturbationHandler.cpp` | `src/inertia.rs` | Regularization when KKT has incorrect inertia |
| Filter | `IpFilterLSAcceptor.cpp` | `src/filter.rs` | Filter acceptance criteria (Wächter & Biegler 2005) |
| Restoration | `IpRestoPhase.cpp` | `src/restoration.rs` | Feasibility restoration when filter rejects step |
| Warm-starting | `IpWarmStartIterateInitializer.cpp` | `src/warmstart.rs` | Re-solve from previous solution |
| Convergence | `IpOrigIpoptNLP.cpp` (termination) | `src/convergence.rs` | Convergence and termination criteria |
| Problem interface | `IpTNLP.hpp` | `src/problem.rs` | `NlpProblem` trait: objective, gradient, Jacobian, Hessian callbacks |
| Options | (various) | `src/options.rs` | Solver configuration (tolerances, iteration limits) |
| Result | (various) | `src/result.rs` | Solution struct: x, multipliers, status |

### Out of scope (not translated)

| Component | Reason | Replacement |
|-----------|--------|-------------|
| MUMPS / MA27 / MA57 internals | Separate libraries; performance-critical code that needs its own strategy | See Section 4 — `LinearSolver` trait with pluggable backends |
| HSL linear solvers | Proprietary license | Excluded |
| AMPL `.nl` file reader | Already handled by discopt WS1 | discopt's own parser |
| Options parsing infrastructure | Over-engineered for our needs | Simple Rust config struct |
| IpStdCInterface (C API) | Not needed | Native Rust API |

---

## 6. Crate Architecture

```
ripopt/
├── Cargo.toml
├── LICENSE-EPL-2.0
├── NOTICE                  (attribution to COIN-OR Ipopt project)
├── README.md
├── src/
│   ├── lib.rs              (public API: solve(), NlpProblem, SolveResult)
│   ├── ipm.rs              (filter line search IPM main loop)
│   ├── kkt.rs              (KKT system assembly + delegation to LinearSolver)
│   ├── linear_solver.rs    (LinearSolver trait + Inertia type)
│   ├── linear_solver/      (pluggable backends — decided by Task R0)
│   │   ├── mod.rs
│   │   ├── faer_backend.rs (if faer provides inertia)
│   │   ├── mumps_ffi.rs    (if FFI path chosen for bootstrap)
│   │   └── dense.rs        (simple dense LDL^T for small problems / testing)
│   ├── inertia.rs          (inertia correction / regularization)
│   ├── filter.rs           (filter acceptance criteria)
│   ├── restoration.rs      (feasibility restoration phase)
│   ├── warmstart.rs        (warm-starting from previous solution)
│   ├── convergence.rs      (convergence and termination criteria)
│   ├── problem.rs          (NlpProblem trait — the public interface)
│   ├── options.rs          (solver configuration)
│   └── result.rs           (solution struct: x, multipliers, status)
├── tests/
│   ├── correctness.rs      (output matching vs C++ Ipopt on benchmark problems)
│   ├── edge_cases.rs       (degenerate Jacobians, near-singular KKT, cycling)
│   ├── linear_solver.rs    (inertia correctness, factorization accuracy)
│   └── benchmarks.rs       (performance comparison vs C++ Ipopt)
└── examples/
    ├── rosenbrock.rs        (unconstrained NLP)
    ├── hs071.rs             (Hock-Schittkowski #71 — constrained NLP)
    └── netlib_afiro.rs      (LP as special case of NLP)
```

### Key types

```rust
/// The problem trait that users implement.
pub trait NlpProblem {
    fn num_variables(&self) -> usize;
    fn num_constraints(&self) -> usize;
    fn bounds(&self) -> (Vec<f64>, Vec<f64>);           // (lower, upper) for variables
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>); // (lower, upper) for constraints
    fn initial_point(&self) -> Vec<f64>;

    fn objective(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64], grad: &mut [f64]);
    fn constraints(&self, x: &[f64], g: &mut [f64]);
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>); // (rows, cols) sparsity
    fn jacobian_values(&self, x: &[f64], vals: &mut [f64]);
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>);  // (rows, cols) sparsity
    fn hessian_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], vals: &mut [f64]);
}

/// Linear solver with inertia detection.
pub trait LinearSolver {
    fn factor(&mut self, matrix: &SymmetricMatrix) -> Result<Inertia, SolverError>;
    fn solve(&self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError>;
    fn provides_inertia(&self) -> bool;
}

pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

pub struct SolveResult {
    pub x: Vec<f64>,              // primal solution
    pub objective: f64,            // optimal objective value
    pub multipliers: Vec<f64>,     // constraint multipliers
    pub status: SolveStatus,       // Optimal, Infeasible, MaxIterations, etc.
    pub iterations: usize,
    pub solve_time: Duration,
}

pub struct SolverOptions {
    pub tol: f64,                  // convergence tolerance (default 1e-8)
    pub max_iter: usize,           // iteration limit (default 3000)
    pub acceptable_tol: f64,       // acceptable convergence (default 1e-6)
    pub mu_init: f64,              // initial barrier parameter
    pub warm_start: bool,          // enable warm-starting
    // ... other Ipopt-equivalent options
}

pub fn solve(problem: &dyn NlpProblem, options: &SolverOptions) -> SolveResult;
```

### Dependencies

```toml
[dependencies]
faer = "0.20"           # Linear algebra (candidate — pending R0 evaluation)

[dev-dependencies]
approx = "0.5"          # Floating-point comparison in tests
criterion = "0.5"       # Benchmarking
```

---

## 7. Translation Approach

### Source material
- **Ipopt source**: https://github.com/coin-or/Ipopt (EPL-2.0)
- **Key C++ files**: `Ipopt/src/Algorithm/` directory, particularly:
  - `IpIpoptAlg.cpp` — main algorithm loop (~800 lines)
  - `IpFilterLSAcceptor.cpp` — filter line search (~600 lines)
  - `IpPDFullSpaceSolver.cpp` — KKT system (~500 lines)
  - `IpRestoPhase.cpp` — restoration (~400 lines)
  - `IpPDPerturbationHandler.cpp` — inertia correction (~300 lines)
- **Total core algorithm**: ~3,000-4,000 lines of C++ → estimated ~2,500-3,500 lines of Rust

### Translation strategy
1. **Module by module.** Translate one C++ file at a time, starting with the problem interface (`TNLP` → `NlpProblem` trait), then working inward to the core loop.
2. **LLM-assisted.** Use LLMs for mechanical C++-to-Rust conversion. Human reviews every function for numerical correctness, edge cases, and idiomatic Rust.
3. **Test as you go.** Each translated module gets unit tests that compare output against C++ Ipopt (via cyipopt) on the same inputs.
4. **Replace globals with structs.** Ipopt uses registered options and SmartPtr reference counting. Replace with explicit `SolverState` struct — this is the key architectural change that enables future vmappability.
5. **Linear solver behind trait.** The `LinearSolver` trait is defined early. Initial implementation is determined by Task R0. The rest of the IPM code is agnostic to which backend is used.

### Suggested translation order
1. `src/problem.rs` — NlpProblem trait (defines the interface everything depends on)
2. `src/options.rs` + `src/result.rs` — configuration and output types
3. `src/linear_solver.rs` — LinearSolver trait + Inertia type (interface only)
4. **Task R0: Linear solver research** (parallel with steps 1-3)
5. `src/kkt.rs` — KKT system assembly (delegates to LinearSolver; **blocked on R0 for factorization backend**)
6. `src/inertia.rs` — inertia correction / regularization
7. `src/filter.rs` — filter acceptance criteria (relatively self-contained)
8. `src/ipm.rs` — main IPM loop (ties everything together)
9. `src/restoration.rs` — feasibility restoration (called from IPM loop)
10. `src/warmstart.rs` — warm-starting logic
11. `src/convergence.rs` — termination criteria

Note: Steps 6-7 can proceed in parallel with R0 since they don't depend on the linear solver implementation, only on the trait interface.

---

## 8. Verification Criteria

### Output matching (correctness)
- Rust ripopt matches C++ Ipopt solution on **all 7 NLP benchmark problems** within `rel_tol=1e-6` (objective value, variable values, multipliers)
- Matches on **all 7 Netlib LP instances** (LP is a special case of NLP) within `abs_tol=1e-6`
- On well-conditioned problems, Rust and C++ Ipopt take the **same number of iterations** (validates algorithm fidelity)

### Failure mode handling
- Correctly detects and reports **infeasibility** (3 infeasible problems)
- Correctly detects and reports **unboundedness** (2 problems)
- Correctly handles **iteration limit** (1 problem with tight limit)

### Inertia correctness (linear solver)
- On KKT systems from benchmark problems, reported inertia matches MUMPS/MA57 inertia
- Inertia correction triggers correctly on ill-conditioned problems (same regularization as C++ Ipopt)

### Warm-starting
- Re-solve after parameter perturbation converges in **fewer iterations** than cold start

### Performance
- Wall-clock within **2x of C++ Ipopt** on single instances (stretch goal: parity)
- Linear solver factorization time benchmarked against MUMPS on representative KKT systems

### Code quality
- `cargo clippy -- -D warnings` clean
- **Zero `unsafe` blocks** in core algorithm (FFI bindings may use unsafe if MUMPS path chosen)
- All public functions documented
- `cargo test` passes on macOS ARM64 and Linux x86_64

### Integration test (with discopt)
- Replacing cyipopt with ripopt in the discopt solver loop produces **identical results** on all 24 MINLPLib instances

---

## 9. Integration with discopt

ripopt is consumed by discopt as a Cargo crate dependency:

```toml
# In discopt/Cargo.toml
[workspace.dependencies]
ripopt = { path = "../ripopt" }    # local during development
# ripopt = "0.1"                   # crates.io for releases
```

**Integration points in discopt:**
- `crates/discopt-python/src/nlp_ripopt.rs` — PyO3 bindings exposing ripopt to Python
- `python/discopt/solvers/nlp_ripopt.py` — Python wrapper implementing `SolverBackend` protocol
- `SolverBackend` ABC — the common protocol that HiGHS, cyipopt, and ripopt all implement

**SolverBackend protocol:**
```python
class SolverBackend(ABC):
    @abstractmethod
    def solve_nlp(self, evaluator, x0, bounds, constraints) -> NLPResult: ...
```

ripopt implements `NlpProblem` on the Rust side; the PyO3 binding adapts JAX-compiled gradient/Hessian callbacks from discopt's `NLPEvaluator` into the `NlpProblem` trait.

---

## 10. Future Evolution (Phase 2 of discopt)

After ripopt is working and validated, discopt's WS4 adapts it for GPU-accelerated batch solving:

1. **Eliminate global state** — verify all solver state is in explicit `SolverState` struct (ripopt should already do this)
2. **Struct-of-arrays layout** — refactor internal arrays from AoS (one solver instance) to SoA (batch of N instances sharing problem structure but different data)
3. **Dense linear algebra for GPU** — for batch GPU solving, replace sparse factorization with `jax.scipy.linalg.cholesky` for problems up to ~5,000 variables. Batch throughput matters more than single-instance efficiency. This is a separate code path from ripopt's CPU solver.
4. **JAX integration** — expose IPM iteration as JAX-callable via `jax.pure_callback` or custom XLA extension, enabling `jax.vmap` over batched problem data
5. **Differentiability** — add `custom_jvp` rule using KKT system solution for backward pass

This is an *evolution* of ripopt's known-good code, not a from-scratch build. The hardest part (getting a correct IPM) is already done.

---

## 11. Timeline

| Milestone | Target Month | Description |
|-----------|-------------|-------------|
| Repo created, Cargo.toml, NlpProblem trait | Month 1-2 | Basic structure, problem interface |
| **Task R0: Linear solver research** | **Month 1-3** | **Evaluate faer inertia, Clarabel approach, MUMPS FFI, inertia-free variants. Decide path.** |
| Core IPM loop translating | Month 2-4 | ipm.rs, filter.rs, restoration.rs (can proceed in parallel with R0) |
| Linear solver backend implemented | Month 3-5 | Based on R0 findings |
| KKT + inertia correction working | Month 4-6 | Full KKT solve with correct inertia handling |
| Core algorithm working | Month 5-7 | Solves Rosenbrock, HS071, and simple NLPs |
| Edge case validation | Month 6-8 | Degenerate problems, restoration phase, warm-starting |
| Matches C++ Ipopt on all benchmarks | Month 7-9 | Full correctness validation |
| Replaces cyipopt in discopt | Month 8-10 | Drop-in replacement, cyipopt becomes optional |
| Publication draft | Month 12-20 | Paper 7: "ripopt: A Memory-Safe Interior Point Method in Rust" |

---

## 12. Publication Opportunity

**Paper 7: ripopt — A Memory-Safe Interior Point Method in Rust**

- **Venue**: Mathematical Programming Computation (MPC), Optimization Methods and Software (OMS), or SoftwareX
- **Contribution**: Faithful translation of Ipopt's core IPM from C++ to Rust, producing a standalone memory-safe NLP solver. Demonstrates that translation preserves numerical robustness while gaining safety guarantees. If a pure-Rust linear solver with inertia is achieved, this is a significant additional contribution.
- **Minimum results**: Matches C++ Ipopt on 7 NLP + 7 LP benchmarks; handles failure modes; wall-clock within 2x; zero unsafe in core; clean clippy
- **Stretch results**: vmappable Rust IPM with JAX integration (≥10x at batch 64), differentiable via `custom_jvp`, pure-Rust linear solver with competitive performance
- **Strategic value**: Establishes ripopt as a community resource; provides publication evidence for grant applications. A production-quality pure-Rust IPM solver would be a first.

---

## 13. Key References

1. Wächter, A. and Biegler, L.T. (2006). "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming." *Mathematical Programming*, 106(1), 25-57.
2. Wächter, A. and Biegler, L.T. (2005). "Line search filter methods for nonlinear programming: Motivation and global convergence." *SIAM Journal on Optimization*, 16(1), 1-31.
3. Ipopt source code: https://github.com/coin-or/Ipopt
4. faer (Rust linear algebra): https://github.com/sarah-ek/faer-rs
5. Clarabel (Rust conic solver): https://github.com/oxfordcontrol/Clarabel.rs
6. Eclipse Public License 2.0: https://www.eclipse.org/legal/epl-2.0/

---

*This document is the bootstrap specification for ripopt. It is derived from the discopt reconciled development plan v2 and unified publication roadmap. For the full discopt plan, see `discopt-org/discopt/reports/reconciled_development_plan_v2.md`.*
