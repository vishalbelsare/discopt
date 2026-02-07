# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

discopt is a hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend (LP solving, B&B tree management), JAX (automatic differentiation, NLP relaxations, GPU acceleration), and Python orchestration. This repository contains the testing and benchmarking framework that validates correctness and measures performance against solvers like BARON, Couenne, SCIP, and HiGHS.

## Commands

### Install
```bash
cd discopt_benchmarks && pip install -e ".[dev]"
```

### Tests
```bash
pytest python/tests/ -v                                  # discopt tests
pytest discopt_benchmarks/tests/ -v                      # Benchmark suite
pytest discopt_benchmarks/tests/ -m smoke                # Quick CI smoke tests
pytest discopt_benchmarks/tests/ -m "not slow"           # Skip long tests
pytest discopt_benchmarks/tests/ -k test_correctness     # Single test file
pytest discopt_benchmarks/tests/ --cov=benchmarks --cov=utils  # With coverage (≥85% required)
```

### Benchmarking
```bash
python discopt_benchmarks/run_benchmarks.py --suite smoke     # Quick sanity check
python discopt_benchmarks/run_benchmarks.py --suite phase1    # Phase 1 validation
python discopt_benchmarks/run_benchmarks.py --gate phase1     # Check phase gate criteria
python discopt_benchmarks/run_benchmarks.py --suite comparison --solvers discopt,baron
```

### Linting & Type Checking
```bash
ruff check python/
ruff format --check python/
mypy python/discopt/
```

## Architecture

- **`python/discopt/modeling/`** — Python modeling API with expression DAG system for MINLP formulation, supporting continuous/binary/integer variables and operator overloading that maps to Rust AST. Imported as `from discopt import Model` or `import discopt.modeling as dm`.
- **`python/discopt/_jax/`** — JAX DAG compiler, McCormick relaxations, NLP evaluator, relaxation compiler.
- **`python/discopt/solvers/`** — HiGHS LP wrapper, cyipopt NLP wrapper.
- **`python/discopt/solver.py`** — Solver orchestrator: end-to-end `Model.solve()` via B&B.
- **`crates/discopt-core/`** — Rust: Expression IR, B&B tree, .nl parser, FBBT/presolve.
- **`crates/discopt-python/`** — Rust: PyO3 bindings with zero-copy numpy.
- **`discopt_benchmarks/`** — Benchmark orchestration, phase gate criteria, performance testing.
  - **`benchmarks/`** — `runner.py` loads instances, `metrics.py` computes metrics.
  - **`tests/`** — Pytest suite with markers: `smoke`, `correctness`, `regression`, etc.
  - **`config/benchmarks.toml`** — Single source of truth for suites, gates, solver configs.
  - **`utils/`** — Statistical utilities, profiles, report generation.

## Key Constraints

- **Correctness is non-negotiable**: Every phase gate enforces `incorrect_count ≤ 0`. Never weaken this check.
- **Numerical tolerances**: abs=1e-6, rel=1e-4, integrality=1e-5, factorization=1e-12 (defined in `conftest.py`).
- **ruff** line-length is 100 chars, targeting Python 3.10+. Pinned to v0.14.6 across pre-commit and CI.
- **Coverage** must stay ≥85%.
- Tests have a 300-second default timeout (configurable in `pyproject.toml`).
