# discopt Testing & Benchmarking Framework

A comprehensive validation, verification, and performance benchmarking framework
for the discopt solver (Rust + JAX hybrid MINLP solver).

## Architecture

```
discopt_benchmarks/
├── config/
│   ├── benchmarks.toml          # Benchmark suite definitions and targets
│   └── solvers.toml             # Solver configurations and paths
├── benchmarks/
│   ├── runner.py                # Main benchmark orchestrator
│   ├── problem_loader.py        # MINLPLib / Netlib / CUTEst loader
│   ├── solver_interface.py      # Uniform interface for all solvers
│   └── metrics.py               # Performance metrics computation
├── tests/
│   ├── test_correctness.py      # Correctness V&V tests
│   ├── test_subsolver_lp.py     # Rust LP subsolver validation
│   ├── test_subsolver_nlp.py    # Hybrid NLP subsolver validation
│   ├── test_relaxation.py       # McCormick relaxation quality tests
│   ├── test_interop.py          # Rust↔JAX interface tests
│   └── test_regression.py       # Performance regression detection
├── utils/
│   ├── profiles.py              # Dolan-Moré performance profiles
│   ├── statistics.py            # Statistical analysis utilities
│   └── reporting.py             # Report generation (markdown + plots)
├── reports/                     # Generated benchmark reports
├── run_benchmarks.py            # CLI entry point
└── README.md
```

## Quick Start

```bash
# Run correctness tests
pytest tests/ -v

# Run full benchmark suite
python run_benchmarks.py --suite phase1

# Run comparison against BARON
python run_benchmarks.py --suite comparison --solvers discopt,baron

# Generate performance report
python run_benchmarks.py --suite phase2 --report
```

## Benchmark Suites

| Suite       | Purpose                              | Instances | Time Limit |
|-------------|--------------------------------------|-----------|------------|
| `smoke`     | Quick sanity check                   | 10        | 60s        |
| `phase1`    | Phase 1 milestone validation         | ~50       | 3600s      |
| `phase2`    | Phase 2 milestone validation         | ~150      | 3600s      |
| `phase3`    | Phase 3 milestone validation         | ~300      | 3600s      |
| `full`      | Complete MINLPLib                     | ~1700     | 3600s      |
| `comparison`| Head-to-head solver comparison       | ~200      | 3600s      |
| `subsolver` | LP/NLP subsolver-only benchmarks     | varies    | 300s       |
| `nightly`   | CI regression suite                  | ~100      | 1800s      |

## Phase Gate Criteria

Each phase has automated pass/fail criteria defined in `config/benchmarks.toml`.
Run `python run_benchmarks.py --gate phaseN` to check.
