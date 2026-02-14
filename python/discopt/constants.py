"""Centralised numerical constants for the discopt solver.

All magic numbers, tolerances, and sentinel values used across the solver
pipeline are defined here so that they can be tuned from a single location
and referenced by name in the code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Infeasibility / failure sentinels
# ---------------------------------------------------------------------------
# Value stored as the lower bound for nodes whose NLP relaxation failed or
# was declared infeasible.  Must be larger than any realistic objective.
INFEASIBILITY_SENTINEL: float = 1e30

# Threshold for filtering bogus incumbents.  Any incumbent with objective
# >= this value is treated as invalid.  Set slightly below the sentinel so
# that legitimate large objectives (e.g. 1e25) are never filtered out.
SENTINEL_THRESHOLD: float = 1e29

# ---------------------------------------------------------------------------
# Constraint bound "infinity" — used where solvers need finite bounds in
# place of +/- inf (e.g. Ipopt, HiGHS).  NOT a sentinel — this is a
# legitimate large number representing an inactive bound.
# ---------------------------------------------------------------------------
CONSTRAINT_INF: float = 1e20

# ---------------------------------------------------------------------------
# Starting-point generation
# ---------------------------------------------------------------------------
# When variable bounds are infinite, clip to this range for midpoint /
# multi-start starting-point generation.
STARTING_POINT_CLIP: float = 100.0

# Fractions along the [lb, ub] interval for multi-start seeds.
MULTISTART_FRACTIONS: tuple[float, ...] = (0.25, 0.75)

# ---------------------------------------------------------------------------
# Solver tolerances
# ---------------------------------------------------------------------------
# Default Ipopt / IPM convergence tolerance.
DEFAULT_OPTIMALITY_TOL: float = 1e-7

# AlphaBB: eigenvalue threshold — alphas below this are treated as zero
# (i.e. the function is already convex in that direction).
ALPHABB_EPS: float = 1e-8

# AlphaBB: safety margin added to alpha to ensure strict underestimation.
ALPHABB_SAFETY: float = 1e-6
