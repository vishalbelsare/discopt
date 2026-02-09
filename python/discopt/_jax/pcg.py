"""
Preconditioned Conjugate Gradient (PCG) solver for JAX.

Provides a JIT-compatible PCG implementation using jax.lax.while_loop,
suitable for solving symmetric positive definite linear systems Ax = b.

For the IPM augmented KKT system (which is symmetric indefinite), the
recommended approach is the "condensed" normal equations:
    (A W^{-1} A^T + D) dy = rhs
which produces a smaller, positive definite system amenable to PCG,
followed by back-substitution for dx.

Key features:
  - JIT-compatible via jax.lax.while_loop (no Python control flow)
  - Supports diagonal (Jacobi) and custom preconditioners
  - Returns solution, residual norm, iteration count, convergence flag
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class PCGResult(NamedTuple):
    """Result of a PCG solve."""

    x: jnp.ndarray  # (n,) solution vector
    residual_norm: jnp.ndarray  # scalar final ||r||
    iterations: jnp.ndarray  # scalar int iteration count
    converged: jnp.ndarray  # scalar bool


class PCGOptions(NamedTuple):
    """Options for the PCG solver."""

    tol: float = 1e-10
    max_iter: int = 1000
    # Relative tolerance: stop when ||r|| <= tol * ||b||
    use_relative_tol: bool = True


# ---------------------------------------------------------------------------
# Preconditioners
# ---------------------------------------------------------------------------


def diagonal_preconditioner(A: jnp.ndarray) -> Callable:
    """Build a Jacobi (diagonal) preconditioner from matrix A.

    Returns a function M_inv(r) that applies the inverse of diag(A) to r.
    """
    diag_A = jnp.diag(A)
    # Clamp to avoid division by zero for zero diagonal entries
    diag_inv = 1.0 / jnp.where(jnp.abs(diag_A) > 1e-30, diag_A, 1.0)
    return lambda r: diag_inv * r


def identity_preconditioner(n: int) -> Callable:
    """No-op preconditioner (identity)."""
    return lambda r: r


# ---------------------------------------------------------------------------
# PCG solver
# ---------------------------------------------------------------------------


def pcg_solve(
    A: jnp.ndarray,
    b: jnp.ndarray,
    x0: Optional[jnp.ndarray] = None,
    preconditioner: Optional[Callable] = None,
    options: Optional[PCGOptions] = None,
) -> PCGResult:
    """Solve Ax = b using Preconditioned Conjugate Gradient.

    A must be symmetric positive definite. For indefinite systems,
    use the condensed normal equations approach.

    Args:
        A: (n, n) symmetric positive definite matrix.
        b: (n,) right-hand side vector.
        x0: (n,) initial guess. If None, uses zeros.
        preconditioner: Function M_inv(r) -> z that applies the
            inverse preconditioner. If None, uses identity.
        options: PCGOptions controlling tolerance and max iterations.

    Returns:
        PCGResult with solution, residual norm, iteration count,
        and convergence flag.
    """
    opts = options if options is not None else PCGOptions()

    if x0 is None:
        x0 = jnp.zeros_like(b)

    if preconditioner is None:
        M_inv = lambda r: r  # noqa: E731
    else:
        M_inv = preconditioner

    # Initial residual
    r0 = b - A @ x0
    z0 = M_inv(r0)
    p0 = z0
    rz0 = jnp.dot(r0, z0)

    # Tolerance threshold
    b_norm = jnp.linalg.norm(b)
    tol_threshold = jnp.where(
        opts.use_relative_tol,
        opts.tol * jnp.maximum(b_norm, 1e-30),
        jnp.array(opts.tol, dtype=b.dtype),
    )

    # Check if already converged
    r0_norm = jnp.linalg.norm(r0)

    # State: (x, r, z, p, rz, iteration, converged)
    init_state = (x0, r0, z0, p0, rz0, jnp.array(0, dtype=jnp.int32), r0_norm <= tol_threshold)

    def cond_fn(state):
        _, _, _, _, _, iteration, converged = state
        return (~converged) & (iteration < opts.max_iter)

    def body_fn(state):
        x, r, z, p, rz, iteration, _ = state

        # Matrix-vector product
        Ap = A @ p

        # Step size
        pAp = jnp.dot(p, Ap)
        alpha = rz / jnp.where(jnp.abs(pAp) > 1e-30, pAp, 1e-30)

        # Update solution and residual
        x_new = x + alpha * p
        r_new = r - alpha * Ap

        # Check convergence
        r_norm = jnp.linalg.norm(r_new)
        converged = r_norm <= tol_threshold

        # Apply preconditioner
        z_new = M_inv(r_new)
        rz_new = jnp.dot(r_new, z_new)

        # Conjugate direction update
        beta = rz_new / jnp.where(jnp.abs(rz) > 1e-30, rz, 1e-30)
        p_new = z_new + beta * p

        return (x_new, r_new, z_new, p_new, rz_new, iteration + 1, converged)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_sol, r_final, _, _, _, iters, conv = final_state

    return PCGResult(
        x=x_sol,
        residual_norm=jnp.linalg.norm(r_final),
        iterations=iters,
        converged=conv,
    )


# ---------------------------------------------------------------------------
# Matrix-free PCG (for large-scale problems)
# ---------------------------------------------------------------------------


def pcg_solve_matvec(
    matvec: Callable,
    b: jnp.ndarray,
    x0: Optional[jnp.ndarray] = None,
    preconditioner: Optional[Callable] = None,
    options: Optional[PCGOptions] = None,
) -> PCGResult:
    """Solve Ax = b using PCG with a matrix-vector product function.

    Instead of providing the full matrix A, provide a function
    matvec(v) -> A @ v. This is useful for large sparse systems
    or implicit operators.

    Args:
        matvec: Function v -> A @ v.
        b: (n,) right-hand side vector.
        x0: (n,) initial guess. If None, uses zeros.
        preconditioner: Function M_inv(r) -> z. If None, uses identity.
        options: PCGOptions.

    Returns:
        PCGResult.
    """
    opts = options if options is not None else PCGOptions()

    if x0 is None:
        x0 = jnp.zeros_like(b)

    if preconditioner is None:
        M_inv = lambda r: r  # noqa: E731
    else:
        M_inv = preconditioner

    r0 = b - matvec(x0)
    z0 = M_inv(r0)
    p0 = z0
    rz0 = jnp.dot(r0, z0)

    b_norm = jnp.linalg.norm(b)
    tol_threshold = jnp.where(
        opts.use_relative_tol,
        opts.tol * jnp.maximum(b_norm, 1e-30),
        jnp.array(opts.tol, dtype=b.dtype),
    )

    r0_norm = jnp.linalg.norm(r0)
    init_state = (x0, r0, z0, p0, rz0, jnp.array(0, dtype=jnp.int32), r0_norm <= tol_threshold)

    def cond_fn(state):
        _, _, _, _, _, iteration, converged = state
        return (~converged) & (iteration < opts.max_iter)

    def body_fn(state):
        x, r, z, p, rz, iteration, _ = state

        Ap = matvec(p)
        pAp = jnp.dot(p, Ap)
        alpha = rz / jnp.where(jnp.abs(pAp) > 1e-30, pAp, 1e-30)

        x_new = x + alpha * p
        r_new = r - alpha * Ap

        r_norm = jnp.linalg.norm(r_new)
        converged = r_norm <= tol_threshold

        z_new = M_inv(r_new)
        rz_new = jnp.dot(r_new, z_new)

        beta = rz_new / jnp.where(jnp.abs(rz) > 1e-30, rz, 1e-30)
        p_new = z_new + beta * p

        return (x_new, r_new, z_new, p_new, rz_new, iteration + 1, converged)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_sol, r_final, _, _, _, iters, conv = final_state

    return PCGResult(
        x=x_sol,
        residual_norm=jnp.linalg.norm(r_final),
        iterations=iters,
        converged=conv,
    )


# ---------------------------------------------------------------------------
# Condensed KKT solve via PCG (for IPM integration)
# ---------------------------------------------------------------------------


def solve_kkt_condensed_pcg(
    W: jnp.ndarray,
    J: jnp.ndarray,
    D_reg: jnp.ndarray,
    rhs_x: jnp.ndarray,
    rhs_y: jnp.ndarray,
    pcg_options: Optional[PCGOptions] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, PCGResult]:
    """Solve the augmented KKT system using condensed normal equations + PCG.

    The augmented KKT system is:
        [W     J^T] [dx]   [rhs_x]
        [J    -D  ] [dy] = [rhs_y]

    The condensed approach eliminates dx to get a positive definite system
    for dy:
        (J W^{-1} J^T + D) dy = J W^{-1} rhs_x - rhs_y
    Then back-substitutes:
        dx = W^{-1} (rhs_x - J^T dy)

    For unconstrained problems (m=0), directly solves W dx = rhs_x with PCG.

    Args:
        W: (n, n) SPD matrix (Hessian + barrier + regularization).
        J: (m, n) constraint Jacobian.
        D_reg: (m,) diagonal of regularization D (must be positive).
        rhs_x: (n,) primal RHS.
        rhs_y: (m,) dual RHS.
        pcg_options: PCGOptions for the PCG solve.

    Returns:
        Tuple of (dx, dy, pcg_result).
    """
    n = W.shape[0]
    m = J.shape[0]

    if m == 0:
        # No constraints: solve W dx = rhs_x directly
        precond = diagonal_preconditioner(W)
        result = pcg_solve(W, rhs_x, preconditioner=precond, options=pcg_options)
        return result.x, jnp.zeros(0, dtype=rhs_x.dtype), result

    # Step 1: Compute W^{-1} via Cholesky (W is SPD after inertia correction)
    # For very large systems, this could be replaced with an inner PCG solve,
    # but for T18 we use Cholesky for W^{-1} since W is typically well-conditioned
    # after inertia correction.
    W_inv = jnp.linalg.solve(W, jnp.eye(n))

    # Step 2: Form the condensed system S = J W^{-1} J^T + diag(D_reg)
    JWinv = J @ W_inv  # (m, n)
    S = JWinv @ J.T + jnp.diag(D_reg)  # (m, m) SPD

    # Step 3: Form condensed RHS: J W^{-1} rhs_x - rhs_y
    rhs_condensed = JWinv @ rhs_x - rhs_y

    # Step 4: Solve condensed system with PCG
    precond = diagonal_preconditioner(S)
    result = pcg_solve(S, rhs_condensed, preconditioner=precond, options=pcg_options)
    dy = result.x

    # Step 5: Back-substitute for dx = W^{-1} (rhs_x - J^T dy)
    dx = W_inv @ (rhs_x - J.T @ dy)

    return dx, dy, result
