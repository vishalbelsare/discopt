"""
Pure-JAX QP Interior Point Method (Mehrotra predictor-corrector).

Solves convex QPs of the form:

    min  0.5 x'Qx + c'x
    s.t. A x = b          (m equality constraints)
         x_l <= x <= x_u   (variable bounds)

Implements a primal-dual Mehrotra predictor-corrector IPM with:
  - Augmented KKT system with Q precomputed once (not per iteration)
  - Predictor (affine) step followed by corrector with centering
  - Fraction-to-boundary rule for step sizes
  - Inertia-correcting diagonal perturbation
  - jax.lax.while_loop for JIT-compatible iteration

All data structures are NamedTuples for JAX pytree compatibility.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-20
_INF = 1e20
_SLACK_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Data structures (NamedTuples for JAX pytree compatibility)
# ---------------------------------------------------------------------------


class QPIPMOptions(NamedTuple):
    """Options for the QP IPM solver."""

    tol: float = 1e-8
    max_iter: int = 100
    tau_min: float = 0.99
    bound_push: float = 1e-2


class QPIPMState(NamedTuple):
    """State carried through the while_loop."""

    x: jnp.ndarray  # (n,) primal variables
    y: jnp.ndarray  # (m,) equality constraint multipliers
    z_l: jnp.ndarray  # (n,) lower bound multipliers
    z_u: jnp.ndarray  # (n,) upper bound multipliers
    mu: jnp.ndarray  # scalar barrier parameter
    iteration: jnp.ndarray  # scalar int
    converged: jnp.ndarray  # 0=running, 1=optimal, 3=max_iter
    obj: jnp.ndarray  # scalar objective value


class QPProblemData(NamedTuple):
    """Pre-computed QP problem structure."""

    Q: jnp.ndarray  # (n, n) symmetric positive semi-definite
    c: jnp.ndarray  # (n,)
    A: jnp.ndarray  # (m, n)
    b: jnp.ndarray  # (m,)
    x_l: jnp.ndarray  # (n,) lower variable bounds
    x_u: jnp.ndarray  # (n,) upper variable bounds
    has_lb: jnp.ndarray  # (n,) float mask
    has_ub: jnp.ndarray  # (n,) float mask
    n: int
    m: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fraction_to_boundary(
    vals: jnp.ndarray,
    dvals: jnp.ndarray,
    tau: float,
) -> jnp.ndarray:
    """Max step alpha s.t. vals + alpha*dvals >= (1-tau)*vals."""
    neg_mask = dvals < 0.0
    ratios = jnp.where(
        neg_mask,
        -tau * vals / jnp.where(neg_mask, dvals, -1.0),
        jnp.array(1e20),
    )
    return jnp.clip(jnp.min(ratios, initial=1e20), 0.0, 1.0)


def _make_problem_data(Q, c, A, b, x_l, x_u) -> QPProblemData:
    """Build QPProblemData with bound masks."""
    n = c.shape[0]
    m = b.shape[0]
    has_lb = (x_l > -_INF).astype(jnp.float64)
    has_ub = (x_u < _INF).astype(jnp.float64)
    return QPProblemData(
        Q=Q,
        c=c,
        A=A,
        b=b,
        x_l=x_l,
        x_u=x_u,
        has_lb=has_lb,
        has_ub=has_ub,
        n=n,
        m=m,
    )


def _push_from_bounds(x, x_l, x_u, has_lb, has_ub, bp):
    """Push x away from bounds by at least bp."""
    rng = x_u - x_l
    push = jnp.minimum(bp, 0.5 * rng)
    push = jnp.where(rng < 1e30, push, bp)
    x_new = x
    x_new = jnp.where((has_lb > 0.5) & (x_new < x_l + push), x_l + push, x_new)
    x_new = jnp.where((has_ub > 0.5) & (x_new > x_u - push), x_u - push, x_new)
    x_new = jnp.where(has_lb > 0.5, jnp.maximum(x_new, x_l), x_new)
    x_new = jnp.where(has_ub > 0.5, jnp.minimum(x_new, x_u), x_new)
    return x_new


def _objective(Q, c, x):
    """Evaluate 0.5 x'Qx + c'x."""
    return 0.5 * x @ Q @ x + c @ x


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_state(pd, opts):
    """Create the initial QPIPMState."""
    m = pd.m
    mu = jnp.array(0.1, dtype=jnp.float64)

    # Start at midpoint of bounds where finite, else 0
    x0 = jnp.where(
        (pd.has_lb > 0.5) & (pd.has_ub > 0.5),
        0.5 * (pd.x_l + pd.x_u),
        jnp.where(pd.has_lb > 0.5, pd.x_l + 1.0, jnp.where(pd.has_ub > 0.5, pd.x_u - 1.0, 0.0)),
    )
    x = _push_from_bounds(x0, pd.x_l, pd.x_u, pd.has_lb, pd.has_ub, opts.bound_push)

    s_l = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
    s_u = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub
    z_l = jnp.where(pd.has_lb > 0.5, mu / jnp.maximum(s_l, _SLACK_FLOOR), 0.0)
    z_u = jnp.where(pd.has_ub > 0.5, mu / jnp.maximum(s_u, _SLACK_FLOOR), 0.0)
    y = jnp.zeros(m, dtype=jnp.float64)
    obj = _objective(pd.Q, pd.c, x)

    return QPIPMState(
        x=x,
        y=y,
        z_l=z_l,
        z_u=z_u,
        mu=mu,
        iteration=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(0, dtype=jnp.int32),
        obj=jnp.array(obj, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Augmented system solve
# ---------------------------------------------------------------------------


def _solve_augmented(W, A, rhs_x, rhs_y, n, m):
    """Solve the augmented KKT system.

    [(Q + Sigma)  -A'] [dx]   [rhs_x]
    [A             0 ] [dy] = [rhs_y]

    For m=0, solves just W @ dx = rhs_x.
    """
    if m > 0:
        KKT = jnp.block(
            [
                [W, -A.T],
                [A, jnp.zeros((m, m), dtype=jnp.float64)],
            ]
        )
        rhs = jnp.concatenate([rhs_x, rhs_y])
        sol = jnp.linalg.solve(KKT, rhs)
        return sol[:n], sol[n:]
    else:
        dx = jnp.linalg.solve(W, rhs_x)
        return dx, jnp.zeros(0, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Core iteration body (Mehrotra predictor-corrector)
# ---------------------------------------------------------------------------


def _make_iteration_body(pd, opts):
    """Build the while_loop body for one QP IPM iteration.

    Uses Mehrotra predictor-corrector:
      1. Affine step (mu=0) to get search direction
      2. Centering parameter sigma = (mu_aff / mu)^3
      3. Corrected step with centering and second-order terms
    """
    n, m = pd.n, pd.m
    Q, c, A, b = pd.Q, pd.c, pd.A, pd.b

    def body(state):
        x, y, mu = state.x, state.y, state.mu
        z_l, z_u = state.z_l, state.z_u
        tau = jnp.maximum(1.0 - mu, opts.tau_min)

        # Slacks from bounds
        s_l = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
        s_u = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub

        # Barrier Hessian diagonal: Sigma = z_l/s_l + z_u/s_u
        Sig = pd.has_lb * z_l / jnp.maximum(s_l, _EPS) + pd.has_ub * z_u / jnp.maximum(s_u, _EPS)

        # Build W = Q + diag(Sigma) (before inertia correction)
        W_base = Q + jnp.diag(Sig)

        # Inertia correction: ensure W is positive definite
        def _needs_more_reg(carry):
            dw, attempt = carry
            W_trial = W_base + dw * jnp.eye(n)
            eig_min = jnp.min(jnp.linalg.eigvalsh(W_trial))
            bad = jnp.any(jnp.isnan(eig_min)) | (eig_min < 1e-8)
            return bad & (attempt < 10)

        def _increase_reg(carry):
            dw, attempt = carry
            dw_next = jnp.maximum(dw * 8.0, 1e-4)
            return (dw_next, attempt + 1)

        final_dw, _ = jax.lax.while_loop(
            _needs_more_reg,
            _increase_reg,
            (jnp.array(0.0, dtype=jnp.float64), jnp.array(0, dtype=jnp.int32)),
        )
        W = W_base + final_dw * jnp.eye(n)

        # Residuals
        # Stationarity: Qx + c - A'y - z_l + z_u = 0
        r_dual = Q @ x + c - z_l + z_u
        if m > 0:
            r_dual = r_dual - A.T @ y

        # Primal feasibility: Ax - b = 0
        if m > 0:
            r_prim = A @ x - b
        else:
            r_prim = jnp.zeros(0, dtype=jnp.float64)

        # Complementarity: (x - x_l) * z_l and (x_u - x) * z_u
        r_comp_l = pd.has_lb * s_l * z_l
        r_comp_u = pd.has_ub * s_u * z_u

        # =====================================================================
        # Step 1: Affine (predictor) direction (mu = 0)
        # =====================================================================
        # rhs_x = -r_dual - has_lb*(r_comp_l/s_l) + has_ub*(r_comp_u/s_u)
        # Simplified: -r_dual - has_lb*z_l + has_ub*z_u  (since r_comp/s = z)
        # But more precisely for the affine step:
        # rhs_x_aff = -(r_dual) - Sig_l * s_l_resid + Sig_u * s_u_resid
        # where for affine: s_l_resid = s_l * z_l / s_l = z_l, etc.
        # Actually: rhs_x = -r_dual + has_lb*(mu_target/s_l - z_l) - has_ub*(mu_target/s_u - z_u)
        # For affine step, mu_target = 0:
        rhs_x_aff = -r_dual - pd.has_lb * z_l + pd.has_ub * z_u
        rhs_y_aff = -r_prim

        dx_aff, dy_aff = _solve_augmented(W, A, rhs_x_aff, rhs_y_aff, n, m)

        # Recover affine dual steps
        dz_l_aff = pd.has_lb * ((-r_comp_l - z_l * dx_aff) / jnp.maximum(s_l, _EPS))
        dz_u_aff = pd.has_ub * ((-r_comp_u + z_u * dx_aff) / jnp.maximum(s_u, _EPS))

        # Affine step sizes (fraction to boundary)
        alpha_aff_p = jnp.array(1.0)
        alpha_aff_p = jnp.minimum(
            alpha_aff_p,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, s_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dx_aff, 0.0),
                1.0,
            ),
        )
        alpha_aff_p = jnp.minimum(
            alpha_aff_p,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, s_u, 1.0),
                jnp.where(pd.has_ub > 0.5, -dx_aff, 0.0),
                1.0,
            ),
        )

        alpha_aff_d = jnp.array(1.0)
        alpha_aff_d = jnp.minimum(
            alpha_aff_d,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dz_l_aff, 0.0),
                1.0,
            ),
        )
        alpha_aff_d = jnp.minimum(
            alpha_aff_d,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                jnp.where(pd.has_ub > 0.5, dz_u_aff, 0.0),
                1.0,
            ),
        )

        # =====================================================================
        # Step 2: Centering parameter sigma = (mu_aff / mu)^3
        # =====================================================================
        n_pairs = jnp.maximum(jnp.sum(pd.has_lb) + jnp.sum(pd.has_ub), 1.0)

        # Affine complementarity after affine step
        s_l_aff = s_l + alpha_aff_p * dx_aff
        s_u_aff = s_u - alpha_aff_p * dx_aff
        z_l_aff = z_l + alpha_aff_d * dz_l_aff
        z_u_aff = z_u + alpha_aff_d * dz_u_aff

        mu_aff = (
            jnp.sum(pd.has_lb * s_l_aff * z_l_aff) + jnp.sum(pd.has_ub * s_u_aff * z_u_aff)
        ) / n_pairs

        sigma = (mu_aff / jnp.maximum(mu, _EPS)) ** 3
        sigma = jnp.clip(sigma, 0.0, 1.0)
        mu_target = sigma * mu

        # =====================================================================
        # Step 3: Corrected (centering + second-order) direction
        # =====================================================================
        # Add centering (mu_target) and Mehrotra second-order correction
        # (dx_aff * dz_l_aff for lower, -dx_aff * dz_u_aff for upper)
        corr_l = pd.has_lb * (dx_aff * dz_l_aff) / jnp.maximum(s_l, _EPS)
        corr_u = pd.has_ub * (dx_aff * dz_u_aff) / jnp.maximum(s_u, _EPS)

        rhs_x_cc = (
            -r_dual
            + pd.has_lb * (mu_target / jnp.maximum(s_l, _EPS) - z_l)
            - pd.has_ub * (mu_target / jnp.maximum(s_u, _EPS) - z_u)
            - corr_l
            - corr_u
        )
        rhs_y_cc = -r_prim

        dx, dy = _solve_augmented(W, A, rhs_x_cc, rhs_y_cc, n, m)

        # Recover corrected dual steps
        dz_l = pd.has_lb * (
            (mu_target - z_l * (s_l + dx) - dx_aff * dz_l_aff) / jnp.maximum(s_l, _EPS)
        )
        dz_u = pd.has_ub * (
            (mu_target - z_u * (s_u - dx) + dx_aff * dz_u_aff) / jnp.maximum(s_u, _EPS)
        )

        # =====================================================================
        # Step sizes (fraction to boundary with tau)
        # =====================================================================
        alpha_p = jnp.array(1.0)
        alpha_p = jnp.minimum(
            alpha_p,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, s_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dx, 0.0),
                tau,
            ),
        )
        alpha_p = jnp.minimum(
            alpha_p,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, s_u, 1.0),
                jnp.where(pd.has_ub > 0.5, -dx, 0.0),
                tau,
            ),
        )

        alpha_d = jnp.array(1.0)
        alpha_d = jnp.minimum(
            alpha_d,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dz_l, 0.0),
                tau,
            ),
        )
        alpha_d = jnp.minimum(
            alpha_d,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                jnp.where(pd.has_ub > 0.5, dz_u, 0.0),
                tau,
            ),
        )

        # =====================================================================
        # Update variables
        # =====================================================================
        x_new = x + alpha_p * dx
        x_new = jnp.where(
            pd.has_lb > 0.5,
            jnp.maximum(x_new, pd.x_l + _SLACK_FLOOR),
            x_new,
        )
        x_new = jnp.where(
            pd.has_ub > 0.5,
            jnp.minimum(x_new, pd.x_u - _SLACK_FLOOR),
            x_new,
        )

        z_l_new = jnp.maximum(z_l + alpha_d * dz_l, _EPS) * pd.has_lb
        z_u_new = jnp.maximum(z_u + alpha_d * dz_u, _EPS) * pd.has_ub

        if m > 0:
            y_new = y + alpha_d * dy
        else:
            y_new = y

        # =====================================================================
        # Stationarity-based z recovery at active bounds
        # =====================================================================
        # When x is at a bound (slack ~ 0), the Newton step for z is
        # numerically unstable. Recover z from stationarity instead:
        #   Qx + c - A'y - z_l + z_u = 0
        s_l_new = jnp.maximum(x_new - pd.x_l, _SLACK_FLOOR) * pd.has_lb
        s_u_new = jnp.maximum(pd.x_u - x_new, _SLACK_FLOOR) * pd.has_ub

        grad_stat = Q @ x_new + c
        if m > 0:
            grad_stat = grad_stat - A.T @ y_new
        # At upper bound: z_u = -(grad_stat) + z_l  => z_u = -(Qx+c-A'y) + z_l
        # At lower bound: z_l = (grad_stat) + z_u    => z_l = (Qx+c-A'y) + z_u
        z_u_stat = jnp.maximum(-(grad_stat - z_l_new), _EPS)
        z_l_stat = jnp.maximum(grad_stat + z_u_new, _EPS)
        at_ub = pd.has_ub * (s_u_new <= _SLACK_FLOOR * 2.0).astype(jnp.float64)
        at_lb = pd.has_lb * (s_l_new <= _SLACK_FLOOR * 2.0).astype(jnp.float64)
        # Avoid circular dependency when both bounds active
        both = at_lb * at_ub
        at_lb = at_lb * (1.0 - both)
        at_ub = at_ub * (1.0 - both)
        z_u_new = jnp.where(at_ub > 0.5, z_u_stat, z_u_new)
        z_l_new = jnp.where(at_lb > 0.5, z_l_stat, z_l_new)

        # =====================================================================
        # Update barrier parameter
        # =====================================================================
        compl = jnp.sum(pd.has_lb * z_l_new * s_l_new) + jnp.sum(pd.has_ub * z_u_new * s_u_new)
        mu_new = compl / jnp.maximum(n_pairs, 1.0)
        # Never increase mu beyond current value
        mu_new = jnp.minimum(mu_new, mu)
        mu_new = jnp.maximum(mu_new, _EPS)

        # =====================================================================
        # Check convergence
        # =====================================================================
        # Dual infeasibility: ||Qx + c - A'y - z_l + z_u|| / (1 + ||c||)
        r_dual_new = Q @ x_new + c - z_l_new + z_u_new
        if m > 0:
            r_dual_new = r_dual_new - A.T @ y_new
        dual_inf = jnp.max(jnp.abs(r_dual_new)) / (1.0 + jnp.max(jnp.abs(c)))

        # Primal infeasibility: ||Ax - b|| / (1 + ||b||)
        if m > 0:
            r_prim_new = A @ x_new - b
            primal_inf = jnp.max(jnp.abs(r_prim_new)) / (1.0 + jnp.max(jnp.abs(b)))
        else:
            primal_inf = jnp.array(0.0)

        # Gap: |obj_p - obj_d| / (1 + |obj_p|)
        obj_p = _objective(Q, c, x_new)
        # Dual objective: 0.5 x'Qx + c'x is same form; gap from complementarity
        # Use complementarity gap = mu as proxy
        gap = mu_new / (1.0 + jnp.abs(obj_p))

        converged = (primal_inf <= opts.tol) & (dual_inf <= opts.tol) & (gap <= opts.tol)
        new_iter = state.iteration + 1
        at_max = new_iter >= opts.max_iter

        code = jnp.where(converged, jnp.int32(1), jnp.int32(0))
        code = jnp.where((code == 0) & at_max, jnp.int32(3), code)

        return QPIPMState(
            x=x_new,
            y=y_new,
            z_l=z_l_new,
            z_u=z_u_new,
            mu=mu_new,
            iteration=new_iter,
            converged=code.astype(jnp.int32),
            obj=obj_p,
        )

    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def qp_ipm_solve(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    x_l: jnp.ndarray,
    x_u: jnp.ndarray,
    options: QPIPMOptions | None = None,
) -> QPIPMState:
    """Solve a QP using a pure-JAX Mehrotra predictor-corrector IPM.

    Solves:
        min  0.5 x'Qx + c'x
        s.t. A x = b
             x_l <= x <= x_u

    Q is precomputed once and remains constant throughout all iterations.
    No calls to jax.hessian or jax.grad are made inside the iteration loop.

    Args:
        Q: (n, n) symmetric positive semi-definite Hessian.
        c: (n,) linear cost vector.
        A: (m, n) equality constraint matrix. Pass jnp.zeros((0, n)) for no
           equality constraints.
        b: (m,) equality constraint rhs. Pass jnp.zeros(0) for none.
        x_l: (n,) lower variable bounds. Use -1e20 for unbounded below.
        x_u: (n,) upper variable bounds. Use 1e20 for unbounded above.
        options: QPIPMOptions tuning parameters.

    Returns:
        QPIPMState with the solution in state.x and convergence info in
        state.converged (1=optimal, 3=max_iter).
    """
    opts = options if options is not None else QPIPMOptions()
    Q = jnp.asarray(Q, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    x_l = jnp.asarray(x_l, dtype=jnp.float64)
    x_u = jnp.asarray(x_u, dtype=jnp.float64)

    pd = _make_problem_data(Q, c, A, b, x_l, x_u)
    state = _initialize_state(pd, opts)
    body = _make_iteration_body(pd, opts)

    def cond(st):
        return st.converged == 0

    return jax.lax.while_loop(cond, body, state)  # type: ignore[no-any-return]


def qp_ipm_solve_batch(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    xl_batch: jnp.ndarray,
    xu_batch: jnp.ndarray,
    options: QPIPMOptions | None = None,
) -> QPIPMState:
    """Solve a batch of QPs in parallel using jax.vmap over variable bounds.

    All instances share Q, c, A, b but have per-instance variable bounds.

    Args:
        Q: (n, n) shared Hessian.
        c: (n,) shared linear cost.
        A: (m, n) shared constraint matrix.
        b: (m,) shared constraint rhs.
        xl_batch: (batch, n) per-instance lower bounds.
        xu_batch: (batch, n) per-instance upper bounds.
        options: QPIPMOptions.

    Returns:
        QPIPMState with batched arrays (batch, ...).
    """

    def _solve_single(xl_single, xu_single):
        return qp_ipm_solve(Q, c, A, b, xl_single, xu_single, options)

    return jax.vmap(_solve_single)(xl_batch, xu_batch)  # type: ignore[no-any-return]
