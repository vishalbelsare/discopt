"""
Convex Envelopes for Composite Operations.

Provides relaxation functions for trilinear products (x*y*z), fractional
expressions (x/y), and signomial terms (x^a for non-integer a).

All functions return (cv, cc) tuples where cv <= f(x) <= cc, are pure JAX,
and are compatible with jax.jit, jax.grad, and jax.vmap.
"""

from __future__ import annotations

import jax.numpy as jnp

from discopt._jax.mccormick import (
    _secant,
    relax_bilinear,
)


def relax_trilinear(x, y, z, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub):
    """McCormick relaxation of x*y*z via nested bilinear decomposition.

    Decomposes as (x*y)*z (Meyer-Floudas approach). First relaxes the
    bilinear term w = x*y, then relaxes w*z.

    Args:
        x, y, z: point values
        x_lb, x_ub: bounds on x
        y_lb, y_ub: bounds on y
        z_lb, z_ub: bounds on z

    Returns:
        (cv, cc) where cv <= x*y*z <= cc
    """
    # Step 1: relax w = x*y
    w_cv, w_cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)

    # Compute range bounds for w = x*y
    corners = jnp.array([x_lb * y_lb, x_lb * y_ub, x_ub * y_lb, x_ub * y_ub])
    w_lb = jnp.min(corners)
    w_ub = jnp.max(corners)

    # Step 2: relax w*z using bilinear relaxation
    # Use w_cv for underestimator composition and w_cc for overestimator
    # For soundness, compose both and take the tightest
    cv1, cc1 = relax_bilinear(w_cv, z, w_lb, w_ub, z_lb, z_ub)
    cv2, cc2 = relax_bilinear(w_cc, z, w_lb, w_ub, z_lb, z_ub)

    cv = jnp.minimum(cv1, cv2)
    cc = jnp.maximum(cc1, cc2)

    return cv, cc


def relax_fractional(x, y, x_lb, x_ub, y_lb, y_ub):
    """McCormick relaxation of x/y where y bounds exclude zero.

    Composes: x/y = x * (1/y). First relaxes 1/y using convex envelope
    (1/y is convex on positive or negative half-line), then uses bilinear
    relaxation for x * (1/y).

    Args:
        x: point value for numerator
        y: point value for denominator
        x_lb, x_ub: bounds on x
        y_lb, y_ub: bounds on y (must not contain 0)

    Returns:
        (cv, cc) where cv <= x/y <= cc
    """

    def recip(t):
        return 1.0 / t

    # 1/y is convex on (0, inf) and on (-inf, 0)
    recip_val = recip(y)
    recip_sec = _secant(recip, y, y_lb, y_ub)

    # cv(1/y) = 1/y (function value for convex), cc(1/y) = secant
    recip_cv = recip_val
    recip_cc = recip_sec

    # Bounds on 1/y (note the swap for positive y)
    r1 = recip(y_lb)
    r2 = recip(y_ub)
    recip_lb = jnp.minimum(r1, r2)
    recip_ub = jnp.maximum(r1, r2)

    # Compose x * (1/y) via bilinear relaxation using both cv and cc of 1/y
    cv1, cc1 = relax_bilinear(x, recip_cv, x_lb, x_ub, recip_lb, recip_ub)
    cv2, cc2 = relax_bilinear(x, recip_cc, x_lb, x_ub, recip_lb, recip_ub)

    cv = jnp.minimum(cv1, cv2)
    cc = jnp.maximum(cc1, cc2)

    return cv, cc


def relax_signomial(x, lb, ub, a):
    """McCormick relaxation of x^a for non-integer exponent a on [lb, ub].

    Handles different regimes based on sign of a and convexity:
      - a in (0,1): concave on (0,inf), cv = secant, cc = x^a
      - a > 1: convex on (0,inf), cv = x^a, cc = secant
      - a < 0: convex on (0,inf), cv = x^a, cc = secant

    Requires lb > 0 for all cases.

    Args:
        x: point value
        lb: lower bound (must be > 0)
        ub: upper bound
        a: real exponent

    Returns:
        (cv, cc) where cv <= x^a <= cc
    """

    def f(t):
        return t**a

    f_val = f(x)
    sec_val = _secant(f, x, lb, ub)

    # a in (0, 1): concave => cv = secant, cc = f(x)
    concave_cv = sec_val
    concave_cc = f_val

    # a > 1 or a < 0: convex => cv = f(x), cc = secant
    convex_cv = f_val
    convex_cc = sec_val

    is_concave = (a > 0.0) & (a < 1.0)

    cv = jnp.where(is_concave, concave_cv, convex_cv)
    cc = jnp.where(is_concave, concave_cc, convex_cc)

    return cv, cc
