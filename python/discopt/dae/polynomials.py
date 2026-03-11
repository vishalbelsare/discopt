"""Collocation point schemes, Lagrange basis functions, and differentiation matrices.

Pure numpy utilities -- no discopt dependency. These support both Radau IIA and
Gauss-Legendre collocation on finite elements.
"""

from __future__ import annotations

import numpy as np

# Radau IIA roots on [0, 1] (includes tau=1), hardcoded for ncp=1..5.
_RADAU_ROOTS: dict[int, np.ndarray] = {
    1: np.array([1.0]),
    2: np.array([1.0 / 3.0, 1.0]),
    3: np.array(
        [
            (4.0 - np.sqrt(6.0)) / 10.0,
            (4.0 + np.sqrt(6.0)) / 10.0,
            1.0,
        ]
    ),
    4: np.array(
        [
            0.088587959512704_4,
            0.409466864440735_6,
            0.787659461760847_1,
            1.0,
        ]
    ),
    5: np.array(
        [
            0.057104196114518_2,
            0.276843013638124_0,
            0.583590432368917_0,
            0.860240135656219_5,
            1.0,
        ]
    ),
}


def radau_roots(ncp: int) -> np.ndarray:
    """Radau IIA collocation points on [0, 1].

    The last point is always tau=1 (the element boundary). This ensures
    C^0 continuity between elements without an explicit continuity equation
    when using the Radau scheme.

    Parameters
    ----------
    ncp : int
        Number of collocation points (1-5).

    Returns
    -------
    np.ndarray
        Sorted collocation points, shape ``(ncp,)``.
    """
    if ncp not in _RADAU_ROOTS:
        raise ValueError(f"Radau roots only available for ncp=1..5, got {ncp}")
    return np.array(_RADAU_ROOTS[ncp])


def legendre_roots(ncp: int) -> np.ndarray:
    """Gauss-Legendre collocation points on (0, 1).

    Points are in the open interval -- neither endpoint is included. This
    gives higher order but requires separate continuity constraints.

    Parameters
    ----------
    ncp : int
        Number of collocation points (>= 1).

    Returns
    -------
    np.ndarray
        Sorted collocation points, shape ``(ncp,)``.
    """
    if ncp < 1:
        raise ValueError(f"ncp must be >= 1, got {ncp}")
    roots, _ = np.polynomial.legendre.leggauss(ncp)
    # Transform from [-1, 1] to [0, 1]
    return (roots + 1.0) / 2.0


def lagrange_basis(tau: np.ndarray, t: float, j: int) -> float:
    """Evaluate the j-th Lagrange basis polynomial at t.

    Parameters
    ----------
    tau : np.ndarray
        Interpolation nodes, shape ``(n,)``.
    t : float
        Evaluation point.
    j : int
        Index of the basis polynomial to evaluate.

    Returns
    -------
    float
        L_j(t) = prod_{k != j} (t - tau[k]) / (tau[j] - tau[k]).
    """
    n = len(tau)
    result = 1.0
    for k in range(n):
        if k != j:
            result *= (t - tau[k]) / (tau[j] - tau[k])
    return result


def collocation_matrix(ncp: int, scheme: str = "radau") -> tuple[np.ndarray, np.ndarray]:
    """Build the differentiation matrix and continuity weights for collocation.

    The full node set is ``[0, tau_1, ..., tau_ncp]`` where ``tau_i`` are the
    collocation points. The differentiation matrix ``A`` maps function values
    at all nodes to derivative values at the collocation points.

    Parameters
    ----------
    ncp : int
        Number of collocation points.
    scheme : str
        ``"radau"`` or ``"legendre"``.

    Returns
    -------
    A : np.ndarray, shape ``(ncp, ncp + 1)``
        Differentiation matrix. ``A[j, k]`` is the derivative of the k-th
        Lagrange basis polynomial (through the full node set) evaluated at
        collocation point ``tau_j``.
    w : np.ndarray, shape ``(ncp + 1,)``
        Continuity weights: Lagrange basis values at ``tau = 1``. Used for
        inter-element continuity: ``x_{i+1,0} = w @ x_{i,:}``.
    """
    if scheme == "radau":
        cp = radau_roots(ncp)
    elif scheme == "legendre":
        cp = legendre_roots(ncp)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}, expected 'radau' or 'legendre'")

    # Full node set: tau=0 followed by collocation points
    nodes = np.concatenate([[0.0], cp])  # shape (ncp+1,)

    # Differentiation matrix: derivative of j-th Lagrange basis at cp[i]
    A = np.zeros((ncp, ncp + 1))
    for i in range(ncp):
        for j in range(ncp + 1):
            # d/dt L_j(t) at t = cp[i]
            A[i, j] = _lagrange_deriv(nodes, cp[i], j)

    # Continuity weights: L_j(1) for each basis function
    w = np.array([lagrange_basis(nodes, 1.0, j) for j in range(ncp + 1)])

    return A, w


def _lagrange_deriv(tau: np.ndarray, t: float, j: int) -> float:
    """Derivative of the j-th Lagrange basis polynomial at t.

    Uses the standard formula:
        L_j'(t) = sum_{m != j} [ prod_{k != j, k != m} (t - tau[k]) ]
                  / prod_{k != j} (tau[j] - tau[k])
    """
    n = len(tau)
    denom = 1.0
    for k in range(n):
        if k != j:
            denom *= tau[j] - tau[k]

    result = 0.0
    for m in range(n):
        if m == j:
            continue
        prod = 1.0
        for k in range(n):
            if k != j and k != m:
                prod *= t - tau[k]
        result += prod
    return result / denom
