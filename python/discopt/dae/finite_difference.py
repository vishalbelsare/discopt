"""Finite-difference discretization for ODE systems.

Provides :class:`FDBuilder` with the same high-level interface as
:class:`~discopt.dae.collocation.DAEBuilder` but using finite-difference
stencils (backward Euler, forward Euler, or central differences) instead
of orthogonal collocation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from discopt.dae.collocation import (
    ContinuousSet,
    ControlVar,
    StateVar,
)


class FDBuilder:
    """Transcribe an ODE system using finite-difference stencils.

    State variables have shape ``(nfe + 1,)`` — values at grid points
    (including both endpoints). Controls are piecewise constant per interval.

    Parameters
    ----------
    model : discopt.Model
        The optimization model.
    continuous_set : ContinuousSet
        Time domain (``ncp`` is ignored; grid spacing is ``h = (tf-t0)/nfe``).
    method : str
        ``"backward"`` (implicit Euler), ``"forward"`` (explicit Euler),
        or ``"central"`` (central differences).
    """

    def __init__(self, model, continuous_set: ContinuousSet, method: str = "backward"):
        if method not in ("backward", "forward", "central"):
            raise ValueError(f"Unknown method {method!r}")
        self._model = model
        self._cs = continuous_set
        self._method = method
        self._states: list[StateVar] = []
        self._controls: list[ControlVar] = []
        self._ode_rhs: Callable | None = None
        self._vars: dict[str, Any] = {}
        self._discretized = False

        if continuous_set.element_boundaries is not None:
            self._h_vec = np.diff(continuous_set.element_boundaries)
        else:
            h_uniform = (continuous_set.bounds[1] - continuous_set.bounds[0]) / continuous_set.nfe
            self._h_vec = np.full(continuous_set.nfe, h_uniform)

    def add_state(
        self,
        name: str,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
        initial: float | np.ndarray | None = None,
    ) -> StateVar:
        """Declare a state variable (same interface as DAEBuilder)."""
        sv = StateVar(name, n_components, bounds, initial)
        self._states.append(sv)
        return sv

    def add_control(
        self,
        name: str,
        n_components: int = 1,
        bounds: tuple[float, float] = (-1e20, 1e20),
    ) -> ControlVar:
        """Declare a piecewise-constant control variable."""
        cv = ControlVar(name, n_components, bounds)
        self._controls.append(cv)
        return cv

    def set_ode(self, rhs: Callable) -> None:
        """Set the ODE right-hand side.

        Parameters
        ----------
        rhs : callable
            ``rhs(t, states, algebraics, controls) -> dict[str, Expression]``
            Same signature as :meth:`DAEBuilder.set_ode`.
        """
        self._ode_rhs = rhs

    def discretize(self) -> dict[str, object]:
        """Generate all variables and finite-difference constraints.

        Returns
        -------
        dict[str, Variable]
        """
        if self._discretized:
            raise RuntimeError("FDBuilder.discretize() has already been called")
        self._discretized = True

        m = self._model
        cs = self._cs
        nfe = cs.nfe

        # State variables: shape (nfe+1,) or (nfe+1, n_comp)
        for sv in self._states:
            shape: tuple[int, ...] = (nfe + 1,)
            if sv.n_components > 1:
                shape = (nfe + 1, sv.n_components)

            lb = np.full(shape, sv.bounds[0], dtype=np.float64)
            ub = np.full(shape, sv.bounds[1], dtype=np.float64)
            if sv.initial is not None:
                if sv.n_components == 1:
                    lb[0] = sv.initial
                    ub[0] = sv.initial
                else:
                    lb[0, ...] = sv.initial
                    ub[0, ...] = sv.initial

            self._vars[sv.name] = m.continuous(f"{cs.name}_{sv.name}", shape=shape, lb=lb, ub=ub)

        # Control variables: shape (nfe,) or (nfe, n_comp)
        for cv in self._controls:
            cshape: tuple[int, ...] = (nfe,)
            if cv.n_components > 1:
                cshape = (nfe, cv.n_components)

            self._vars[cv.name] = m.continuous(
                f"{cs.name}_{cv.name}",
                shape=cshape,
                lb=cv.bounds[0],
                ub=cv.bounds[1],
            )

        # Add FD constraints
        if self._ode_rhs is None:
            raise RuntimeError("No ODE RHS set. Call set_ode() first.")

        tp = self.time_points()
        constraints = []

        if self._method == "backward":
            # (x[k] - x[k-1]) / h_k == f(t[k], x[k])  for k=1..nfe
            for k in range(1, nfe + 1):
                h_k = float(self._h_vec[k - 1])
                states_k = self._state_dict_at(k)
                ctrl_k = self._control_dict_at(k - 1)  # interval [k-1, k]
                derivs = self._ode_rhs(tp[k], states_k, {}, ctrl_k)

                for sv in self._states:
                    if sv.name not in derivs:
                        continue
                    var = self._vars[sv.name]
                    if sv.n_components == 1:
                        lhs = (var[k] - var[k - 1]) / h_k
                        constraints.append(lhs == derivs[sv.name])
                    else:
                        for c in range(sv.n_components):
                            lhs = (var[k, c] - var[k - 1, c]) / h_k
                            constraints.append(lhs == derivs[sv.name][c])

        elif self._method == "forward":
            # (x[k+1] - x[k]) / h_k == f(t[k], x[k])  for k=0..nfe-1
            for k in range(nfe):
                h_k = float(self._h_vec[k])
                states_k = self._state_dict_at(k)
                ctrl_k = self._control_dict_at(k)
                derivs = self._ode_rhs(tp[k], states_k, {}, ctrl_k)

                for sv in self._states:
                    if sv.name not in derivs:
                        continue
                    var = self._vars[sv.name]
                    if sv.n_components == 1:
                        lhs = (var[k + 1] - var[k]) / h_k
                        constraints.append(lhs == derivs[sv.name])
                    else:
                        for c in range(sv.n_components):
                            lhs = (var[k + 1, c] - var[k, c]) / h_k
                            constraints.append(lhs == derivs[sv.name][c])

        elif self._method == "central":
            # (x[k+1] - x[k-1]) / (h_{k-1} + h_k) == f(t[k], x[k])
            for k in range(1, nfe):
                h_span = float(self._h_vec[k - 1] + self._h_vec[k])
                states_k = self._state_dict_at(k)
                ctrl_k = self._control_dict_at(k - 1 if k > 0 else k)
                derivs = self._ode_rhs(tp[k], states_k, {}, ctrl_k)

                for sv in self._states:
                    if sv.name not in derivs:
                        continue
                    var = self._vars[sv.name]
                    if sv.n_components == 1:
                        lhs = (var[k + 1] - var[k - 1]) / h_span
                        constraints.append(lhs == derivs[sv.name])
                    else:
                        for c in range(sv.n_components):
                            lhs = (var[k + 1, c] - var[k - 1, c]) / h_span
                            constraints.append(lhs == derivs[sv.name][c])

            # Also need the last point: backward at k=nfe
            k = nfe
            h_k = float(self._h_vec[k - 1])
            states_k = self._state_dict_at(k)
            ctrl_k = self._control_dict_at(k - 1)
            derivs = self._ode_rhs(tp[k], states_k, {}, ctrl_k)
            for sv in self._states:
                if sv.name not in derivs:
                    continue
                var = self._vars[sv.name]
                if sv.n_components == 1:
                    lhs = (var[k] - var[k - 1]) / h_k
                    constraints.append(lhs == derivs[sv.name])
                else:
                    for c in range(sv.n_components):
                        lhs = (var[k, c] - var[k - 1, c]) / h_k
                        constraints.append(lhs == derivs[sv.name][c])

        if constraints:
            m.subject_to(constraints, name=f"{cs.name}_fd")

        return dict(self._vars)

    def get_state(self, name: str):
        """Get the discopt Variable for a state by name."""
        if name not in self._vars:
            raise KeyError(f"Unknown variable {name!r}")
        return self._vars[name]

    def time_points(self) -> np.ndarray:
        """Return grid points as a flat array, shape ``(nfe + 1,)``."""
        cs = self._cs
        if cs.element_boundaries is not None:
            return np.array(cs.element_boundaries)
        return np.linspace(cs.bounds[0], cs.bounds[1], cs.nfe + 1)

    def extract_solution(self, result, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series from a solve result."""
        var = self._vars[name]
        val = result.value(var)
        return self.time_points(), val

    def least_squares(
        self,
        state_name: str,
        t_data: np.ndarray,
        y_data: np.ndarray,
        component: int | None = None,
    ) -> object:
        """Build a sum-of-squared-residuals expression for parameter estimation.

        Maps each measurement time to the nearest grid point and returns
        ``sum((x[nearest] - y_data[i])^2)`` as a discopt expression.

        Must be called after :meth:`discretize`.

        Parameters
        ----------
        state_name : str
            Name of the state variable to fit.
        t_data : np.ndarray
            Measurement times, shape ``(n_obs,)``.
        y_data : np.ndarray
            Observed values, shape ``(n_obs,)``.
        component : int, optional
            For vector-valued states, which component to fit.

        Returns
        -------
        Expression
        """
        if not self._discretized:
            raise RuntimeError("Call discretize() before least_squares()")
        if state_name not in self._vars:
            raise KeyError(f"Unknown state {state_name!r}")

        t_data = np.asarray(t_data, dtype=np.float64)
        y_data = np.asarray(y_data, dtype=np.float64)

        var = self._vars[state_name]
        tp = self.time_points()

        terms = []
        for i in range(len(t_data)):
            idx = int(np.argmin(np.abs(tp - t_data[i])))
            if component is not None:
                x_expr = var[idx, component]
            else:
                x_expr = var[idx]
            terms.append((x_expr - float(y_data[i])) ** 2)

        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    # ── Internal helpers ──

    def _state_dict_at(self, k: int) -> dict:
        """Build states dict at grid point k."""
        d = {}
        for sv in self._states:
            var = self._vars[sv.name]
            if sv.n_components == 1:
                d[sv.name] = var[k]
            else:
                d[sv.name] = [var[k, c] for c in range(sv.n_components)]
        return d

    def _control_dict_at(self, k: int) -> dict:
        """Build controls dict at interval k."""
        d = {}
        for cv in self._controls:
            var = self._vars[cv.name]
            if cv.n_components == 1:
                d[cv.name] = var[k]
            else:
                d[cv.name] = [var[k, c] for c in range(cv.n_components)]
        return d
