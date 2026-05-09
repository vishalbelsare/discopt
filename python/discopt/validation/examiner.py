"""Examiner-style post-solve validation.

Replicates the structure of GAMS Examiner (https://www.gams.com/latest/docs/S_EXAMINER.html)
in pure Python: re-evaluate the model at the returned point, recover
multipliers from the active set when the solver does not expose them, and
report the seven first-order KKT residuals plus integrality and objective
consistency. We do not call GAMS — every check uses ``NLPEvaluator`` and
``scipy.optimize.lsq_linear``.

Tolerances follow Examiner's defaults where they exist; integrality has no
Examiner counterpart (Examiner fixes integers and runs continuous KKT) and
uses a discopt-conventional ``1e-5``.

For MIP / MINLP the discrete variables are fixed at their incumbent
(rounded) value before dual recovery, so the KKT system is solved only over
the continuous columns of the Jacobian — matching Examiner's "fix and
re-check" handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import lsq_linear

from discopt.modeling.core import Model, ObjectiveSense, VarType

PRIMAL_FEAS_TOL = 1e-6
DUAL_FEAS_TOL = 1e-6
PRIMAL_CS_TOL = 1e-7
DUAL_CS_TOL = 1e-7
INTEGRALITY_TOL = 1e-5
OBJ_TOL = 1e-6
SHOW_TOL = 1e-4
ACTIVE_TOL = 1e-6


@dataclass
class CheckResult:
    """One Examiner check outcome."""

    name: str
    passed: bool
    tolerance: float
    max_violation: float = 0.0
    norm2_violation: float = 0.0
    worst_label: str = ""
    detail: str = ""
    violators: list[tuple[str, float]] = field(default_factory=list)

    def line(self) -> str:
        if self.passed:
            return f"  [PASS] {self.name} (tol={self.tolerance:.1e})"
        return (
            f"  [FAIL] {self.name}: max={self.max_violation:.3e} "
            f"(2-norm={self.norm2_violation:.3e}) at {self.worst_label} "
            f"-- {self.detail}"
        )


@dataclass
class ExaminerReport:
    """Aggregate of all Examiner checks at a returned point."""

    checks: list[CheckResult]
    merit: float = 0.0
    n_active_constraints: int = 0
    n_active_bounds: int = 0
    duals_recovered: bool = False
    dual_recovery_residual: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def first_failure(self) -> Optional[CheckResult]:
        for c in self.checks:
            if not c.passed:
                return c
        return None

    def summary(self, *, verbose: bool = False) -> str:
        head = (
            f"ExaminerReport: {'PASS' if self.passed else 'FAIL'} "
            f"merit={self.merit:.3e} "
            f"active_cons={self.n_active_constraints} "
            f"active_bounds={self.n_active_bounds}"
        )
        if not verbose and self.passed:
            return head
        body = [head] + [c.line() for c in self.checks]
        if verbose:
            for c in self.checks:
                if c.violators:
                    body.append(
                        f"    {c.name} top violators (>{SHOW_TOL:.0e}): "
                        + ", ".join(f"{lbl}={v:.3e}" for lbl, v in c.violators[:8])
                    )
        return "\n".join(body)


def examine(
    result,
    model: Model,
    *,
    primal_feas_tol: float = PRIMAL_FEAS_TOL,
    dual_feas_tol: float = DUAL_FEAS_TOL,
    primal_cs_tol: float = PRIMAL_CS_TOL,
    dual_cs_tol: float = DUAL_CS_TOL,
    integrality_tol: float = INTEGRALITY_TOL,
    obj_tol: float = OBJ_TOL,
    active_tol: float = ACTIVE_TOL,
    show_tol: float = SHOW_TOL,
    recover_duals: bool = True,
) -> ExaminerReport:
    """Run all Examiner checks at ``result.x``.

    Returns an :class:`ExaminerReport`; never raises. The aggregate
    ``merit`` field is the sum of squared violations across checks (zero
    at a true KKT point).
    """
    if result.x is None:
        raise ValueError("examine() requires a primal point in result.x")

    from discopt._jax.nlp_evaluator import NLPEvaluator

    parts: list[np.ndarray] = []
    is_integer_mask: list[bool] = []
    var_names: list[str] = []
    for v in model._variables:
        sz = int(v.size)
        if v.name not in result.x:
            raise ValueError(f"Variable {v.name!r} missing from result.x")
        parts.append(np.asarray(result.x[v.name], dtype=float).reshape(-1))
        is_integer_mask.extend([v.var_type in (VarType.BINARY, VarType.INTEGER)] * sz)
        var_names.extend([f"{v.name}[{i}]" if sz > 1 else v.name for i in range(sz)])
    x_flat = np.concatenate(parts) if parts else np.empty(0, dtype=float)
    is_integer = np.asarray(is_integer_mask, dtype=bool)
    is_continuous = ~is_integer

    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    assert model._objective is not None  # NLPEvaluator() raises otherwise
    obj_sense = model._objective.sense

    sense_arr, rhs_arr, row_labels = _row_metadata(evaluator)
    body = evaluator.evaluate_constraints(x_flat) if sense_arr.size else np.empty(0, dtype=float)

    checks: list[CheckResult] = []

    # ── 1. Primal variable feasibility ──────────────────────────────────────
    checks.append(_check_var_bounds(x_flat, lb, ub, var_names, primal_feas_tol, show_tol))

    # ── 2. Primal constraint feasibility (unscaled + scaled) ────────────────
    jac = (
        evaluator.evaluate_jacobian(x_flat)
        if sense_arr.size
        else np.empty((0, x_flat.size), dtype=float)
    )
    if body.size:
        viol_unscaled, _signed = _constraint_violations(body, sense_arr, rhs_arr)
        checks.append(
            _build_check(
                "primal_con_feas (unscaled)",
                viol_unscaled,
                row_labels,
                primal_feas_tol,
                show_tol,
                detail_template="row {label}: body={body:.6g} {sense} {rhs:.6g}",
                body=body,
                sense=sense_arr,
                rhs=rhs_arr,
            )
        )
        # Examiner's scaled mode: row_scale = max(|RHS|, max|J_row|·max(1,|x|)).
        if jac.size:
            jac_scale = np.max(np.abs(jac) * np.maximum(1.0, np.abs(x_flat))[None, :], axis=1)
            row_scale = np.maximum(np.abs(rhs_arr), jac_scale)
            row_scale = np.maximum(row_scale, 1.0)
            viol_scaled = viol_unscaled / row_scale
            checks.append(
                _build_check(
                    "primal_con_feas (scaled)",
                    viol_scaled,
                    row_labels,
                    primal_feas_tol,
                    show_tol,
                    detail_template="row {label} (scale={scale:.3g})",
                    scale=row_scale,
                )
            )

    # ── 3. Integrality ──────────────────────────────────────────────────────
    if is_integer.any():
        int_resid = np.where(is_integer, np.abs(x_flat - np.round(x_flat)), 0.0)
        checks.append(
            _build_check(
                "integrality",
                int_resid,
                var_names,
                integrality_tol,
                show_tol,
                detail_template="var {label} = {value:.6g}",
                value=x_flat,
            )
        )

    # ── 4. Objective consistency ────────────────────────────────────────────
    if result.objective is not None:
        re_obj_min = evaluator.evaluate_objective(x_flat)
        re_obj = -re_obj_min if obj_sense == ObjectiveSense.MAXIMIZE else re_obj_min
        drift = abs(re_obj - result.objective)
        tol = obj_tol + 1e-4 * abs(result.objective)
        checks.append(
            CheckResult(
                name="obj_consistency",
                passed=drift <= tol,
                tolerance=tol,
                max_violation=drift,
                norm2_violation=drift,
                worst_label="objective",
                detail=f"returned={result.objective:.10g} re-eval={re_obj:.10g}",
            )
        )

    # ── 5-7. Dual-side KKT checks via active-set recovery ───────────────────
    n_active_cons = 0
    n_active_bounds = 0
    duals_recovered = False
    dual_residual = 0.0
    if recover_duals and x_flat.size:
        kkt = _recover_and_check_kkt(
            evaluator=evaluator,
            x_flat=x_flat,
            lb=lb,
            ub=ub,
            body=body,
            sense_arr=sense_arr,
            rhs_arr=rhs_arr,
            jac=jac,
            row_labels=row_labels,
            var_names=var_names,
            is_continuous=is_continuous,
            active_tol=active_tol,
            dual_feas_tol=dual_feas_tol,
            primal_cs_tol=primal_cs_tol,
            dual_cs_tol=dual_cs_tol,
            sense_max=obj_sense,
        )
        checks.extend(kkt["checks"])
        n_active_cons = kkt["n_active_cons"]
        n_active_bounds = kkt["n_active_bounds"]
        duals_recovered = kkt["recovered"]
        dual_residual = kkt["residual"]

    merit = float(sum(c.norm2_violation**2 for c in checks))
    return ExaminerReport(
        checks=checks,
        merit=merit,
        n_active_constraints=n_active_cons,
        n_active_bounds=n_active_bounds,
        duals_recovered=duals_recovered,
        dual_recovery_residual=dual_residual,
    )


def assert_examined(result, model: Model, name: str, **kwargs) -> None:
    """Run :func:`examine` and raise ``AssertionError`` on the first failure.

    The error message names the failing check and the worst-row violation,
    so benchmark output distinguishes wrong-objective from
    infeasible-point-of-x bugs as required by issue #55.
    """
    report = examine(result, model, **kwargs)
    if not report.passed:
        first = report.first_failure()
        assert first is not None
        raise AssertionError(
            f"[{name}] Examiner check failed: {first.name} "
            f"max={first.max_violation:.3e} > tol {first.tolerance:.3e} "
            f"at {first.worst_label} -- {first.detail}\n"
            f"Full report:\n{report.summary(verbose=True)}"
        )


# ── internals ────────────────────────────────────────────────────────────────


def _row_metadata(evaluator):
    senses: list[str] = []
    rhss: list[float] = []
    labels: list[str] = []
    for c, sz in zip(evaluator._source_constraints, evaluator._constraint_flat_sizes):
        sz = int(sz)
        senses.extend([c.sense] * sz)
        rhss.extend([float(c.rhs)] * sz)
        cname = getattr(c, "name", None) or repr(c.body)[:40]
        if sz > 1:
            labels.extend([f"{cname}[{i}]" for i in range(sz)])
        else:
            labels.append(str(cname))
    return np.asarray(senses), np.asarray(rhss, dtype=float), labels


def _check_var_bounds(
    x: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    labels: list[str],
    tol: float,
    show_tol: float,
) -> CheckResult:
    if x.size == 0:
        return CheckResult(name="primal_var_feas", passed=True, tolerance=tol)
    lb_viol = np.maximum(lb - x, 0.0)
    ub_viol = np.maximum(x - ub, 0.0)
    viol = np.maximum(lb_viol, ub_viol)
    return _build_check(
        "primal_var_feas",
        viol,
        labels,
        tol,
        show_tol,
        detail_template="var {label}: {lb:.6g} <= {value:.6g} <= {ub:.6g}",
        lb=lb,
        ub=ub,
        value=x,
    )


def _constraint_violations(body, sense_arr, rhs_arr):
    viol = np.zeros_like(body)
    le = sense_arr == "<="
    ge = sense_arr == ">="
    eq = sense_arr == "=="
    viol[le] = np.maximum(body[le] - rhs_arr[le], 0.0)
    viol[ge] = np.maximum(rhs_arr[ge] - body[ge], 0.0)
    viol[eq] = np.abs(body[eq] - rhs_arr[eq])
    signed = body - rhs_arr
    return viol, signed


def _build_check(
    name: str,
    viol: np.ndarray,
    labels: list[str],
    tol: float,
    show_tol: float,
    *,
    detail_template: str = "",
    **fields,
) -> CheckResult:
    if viol.size == 0:
        return CheckResult(name=name, passed=True, tolerance=tol)
    max_viol = float(np.max(viol))
    norm2 = float(np.linalg.norm(viol))
    worst = int(np.argmax(viol))
    passed = max_viol <= tol
    detail = ""
    if detail_template and not passed:
        ctx: dict[str, object] = {}
        for k, v in fields.items():
            if hasattr(v, "__len__") and not isinstance(v, str):
                try:
                    ctx[k] = v[worst]
                except (IndexError, TypeError):
                    ctx[k] = v
            else:
                ctx[k] = v
        ctx["label"] = labels[worst] if labels else str(worst)
        try:
            detail = detail_template.format(**ctx)
        except (KeyError, IndexError, ValueError):
            detail = ""
    violators: list[tuple[str, float]] = []
    if labels:
        order = np.argsort(viol)[::-1]
        for idx in order:
            if viol[idx] <= show_tol:
                break
            violators.append((labels[idx], float(viol[idx])))
            if len(violators) >= 16:
                break
    return CheckResult(
        name=name,
        passed=passed,
        tolerance=tol,
        max_violation=max_viol,
        norm2_violation=norm2,
        worst_label=labels[worst] if labels else str(worst),
        detail=detail,
        violators=violators,
    )


def _recover_and_check_kkt(
    *,
    evaluator,
    x_flat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    body: np.ndarray,
    sense_arr: np.ndarray,
    rhs_arr: np.ndarray,
    jac: np.ndarray,
    row_labels: list[str],
    var_names: list[str],
    is_continuous: np.ndarray,
    active_tol: float,
    dual_feas_tol: float,
    primal_cs_tol: float,
    dual_cs_tol: float,
    sense_max: ObjectiveSense,
) -> dict:
    """Recover multipliers from the active set and run dual-side KKT checks.

    Builds A y = -∇f over the active set, solves with sign bounds via
    :func:`scipy.optimize.lsq_linear`, then reports stationarity residual
    plus complementary-slackness proxies. Integer columns are dropped --
    Examiner fixes integers and runs continuous KKT, which is exactly what
    omitting those columns does.
    """
    grad = evaluator.evaluate_gradient(x_flat)
    cont_idx = np.where(is_continuous)[0]
    if cont_idx.size == 0:
        return {
            "checks": [],
            "n_active_cons": 0,
            "n_active_bounds": 0,
            "recovered": False,
            "residual": 0.0,
        }

    grad_c = grad[cont_idx]
    jac_c = jac[:, cont_idx] if jac.size else np.empty((0, cont_idx.size))
    lb_c = lb[cont_idx]
    ub_c = ub[cont_idx]
    x_c = x_flat[cont_idx]
    cont_names = [var_names[i] for i in cont_idx]

    if body.size:
        signed = body - rhs_arr
        is_le = sense_arr == "<="
        is_ge = sense_arr == ">="
        is_eq = sense_arr == "=="
        active_le = is_le & (np.abs(signed) <= active_tol)
        active_ge = is_ge & (np.abs(signed) <= active_tol)
        active_rows = active_le | active_ge | is_eq
        row_select = np.where(active_rows)[0]
    else:
        row_select = np.zeros(0, dtype=int)

    lb_active = np.where(np.isfinite(lb_c) & (x_c - lb_c <= active_tol))[0]
    ub_active = np.where(np.isfinite(ub_c) & (ub_c - x_c <= active_tol))[0]

    n_mu = row_select.size
    n_llb = lb_active.size
    n_lub = ub_active.size
    n_unknowns = n_mu + n_llb + n_lub

    if n_unknowns == 0:
        residual = grad_c
        max_r = float(np.max(np.abs(residual))) if residual.size else 0.0
        norm_r = float(np.linalg.norm(residual))
        worst = int(np.argmax(np.abs(residual))) if residual.size else 0
        return {
            "checks": [
                CheckResult(
                    name="stationarity",
                    passed=max_r <= dual_feas_tol,
                    tolerance=dual_feas_tol,
                    max_violation=max_r,
                    norm2_violation=norm_r,
                    worst_label=cont_names[worst] if cont_names else "",
                    detail="no active set; ‖∇f‖∞ at returned x",
                )
            ],
            "n_active_cons": 0,
            "n_active_bounds": 0,
            "recovered": True,
            "residual": norm_r,
        }

    cols = []
    var_lb_y: list[float] = []
    var_ub_y: list[float] = []
    if n_mu:
        sub_sense = sense_arr[row_select]
        sub_jac = jac_c[row_select, :].copy()
        flip = sub_sense == ">="
        sub_jac[flip, :] *= -1.0
        cols.append(sub_jac.T)
        for s in sub_sense:
            var_lb_y.append(-np.inf if s == "==" else 0.0)
            var_ub_y.append(np.inf)
    if n_llb:
        I_lb = np.zeros((cont_idx.size, n_llb))
        for k, j in enumerate(lb_active):
            I_lb[j, k] = -1.0
        cols.append(I_lb)
        var_lb_y.extend([0.0] * n_llb)
        var_ub_y.extend([np.inf] * n_llb)
    if n_lub:
        I_ub = np.zeros((cont_idx.size, n_lub))
        for k, j in enumerate(ub_active):
            I_ub[j, k] = 1.0
        cols.append(I_ub)
        var_lb_y.extend([0.0] * n_lub)
        var_ub_y.extend([np.inf] * n_lub)

    A = np.concatenate(cols, axis=1) if cols else np.zeros((cont_idx.size, 0))
    b = -grad_c

    try:
        sol = lsq_linear(A, b, bounds=(np.asarray(var_lb_y), np.asarray(var_ub_y)))
        y = sol.x
    except Exception as e:
        msg = f"dual recovery failed: {e}"
        return {
            "checks": [
                CheckResult(name="stationarity", passed=False, tolerance=dual_feas_tol, detail=msg),
                CheckResult(name="primal_cs", passed=False, tolerance=primal_cs_tol, detail=msg),
                CheckResult(name="dual_cs", passed=False, tolerance=dual_cs_tol, detail=msg),
            ],
            "n_active_cons": int(n_mu),
            "n_active_bounds": int(n_llb + n_lub),
            "recovered": False,
            "residual": float("inf"),
        }

    stat_resid = A @ y - b
    stat_max = float(np.max(np.abs(stat_resid))) if stat_resid.size else 0.0
    stat_norm = float(np.linalg.norm(stat_resid))
    worst_var = int(np.argmax(np.abs(stat_resid))) if stat_resid.size else 0
    stationarity = CheckResult(
        name="stationarity",
        passed=stat_max <= dual_feas_tol,
        tolerance=dual_feas_tol,
        max_violation=stat_max,
        norm2_violation=stat_norm,
        worst_label=cont_names[worst_var] if cont_names else "",
        detail=(
            f"recovered |μ_act|={n_mu}, |λ_lb_act|={n_llb}, |λ_ub_act|={n_lub}; "
            f"sense={'max' if sense_max == ObjectiveSense.MAXIMIZE else 'min'}"
        ),
    )

    mu = y[:n_mu] if n_mu else np.zeros(0)
    lam_lb = y[n_mu : n_mu + n_llb] if n_llb else np.zeros(0)
    lam_ub = y[n_mu + n_llb :] if n_lub else np.zeros(0)

    cs_p = np.zeros(cont_idx.size)
    for k, j in enumerate(lb_active):
        cs_p[j] = max(cs_p[j], abs(lam_lb[k]) * abs(x_c[j] - lb_c[j]))
    for k, j in enumerate(ub_active):
        cs_p[j] = max(cs_p[j], abs(lam_ub[k]) * abs(ub_c[j] - x_c[j]))
    cs_p_max = float(np.max(cs_p)) if cs_p.size else 0.0
    cs_p_norm = float(np.linalg.norm(cs_p))
    primal_cs = CheckResult(
        name="primal_cs",
        passed=cs_p_max <= primal_cs_tol,
        tolerance=primal_cs_tol,
        max_violation=cs_p_max,
        norm2_violation=cs_p_norm,
        worst_label=(cont_names[int(np.argmax(cs_p))] if cs_p.size and cont_names else ""),
        detail="max |λ|·|x − bound| over active continuous bounds",
    )

    if n_mu:
        sub_signed = (body - rhs_arr)[row_select]
        cs_d = np.abs(mu) * np.abs(sub_signed)
        cs_d_max = float(np.max(cs_d))
        cs_d_norm = float(np.linalg.norm(cs_d))
        worst_row = int(np.argmax(cs_d))
        dual_cs = CheckResult(
            name="dual_cs",
            passed=cs_d_max <= dual_cs_tol,
            tolerance=dual_cs_tol,
            max_violation=cs_d_max,
            norm2_violation=cs_d_norm,
            worst_label=row_labels[row_select[worst_row]] if row_labels else "",
            detail="max |μ|·|body−rhs| over recovered active rows",
        )
    else:
        dual_cs = CheckResult(name="dual_cs", passed=True, tolerance=dual_cs_tol)

    return {
        "checks": [stationarity, primal_cs, dual_cs],
        "n_active_cons": int(n_mu),
        "n_active_bounds": int(n_llb + n_lub),
        "recovered": True,
        "residual": stat_norm,
    }
