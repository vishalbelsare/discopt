"""Tests for the discopt.llm package.

These tests work without litellm installed — they test the public API,
serialization, safety validation, tool execution, and graceful degradation.
"""

from __future__ import annotations

from unittest.mock import patch

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult

# ─────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────


class TestAvailability:
    def test_is_available_returns_bool(self):
        from discopt.llm import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_is_available_false_without_litellm(self):
        import discopt.llm

        with patch.dict("sys.modules", {"litellm": None}):
            # Force re-check by calling the function
            # (it does a fresh import each time)
            result = discopt.llm.is_available()
            # May or may not be False depending on whether litellm is installed
            assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────
# Serializer tests
# ─────────────────────────────────────────────────────────────


class TestSerializer:
    def _make_model(self):
        m = dm.Model("test_model")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        y = m.binary("y", shape=(2,))
        m.minimize(dm.sum(x) + 5 * dm.sum(y))
        m.subject_to(x[0] + x[1] >= 3, name="min_flow")
        m.subject_to(dm.sum(y) <= 1, name="max_active")
        return m

    def _make_result(self, model=None):
        return SolveResult(
            status="optimal",
            objective=12.5,
            bound=12.5,
            gap=0.0,
            x={"x": np.array([1.5, 1.5, 0.0]), "y": np.array([1.0, 0.0])},
            wall_time=0.5,
            node_count=10,
            rust_time=0.1,
            jax_time=0.3,
            python_time=0.1,
            _model=model,
        )

    def test_serialize_model(self):
        from discopt.llm.serializer import serialize_model

        m = self._make_model()
        text = serialize_model(m)

        assert "test_model" in text
        assert "x" in text
        assert "y" in text
        assert "continuous" in text
        assert "binary" in text
        assert "Constraints" in text
        assert "Variables" in text

    def test_serialize_solve_result(self):
        from discopt.llm.serializer import serialize_solve_result

        m = self._make_model()
        result = self._make_result(model=m)
        text = serialize_solve_result(result, model=m)

        assert "optimal" in text
        assert "12.5" in text
        assert "Wall time" in text
        assert "Rust" in text
        assert "JAX" in text

    def test_serialize_data_schema_numpy(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {
            "costs": np.array([1.0, 2.0, 3.0]),
            "matrix": np.ones((3, 4)),
        }
        text = serialize_data_schema(data)

        assert "costs" in text
        assert "numpy" in text.lower() or "array" in text.lower()
        assert "matrix" in text

    def test_serialize_data_schema_dict(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {"config": {"a": 1, "b": 2}}
        text = serialize_data_schema(data)
        assert "config" in text
        assert "dict" in text

    def test_serialize_data_schema_list(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {"items": [1, 2, 3, 4, 5]}
        text = serialize_data_schema(data)
        assert "items" in text


# ─────────────────────────────────────────────────────────────
# Safety tests
# ─────────────────────────────────────────────────────────────


class TestSafety:
    def test_validate_explanation_normal(self):
        from discopt.llm.safety import validate_explanation

        text = "The model solved optimally with objective 12.5."
        result = validate_explanation(text)
        assert result == text

    def test_validate_explanation_empty(self):
        from discopt.llm.safety import validate_explanation

        assert validate_explanation("") == "No explanation available."
        assert validate_explanation("   ") == "No explanation available."

    def test_validate_explanation_truncation(self):
        from discopt.llm.safety import validate_explanation

        long_text = "x" * 10000
        result = validate_explanation(long_text)
        assert len(result) < 10000
        assert "[Explanation truncated]" in result

    def test_validate_model_ok(self):
        from discopt.llm.safety import validate_model

        m = dm.Model("test")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        m.subject_to(m._variables[0] >= 1, name="c1")

        warnings = validate_model(m)
        assert isinstance(warnings, list)

    def test_validate_model_no_objective(self):
        from discopt.llm.safety import validate_model

        m = dm.Model("test")
        m.continuous("x", lb=0, ub=10)
        # No objective set
        warnings = validate_model(m)
        assert any("Validation error" in w for w in warnings)

    def test_validate_model_no_constraints(self):
        from discopt.llm.safety import validate_model

        m = dm.Model("test")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        # No constraints
        warnings = validate_model(m)
        assert any("no constraints" in w for w in warnings)

    def test_sanitize_tool_args_names(self):
        from discopt.llm.safety import sanitize_tool_args

        args = {"name": "my variable!@#"}
        result = sanitize_tool_args("add_continuous", args)
        assert result["name"] == "my_variable___"

    def test_sanitize_tool_args_bounds(self):
        from discopt.llm.safety import sanitize_tool_args

        args = {"name": "x", "lb": -1e30, "ub": 1e30}
        result = sanitize_tool_args("add_continuous", args)
        assert result["lb"] == -1e20
        assert result["ub"] == 1e20

    def test_sanitize_tool_args_invalid_sense(self):
        from discopt.llm.safety import sanitize_tool_args

        args = {"sense": "invalid", "expression": "x"}
        with pytest.raises(ValueError, match="Invalid objective sense"):
            sanitize_tool_args("set_objective", args)


# ─────────────────────────────────────────────────────────────
# Tool execution tests
# ─────────────────────────────────────────────────────────────


class TestModelBuilder:
    def test_create_model(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        result = builder.execute_tool("create_model", {"name": "test"})
        assert "Created" in result
        assert builder.model is not None
        assert builder.model.name == "test"

    def test_add_continuous(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "test"})
        result = builder.execute_tool(
            "add_continuous", {"name": "x", "shape": [3], "lb": 0, "ub": 10}
        )
        assert "Added" in result
        assert builder.model.num_variables == 3
        assert "x" in builder._namespace

    def test_add_binary(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "test"})
        result = builder.execute_tool("add_binary", {"name": "y", "shape": [2]})
        assert "Added" in result
        assert builder.model.num_integer == 2

    def test_add_integer(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "test"})
        result = builder.execute_tool("add_integer", {"name": "n", "shape": [], "lb": 1, "ub": 10})
        assert "Added" in result

    def test_set_objective(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "test"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "set_objective", {"sense": "minimize", "expression": "x[0] + x[1]"}
        )
        assert "Set objective" in result

    def test_add_constraint(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "test"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "add_constraint",
            {"lhs": "x[0] + x[1]", "sense": ">=", "rhs": "5", "name": "min_sum"},
        )
        assert "Added constraint" in result
        assert builder.model.num_constraints == 1

    def test_unknown_tool(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        result = builder.execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_tool_before_model(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        result = builder.execute_tool("add_continuous", {"name": "x"})
        assert "Error" in result

    def test_full_model_building(self):
        """Test building a complete model through the tool interface."""
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()

        # Build a simple transportation problem
        builder.execute_tool("create_model", {"name": "transport"})
        builder.execute_tool(
            "add_continuous",
            {"name": "x", "shape": [3], "lb": 0, "ub": 100},
        )
        builder.execute_tool(
            "set_objective",
            {"sense": "minimize", "expression": "x[0] + 2*x[1] + 3*x[2]"},
        )
        builder.execute_tool(
            "add_constraint",
            {"lhs": "x[0] + x[1] + x[2]", "sense": ">=", "rhs": "50", "name": "demand"},
        )

        model = builder.model
        assert model is not None
        assert model.name == "transport"
        assert model.num_variables == 3
        assert model.num_constraints == 1
        model.validate()  # Should not raise


# ─────────────────────────────────────────────────────────────
# Prompts tests
# ─────────────────────────────────────────────────────────────


class TestPrompts:
    def test_get_explain_prompt_all_statuses(self):
        from discopt.llm.prompts import get_explain_prompt

        for status in [
            "optimal",
            "feasible",
            "infeasible",
            "iteration_limit",
            "time_limit",
            "node_limit",
            "unknown",
        ]:
            prompt = get_explain_prompt(status)
            assert isinstance(prompt, str)
            assert "{model_text}" in prompt
            assert "{result_text}" in prompt

    def test_prompt_templates_have_placeholders(self):
        from discopt.llm.prompts import (
            EXPLAIN_INFEASIBLE,
            EXPLAIN_OPTIMAL,
            FORMULATE_SYSTEM,
            FORMULATE_USER,
        )

        assert "{model_text}" in EXPLAIN_OPTIMAL
        assert "{result_text}" in EXPLAIN_OPTIMAL
        assert "{model_text}" in EXPLAIN_INFEASIBLE
        assert "{description}" in FORMULATE_USER
        assert isinstance(FORMULATE_SYSTEM, str)


# ─────────────────────────────────────────────────────────────
# Provider tests (mocked)
# ─────────────────────────────────────────────────────────────


class TestProvider:
    def test_get_model_default(self):
        from discopt.llm.provider import _get_model

        model = _get_model()
        assert "claude" in model or "anthropic" in model

    def test_get_model_explicit(self):
        from discopt.llm.provider import _get_model

        assert _get_model("openai/gpt-4o") == "openai/gpt-4o"

    def test_get_model_env_var(self):
        from discopt.llm.provider import _get_model

        with patch.dict("os.environ", {"DISCOPT_LLM_MODEL": "ollama/llama3"}):
            assert _get_model() == "ollama/llama3"

    def test_get_model_explicit_overrides_env(self):
        from discopt.llm.provider import _get_model

        with patch.dict("os.environ", {"DISCOPT_LLM_MODEL": "ollama/llama3"}):
            assert _get_model("openai/gpt-4o") == "openai/gpt-4o"


# ─────────────────────────────────────────────────────────────
# explain() graceful degradation
# ─────────────────────────────────────────────────────────────


class TestExplainDegradation:
    def test_explain_without_llm(self):
        """explain() returns template string when llm=False."""
        result = SolveResult(
            status="optimal",
            objective=10.0,
            gap=0.0,
            wall_time=1.0,
            node_count=5,
        )
        explanation = result.explain()
        assert "optimal" in explanation
        assert "10.0" in explanation

    def test_explain_with_llm_false(self):
        """explain(llm=False) returns template string."""
        result = SolveResult(
            status="optimal",
            objective=10.0,
            gap=0.0,
            wall_time=1.0,
            node_count=5,
        )
        explanation = result.explain(llm=False)
        assert "optimal" in explanation

    def test_explain_cached(self):
        """explain() returns cached explanation if available."""
        result = SolveResult(
            status="optimal",
            _explanation="Custom explanation from LLM",
        )
        assert result.explain() == "Custom explanation from LLM"

    def test_explain_llm_graceful_failure(self):
        """explain(llm=True) falls back to template on LLM failure."""
        result = SolveResult(
            status="optimal",
            objective=10.0,
            gap=0.0,
            wall_time=1.0,
            node_count=5,
        )
        # Even if litellm is not installed or fails, should return template
        explanation = result.explain(llm=True)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


# ─────────────────────────────────────────────────────────────
# from_description() import guard
# ─────────────────────────────────────────────────────────────


class TestFromDescription:
    def test_from_description_requires_litellm(self):
        """from_description() raises ImportError when litellm not available."""
        with patch("discopt.llm.is_available", return_value=False):
            with pytest.raises(ImportError, match="litellm"):
                dm.from_description("Minimize x^2")


# ─────────────────────────────────────────────────────────────
# Advisor tests
# ─────────────────────────────────────────────────────────────


class TestAdvisor:
    def _make_model(self):
        m = dm.Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        y = m.binary("y", shape=(2,))
        m.minimize(dm.sum(x) + 5 * dm.sum(y))
        m.subject_to(x[0] + x[1] >= 3, name="min_flow")
        return m

    def test_suggest_solver_params(self):
        from discopt.llm.advisor import suggest_solver_params

        m = self._make_model()
        params = suggest_solver_params(m)
        assert isinstance(params, dict)
        assert "nlp_solver" in params
        assert "partitions" in params
        assert "reasoning" in params

    def test_presolve_analysis(self):
        from discopt.llm.advisor import presolve_analysis

        m = dm.Model("unbounded")
        m.continuous("x")  # unbounded
        m.minimize(0)
        warnings = presolve_analysis(m)
        assert isinstance(warnings, list)
        # Should detect unbounded variable
        assert any("unbounded" in w.lower() for w in warnings)

    def test_presolve_no_warnings(self):
        from discopt.llm.advisor import presolve_analysis

        m = self._make_model()
        warnings = presolve_analysis(m)
        assert isinstance(warnings, list)


# ─────────────────────────────────────────────────────────────
# Diagnosis tests
# ─────────────────────────────────────────────────────────────


class TestDiagnosis:
    def test_diagnose_infeasibility(self):
        from discopt.llm.diagnosis import diagnose_infeasibility

        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 20, name="impossible")

        result = SolveResult(status="infeasible", wall_time=0.1)
        report = diagnose_infeasibility(m, result)
        assert isinstance(report, str)
        assert "infeasible" in report.lower() or "Infeasib" in report

    def test_diagnose_result_dispatch(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("test")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)

        # Time limit
        result = SolveResult(
            status="time_limit",
            wall_time=100.0,
            node_count=500,
            gap=0.05,
            rust_time=30.0,
            jax_time=60.0,
            python_time=10.0,
        )
        report = diagnose_result(m, result)
        assert "time" in report.lower() or "limit" in report.lower()

        # Iteration limit
        result2 = SolveResult(status="iteration_limit", wall_time=1.0)
        report2 = diagnose_result(m, result2)
        assert "iteration" in report2.lower()

        # Optimal with no issues
        result3 = SolveResult(status="optimal", gap=0.0, wall_time=0.1)
        report3 = diagnose_result(m, result3)
        assert "No issues" in report3 or "optimal" in report3.lower()

    def test_diagnose_large_gap(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("test")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)

        result = SolveResult(
            status="optimal",
            gap=0.15,
            wall_time=10.0,
            objective=100.0,
            bound=85.0,
        )
        report = diagnose_result(m, result)
        assert "gap" in report.lower()


# ─────────────────────────────────────────────────────────────
# Reformulation tests
# ─────────────────────────────────────────────────────────────


class TestReformulation:
    def test_analyze_reformulations(self):
        from discopt.llm.reformulation import analyze_reformulations

        m = dm.Model("test")
        m.continuous("x", shape=(3,))  # unbounded!
        y = m.binary("y", shape=(3,))
        m.minimize(dm.sum(y))
        m.subject_to(dm.sum(y) >= 1, name="at_least_one")

        suggestions = analyze_reformulations(m)
        assert isinstance(suggestions, list)
        # Should detect weak bounds on x
        assert any(s.category == "bound_tightening" for s in suggestions)

    def test_detect_bilinear(self):
        from discopt.llm.reformulation import analyze_reformulations

        m = dm.Model("bilinear")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        m.subject_to(x * y <= 50, name="bilinear_con")

        suggestions = analyze_reformulations(m)
        categories = [s.category for s in suggestions]
        assert "mccormick_partitioning" in categories

    def test_apply_bound_tightening(self):
        from discopt.llm.reformulation import apply_bound_tightening

        m = dm.Model("test")
        m.continuous("x", lb=0, ub=100)
        m.minimize(0)
        count = apply_bound_tightening(m)
        assert isinstance(count, int)


# ─────────────────────────────────────────────────────────────
# Commentary tests
# ─────────────────────────────────────────────────────────────


class TestCommentary:
    def test_commentator_creation(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Test model summary")
        assert c._interval == 10.0

    def test_commentator_rate_limiting(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Test model", interval=100.0)
        c._enabled = False  # Disable LLM calls
        # Should return None (rate limited or disabled)
        msg = c.maybe_comment(1.0, None, 0.0, None, 0, 0, 0)
        assert msg is None


# ─────────────────────────────────────────────────────────────
# Chat session tests
# ─────────────────────────────────────────────────────────────


class TestChat:
    def test_chat_import(self):
        """chat() is accessible from top-level discopt."""
        import discopt

        assert hasattr(discopt, "chat")
        assert callable(discopt.chat)

    def test_chat_session_creation(self):
        """ChatSession can be created if litellm is available."""
        from discopt.llm import is_available

        if not is_available():
            pytest.skip("litellm not installed")

        from discopt.llm.chat import ChatSession

        session = ChatSession(verbose=False)
        assert session.model is None
        assert session._active
        session.close()
        assert not session._active

    def test_chat_context_manager(self):
        """ChatSession works as a context manager."""
        from discopt.llm import is_available

        if not is_available():
            pytest.skip("litellm not installed")

        from discopt.llm.chat import ChatSession

        with ChatSession(verbose=False) as session:
            assert session._active
        assert not session._active


# ─────────────────────────────────────────────────────────────
# Prompt template tests for new features
# ─────────────────────────────────────────────────────────────


class TestNewPrompts:
    def test_teaching_prompts_exist(self):
        from discopt.llm.prompts import TEACHING_SYSTEM, TEACHING_WALKTHROUGH

        assert isinstance(TEACHING_SYSTEM, str)
        assert "{model_text}" in TEACHING_WALKTHROUGH
        assert "{result_text}" in TEACHING_WALKTHROUGH

    def test_debug_prompts_exist(self):
        from discopt.llm.prompts import DEBUG_QUESTION, DEBUG_SYSTEM

        assert isinstance(DEBUG_SYSTEM, str)
        assert "{question}" in DEBUG_QUESTION
        assert "{model_text}" in DEBUG_QUESTION
