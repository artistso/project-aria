"""Solving: correct answers *and* sound step-by-step derivations."""

import pytest

from aria.core.parse import parse, parse_problem
from aria.reason.engine import SymbolicEngine
from aria.verify.verifier import Verifier


@pytest.fixture(scope="module")
def engine():
    return SymbolicEngine()


@pytest.fixture(scope="module")
def verifier():
    return Verifier()


@pytest.mark.parametrize(
    "prompt,solutions",
    [
        ("solve for x in 2x + 4 = 10", ["3"]),
        ("solve for x in 3x - 7 = 2x + 5", ["12"]),
        ("solve x^2 - 9 = 0", ["-3", "3"]),
        ("solve x^2 + 5x + 6 = 0", ["-3", "-2"]),
        ("solve x^2 + 2x + 1 = 0", ["-1"]),
        ("solve 5x = 3", ["3/5"]),
    ],
)
def test_solutions_are_exact(engine, verifier, prompt, solutions):
    derivation = engine.run(parse_problem(prompt))
    assert derivation.status == "solved", derivation.notes
    assert len(derivation.solutions) == len(solutions)
    for expected in solutions:
        target = parse(expected)
        assert any(verifier.equivalent(value, target).ok for value in derivation.solutions)


def test_all_steps_verified(engine):
    derivation = engine.run(parse_problem("solve x^2 + 5x + 6 = 0"))
    assert derivation.steps
    assert derivation.valid
    assert derivation.invalid_steps == []


def test_irrational_roots(engine, verifier):
    derivation = engine.run(parse_problem("solve x^2 - 2 = 0"))
    assert derivation.status == "solved"
    squared = [value * value for value in derivation.solutions]
    for value in squared:
        simplified = engine.simplify(value).result
        assert verifier.equivalent(simplified, parse("2")).ok


def test_complex_roots_are_labelled(engine):
    derivation = engine.run(parse_problem("solve x^2 + 2x + 5 = 0"))
    assert derivation.status == "solved"
    assert "complex" in derivation.notes


def test_contradiction_is_refuted(engine):
    derivation = engine.run(parse_problem("solve 2x + 1 = 2x + 5"))
    assert derivation.status == "refuted"


def test_identity_equation(engine):
    derivation = engine.run(parse_problem("solve 4x - 8 = 4x - 8"))
    assert derivation.status == "solved"
    assert "every value" in derivation.notes


def test_modular_solutions(engine):
    derivation = engine.run(parse_problem("solve 5x ≡ 3 mod 7"))
    assert derivation.status == "solved"
    assert [str(value) for value in derivation.solutions] == ["2"]


def test_modular_no_solution(engine):
    derivation = engine.run(parse_problem("solve 4x ≡ 6 mod 8"))
    assert derivation.status.lower() in {"solved", "open"}
    assert derivation.solutions == ()
    assert "no solution" in derivation.notes


def test_literal_tampering_is_detected(engine):
    """A deliberately corrupted step must not verify (soundness guard)."""
    from aria.reason.derivation import DerivationStep

    step = DerivationStep(
        rule="move_all_to_one_side",
        before=parse("2x + 4 = 10"),
        after=parse("2x + 4 = 11"),
        justification="corrupted",
        check={"kind": "equation_scaled"},
    )
    assert engine.verifier.step(step).status == "refuted"
