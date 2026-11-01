"""Verification: proven / refuted / unknown, and tamper detection."""

import pytest

from aria.core.parse import parse
from aria.verify.verifier import Verifier


@pytest.fixture(scope="module")
def verifier():
    return Verifier()


@pytest.mark.parametrize(
    "left,right",
    [
        ("(x+1)^2", "x^2 + 2x + 1"),
        ("(a+b)^2 - (a-b)^2", "4 a b"),
        ("(x^2-1)/(x-1)", "x + 1"),
        ("log(a*b)", "log(a) + log(b)"),
        ("sin(x)^2 + cos(x)^2", "1"),
        ("2^(1/2) * 2^(1/2)", "2"),
    ],
)
def test_provable_equivalences(verifier, left, right):
    verdict = verifier.equivalent(parse(left), parse(right))
    assert verdict.ok, verdict


@pytest.mark.parametrize(
    "left,right",
    [
        ("(x+1)^2", "x^2 + 1"),
        ("log(a+b)", "log(a) + log(b)"),
        ("3/4", "4/5"),
    ],
)
def test_refutations(verifier, left, right):
    verdict = verifier.equivalent(parse(left), parse(right))
    assert verdict.status == "refuted", verdict


def test_numeric_identity(tmp_path):
    verifier = Verifier()
    assert verifier.claim(parse("2 + 2 = 4")).ok
    assert verifier.claim(parse("2 + 2 = 5")).status == "refuted"


def test_unknown_backend_reports_unknown_without_crashing():
    verifier = Verifier(backend="internal")
    verdict = verifier.equivalent(parse("log(a*b)"), parse("log(a) + log(b)"))
    assert verdict.status in ("unknown", "refuted")
    # The internal backend must still decide polynomial equivalence exactly.
    assert verifier.equivalent(parse("(x+1)^2"), parse("x^2 + 2x + 1")).ok


def test_invalid_equation_step_is_caught(verifier):
    from aria.reason.derivation import DerivationStep

    step = DerivationStep(
        rule="bogus",
        before=parse("2x + 4 = 10"),
        after=parse("2x = 10"),
        justification="claim: dropping the constant",
        check={"kind": "equation_scaled"},
    )
    assert verifier.step(step).status == "refuted"


def test_valid_equation_step_passes(verifier):
    from aria.reason.derivation import DerivationStep

    step = DerivationStep(
        rule="divide_by_coefficient",
        before=parse("2x = 6"),
        after=parse("x = 3"),
        justification="divide both sides by 2",
        check={"kind": "both_sides_mul"},
    )
    assert verifier.step(step).ok
