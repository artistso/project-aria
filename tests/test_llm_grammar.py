"""GBNF grammars: structurally valid, and they actually constrain the contract."""

import pytest

from aria.llm.grammar import build_grammar, rules_of, validate_grammar


@pytest.mark.parametrize("mode", ["loose", "strict"])
def test_generated_grammars_are_internally_consistent(mode):
    ok, problems = validate_grammar(build_grammar(mode))
    assert ok, problems


@pytest.mark.parametrize("mode", ["loose", "strict"])
def test_every_goal_tag_is_reachable(mode):
    grammar = build_grammar(mode)
    assert "goal" in rules_of(grammar)
    for tag in ("solve", "simplify", "factor", "expand", "verify", "differentiate"):
        assert tag in grammar


def test_undefined_rule_reference_is_reported():
    broken = 'root ::= "x" missing-rule\n'
    ok, problems = validate_grammar(broken)
    assert not ok
    assert any("undefined rule" in problem for problem in problems)


def test_missing_root_is_reported():
    ok, problems = validate_grammar('expr ::= [a-z]+\n')
    assert not ok
    assert any("root" in problem for problem in problems)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_grammar("paranoid")
