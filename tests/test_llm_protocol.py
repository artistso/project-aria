"""The math channel contract: what the model is allowed to say."""

import pytest

from aria.core.state import GoalKind
from aria.llm.orchestrator import deterministic_translator
from aria.llm.protocol import ContractError, MathRequest, parse_contract, to_state


def test_valid_contract_parses():
    request = parse_contract("goal: solve\nvar: x\nexpr: 2x + 4 = 10")
    assert request.goal == "solve"
    assert request.variable == "x"
    assert request.expression == "2x + 4 = 10"


def test_contract_tolerates_markdown_fences_and_extra_blank_lines():
    raw = "```\ngoal: simplify\n\nexpr: (x+1)^2\n```"
    request = parse_contract(raw)
    assert request.goal == "simplify"


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("Here is the answer\ngoal: simplify\nexpr: x + 1", "unexpected content"),
        ("goal: simplify\nnote: x + 1\nexpr: x + 1", "unexpected contract field"),
        ("goal: simplify\ngoal: factor\nexpr: x + 1", "duplicate 'goal'"),
        ("expr: x + 1\ngoal: simplify", "ordered as goal"),
    ],
)
def test_contract_rejects_untrusted_extra_duplicate_or_misordered_content(raw, reason):
    with pytest.raises(ContractError, match=reason):
        parse_contract(raw)


def test_variable_is_not_allowed_for_goals_that_do_not_use_one():
    with pytest.raises(ContractError, match="does not accept a 'var:'"):
        parse_contract("goal: simplify\nvar: x\nexpr: x + 1")


def test_expression_is_validated_not_trusted():
    with pytest.raises(ContractError, match="does not parse"):
        parse_contract("goal: simplify\nexpr: x +")


def test_missing_goal_is_rejected():
    with pytest.raises(ContractError, match="missing 'goal:'"):
        parse_contract("expr: x + 1")


def test_unknown_goal_is_rejected():
    with pytest.raises(ContractError, match="unknown goal"):
        parse_contract("goal: hallucinate\nexpr: x + 1")


def test_solve_requires_a_variable():
    with pytest.raises(ContractError, match="requires a 'var:'"):
        parse_contract("goal: solve\nexpr: 2x + 4 = 10")


def test_state_carries_the_typed_goal():
    request = MathRequest(goal="solve", expression="2x + 4 = 10", variable="x")
    state = to_state(request)
    assert state.goal.kind is GoalKind.SOLVE
    assert state.goal.var.name == "x"


def test_variable_can_be_inferred_when_unambiguous():
    request = MathRequest(goal="differentiate", expression="x^2")
    state = to_state(request)
    assert state.goal.var.name == "x"


def test_deterministic_translator_emits_a_valid_contract():
    text = deterministic_translator("solve for x in 2x + 4 = 10")
    request = parse_contract(text)
    assert request.goal == "solve"
    assert request.variable == "x"


@pytest.mark.parametrize(
    "prompt,goal",
    [
        ("simplify (x+1)^2", "simplify"),
        ("factor x^2 - 9", "factor"),
        ("verify log(a*b) = log(a) + log(b)", "verify"),
        ("differentiate x^2 with respect to x", "differentiate"),
    ],
)
def test_translator_maps_instruction_words_to_goals(prompt, goal):
    assert parse_contract(deterministic_translator(prompt)).goal == goal
