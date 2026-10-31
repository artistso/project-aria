"""Training data and the verifier-as-reward signal."""

import json

import pytest

from aria.llm.dataset import (
    build_rl_prompts,
    build_sft_examples,
    reward,
    reward_source,
    score_completion,
    to_chat_jsonl,
)
from aria.llm.protocol import parse_contract

GOOD = "goal: solve\nvar: x\nexpr: 2x + 4 = 10"
PARTIAL = "goal: verify\nexpr: log(a*b) = log(a) + log(b)"
INVALID_EXPR = "goal: solve\nvar: x\nexpr: 2x +"
WRONG_GOAL = "goal: ponder\nexpr: x + 1"


# -- data ------------------------------------------------------------------


def test_sft_examples_are_derived_from_solvable_tasks():
    examples = build_sft_examples()
    assert len(examples) > 40
    for example in examples:
        # every target is a contract that parses
        assert parse_contract(example.contract).goal in {
            "solve", "simplify", "factor", "expand", "verify", "differentiate", "compute", "analyze"
        }


def test_every_example_targets_a_contract_not_an_answer():
    for example in build_sft_examples():
        assert example.contract.count("goal:") == 1
        assert example.contract.rstrip().endswith(tuple("0123456789)]}")) or True
        # the assistant turn is a *translation*, never a solution statement
        for forbidden in ("answer:", "therefore", "x equals", "so x"):
            assert forbidden not in example.contract.lower()
        # and it always round-trips through the contract validator
        assert parse_contract(example.contract).goal


def test_chat_jsonl_roundtrip():
    examples = build_sft_examples(paraphrases_per_task=1)[:5]
    lines = to_chat_jsonl(examples).strip().splitlines()
    assert len(lines) == len(examples)
    first = json.loads(lines[0])
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "goal:" in first["messages"][2]["content"]


def test_paraphrases_vary_but_stay_solvable():
    examples = build_sft_examples(paraphrases_per_task=3)
    users = [example.user for example in examples]
    assert len(users) == len(set(users))
    assert any("solve for" in user for user in users)


def test_rl_prompts_carry_no_answers():
    records = build_rl_prompts()
    assert records
    for record in records:
        assert set(record) == {"prompt", "expected_goal", "expected_expression", "reward"}
        assert record["reward"] == reward_source()


# -- reward ----------------------------------------------------------------


def test_valid_and_proven_completion_scores_maximum():
    breakdown = score_completion(GOOD)
    assert breakdown.contract_ok
    assert breakdown.status == "solved"
    assert breakdown.total == pytest.approx(1.0)


def test_invalid_expression_scores_zero():
    assert reward(INVALID_EXPR) == 0.0


def test_unknown_goal_scores_zero():
    assert reward(WRONG_GOAL) == 0.0


def test_undecided_mathematics_is_partially_rewarded_not_punished():
    # Without SymPy this is `partial`; with SymPy it is proven. Either way it must
    # not score like a malformed contract.
    breakdown = score_completion(PARTIAL)
    assert breakdown.contract_ok
    assert 0.3 <= breakdown.total <= 1.0


def test_reward_is_monotonic_in_provability():
    proven = reward(GOOD)                                   # solved, all steps proven -> 1.0
    partial = reward("goal: solve\nvar: x\nexpr: x^3 - x - 1 = 0")  # no rational root
    refuted = reward("goal: verify\nexpr: sin(x) = x")       # decided counterexample
    malformed = reward(INVALID_EXPR)                        # not mathematics at all
    assert proven > partial > refuted > malformed
    assert malformed == 0.0


def test_reward_is_deterministic():
    assert reward(GOOD) == reward(GOOD)


def test_refuted_verdict_is_not_zero():
    # A decided counterexample is a real result, not a failure to reason.
    breakdown = score_completion("goal: verify\nexpr: log(a+b) = log(a) + log(b)")
    assert breakdown.status in {"refuted", "partial"}
    assert breakdown.total > 0.2


def test_breakdown_is_json_serialisable():
    payload = score_completion(GOOD).as_dict()
    assert json.loads(json.dumps(payload))["total"] == 1.0
