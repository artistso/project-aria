"""Tests for the ledger: ARIA's record of its own correct and incorrect work."""

from __future__ import annotations

import json

import pytest

from aria.llm import ledger as L


def contract(prompt: str) -> str:
    text = L.contract_of(prompt)
    assert text is not None, f"prompt did not parse: {prompt}"
    return text


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("verify 2 + 2 = 4", "arithmetic"),
        ("simplify 2x + 3x - x", "simplify"),
        ("expand (x + 1)^3", "expand"),
        ("solve 9x - 3 = 24", "linear_equation"),
        ("factor x^2 - 9", "factor"),
        ("solve x^2 + x - 6 = 0", "quadratic_equation"),
        ("differentiate 4x^3 - 2x with respect to x", "derivative_polynomial"),
        ("differentiate sin(x) with respect to x", "derivative_transcendental"),
        ("verify log(a*b) = log(a) + log(b)", "log_exp"),
        ("verify sin(x)^2 + cos(x)^2 = 1", "trig"),
        ("verify (a + b)^2 = a^2 + 2ab + b^2", "identity"),
    ],
)
def test_classify_names_the_rung(prompt: str, expected: str) -> None:
    verdict = L.judge(contract(prompt))
    assert verdict.skill == expected
    assert verdict.demonstrated is True


def test_congruence_is_its_own_rung() -> None:
    verdict = L.judge(contract("solve 5x ≡ 3 mod 7"))
    assert verdict.skill == "modular"


# ---------------------------------------------------------------------------
# Judgement: the verifier is the only author of verdicts
# ---------------------------------------------------------------------------


def test_judge_rejects_text_that_is_not_a_contract() -> None:
    verdict = L.judge("the answer is 3")
    assert verdict.contract_ok is False
    assert verdict.outcome == "invalid"
    assert verdict.demonstrated is False


def test_judge_records_a_refutation_as_a_proven_result() -> None:
    verdict = L.judge(contract("verify 2 + 2 = 5"))
    assert verdict.outcome == "refuted"
    assert verdict.demonstrated is True


def test_judge_records_an_unfinished_run_as_not_demonstrated() -> None:
    verdict = L.judge("goal: solve\nvar: x\nexpr: x^3 - x - 1 = 0")
    assert verdict.outcome == "partial"
    assert verdict.demonstrated is False


def test_steps_carry_the_process_not_just_the_answer() -> None:
    verdict = L.judge(contract("solve 9x - 3 = 24"))
    assert len(verdict.steps) >= 2
    assert all(step["ok"] for step in verdict.steps)
    assert verdict.steps[0]["rule"]


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def test_record_appends_one_line_and_numbers_it(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    first = book.record("solve 9x - 3 = 24", contract("solve 9x - 3 = 24"))
    second = book.record("solve 2x + 7 = 15", contract("solve 2x + 7 = 15"))

    assert (first.seq, second.seq) == (1, 2)
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["skill"] == "linear_equation"


def test_authorship_is_split_verifier_verdicts_model_note(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    book.record("solve 9x - 3 = 24", contract("solve 9x - 3 = 24"), note="divide both sides by 9")
    data = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])
    assert data["author"] == {"verdict": "verifier", "note": "model"}
    assert data["outcome"] == "solved"


def test_corrupt_lines_are_skipped_not_fatal(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    book = L.Ledger(path)
    book.record("solve 9x - 3 = 24", contract("solve 9x - 3 = 24"))
    with path.open("a") as handle:
        handle.write("\n{not json\n")
    assert len(book.entries()) == 1


# ---------------------------------------------------------------------------
# The school: holding, repair, blocked, graduation
# ---------------------------------------------------------------------------


def _work(book: L.Ledger, prompt: str, note: str = "") -> None:
    book.record(prompt, contract(prompt), note=note)


def test_three_distinct_proofs_hold_a_rung(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    prompts = ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19")
    for prompt in prompts[:2]:
        _work(book, prompt)
        assert book.skills()["linear_equation"].state != "held"
    _work(book, prompts[2])
    record = book.skills()["linear_equation"]
    assert record.state == "held"
    assert record.distinct == 3


def test_repeating_one_problem_does_not_hold_a_rung(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for _ in range(5):
        _work(book, "solve 9x - 3 = 24")
    record = book.skills()["linear_equation"]
    assert record.streak == 5
    assert record.distinct == 1
    assert record.state == "learning"


#: A run that ends "solved" while one of its steps failed to verify. The reported
#: solution happens to be right, but the engine's final transformation is unchecked.
BROKEN_PROCESS = "goal: solve\nvar: x\nexpr: 9/x - 3 = 24"


def test_a_failure_does_not_remove_the_grade_but_opens_a_repair(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for _level, _skill, _label, seeds in L.CURRICULUM[:4]:
        for prompt in seeds[:3]:
            _work(book, prompt)
    assert book.grade() == 5
    assert book.skills()["linear_equation"].state == "held"

    entry = book.record("solve 9/x - 3 = 24", BROKEN_PROCESS)
    assert entry.outcome == "solved"
    assert entry.demonstrated is False  # an unverified step is not a proof
    record = book.skills()["linear_equation"]
    assert record.state == "repair"
    assert record.held_once is True
    assert record.demonstrated == 3  # the grade itself is not taken away
    assert book.grade() == 5
    card = book.card()
    assert "linear_equation" in card["held"]
    assert "linear_equation" in card["open_repairs"]
    assert any(row["skill"] == "linear_equation" for row in book.to_sft_records())


def test_repair_comes_before_new_material(tmp_path) -> None:
    book = L.Ledger("/tmp/x.jsonl") if False else L.Ledger(tmp_path / "ledger.jsonl")
    for prompt in ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19"):
        _work(book, prompt)
    book.record("solve 9/x - 3 = 24", BROKEN_PROCESS)
    assert book.frontier() == "linear_equation"
    assert "linear_equation" in book.card()["open_repairs"]


def test_repair_closes_after_two_clean_distinct_attempts(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for prompt in ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19"):
        _work(book, prompt)
    book.record("solve 9/x - 3 = 24", BROKEN_PROCESS)
    assert book.skills()["linear_equation"].state == "repair"
    _work(book, "solve 5x - 2 = 8")
    assert book.skills()["linear_equation"].state == "repair"
    _work(book, "solve 4x + 3 = 19")
    assert book.skills()["linear_equation"].state == "held"


def test_an_undecidable_rung_is_dark_not_failed(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    _work(book, "solve 5x ≡ 3 mod 7")
    record = book.skills()["modular"]
    assert record.state == "blocked"
    assert record.demonstrated == 0
    assert "modular" in book.card()["blocked"]


def test_graduation_needs_every_gradable_rung(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for level, skill, _label, seeds in L.CURRICULUM:
        for prompt in seeds[:3]:
            book.record(prompt, contract(prompt))
    assert book.grade() == L.COLLEGE
    assert book.card()["college"] is True
    assert "graduated" in L.report(book)


def test_grade_advances_one_rung_at_a_time(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    assert book.grade() == 1
    for prompt in ("verify 2 + 2 = 4", "verify 17 * 23 = 391", "verify 2^10 = 1024"):
        _work(book, prompt)
    assert book.grade() == 2  # arithmetic held, simplify is next


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def test_next_task_starts_at_grade_one_and_names_its_source(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    following = book.next_task()
    assert following is not None
    assert following["skill"] == "arithmetic"
    assert following["prompt"]
    assert following["source"] in ("bench", "seed")


def test_next_task_skips_problems_already_filed(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    first = book.next_task()
    book.record(first["prompt"], contract(first["prompt"]))
    second = book.next_task()
    assert second["prompt"] != first["prompt"]


def test_next_task_moves_to_the_next_rung_once_a_rung_is_held(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for prompt in ("verify 2 + 2 = 4", "verify 17 * 23 = 391", "verify 2^10 = 1024"):
        _work(book, prompt)
    following = book.next_task()
    assert following["skill"] == "simplify"


# ---------------------------------------------------------------------------
# Training data comes from the record, not from a score
# ---------------------------------------------------------------------------


def test_sft_exports_only_proven_processes(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    _work(book, "solve 9x - 3 = 24")
    book.record("solve x^3 - x - 1 = 0", "goal: solve\nvar: x\nexpr: x^3 - x - 1 = 0")
    records = book.to_sft_records(held_only=False)
    assert len(records) == 1
    assert records[0]["messages"][0]["role"] == "system"
    assert "9 x" in records[0]["messages"][-1]["content"]


def test_sft_default_advances_only_with_held_school_rungs(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    prompts = ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19")
    for prompt in prompts[:2]:
        _work(book, prompt)
    assert book.to_sft_records() == []

    _work(book, prompts[2])
    staged = book.to_sft_records()
    assert len(staged) == 3
    assert {record["skill"] for record in staged} == {"linear_equation"}
    assert book.to_sft_records(max_level=1) == []
    assert len(book.to_sft_records(max_level=4)) == 3

    _work(book, "simplify 3x + 4x - 2x")
    assert len(book.to_sft_records()) == 3
    assert len(book.to_sft_records(held_only=False)) == 4


def test_a_broken_process_is_never_exported_even_after_the_rung_is_held(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    book.record("solve 9/x - 3 = 24", BROKEN_PROCESS)
    assert book.to_sft_records(held_only=False) == []

    prompts = ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19")
    for prompt in prompts:
        _work(book, prompt)

    records = book.to_sft_records()
    assert len(records) == 3
    assert all("9/x" not in record["messages"][-1]["content"] for record in records)
    assert {record["skill"] for record in records} == {"linear_equation"}


def test_the_scalar_reward_can_pay_for_a_process_that_fails() -> None:
    """A shaped scalar can pay for an unchecked process with a correct endpoint.

    The engine reports x = 1/3, which satisfies this equation, but one produced
    step has an invalid justification. Two other steps pass, yielding 0.90;
    the ledger correctly refuses to count the process as demonstrated.
    """
    from aria.llm.dataset import score_completion

    assert score_completion(BROKEN_PROCESS).total == pytest.approx(0.9)
    assert L.judge(BROKEN_PROCESS).demonstrated is False


def test_report_card_shows_the_states(tmp_path) -> None:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for prompt in ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19"):
        _work(book, prompt)
    _work(book, "solve 5x ≡ 3 mod 7")
    text = L.report(book)
    assert "HELD" in text
    assert "DARK" in text
    assert "linear_equation" in text
    assert "study" in text.lower() or "studying" in text.lower()


@pytest.mark.parametrize(
    "level, skill, seeds",
    [(level, skill, seeds) for level, skill, _label, seeds in L.CURRICULUM if skill != "modular"],
)
def test_every_seed_problem_belongs_to_its_own_rung(level, skill, seeds) -> None:
    """A rung is only a rung if its problems are demonstrable and land where claimed."""
    assert L.LEVEL_OF[skill] == level
    for prompt in seeds:
        text = L.contract_of(prompt)
        assert text is not None, prompt
        verdict = L.judge(text)
        assert verdict.skill == skill, (prompt, verdict.skill)
        assert verdict.demonstrated is True, (prompt, verdict.outcome)


def test_modular_seeds_are_the_dark_rung() -> None:
    for prompt in L.SEEDS["modular"]:
        verdict = L.judge(L.contract_of(prompt))
        assert verdict.outcome == "partial"
        assert verdict.demonstrated is False
