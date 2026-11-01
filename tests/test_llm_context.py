"""Cross-scale reasoning, held-state, and recoverable context semantics."""

from __future__ import annotations

import json

import pytest

from aria.llm.client import StubClient
from aria.llm.context import (
    AuditState,
    BranchMode,
    CandidateRoute,
    ContextBlock,
    ContextBudgetError,
    ContextBuilder,
    HeldState,
    MesoBranch,
    MicroEvidence,
    ReasoningFrame,
    Scale,
    StateEvent,
    compile_branch_states,
    least_resistance_route,
)
from aria.llm.orchestrator import MathChannel
from aria.llm.protocol import parse_contract, to_state
from aria.reason.engine import SymbolicEngine

GOOD = "goal: solve\nvar: x\nexpr: 2x + 4 = 10"


def test_branch_compilation_preserves_held_states():
    assert compile_branch_states([]) is AuditState.HELD
    assert compile_branch_states([AuditState.PASS, AuditState.PASS]) is AuditState.PASS
    assert compile_branch_states([AuditState.PASS, AuditState.HELD]) is AuditState.HELD
    assert compile_branch_states([AuditState.PASS, AuditState.FAIL]) is AuditState.FAIL
    assert compile_branch_states([AuditState.FAIL], BranchMode.ANY) is AuditState.FAIL
    assert compile_branch_states([AuditState.FAIL, AuditState.HELD], BranchMode.ANY) is AuditState.HELD
    assert compile_branch_states([AuditState.FAIL, AuditState.PASS], BranchMode.ANY) is AuditState.PASS


def test_task_refutation_is_not_a_failed_process_audit():
    derivation = SymbolicEngine().run(
        to_state(parse_contract("goal: verify\nexpr: 2 + 2 = 5"))
    )
    frame = ReasoningFrame.from_derivation(derivation)
    assert frame.task_outcome == "refuted"
    assert frame.audit_state is AuditState.PASS
    assert frame.held_state.state is AuditState.PASS
    with pytest.raises(ValueError, match="task outcome"):
        AuditState.from_verifier_status("refuted")


def test_bad_steps_keep_the_macro_audit_held():
    derivation = SymbolicEngine().run(
        to_state(parse_contract("goal: solve\nvar: x\nexpr: 9/x - 3 = 24"))
    )
    frame = ReasoningFrame.from_derivation(derivation)
    assert frame.task_outcome == "solved"
    assert frame.audit_state is AuditState.HELD
    assert frame.open_obligations


def test_held_state_changes_only_for_evidenced_verifier_events():
    held = HeldState()
    assert held.on_event(StateEvent.CONTEXT_REDUCED) is held
    assert held.on_event(StateEvent.CONTEXT_RECOVERED) is held
    assert held.on_event(StateEvent.NARRATION) is held
    with pytest.raises(ValueError, match="evidence reference"):
        held.on_event(StateEvent.VERIFIER_PASSED)

    passed = held.on_event(
        StateEvent.VERIFIER_PASSED,
        evidence_ref="verifier-run-17",
        reason="all required checks passed",
    )
    assert passed.state is AuditState.PASS
    assert passed.revision == held.revision + 1
    assert passed.evidence_ref == "verifier-run-17"


def test_least_resistance_ignores_unverified_and_failed_routes():
    routes = [
        CandidateRoute("cheap-held", 0.01, AuditState.HELD, ""),
        CandidateRoute("cheap-failed", 0.02, AuditState.FAIL, "counterexample"),
        CandidateRoute("long-valid", 5.0, AuditState.PASS, "proof-long"),
        CandidateRoute("short-valid", 1.0, AuditState.PASS, "proof-short"),
    ]
    assert least_resistance_route(routes).id == "short-valid"
    assert least_resistance_route(routes[:2]) is None


def test_context_omits_only_whole_blocks_and_can_rebuild():
    def measure(text: str) -> int:
        return len(text) + (1000 if "optional-detail-payload" in text else 0)

    blocks = (
        ContextBlock(
            "task",
            Scale.MACRO,
            "The checked task and outcome.",
            "aria.reason.engine",
            AuditState.PASS,
            required=True,
        ),
        ContextBlock(
            "step-001",
            Scale.MICRO,
            "optional-detail-payload with full step provenance",
            "aria.verify.verifier",
            AuditState.PASS,
            required=False,
            priority=1,
        ),
    )
    pack = ContextBuilder(600, measure=measure).build(blocks, phase="narration")
    assert pack.reduced
    assert pack.included_ids == ("task",)
    assert pack.omitted_ids == ("step-001",)
    assert "CONTEXT_COMPLETENESS: REDUCED" in pack.text
    assert "optional-detail-payload" not in pack.text
    assert pack.rebuild().completeness.value == "complete"
    assert "optional-detail-payload" in pack.rebuild().text

    serialized = pack.as_dict()
    assert serialized["source_blocks"][1]["text"].startswith("optional-detail-payload")
    assert serialized["recoverable"] is True
    json.dumps(serialized)


def test_mandatory_context_overflow_fails_closed_and_is_recoverable():
    block = ContextBlock("task", Scale.MACRO, "mandatory task", "test")
    with pytest.raises(ContextBudgetError) as raised:
        ContextBuilder(0).build((block,), phase="translation")

    error = raised.value
    assert "nothing was clipped or sent" in str(error)
    assert error.as_dict()["source_blocks"][0]["text"] == "mandatory task"
    assert error.rebuild(10_000).completeness.value == "complete"


def test_translation_overflow_does_not_call_the_model():
    client = StubClient([GOOD])
    result = MathChannel(client, narrate=False, context_budget=1).ask("solve for x in 2x + 4 = 10")
    assert not result.ok
    assert result.derivation is None
    assert client.calls == []
    assert result.context_failures[0].as_dict()["recoverable"] is True


def test_translation_sends_the_measured_pack_not_the_original_messages():
    prompt = "solve for x in 2x + 4 = 10"
    client = StubClient([GOOD])
    result = MathChannel(client, narrate=False, context_budget=5000).ask(prompt)

    assert result.ok
    pack = result.context_uses[0].pack
    sent_messages = client.calls[0]["messages"]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[0]["content"] == pack.reserved_text
    assert sent_messages[1] == {"role": "user", "content": pack.text}
    assert sent_messages[1]["content"] != prompt
    assert prompt in pack.text
    assert pack.used_units <= 5000


def test_narration_overflow_keeps_the_checked_derivation():
    def measure(text: str) -> int:
        return 1000 if "ARIA FRAME" in text else 1

    client = StubClient([GOOD])
    result = MathChannel(
        client,
        context_budget=100,
        context_measure=measure,
    ).ask("solve for x in 2x + 4 = 10")
    assert result.ok
    assert result.derivation.solutions[0] == 3
    assert "narration held" in result.explanation
    assert len(client.calls) == 1
    payload = result.as_dict()
    assert payload["context_failures"][0]["source_blocks"]
    json.dumps(payload)
