"""End-to-end math channel with scripted and injected transports."""

import json

import pytest

from aria.llm.client import ClientError, Message, OpenAICompatClient, StubClient
from aria.llm.orchestrator import MathChannel, dry_run_client

GOOD = "goal: solve\nvar: x\nexpr: 2x + 4 = 10"
BAD = "goal: solve\nvar: x\nexpr: 2x +"


def test_channel_produces_a_verified_derivation():
    client = StubClient([GOOD, "x is 3 because subtracting 4 and halving gives 3."])
    result = MathChannel(client).ask("solve for x in 2x + 4 = 10")
    assert result.ok
    assert result.derivation.status == "solved"
    assert [str(v) for v in result.derivation.solutions] == ["3"]
    assert result.derivation.valid
    assert "3" in result.explanation


def test_grammar_is_sent_with_the_translation_request_only():
    client = StubClient([GOOD, "explanation"])
    MathChannel(client).ask("solve for x in 2x + 4 = 10")
    assert client.calls[0]["grammar"] is True      # translation is constrained
    assert client.calls[1]["grammar"] is False     # narration is free text


def test_invalid_contract_is_repaired_then_accepted():
    client = StubClient([BAD, GOOD, "ok"])
    result = MathChannel(client, max_repairs=2).ask("solve for x in 2x + 4 = 10")
    assert result.ok
    assert result.repairs == 1
    # the repair turn must carry the actual error, not a generic apology
    repair_text = client.calls[1]["messages"][-1]["content"]
    assert "does not parse" in repair_text


def test_persistent_violation_fails_closed():
    client = StubClient([BAD, BAD, BAD])
    result = MathChannel(client, max_repairs=1).ask("solve for x in 2x + 4 = 10")
    assert not result.ok
    assert "contract rejected" in result.error
    assert result.derivation is None


def test_unreachable_server_is_reported_not_crashed():
    class Broken:
        def complete(self, *args, **kwargs):
            raise ClientError("connection refused")

    result = MathChannel(Broken()).ask("solve for x in 2x + 4 = 10")
    assert not result.ok
    assert "connection refused" in result.error


def test_dry_run_needs_no_model():
    result = MathChannel(dry_run_client(), narrate=False).ask("solve for x in 2x + 4 = 10")
    assert result.ok
    assert result.derivation.solutions[0] == 3


def test_narration_failure_does_not_lose_the_mathematics():
    class FlakyNarrator(StubClient):
        def complete(self, messages, **kwargs):
            if messages[-1].content.startswith("User question"):
                raise ClientError("context overflow")
            return super().complete(messages, **kwargs)

    result = MathChannel(FlakyNarrator([GOOD])).ask("solve for x in 2x + 4 = 10")
    assert result.ok
    assert "narration unavailable" in result.explanation


def test_openai_client_payload_and_response_parsing():
    captured = {}

    def transport(url, payload, headers, timeout):
        captured.update({"url": url, "payload": payload, "headers": headers})
        return json.dumps({"choices": [{"message": {"content": GOOD}}]})

    client = OpenAICompatClient(endpoint="http://localhost:9999/v1/chat/completions", transport=transport)
    text = client.complete([Message("user", "hi")], grammar="root ::= \"a\"")
    assert text == GOOD
    assert captured["payload"]["grammar"] == 'root ::= "a"'
    assert captured["payload"]["temperature"] == 0.0
    assert captured["url"].endswith("/v1/chat/completions")


def test_openai_client_surfaces_bad_responses():
    client = OpenAICompatClient(transport=lambda *args: "not json")
    with pytest.raises(ClientError):
        client.complete([Message("user", "hi")])
