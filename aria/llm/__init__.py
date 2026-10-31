"""Local-model integration: a math-only channel between text and symbols.

    from aria.llm import MathChannel, OpenAICompatClient

    client = OpenAICompatClient("http://127.0.0.1:8080/v1/chat/completions")
    result = MathChannel(client).ask("solve for x in 2x + 4 = 10")
    print(result.to_text())
"""

from .client import ChatClient, ClientError, Message, OpenAICompatClient, StubClient
from .grammar import GRAMMAR_MODES, build_grammar, validate_grammar
from .orchestrator import ChannelResult, MathChannel, deterministic_translator, dry_run_client
from .protocol import ContractError, MathRequest, parse_contract, to_state

__all__ = [
    "ChannelResult",
    "ChatClient",
    "ClientError",
    "ContractError",
    "GRAMMAR_MODES",
    "MathChannel",
    "MathRequest",
    "Message",
    "OpenAICompatClient",
    "StubClient",
    "build_grammar",
    "deterministic_translator",
    "dry_run_client",
    "parse_contract",
    "to_state",
    "validate_grammar",
]
