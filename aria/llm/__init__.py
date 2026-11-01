"""LLM integration: a structured math channel between natural language and symbols.

    from aria.llm import MathChannel, OpenAICompatClient

    client = OpenAICompatClient("http://127.0.0.1:8080/v1/chat/completions")
    result = MathChannel(client).ask("solve for x in 2x + 4 = 10")
    print(result.to_text())
"""

from .client import ChatClient, ClientError, Message, OpenAICompatClient, StubClient
from .context import (
    AuditState,
    BranchMode,
    CandidateRoute,
    ContextBlock,
    ContextBudgetError,
    ContextBuilder,
    ContextCompleteness,
    ContextPack,
    ContextUse,
    HeldState,
    MesoBranch,
    MicroEvidence,
    ReasoningFrame,
    Scale,
    StateEvent,
    compile_branch_states,
    least_resistance_route,
)
from .grammar import GRAMMAR_MODES, build_grammar, validate_grammar
from .orchestrator import ChannelResult, MathChannel, deterministic_translator, dry_run_client
from .protocol import ContractError, MathRequest, parse_contract, to_state

__all__ = [
    "AuditState",
    "BranchMode",
    "CandidateRoute",
    "ChannelResult",
    "ChatClient",
    "ClientError",
    "ContextBlock",
    "ContextBudgetError",
    "ContextBuilder",
    "ContextCompleteness",
    "ContextPack",
    "ContextUse",
    "ContractError",
    "HeldState",
    "MesoBranch",
    "MicroEvidence",
    "ReasoningFrame",
    "Scale",
    "StateEvent",
    "GRAMMAR_MODES",
    "MathChannel",
    "MathRequest",
    "Message",
    "OpenAICompatClient",
    "StubClient",
    "build_grammar",
    "compile_branch_states",
    "deterministic_translator",
    "dry_run_client",
    "parse_contract",
    "to_state",
    "validate_grammar",
    "least_resistance_route",
]
