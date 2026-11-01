"""Policy-guided search over symbolic rewrites.

Search is where the neural component earns its keep: the rule engine defines
*what* is legal, the verifier decides *what is true*, and the policy decides
*what to try first*.  A better policy means fewer expansions to reach a verified
target -- it never changes what counts as an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.ast import Expr
from ..reason import rules as R
from ..reason.engine import SymbolicEngine
from ..verify.verifier import Verifier
from .policy import StepFeatures, StepPolicy, features


def _same_form(left: Expr, right: Expr) -> bool:
    """Structural target test: the search drives rewrites toward a normal form.

    Verification is deliberately *not* used as the search goal: a sound verifier
    can already decide equivalence of the start state, which would make the
    search vacuous.  Here the target is a specific canonical form, and the
    verifier is applied to the answer the search produces.
    """
    return left == right or left.canonical_key() == right.canonical_key()

__all__ = ["SearchResult", "PolicySearch"]


@dataclass
class SearchResult:
    found: bool
    start: Expr
    goal: Optional[Expr]
    result: Optional[Expr] = None
    path: List[Tuple[str, Expr]] = field(default_factory=list)
    expansions: int = 0
    visited: int = 0
    verified: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "found": self.found,
            "expansions": self.expansions,
            "visited": self.visited,
            "verified": self.verified,
            "path": [rule for rule, _ in self.path],
        }


class PolicySearch:
    """Beam search over rule applications, ordered by a learned policy."""

    def __init__(
        self,
        engine: Optional[SymbolicEngine] = None,
        policy: Optional[StepPolicy] = None,
        beam_width: int = 4,
        max_expansions: int = 120,
    ) -> None:
        self.engine = engine or SymbolicEngine()
        self.verifier = self.engine.verifier
        self.policy = policy or StepPolicy()
        self.beam_width = beam_width
        self.max_expansions = max_expansions

    def search(
        self,
        start: Expr,
        goal: Optional[Expr],
        rule_names: Optional[Sequence[str]] = None,
        max_depth: int = 6,
    ) -> SearchResult:
        rule_names = rule_names or list(R.SIMPLIFY_RULES)
        rules = [R.RULES[name] for name in rule_names if name in R.RULES]
        result = SearchResult(found=False, start=start, goal=goal)
        if goal is not None and _same_form(start, goal):
            result.found = True
            result.result = start
            result.verified = self.verifier.equivalent(start, goal).ok
            return result

        beam: List[Tuple[Expr, List[Tuple[str, Expr]]]] = [(start, [])]
        seen = {start.canonical_key()}
        for _ in range(max_depth):
            candidates: List[Tuple[float, Expr, List[Tuple[str, Expr]]]] = []
            for state, path in beam:
                for rule in rules:
                    candidate, _why = R.apply_rule_everywhere(state, rule)
                    if candidate == state:
                        continue
                    result.expansions += 1
                    if result.expansions > self.max_expansions:
                        return result
                    key = candidate.canonical_key()
                    if key in seen:
                        continue
                    seen.add(key)
                    result.visited += 1
                    values = features(state, candidate, goal, rule.name)
                    candidates.append((self.policy.score(values), candidate, path + [(rule.name, candidate)]))
            if not candidates:
                return result
            candidates.sort(key=lambda item: item[0], reverse=True)
            beam = [(candidate, path) for _score, candidate, path in candidates[: self.beam_width]]
            for _score, candidate, path in candidates[: self.beam_width]:
                if goal is not None and _same_form(candidate, goal):
                    result.found = True
                    result.result = candidate
                    result.path = path
                    result.verified = self.verifier.equivalent(candidate, goal).ok
                    return result
        if goal is None and beam:
            result.result = beam[0][0]
            result.path = beam[0][1]
        return result


# ---------------------------------------------------------------------------
# Baseline used for comparison: no learned ranking (rule declaration order)
# ---------------------------------------------------------------------------


class OrderedSearch(PolicySearch):
    """Same beam search, but candidates are ranked by a fixed rule order."""

    def search(  # type: ignore[override]
        self,
        start: Expr,
        goal: Optional[Expr],
        rule_names: Optional[Sequence[str]] = None,
        max_depth: int = 6,
    ) -> SearchResult:
        rule_names = rule_names or list(R.SIMPLIFY_RULES)
        rules = [R.RULES[name] for name in rule_names if name in R.RULES]
        result = SearchResult(found=False, start=start, goal=goal)
        if goal is not None and _same_form(start, goal):
            result.found = True
            result.result = start
            result.verified = self.verifier.equivalent(start, goal).ok
            return result
        beam: List[Tuple[Expr, List[Tuple[str, Expr]]]] = [(start, [])]
        seen = {start.canonical_key()}
        for _ in range(max_depth):
            next_beam: List[Tuple[Expr, List[Tuple[str, Expr]]]] = []
            for state, path in beam:
                for rule in rules:
                    candidate, _why = R.apply_rule_everywhere(state, rule)
                    if candidate == state:
                        continue
                    result.expansions += 1
                    if result.expansions > self.max_expansions:
                        return result
                    key = candidate.canonical_key()
                    if key in seen:
                        continue
                    seen.add(key)
                    result.visited += 1
                    next_beam.append((candidate, path + [(rule.name, candidate)]))
                    if goal is not None and _same_form(candidate, goal):
                        result.found = True
                        result.result = candidate
                        result.path = path + [(rule.name, candidate)]
                        result.verified = self.verifier.equivalent(candidate, goal).ok
                        return result
                    if len(next_beam) >= self.beam_width:
                        break
            if not next_beam:
                return result
            beam = next_beam[: self.beam_width]
        if goal is None and beam:
            result.result = beam[0][0]
            result.path = beam[0][1]
        return result


class RandomOrderSearch(OrderedSearch):
    """Beam search with an unlearned, randomly shuffled rule order (control)."""

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed

    def search(  # type: ignore[override]
        self,
        start: Expr,
        goal: Optional[Expr],
        rule_names: Optional[Sequence[str]] = None,
        max_depth: int = 6,
    ) -> SearchResult:
        names = list(rule_names or R.SIMPLIFY_RULES)
        import random

        random.Random(self.seed).shuffle(names)
        return super().search(start, goal, names, max_depth)
