"""Neural reasoning core: a learned policy over symbolic transformations.

The neural component in ARIA does not emit expressions as token sequences.  It
scores *candidate symbolic transformations* -- pairs ``(state, rewrite)`` --
using structural features, and the search loop uses those scores to order what
to try next.  Answers always come from the rule engine and are always verified,
so a miscalibrated policy costs search time, never correctness.

The model is a logistic ranker trained by SGD on preference pairs mined from the
engine's own verified derivations.  It is intentionally dependency-free: the
same code runs with or without a deep-learning stack, and the learned weights
are a small JSON file you can inspect.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.ast import Expr, Number, Symbol
from ..embed.tree_embed import cosine, embed

__all__ = ["StepFeatures", "StepPolicy", "features", "PreferenceExample", "DEFAULT_WEIGHTS"]

FeatureDict = Dict[str, float]


@dataclass(frozen=True)
class PreferenceExample:
    """A single ranked choice: ``chosen`` is preferable to ``rejected``."""

    state_features: FeatureDict
    chosen_features: FeatureDict
    rejected_features: FeatureDict
    rule_chosen: str
    rule_rejected: str


def _complexity(expr: Expr) -> Dict[str, float]:
    numbers = 0
    non_integer = 0
    functions = 0
    negative_powers = 0
    for node in expr.walk():
        if isinstance(node, Number):
            numbers += 1
            if not node.is_integer:
                non_integer += 1
        elif getattr(node, "head", "") == "Func":
            functions += 1
        elif getattr(node, "head", "") == "Pow":
            exponent = getattr(node, "exp", None)
            if isinstance(exponent, Number) and exponent.is_negative:
                negative_powers += 1
    return {
        "numbers": float(numbers),
        "non_integer": float(non_integer),
        "functions": float(functions),
        "negative_powers": float(negative_powers),
    }


@dataclass(frozen=True)
class StepFeatures:
    name: str
    values: FeatureDict

    def as_dict(self) -> FeatureDict:
        return dict(self.values)


def features(before: Expr, after: Expr, goal: Optional[Expr], rule_name: str) -> FeatureDict:
    """Structural features describing one candidate rewrite."""
    before_embed = embed(before, dim=128)
    after_embed = embed(after, dim=128)
    goal_embed = embed(goal, dim=128) if goal is not None else None

    before_complexity = _complexity(before)
    after_complexity = _complexity(after)

    values: FeatureDict = {
        "bias": 1.0,
        f"rule={rule_name}": 1.0,
        "size_delta": (after.size - before.size) / 10.0,
        "depth_delta": (after.depth - before.depth) / 5.0,
        "size_shrank": 1.0 if after.size < before.size else 0.0,
        "depth_shrank": 1.0 if after.depth < before.depth else 0.0,
        "self_similarity": cosine(before_embed.vector, after_embed.vector),
        "numbers_delta": (after_complexity["numbers"] - before_complexity["numbers"]) / 5.0,
        "functions_delta": (after_complexity["functions"] - before_complexity["functions"]) / 3.0,
        "negative_powers_delta": after_complexity["negative_powers"] - before_complexity["negative_powers"],
        "shared_symbols": float(
            len({s.name for s in before.free_symbols} & {s.name for s in after.free_symbols})
        )
        / 4.0,
    }
    if goal_embed is not None:
        values["goal_similarity"] = cosine(after_embed.vector, goal_embed.vector)
        values["goal_similarity_delta"] = values["goal_similarity"] - cosine(
            before_embed.vector, goal_embed.vector
        )
    else:
        values["goal_similarity"] = 0.0
        values["goal_similarity_delta"] = 0.0
    return values


#: Hand-set starting point (also the fallback when no model file is present).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": -0.2,
    "size_delta": -0.8,
    "depth_delta": -0.4,
    "size_shrank": 1.0,
    "depth_shrank": 0.5,
    "self_similarity": 0.9,
    "numbers_delta": -0.2,
    "functions_delta": -0.4,
    "negative_powers_delta": -0.4,
    "shared_symbols": 0.6,
    "goal_similarity": 1.4,
    "goal_similarity_delta": 1.6,
}


@dataclass
class StepPolicy:
    """Logistic ranker over structural step features."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    history: List[float] = field(default_factory=list)

    # -- scoring -----------------------------------------------------------

    def score(self, values: FeatureDict) -> float:
        total = 0.0
        for name, value in values.items():
            total += self.weights.get(name, 0.0) * value
        return total

    def probability(self, values: FeatureDict) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.score(values)))))

    def rank(self, candidates: Sequence[StepFeatures]) -> List[StepFeatures]:
        return sorted(candidates, key=lambda c: self.score(c.values), reverse=True)

    # -- training ----------------------------------------------------------

    def train(
        self,
        examples: Sequence[PreferenceExample],
        epochs: int = 60,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Ranking loss (Bradley-Terry) with SGD; returns training metrics."""
        rng = random.Random(seed)
        order = list(range(len(examples)))
        for epoch in range(epochs):
            rng.shuffle(order)
            total_loss = 0.0
            for index in order:
                example = examples[index]
                margin = self.score(example.chosen_features) - self.score(example.rejected_features)
                loss = math.log1p(math.exp(-max(-30.0, min(30.0, margin))))
                total_loss += loss
                gradient_scale = -1.0 / (1.0 + math.exp(max(-30.0, min(30.0, margin))))
                for name, value in example.chosen_features.items():
                    self.weights[name] = self.weights.get(name, 0.0) - learning_rate * (
                        gradient_scale * value + l2 * self.weights.get(name, 0.0)
                    )
                for name, value in example.rejected_features.items():
                    self.weights[name] = self.weights.get(name, 0.0) + learning_rate * (
                        gradient_scale * value - l2 * self.weights.get(name, 0.0)
                    )
            self.history.append(total_loss / max(1, len(examples)))
        return {
            "examples": float(len(examples)),
            "epochs": float(epochs),
            "final_loss": self.history[-1] if self.history else float("nan"),
        }

    def accuracy(self, examples: Sequence[PreferenceExample]) -> float:
        if not examples:
            return float("nan")
        correct = sum(
            1
            for example in examples
            if self.score(example.chosen_features) >= self.score(example.rejected_features)
        )
        return correct / len(examples)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"weights": self.weights}, indent=2, sort_keys=True))
        return target

    @classmethod
    def load(cls, path: str | Path) -> "StepPolicy":
        data = json.loads(Path(path).read_text())
        return cls(weights={str(k): float(v) for k, v in data.get("weights", {}).items()})
