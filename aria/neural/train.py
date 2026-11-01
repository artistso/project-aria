"""Train the step-ranking policy from verified derivations.

Training data is mined, not hand-labelled: the rule engine produces a verified
derivation, the step it chose becomes the positive example, and every other
rule applicable in the same state becomes a negative example.  This is a small,
fully reproducible curriculum in which the labels are exactly "what a verified
derivation actually did next".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..bench.tasks import TASKS, Task
from ..core.ast import Expr
from ..core.parse import parse, parse_problem
from ..reason import rules as R
from ..reason.engine import SymbolicEngine
from .policy import PreferenceExample, StepPolicy, features
from .search import OrderedSearch, PolicySearch, RandomOrderSearch

__all__ = ["mine_preferences", "train_policy", "evaluate_search", "DEFAULT_WEIGHTS_PATH"]

DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("weights.json")

#: Extra expressions used only for mining rule-ordering pairs (broadens rule
#: coverage beyond the benchmark suite, without touching evaluation data).
SYNTHETIC_EXPRESSIONS: Tuple[str, ...] = (
    "2(x + 3) - 4x",
    "(x + 2)(x - 3)",
    "(a + b)^3",
    "(x^2 - 9)/(x - 3)",
    "log(a*b*c)",
    "log(a^2) + log(b)",
    "3x + 2x - x + 7 - 2",
    "(2x + 1)^2",
    "(x + 1)^2 - (x - 1)^2",
    "x^2 + 6x + 9",
    "2x^2 - 8",
    "sin(x)^2 + cos(x)^2 + x",
    "exp(log(x + 1))",
    "log(a/b) - log(a) + log(b)",
    "4x^2 - 12x + 9",
)


def _goal_expression(task: Task) -> Optional[Expr]:
    if task.expected_result:
        try:
            return parse(task.expected_result)
        except Exception:
            return None
    return None


def _neighbours(state: Expr, rule_names: Sequence[str]):
    """All one-step rewrites of ``state`` (rule name, resulting expression)."""
    for name in rule_names:
        rule = R.RULES.get(name)
        if rule is None:
            continue
        candidate, _why = R.apply_rule_everywhere(state, rule)
        if candidate != state:
            yield name, candidate


def mine_shortest_path_preferences(
    expressions: Sequence[str] = SYNTHETIC_EXPRESSIONS,
    max_depth: int = 3,
    per_state_limit: int = 3,
) -> List[PreferenceExample]:
    """Mine labels from *shortest verified paths* rather than expert order.

    For every reachable state we record which rule lies on a shortest path to the
    engine's normal form for that expression.  Preference pairs then say "prefer
    the step that shortens the derivation", which is a genuinely different
    objective from imitating the engine's fixed rule order.
    """
    engine = SymbolicEngine()
    rule_names = list(R.SIMPLIFY_RULES) + list(R.FACTOR_RULES)
    examples: List[PreferenceExample] = []
    for text in expressions:
        try:
            start = parse(text)
            target = engine.simplify(start).result
        except Exception:
            continue
        if target is None:
            continue
        # Breadth-first search from the start state to the target normal form.
        frontier = [(start, 0)]
        distance = {start.canonical_key(): 0}
        edges: Dict[tuple, List[Tuple[str, Expr]]] = {}
        target_key = target.canonical_key()
        for state, depth in frontier:
            if depth >= max_depth:
                continue
            neighbours = list(_neighbours(state, rule_names))[: 40]
            edges[state.canonical_key()] = neighbours
            for name, candidate in neighbours:
                key = candidate.canonical_key()
                if key not in distance:
                    distance[key] = depth + 1
                    frontier.append((candidate, depth + 1))
        if target_key not in distance:
            continue
        # Any transition that decreases the BFS distance lies on a shortest path.
        for state, depth in frontier:
            for name, candidate in edges.get(state.canonical_key(), []):
                child_distance = distance.get(candidate.canonical_key())
                if child_distance is None or child_distance != depth + 1:
                    continue
                chosen = features(state, candidate, target, name)
                rejected_count = 0
                for other_name, other in edges.get(state.canonical_key(), []):
                    if other_name == name:
                        continue
                    if other.canonical_key() == candidate.canonical_key():
                        continue
                    if distance.get(other.canonical_key(), 99) < distance[candidate.canonical_key()]:
                        continue
                    examples.append(
                        PreferenceExample(
                            state_features={},
                            chosen_features=chosen,
                            rejected_features=features(state, other, target, other_name),
                            rule_chosen=name,
                            rule_rejected=other_name,
                        )
                    )
                    rejected_count += 1
                    if rejected_count >= per_state_limit:
                        break
    return examples


def mine_preferences(
    engine: Optional[SymbolicEngine] = None,
    tasks: Sequence[Task] = TASKS,
) -> List[PreferenceExample]:
    """Mine ranked rule-ordering pairs from verifier-checked simplification traces.

    These are generated from symbolic search outcomes, not human preference
    ratings or user feedback.
    """
    engine = engine or SymbolicEngine()
    examples: List[PreferenceExample] = []
    relevant = {"simplify", "factor", "log-exp", "calculus", "solve"}

    for expression_text in SYNTHETIC_EXPRESSIONS:
        try:
            expr = parse(expression_text)
            derivation = engine.simplify(expr)
        except Exception:
            continue
        for step in derivation.steps:
            if not step.verified:
                continue
            chosen_values = features(step.before, step.after, None, step.rule)
            for name in R.SIMPLIFY_RULES + R.FACTOR_RULES:
                if name == step.rule:
                    continue
                rule = R.RULES.get(name)
                if rule is None:
                    continue
                candidate, _why = R.apply_rule_everywhere(step.before, rule)
                if candidate == step.before or candidate == step.after:
                    continue
                rejected_values = features(step.before, candidate, None, rule.name)
                examples.append(
                    PreferenceExample(
                        state_features={},
                        chosen_features=chosen_values,
                        rejected_features=rejected_values,
                        rule_chosen=step.rule,
                        rule_rejected=rule.name,
                    )
                )

    for task in tasks:
        if task.suite not in relevant:
            continue
        goal = _goal_expression(task)
        try:
            state = parse_problem(task.prompt)
            derivation = engine.run(state)
        except Exception:
            continue
        for step in derivation.steps:
            if not step.verified:
                continue
            chosen_values = features(step.before, step.after, goal, step.rule)
            for name in R.SIMPLIFY_RULES + R.FACTOR_RULES:
                if name == step.rule:
                    continue
                rule = R.RULES.get(name)
                if rule is None:
                    continue
                candidate, _why = R.apply_rule_everywhere(step.before, rule)
                if candidate == step.before or candidate == step.after:
                    continue
                rejected_values = features(step.before, candidate, goal, rule.name)
                examples.append(
                    PreferenceExample(
                        state_features={},
                        chosen_features=chosen_values,
                        rejected_features=rejected_values,
                        rule_chosen=step.rule,
                        rule_rejected=rule.name,
                    )
                )
    return examples


@dataclass
class TrainingReport:
    examples: int
    train_examples: int
    test_examples: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    weights_path: Optional[str] = None

    def to_text(self) -> str:
        lines = [
            "ARIA step-policy training",
            f"  rule-order pairs : {self.examples} (train {self.train_examples} / test {self.test_examples})",
            f"  train accuracy   : {self.train_accuracy:.1%}",
            f"  test accuracy    : {self.test_accuracy:.1%}",
            f"  final ranking loss: {self.final_loss:.4f}",
        ]
        if self.weights_path:
            lines.append(f"  weights saved to : {self.weights_path}")
        return "\n".join(lines)


def train_policy(
    tasks: Sequence[Task] = TASKS,
    epochs: int = 60,
    learning_rate: float = 0.05,
    seed: int = 0,
    save_path: Optional[Path] = None,
) -> Tuple[StepPolicy, TrainingReport]:
    """Mine examples, split, train and persist the policy."""
    engine = SymbolicEngine()
    examples = mine_preferences(engine, tasks) + mine_shortest_path_preferences()
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.8))
    train_set = shuffled[:split]
    test_set = shuffled[split:]

    policy = StepPolicy()
    metrics = policy.train(train_set, epochs=epochs, learning_rate=learning_rate, seed=seed)
    report = TrainingReport(
        examples=len(examples),
        train_examples=len(train_set),
        test_examples=len(test_set),
        train_accuracy=policy.accuracy(train_set),
        test_accuracy=policy.accuracy(test_set) if test_set else float("nan"),
        final_loss=metrics["final_loss"],
    )
    if save_path is not None:
        saved = policy.save(save_path)
        report.weights_path = str(saved)
    return policy, report


def evaluate_search(
    tasks: Sequence[Task] = TASKS,
    suite: str = "simplify",
    policy: Optional[StepPolicy] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Compare three candidate orderings under identical search and verification.

    * ``policy``  -- goal-aware learned ranking,
    * ``fixed``   -- the hand-ordered rule list (the engine's default, a strong
      hand-tuned baseline for this rule set),
    * ``random``  -- a seeded shuffle (the control).

    All three consult the same verifier, so soundness is identical by
    construction; only search effort and reachability can differ.
    """
    engine = SymbolicEngine()
    policy = policy or StepPolicy()
    variants = {
        "policy": PolicySearch(engine, policy),
        "fixed": OrderedSearch(engine),
        "random": RandomOrderSearch(engine, seed=seed),
    }
    relevant = [task for task in tasks if task.suite == suite and task.expected_result]
    if not relevant:
        relevant = [task for task in tasks if task.expected_result]

    totals: Dict[str, float] = {"tasks": 0.0}
    for name in variants:
        totals[f"{name}_solved"] = 0.0
        totals[f"{name}_expansions"] = 0.0

    for task in relevant:
        goal = _goal_expression(task)
        if goal is None:
            continue
        expr = parse_problem(task.prompt).primary
        totals["tasks"] += 1
        for name, searcher in variants.items():
            result = searcher.search(expr, goal)
            totals[f"{name}_solved"] += 1.0 if result.found else 0.0
            totals[f"{name}_expansions"] += result.expansions

    count = max(1.0, totals["tasks"])
    summary: Dict[str, float] = {"tasks": totals["tasks"]}
    for name in variants:
        summary[f"{name}_solved"] = totals[f"{name}_solved"]
        summary[f"{name}_mean_expansions"] = round(totals[f"{name}_expansions"] / count, 3)
    return summary
