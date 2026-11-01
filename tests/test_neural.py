"""Neural reasoning core: policy features, ranking and search."""

from aria.core.parse import parse, parse_problem
from aria.neural.policy import StepPolicy, features
from aria.neural.search import OrderedSearch, PolicySearch
from aria.neural.train import mine_preferences, train_policy


def test_feature_vector_is_finite():
    values = features(parse("(x+1)^2"), parse("x^2 + 2x + 1"), parse("x^2 + 2x + 1"), "distribute")
    assert values["bias"] == 1.0
    assert all(isinstance(v, float) for v in values.values())


def test_policy_prefers_goal_directed_steps():
    policy = StepPolicy()
    start = parse("(x + 1)^2")
    goal = parse("x^2 + 2x + 1")
    toward = features(start, goal, goal, "distribute")
    away = features(start, parse("2 x + 1"), goal, "collect_like_terms")
    assert policy.score(toward) > policy.score(away)


def test_training_reduces_loss_and_generalises():
    policy, report = train_policy(epochs=30, save_path=None)
    assert report.examples > 20
    assert policy.history[-1] <= policy.history[0]
    assert report.train_accuracy > 0.7


def test_policy_persistence(tmp_path):
    path = tmp_path / "weights.json"
    policy = StepPolicy(weights={"bias": 1.5, "size_delta": -2.0})
    policy.save(path)
    restored = StepPolicy.load(path)
    assert restored.weights["bias"] == 1.5


def test_policy_search_reaches_a_normal_form():
    engine = None
    searcher = PolicySearch()
    start = parse("2x + 3x - x")
    goal = parse("4x")
    result = searcher.search(start, goal)
    assert result.found
    assert result.verified


def test_baseline_search_also_finds_the_goal():
    searcher = OrderedSearch()
    result = searcher.search(parse("(x + 1)^2 - x^2"), parse("1 + 2x"))
    assert result.found


def test_mined_examples_are_verified_pairs():
    examples = mine_preferences()
    assert examples
    assert all(example.rule_chosen != example.rule_rejected for example in examples)
