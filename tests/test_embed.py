"""Structural embeddings."""

from aria.core.parse import parse
from aria.embed.tree_embed import SymbolicGraph, cosine, embed, node_features


def test_embedding_shape_and_normalisation():
    vector = embed(parse("x^2 + 2x + 1"), dim=64)
    assert len(vector) == 64
    norm = sum(v * v for v in vector.vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embedding_is_deterministic():
    first = embed(parse("(x+1)^2"))
    second = embed(parse("(x+1)^2"))
    assert first.as_tuple() == second.as_tuple()


def test_structurally_similar_expressions_score_higher():
    base = embed(parse("(x + 1)^2"))
    similar = embed(parse("(y + 1)^2"))
    different = embed(parse("log(a b) + sin(z)"))
    assert cosine(base.vector, similar.vector) > cosine(base.vector, different.vector)


def test_graph_view_of_a_tree():
    graph = SymbolicGraph.from_expr(parse("x + 2"))
    assert len(graph) == 3
    assert len(graph.edges) == 2


def test_node_features_are_interpretable():
    features = node_features(parse("sin(x)").base if hasattr(parse("sin(x)"), "base") else parse("sin(x)"))
    assert "fn=sin" in features or "head=Func" in features
