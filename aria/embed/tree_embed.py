"""Symbolic embedding layer.

Neural processing needs a fixed-width numeric view of a symbolic object, but the
view must preserve *structure*, not token identity.  ARIA therefore encodes
expression trees in two complementary ways:

* :class:`SymbolicGraph` -- an explicit node/edge graph suitable for a GNN or a
  tree transformer over node features.
* :class:`TreeEmbedding` -- a fixed-width, hash-bucketed descriptor built from
  root paths, node heads, operator counts and depth statistics, with no learned
  parameters and therefore fully reproducible.

The embedding is deliberately *structural*: two expressions that differ only in
variable naming share the same shape features, and two syntactically similar
expressions with different semantics differ in their operator/path features.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from ..core.ast import (
    Add,
    Congruence,
    Derivative,
    Equation,
    Expr,
    Func,
    Matrix,
    Mul,
    Number,
    Pow,
    Relation,
    Symbol,
)

__all__ = ["SymbolicGraph", "TreeEmbedding", "embed", "cosine", "node_features"]


# ---------------------------------------------------------------------------
# Graph view
# ---------------------------------------------------------------------------


@dataclass
class SymbolicGraph:
    """Node/edge view of an expression tree (pre-order indexing)."""

    features: List[Tuple[str, ...]] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)

    @classmethod
    def from_expr(cls, expr: Expr) -> "SymbolicGraph":
        graph = cls()

        def visit(node: Expr, parent: int = -1) -> int:
            index = len(graph.features)
            graph.features.append(node_features(node))
            if parent >= 0:
                graph.edges.append((parent, index))
            for child in node.children():
                visit(child, index)
            return index

        visit(expr)
        return graph

    def adjacency(self, size: int) -> List[List[int]]:
        """Dense adjacency lists (undirected, self-loops included)."""
        neighbours: List[List[int]] = [[i] for i in range(size)]
        for parent, child in self.edges:
            neighbours[parent].append(child)
            neighbours[child].append(parent)
        return neighbours


def node_features(node: Expr) -> Tuple[str, ...]:
    """Discrete feature tokens for a node (a small, model-friendly vocabulary)."""
    tokens: List[str] = [f"head={node.head}"]
    if isinstance(node, Number):
        tokens.append("numeric=integer" if node.is_integer else "numeric=rational")
        tokens.append("sign=neg" if node.is_negative else "sign=pos")
        tokens.append("magnitude=small" if abs(float(node.value)) < 10 else "magnitude=large")
    elif isinstance(node, Symbol):
        tokens.append("symbol=variable")
        for assumption in sorted(node.assumptions):
            tokens.append(f"assume={assumption}")
    elif isinstance(node, Add):
        tokens.append(f"arity={min(len(node.terms), 6)}")
    elif isinstance(node, Mul):
        tokens.append(f"arity={min(len(node.factors), 6)}")
    elif isinstance(node, Pow):
        if isinstance(node.exp, Number):
            tokens.append("exponent=numeric")
            if node.exp.value == 2:
                tokens.append("exponent=square")
            elif node.exp.value < 0:
                tokens.append("exponent=negative")
        else:
            tokens.append("exponent=symbolic")
    elif isinstance(node, Func):
        tokens.append(f"fn={node.name}")
    elif isinstance(node, Derivative):
        tokens.append("calculus=derivative")
    elif isinstance(node, Equation):
        tokens.append("relation=equation")
    elif isinstance(node, Relation):
        tokens.append(f"relation={node.op}")
    elif isinstance(node, Congruence):
        tokens.append("relation=congruence")
    elif isinstance(node, Matrix):
        tokens.append(f"matrix={node.shape[0]}x{node.shape[1]}")
    return tuple(tokens)


# ---------------------------------------------------------------------------
# Fixed-width embedding
# ---------------------------------------------------------------------------


@dataclass
class TreeEmbedding:
    """A hash-bucketed structural descriptor of an expression tree."""

    vector: List[float]
    dim: int
    dimension: int = 0  # length of the vector

    def __post_init__(self) -> None:
        self.dimension = len(self.vector)
        norm = math.sqrt(sum(v * v for v in self.vector)) or 1.0
        self.vector = [v / norm for v in self.vector]

    def as_tuple(self) -> Tuple[float, ...]:
        return tuple(self.vector)

    def __len__(self) -> int:
        return len(self.vector)

    def __getitem__(self, index: int) -> float:
        return self.vector[index]


def _hash(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def embed(expr: Expr, dim: int = 128, include_values: bool = True) -> TreeEmbedding:
    """Embed an expression into a fixed-width structural vector.

    Features: root-path signatures (depth-weighted), node-head unigrams and
    bigrams, operator counts and normalised depth statistics.  Paths capture
    where in the tree a construct occurs, which is what distinguishes
    ``(x + 1)^2`` from ``x^2 + 1``.
    """
    vector = [0.0] * dim
    paths: Dict[str, float] = {}

    def visit(node: Expr, path: Tuple[str, ...], depth: int) -> None:
        head = node.head if not isinstance(node, Symbol) else "Symbol"
        signature = "/".join(path + (head,))
        paths[signature] = paths.get(signature, 0.0) + 1.0 / (1.0 + depth)
        for child in node.children():
            visit(child, path + (head,), depth + 1)

    visit(expr, (), 0)

    for signature, weight in paths.items():
        vector[_hash("path:" + signature, dim)] += weight

    counts: Dict[str, int] = {}
    for node in expr.walk():
        counts[node.head] = counts.get(node.head, 0) + 1
        if isinstance(node, Func):
            counts["fn:" + node.name] = counts.get("fn:" + node.name, 0) + 1
        if isinstance(node, Pow):
            counts["pow"] = counts.get("pow", 0) + 1
            if isinstance(node.exp, Number) and node.exp.value < 0:
                counts["pow:negative"] = counts.get("pow:negative", 0) + 1
    for token, count in counts.items():
        vector[_hash("count:" + token, dim)] += math.log1p(count)

    vector[_hash(f"depth={min(expr.depth, 12)}", dim)] += 1.0
    vector[_hash(f"size={min(expr.size, 64)}", dim)] += 1.0
    if include_values and isinstance(expr, Number):
        vector[_hash(f"value={expr.value}", dim)] += 1.0

    return TreeEmbedding(vector, dim)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left)) or 1.0
    norm_right = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (norm_left * norm_right)
