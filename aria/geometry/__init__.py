"""Geometry ARIA can check: shapes with their identities, and causal diagrams."""

from .causal import MODELS, checks as causal_checks, diagram_text, structure
from .shapes import PLATONIC, describe, checks as shape_checks, shape_names

__all__ = [
    "MODELS",
    "PLATONIC",
    "describe",
    "shape_names",
    "structure",
    "diagram_text",
    "shape_checks",
    "causal_checks",
]
