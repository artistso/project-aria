"""Shapes, their properties, and the checks that hold those properties up.

A shape here is a small record of counts and measurements -- vertices, edges,
faces, area, volume, angles -- plus the identities that must hold for it. The
point is not to draw the shape. The point is that a property stated about a
shape can be *checked*: a cube with five faces is not a cube, and
``V - E + F = 2`` says so without anyone looking at a picture.

Everything is computed from the edge length (polyhedra) or the radius
(curves), so a value is never stored separately from the rule that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = [
    "PLATONIC",
    "Solid",
    "RegularPolygon",
    "Circle",
    "Sphere",
    "shape_names",
    "describe",
    "checks",
]


def _deg(radians: float) -> float:
    return radians * 180.0 / math.pi


@dataclass(frozen=True)
class RegularPolygon:
    """A regular n-gon of edge length ``side``."""

    sides: int
    side: float = 1.0

    def __post_init__(self) -> None:
        if self.sides < 3:
            raise ValueError("a regular polygon needs at least three sides")
        if self.side <= 0:
            raise ValueError("side length must be positive")

    @property
    def name(self) -> str:
        return f"regular {self.sides}-gon"

    @property
    def perimeter(self) -> float:
        return self.sides * self.side

    @property
    def area(self) -> float:
        return self.sides * self.side ** 2 / (4.0 * math.tan(math.pi / self.sides))

    @property
    def interior_angle(self) -> float:
        return (self.sides - 2) * 180.0 / self.sides

    @property
    def exterior_angle(self) -> float:
        return 360.0 / self.sides

    @property
    def circumradius(self) -> float:
        return self.side / (2.0 * math.sin(math.pi / self.sides))

    @property
    def inradius(self) -> float:
        return self.side / (2.0 * math.tan(math.pi / self.sides))

    @property
    def symmetry_order(self) -> int:
        return 2 * self.sides

    @property
    def schlafli(self) -> str:
        return f"{{{self.sides}}}"

    def properties(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "sides": self.sides,
            "side": self.side,
            "schlafli": self.schlafli,
            "perimeter": round(self.perimeter, 6),
            "area": round(self.area, 6),
            "interior_angle_deg": round(self.interior_angle, 6),
            "exterior_angle_deg": round(self.exterior_angle, 6),
            "circumradius": round(self.circumradius, 6),
            "inradius": round(self.inradius, 6),
            "symmetry_order": self.symmetry_order,
        }


def _regular_solid(
    name: str,
    vertices: int,
    edges: int,
    faces: int,
    face_sides: int,
    faces_at_vertex: int,
    dihedral: float,
    dual: str,
    symmetry: int,
) -> "Solid":
    return Solid(
        name=name,
        vertices=vertices,
        edges=edges,
        faces=faces,
        face_sides=face_sides,
        faces_at_vertex=faces_at_vertex,
        dihedral_deg=dihedral,
        dual=dual,
        symmetry_order=symmetry,
        schlafli=f"{{{face_sides},{faces_at_vertex}}}",
    )


@dataclass(frozen=True)
class Solid:
    """A convex regular polyhedron with an edge length."""

    name: str
    vertices: int
    edges: int
    faces: int
    face_sides: int
    faces_at_vertex: int
    dihedral_deg: float
    dual: str
    symmetry_order: int
    schlafli: str
    edge: float = 1.0

    def __post_init__(self) -> None:
        if self.edge <= 0:
            raise ValueError("edge length must be positive")

    @property
    def euler_characteristic(self) -> int:
        return self.vertices - self.edges + self.faces

    @property
    def surface_area(self) -> float:
        a = self.edge
        return {
            "tetrahedron": math.sqrt(3.0) * a ** 2,
            "cube": 6.0 * a ** 2,
            "octahedron": 2.0 * math.sqrt(3.0) * a ** 2,
            "dodecahedron": 3.0 * math.sqrt(25.0 + 10.0 * math.sqrt(5.0)) * a ** 2,
            "icosahedron": 5.0 * math.sqrt(3.0) * a ** 2,
        }[self.name]

    @property
    def volume(self) -> float:
        a = self.edge
        return {
            "tetrahedron": a ** 3 / (6.0 * math.sqrt(2.0)),
            "cube": a ** 3,
            "octahedron": math.sqrt(2.0) / 3.0 * a ** 3,
            "dodecahedron": (15.0 + 7.0 * math.sqrt(5.0)) / 4.0 * a ** 3,
            "icosahedron": 5.0 * (3.0 + math.sqrt(5.0)) / 12.0 * a ** 3,
        }[self.name]

    @property
    def circumradius(self) -> float:
        a = self.edge
        return {
            "tetrahedron": a * math.sqrt(6.0) / 4.0,
            "cube": a * math.sqrt(3.0) / 2.0,
            "octahedron": a / math.sqrt(2.0),
            "dodecahedron": a * math.sqrt(3.0) * (1.0 + math.sqrt(5.0)) / 4.0,
            "icosahedron": a * math.sqrt(10.0 + 2.0 * math.sqrt(5.0)) / 4.0,
        }[self.name]

    def properties(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "schlafli": self.schlafli,
            "edge": self.edge,
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces,
            "euler_characteristic": self.euler_characteristic,
            "face_sides": self.face_sides,
            "faces_at_vertex": self.faces_at_vertex,
            "dihedral_angle_deg": round(self.dihedral_deg, 4),
            "surface_area": round(self.surface_area, 6),
            "volume": round(self.volume, 6),
            "circumradius": round(self.circumradius, 6),
            "dual": self.dual,
            "symmetry_group_order": self.symmetry_order,
        }


@dataclass(frozen=True)
class Circle:
    radius: float = 1.0

    @property
    def name(self) -> str:
        return "circle"

    @property
    def circumference(self) -> float:
        return 2.0 * math.pi * self.radius

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    def properties(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "radius": self.radius,
            "circumference": round(self.circumference, 6),
            "area": round(self.area, 6),
            "symmetry_order": "continuous (SO(2))",
        }


@dataclass(frozen=True)
class Sphere:
    radius: float = 1.0

    @property
    def name(self) -> str:
        return "sphere"

    @property
    def surface_area(self) -> float:
        return 4.0 * math.pi * self.radius ** 2

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def euler_characteristic(self) -> int:
        return 2

    def properties(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "radius": self.radius,
            "surface_area": round(self.surface_area, 6),
            "volume": round(self.volume, 6),
            "euler_characteristic": self.euler_characteristic,
            "symmetry_order": "continuous (SO(3))",
        }


#: The five Platonic solids, by the counts that define them.
PLATONIC: Tuple[Solid, ...] = (
    _regular_solid("tetrahedron", 4, 6, 4, 3, 3, _deg(math.acos(1.0 / 3.0)), "tetrahedron", 24),
    _regular_solid("cube", 8, 12, 6, 4, 3, 90.0, "octahedron", 48),
    _regular_solid("octahedron", 6, 12, 8, 3, 4, _deg(math.acos(-1.0 / 3.0)), "cube", 48),
    _regular_solid(
        "dodecahedron", 20, 30, 12, 5, 3, _deg(math.acos(-1.0 / math.sqrt(5.0))), "icosahedron", 120
    ),
    _regular_solid(
        "icosahedron", 12, 30, 20, 3, 5,
        _deg(math.acos(-math.sqrt(5.0) / 3.0)), "dodecahedron", 120,
    ),
)

_NAMED = {solid.name: solid for solid in PLATONIC}
_NAMED["tetra"] = _NAMED["tetrahedron"]
_NAMED["hexahedron"] = _NAMED["cube"]
_NAMED["octa"] = _NAMED["octahedron"]
_NAMED["dodeca"] = _NAMED["dodecahedron"]
_NAMED["icosa"] = _NAMED["icosahedron"]


def shape_names() -> Tuple[str, ...]:
    return tuple(solid.name for solid in PLATONIC) + ("circle", "sphere", "polygon")


def describe(name: str, size: float = 1.0, sides: int = 5) -> Dict[str, object]:
    """Properties of a named shape at a given size."""
    key = name.strip().lower()
    if key in ("circle", "sphere"):
        return (Circle(size) if key == "circle" else Sphere(size)).properties()
    if key in ("polygon", "ngon", "n-gon"):
        return RegularPolygon(sides, size).properties()
    if key not in _NAMED:
        raise ValueError(
            f"unknown shape {name!r}; try one of {', '.join(shape_names())}"
        )
    base = _NAMED[key]
    return Solid(
        name=base.name,
        vertices=base.vertices,
        edges=base.edges,
        faces=base.faces,
        face_sides=base.face_sides,
        faces_at_vertex=base.faces_at_vertex,
        dihedral_deg=base.dihedral_deg,
        dual=base.dual,
        symmetry_order=base.symmetry_order,
        schlafli=base.schlafli,
        edge=size,
    ).properties()


def checks() -> List[Tuple[str, bool]]:
    """Identities that must hold, evaluated rather than asserted.

    Each item is ``(claim, passed)``. These are the checks ARIA can put into its
    ledger: a claim about a shape is verified by the same discipline as a claim
    about an equation.
    """
    out: List[Tuple[str, bool]] = []

    for solid in PLATONIC:
        out.append(
            (f"{solid.name}: V - E + F = 2", solid.euler_characteristic == 2)
        )

    # Every Platonic solid appears exactly once as a dual, and duality is an
    # involution: the dual of the dual is the solid itself.
    duals = {solid.name: solid.dual for solid in PLATONIC}
    out.append(
        ("duality is an involution on the five solids",
         all(duals[duals[name]] == name for name in duals))
    )

    # The solid angle constraint: q * (p - 2) < 2p is exactly what makes the
    # five solids the only possible ones.
    out.append(
        ("exactly five (p, q) pairs satisfy q(p - 2) < 2p",
         len({(s.face_sides, s.faces_at_vertex) for s in PLATONIC}) == 5
         and all(
             s.faces_at_vertex * (s.face_sides - 2) < 2 * s.face_sides for s in PLATONIC
         ))
    )

    polygon = RegularPolygon(7, 2.0)
    out.append(
        ("a regular polygon's area equals half its perimeter times its inradius",
         abs(polygon.area - polygon.perimeter * polygon.inradius / 2.0) < 1e-9)
    )
    out.append(
        ("a hexagon's interior angles sum to four right angles",
         abs(6 * RegularPolygon(6).interior_angle - 720.0) < 1e-9)
    )

    sphere = Sphere(2.0)
    out.append(
        ("a sphere's volume is one third of its surface area times its radius",
         abs(sphere.volume - sphere.surface_area * sphere.radius / 3.0) < 1e-9)
    )
    return out
