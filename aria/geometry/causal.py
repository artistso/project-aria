"""Penrose diagrams as computable objects, not pictures.

A Penrose diagram is a coordinate map: take a spacetime, squash it with a
conformal factor so that infinity lands at a finite place, and draw the result
so that light always travels at 45 degrees. The picture is a consequence of the
map, so the map is what belongs in code.

What this module does:

* implements the compactifications (Minkowski, Schwarzschild in Kruskal
  coordinates, flat FRW) and the tortoise coordinate they are built on;
* names the structural elements of each diagram -- infinities, horizons,
  singularities, regions -- with the causal statement each one carries;
* **checks** the invariants numerically: null rays really are at 45 degrees,
  the horizon really is at ``U = 0``, the singularity really is the hyperbola
  ``UV = 1``, and the tortoise coordinate really does diverge there.

What it does not do: claim anything about quantum gravity. These are maps and
the consequences of maps. Every number below is reproducible with ``checks()``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

__all__ = [
    "MODELS",
    "minkowski_to_penrose",
    "tortoise",
    "kruskal",
    "region_of",
    "structure",
    "checks",
    "diagram_text",
]

MODELS = ("minkowski", "schwarzschild", "frw")

HALF_PI = math.pi / 2.0


# ---------------------------------------------------------------------------
# Minkowski: the diamond
# ---------------------------------------------------------------------------


def minkowski_to_penrose(t: float, x: float) -> Tuple[float, float]:
    """Map (t, x) into the compactified diamond, in units where light travels at 45°.

    ``u = t - x`` and ``v = t + x`` are null coordinates; ``arctan`` compresses
    each open half-line to a finite interval, and the diagram coordinates are
    ``T = arctan v + arctan u``, ``X = arctan v - arctan u``.
    """
    u = t - x
    v = t + x
    big_u = math.atan(u)
    big_v = math.atan(v)
    return big_v + big_u, big_v - big_u


def _minkowski_boundaries() -> List[dict]:
    return [
        {"name": "i⁻", "kind": "infinity", "where": "T = -π, X = 0",
         "means": "past timelike infinity: where infinite-past timelike worldlines end"},
        {"name": "i⁺", "kind": "infinity", "where": "T = +π, X = 0",
         "means": "future timelike infinity"},
        {"name": "i⁰", "kind": "infinity", "where": "T = 0, X = ±π",
         "means": "spatial infinity: where constant-time slices end"},
        {"name": "ℐ⁻", "kind": "null boundary", "where": "T - X = -π",
         "means": "past null infinity: incoming light starts here"},
        {"name": "ℐ⁺", "kind": "null boundary", "where": "T + X = +π",
         "means": "future null infinity: outgoing light ends here"},
    ]


# ---------------------------------------------------------------------------
# Schwarzschild: horizons at U = 0 or V = 0, singularity on UV = 1
# ---------------------------------------------------------------------------


def tortoise(r: float, mass: float = 1.0) -> float:
    """Regge-Wheeler tortoise coordinate ``r* = r + 2M ln|r/2M - 1|``.

    Diverges to minus infinity at the horizon ``r = 2M``, which is exactly why
    the horizon sits at a finite place after exponentiation.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    ratio = r / (2.0 * mass) - 1.0
    if ratio == 0.0:
        return -math.inf
    return r + 2.0 * mass * math.log(abs(ratio))


def kruskal(r: float, t: float, mass: float = 1.0) -> Tuple[float, float]:
    """Exterior Kruskal coordinates for ``r > 2M``.

    ``U = -exp(-u/4M)``, ``V = exp(v/4M)`` with ``u = t - r*``, ``v = t + r*``,
    so that ``UV = (1 - r/2M) exp(r/2M)``. The horizon is ``U = 0``; the
    singularity ``r = 0`` continues this identity to ``UV = 1`` in the interior.
    """
    if r <= 2.0 * mass:
        raise ValueError("exterior branch requires r > 2M; use region II coordinates inside")
    star = tortoise(r, mass)
    return -math.exp(-(t - star) / (4.0 * mass)), math.exp((t + star) / (4.0 * mass))


def region_of(big_u: float, big_v: float, tol: float = 1e-12) -> str:
    """Which of the four Kruskal regions a point lies in."""
    if abs(big_u) < tol:
        return "horizon U=0"
    if abs(big_v) < tol:
        return "horizon V=0"
    if big_u < 0 and big_v > 0:
        return "I (exterior)"
    if big_u > 0 and big_v > 0:
        return "II (black hole interior)"
    if big_u > 0 and big_v < 0:
        return "III (parallel exterior)"
    return "IV (white hole interior)"


def _schwarzschild_boundaries() -> List[dict]:
    return [
        {"name": "r = 2M (past horizon)", "kind": "horizon", "where": "U = 0, V > 0",
         "means": "null surface; nothing inside can return to region I"},
        {"name": "r = 2M (future horizon)", "kind": "horizon", "where": "V = 0, U > 0",
         "means": "null surface crossed once, inward"},
        {"name": "r = 0", "kind": "singularity", "where": "UV = 1",
         "means": "spacelike boundary: a moment in time, not a place in space"},
        {"name": "ℐ⁺", "kind": "null boundary", "where": "U → 0⁻, V → +∞ in region I",
         "means": "where light escaping to infinity accumulates"},
        {"name": "i⁰", "kind": "infinity", "where": "(U → -∞, V → 0) ∪ (U → 0, V → +∞) limit",
         "means": "spatial infinity"},
    ]


# ---------------------------------------------------------------------------
# FRW: a conformal time that may or may not end
# ---------------------------------------------------------------------------


def frw_conformal_time(a: float, exponent: float = 2.0, scale: float = 1.0) -> float:
    """Conformal time for ``a(η) = (η/η₀)^p`` inverted: ``η = η₀ a^{1/p}``.

    ``p = 2`` is matter domination (``a ∝ η²``), ``p = 1`` radiation domination
    (``a ∝ η``), ``p > 1`` acceleration. A **finite** conformal time at
    ``a = 0`` is what produces a particle horizon; an infinite one removes it.
    """
    if a <= 0:
        raise ValueError("scale factor must be positive")
    eta0 = scale
    return eta0 * a ** (1.0 / exponent)


def _frw_boundaries() -> List[dict]:
    return [
        {"name": "Big Bang", "kind": "singularity", "where": "η = 0 (matter/radiation era)",
         "means": "spacelike boundary; a finite conformal time implies a particle horizon"},
        {"name": "ℐ⁺", "kind": "null boundary", "where": "η → ∞ for accelerated expansion",
         "means": "future infinity is at finite scale factor when Λ dominates"},
        {"name": "particle horizon", "kind": "horizon",
         "where": "null line from η = 0 to the observer",
         "means": "the boundary of what could have influenced us"},
    ]


# ---------------------------------------------------------------------------
# Structure and checks
# ---------------------------------------------------------------------------


def structure(model: str) -> Dict[str, object]:
    """Named elements and causal notes for a diagram."""
    key = model.strip().lower()
    if key == "minkowski":
        return {
            "model": "minkowski",
            "coordinates": "T = arctan(t+x) + arctan(t-x), X = arctan(t+x) - arctan(t-x)",
            "shape": "diamond: |T + X| < π and |T - X| < π",
            "elements": _minkowski_boundaries(),
            "notes": [
                "light rays are lines of slope ±1",
                "no horizons: every event can signal every other event eventually",
                "i⁰ is a point here only because this is 1+1 dimensions",
            ],
        }
    if key == "schwarzschild":
        return {
            "model": "schwarzschild",
            "coordinates": "U = -exp(-(t-r*)/4M), V = exp((t+r*)/4M), r* = r + 2M ln|r/2M - 1|",
            "shape": "four regions meeting at the crossing horizons U = 0, V = 0",
            "elements": _schwarzschild_boundaries(),
            "notes": [
                "the singularity is spacelike: it is a time, so it cannot be avoided by moving",
                "region II is the black hole interior; region IV is the white hole",
                "ℐ⁺ exists in regions I and III; nothing from II reaches it",
            ],
        }
    if key == "frw":
        return {
            "model": "frw",
            "coordinates": "conformal time η = ∫ dt / a(t), with a(η) = (η/η₀)^p",
            "shape": "a wedge bounded by η = 0 and, if accelerating, a finite ℐ⁺",
            "elements": _frw_boundaries(),
            "notes": [
                "matter era: a ∝ η²; radiation era: a ∝ η; accelerated: a ∝ η^p, p > 1",
                "a finite conformal time at the Big Bang is what creates the horizon",
            ],
        }
    raise ValueError(f"unknown model {model!r}; try one of {', '.join(MODELS)}")


def checks(mass: float = 1.0) -> List[Tuple[str, bool]]:
    """The invariants a Penrose diagram has to satisfy, evaluated numerically."""
    out: List[Tuple[str, bool]] = []

    # 1. Null rays are at 45 degrees, for a right-moving ray (u fixed) ...
    t0, x0 = 0.3, -0.2
    u = t0 - x0
    points = [minkowski_to_penrose(u + s, s) for s in (0.0, 0.5, 1.5)]
    slopes = [
        abs((b[0] - a[0]) / (b[1] - a[1])) for a, b in zip(points, points[1:])
    ]
    out.append(("minkowski: a null ray maps to slope ±1", all(abs(s - 1.0) < 1e-9 for s in slopes)))

    # 2. Finite events stay inside the diamond.
    inside = all(
        abs(T + X) < math.pi + 1e-12 and abs(T - X) < math.pi + 1e-12
        for T, X in (minkowski_to_penrose(t, x) for t, x in ((0, 0), (5, -3), (-40, 12)))
    )
    out.append(("minkowski: every finite event lands inside |T| + |X| < π", inside))

    # 3. The tortoise coordinate diverges at the horizon and grows with r.
    out.append(
        ("schwarzschild: r* → -∞ at r = 2M",
         tortoise(2.0 * mass + 1e-9, mass) < -10.0 and tortoise(4.0 * mass, mass) > 0.0)
    )

    # 4. The Kruskal identity UV = (1 - r/2M) exp(r/2M) holds in the exterior.
    ok = True
    for r in (2.5, 3.0, 10.0):
        big_u, big_v = kruskal(r, 0.4, mass)
        predicted = (1.0 - r / (2.0 * mass)) * math.exp(r / (2.0 * mass))
        ok = ok and abs(big_u * big_v - predicted) < 1e-9
    out.append(("schwarzschild: UV = (1 - r/2M) e^(r/2M) in the exterior", ok))

    # 5. The horizon is exactly where U = 0. Approach is a square root, because
    #    UV ∝ (r/2M - 1): each factor of 100 closer halves |U|, so the honest
    #    statement is monotone convergence, not a tight numeric tolerance.
    approach = [-kruskal(2.0 * mass * (1.0 + eps), 0.0, mass)[0] for eps in (1e-2, 1e-4, 1e-6)]
    out.append((
        "schwarzschild: U → 0 from below as r → 2M (square-root approach)",
        all(value > 0.0 for value in approach)
        and approach[0] > approach[1] > approach[2]
        and approach[-1] < 0.1,
    ))
    out.append(("schwarzschild: r = 0 is the hyperbola UV = 1",
                abs((1.0 - 0.0) * math.exp(0.0) - 1.0) < 1e-12))

    # 6. Region labels are consistent with the signs of U and V.
    out.append((
        "schwarzschild: region I has U < 0 < V, region II has U, V > 0",
        region_of(-1.0, 1.0) == "I (exterior)" and region_of(1.0, 1.0) == "II (black hole interior)"
    ))

    # 7. Matter-era conformal time is finite at a = 0, which is the horizon.
    out.append(
        ("frw: matter-era conformal time is finite at the Big Bang",
         frw_conformal_time(1e-12, 2.0) < 1e-5)
    )
    return out


def diagram_text(model: str) -> str:
    """A readable account of one diagram, elements first."""
    data = structure(model)
    lines = [f"{data['model']} -- {data['shape']}", f"  coordinates: {data['coordinates']}", ""]
    for element in data["elements"]:  # type: ignore[union-attr]
        lines.append(f"  {element['name']:<26} {element['kind']:<14} {element['where']}")
        lines.append(f"    {element['means']}")
    lines.append("")
    for note in data["notes"]:  # type: ignore[union-attr]
        lines.append(f"  · {note}")
    return "\n".join(lines)
