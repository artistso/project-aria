# Geometry: shapes, identities, and Penrose diagrams as computable objects

A shape in ARIA is not a picture. It is a small record of counts and measurements plus
the identities that must hold for them — so a claim about a shape can be *checked* the
same way a claim about an equation can.

```bash
aria geometry list
aria geometry shape icosahedron --size 2
aria geometry penrose schwarzschild
aria geometry check --remember        # run every identity, file the verified ones in memory
```

## 1. What is verified, not asserted

`aria geometry check` evaluates eleven identities and prints PASS/FAIL for each:

| identity | why it is the check |
|---|---|
| `V − E + F = 2` for all five Platonic solids | Euler's formula; a solid with the wrong counts is not that solid |
| duality is an involution | dual of the dual is the solid itself — catches a swapped edge |
| `q(p − 2) < 2p` admits exactly five `(p, q)` pairs | the reason there are five Platonic solids and not more |
| area = ½ · perimeter · inradius | ties three separate formulas together |
| a hexagon's angles sum to 720° | angle sum against the interior-angle formula |
| sphere volume = ⅓ · area · radius | the same relation that makes `dA = 8πr dr` |

Measured values, edge 1: tetrahedron `0.117851`, cube `1`, octahedron `0.471405`,
dodecahedron `7.663119`, icosahedron `2.181695`. Dihedral angles: tetrahedron
`70.5288°`, octahedron `109.4712°`, dodecahedron `116.5651°`, icosahedron `138.1897°`.
Symmetry-group orders: 24, 48, 48, 120, 120.

## 2. Penrose diagrams

A Penrose diagram is a **coordinate map**: squash a spacetime with a conformal factor so
infinity lands somewhere finite, chosen so that light always travels at 45°. The picture
is a consequence of the map, so the map is what belongs in code.

Implemented:

* **Minkowski.** `T = arctan(t+x) + arctan(t−x)`, `X = arctan(t+x) − arctan(t−x)`, which
  sends every finite event into the diamond `|T+X| < π`, `|T−X| < π`, with `i⁻`, `i⁺`,
  `i⁰`, `ℐ⁻`, `ℐ⁺` on the boundary.
* **Schwarzschild**, in Kruskal coordinates built on the tortoise coordinate
  `r* = r + 2M ln|r/2M − 1|`, so that `UV = (1 − r/2M) e^{r/2M}`. Horizons are `U = 0`
  and `V = 0`; the singularity is the hyperbola `UV = 1`; the four regions are labelled
  by the signs of `U` and `V`.
* **FRW.** Conformal time `η = η₀ a^{1/p}` — matter `p = 2`, radiation `p = 1`,
  acceleration `p > 1` — with the finite-at-the-Big-Bang condition that creates a
  particle horizon.

And checks, evaluated rather than claimed:

1. a null ray maps to slope ±1, for both right- and left-moving rays;
2. every finite event lands inside the diamond;
3. `r* → −∞` at `r = 2M`, while `r*(4M) > 0`;
4. `UV = (1 − r/2M) e^{r/2M}` holds at several radii;
5. `U → 0` from below as `r → 2M`, with the **square-root** approach — one factor of 100
   closer halves `|U|`, so the honest statement is monotone convergence, not a tight
   tolerance;
6. `r = 0` is the hyperbola `UV = 1`;
7. region labels match the signs of `U`, `V`;
8. matter-era conformal time is finite at the Big Bang.

### The honest boundary

This is 1+1-dimensional conformal structure: the causal skeleton, which is exactly what a
Penrose diagram is *for*. It is not Kerr, not numerical relativity, and not a claim about
quantum gravity. The structural statements — the singularity is spacelike, the horizon is
null, nothing inside region II reaches `ℐ⁺` — are dictionary facts about the
compactification, and every one of them is reproducible with `aria geometry check`.

## 3. Why this belongs in ARIA

Geometry is the second domain where the verifier can decide things that language models
are famously loose about, and the first where the *object* is not a formula:

* a shape's properties are arithmetic identities, so the existing verifier applies;
* a diagram's structure is a set of coordinate facts, so the existing verifier applies;
* both fail loudly. `aria geometry check` returns a non-zero exit code if any identity
  breaks, which means it can gate a commit or a lesson.

`--remember` files the verified facts into the text memory with `confidence: verified` and
a source pointing at the module that computed them, so Penrose structure and Platonic
counts are available to retrieval — and are marked *checked* rather than merely stated.

### Next step, if this line is worth pursuing

Add `geometry` as a rung on the ledger's ladder rather than a separate command: the
curriculum then asks for *"find the dihedral angle of the icosahedron"*, the model answers
with a contract, and the same engine that proves algebraic steps checks the angle. That is
the point where geometry stops being a calculator and becomes part of the school.
