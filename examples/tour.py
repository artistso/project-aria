"""A tour of the ARIA API -- run with ``python examples/tour.py``.

Demonstrates the five layers: representation, embedding, reasoning,
verification, and scientific dimensional analysis.
"""

from aria import (
    SymbolicEngine,
    Verifier,
    check_law,
    derive_dimension,
    embed,
    parse,
    parse_problem,
)
from aria.neural.policy import StepPolicy, features
from aria.neural.search import PolicySearch


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    engine = SymbolicEngine(Verifier())

    section("1. Representation: text becomes a typed expression tree")
    state = parse_problem("solve for x in 2x + 4 = 10")
    print(f"state     : {state}")
    print(f"goal      : {state.goal}")
    print(f"expression: {state.primary!r}")

    section("2. Reasoning + verification: every step carries a proof")
    derivation = engine.run(state)
    print(derivation.to_text())

    section("3. Verification is three-valued, not optimistic")
    for claim in (
        "log(a*b) = log(a) + log(b)",
        "log(a+b) = log(a) + log(b)",
        "sin(x)^2 + cos(x)^2 = 1",
    ):
        verdict = Verifier().claim(parse(claim))
        print(f"{claim:32s} -> {verdict}")

    section("4. Embeddings: structure, not strings")
    left = embed(parse("(x + 1)^2"))
    similar = embed(parse("(y + 1)^2"))
    different = embed(parse("log(a b) + sin(z)"))
    from aria.embed.tree_embed import cosine

    print(f"cos((x+1)^2, (y+1)^2)        = {cosine(left.vector, similar.vector):.3f}")
    print(f"cos((x+1)^2, log(ab)+sin(z)) = {cosine(left.vector, different.vector):.3f}")

    section("5. Neural policy: ranks rewrites, never decides truth")
    from aria.reason import rules as R

    policy = StepPolicy()
    start = parse("log(a*b) + 2x + 3x")
    goal = parse("log(a) + log(b) + 5 x")
    candidates = []
    for name in R.SIMPLIFY_RULES:
        candidate, _why = R.apply_rule_everywhere(start, R.RULES[name])
        if candidate != start:
            candidates.append((name, candidate))
    scored = sorted(
        candidates,
        key=lambda item: policy.score(features(start, item[1], goal, item[0])),
        reverse=True,
    )
    print(f"state: {start}    target form: {goal}")
    for name, candidate in scored:
        score = policy.score(features(start, candidate, goal, name))
        print(f"  {score:+.3f}  {name:20s} -> {candidate}")
    result = PolicySearch(policy=policy).search(start, goal)
    print(f"policy search found target: {result.found} in {result.expansions} expansion(s)")
    print("the policy only orders these candidates; each one is still rule-produced and verified")

    section("6. Science: dimensional analysis as a constraint system")
    print(check_law("newton_second").to_text())
    print()
    print(f"Solving F = m a for [a]: {derive_dimension('F = m a', {'F': 'kg*m/s^2', 'm': 'kg'}, 'a')}")
    broken = check_law("newton_second")
    from aria.science.units import check_equation

    bad = check_equation("F = m a^2", {"F": "kg*m/s^2", "m": "kg", "a": "m/s^2"})
    print(f"control (F = m a^2) consistent? {bad.consistent}  <- must be False")


if __name__ == "__main__":
    main()
