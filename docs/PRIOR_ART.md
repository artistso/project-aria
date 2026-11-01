# Prior Art

**Short answer: yes — extensively.** ARIA is not a new idea. It is an
implementation of a research programme that is roughly sixty years old in its
symbolic half and fifteen years old in its neuro-symbolic half. This document
records the lineages honestly so that nobody mistakes a working reference
implementation for a breakthrough, and so that the genuinely open questions stay
visible.

Read this before claiming novelty anywhere else.

---

## 1. Lineage A — symbolic computation and automated reasoning (1960s–)

The representation, rewriting and verification layers are classical computer
algebra and automated theorem proving:

| Work | Year | What it established |
|---|---|---|
| Macsyma | 1968 | symbolic algebra on computers as a practical tool |
| Resolution / paramodulation ATP | 1965– | machine-checkable derivations as objects |
| Mathematica, Maple, Maxima | 1988– | canonical symbolic manipulation at scale |
| Coq / Isabelle / HOL | 1989– | proofs as first-class, machine-verified artifacts |
| Z3 and the SMT revolution | 2008 | decision procedures as a service |
| Lean (2013) and mathlib | 2013– | a large, checkable mathematics library |
| SymPy | 2007– | open symbolic algebra in Python |

ARIA's `Equation`/`Congruence`/`Derivative` nodes, its rewrite rules and its
rational normal forms are refinements of this lineage. The `Verifier` is a thin
orchestrator over the same decision procedures.

---

## 2. Lineage B — neuro-symbolic AI as a field

Neuro-symbolic integration is an established subfield with its own surveys and
conferences (the AAAI *"The Future Is Neuro-Symbolic"* retrospective treats
AlphaGeometry/AlphaProof explicitly as neuro-symbolic case studies):

* **Neuro-Symbolic Concept Learner** (Mao et al., 2019) — perception neural,
  reasoning symbolic.
* **DeepProbLog**, **Logic Tensor Networks** — differentiable logic.
* **Garcez & Lamb surveys** — the framing of "System 1 / System 2" integration.

ARIA sits inside this tradition: neural scoring over symbolic structures, with
symbolic rules retaining semantic authority.

---

## 3. Lineage C — neural theorem proving and verified search (the closest work)

This is where ARIA's architecture has the most direct precedent. All of these
systems share one pattern: **a neural component proposes, a symbolic engine
verifies, and search sits between them.**

| System | Mechanism | Scale of result |
|---|---|---|
| **GPT-f** (Polu & Sutskever, 2020) | transformer proposes Lean tactics, best-first search over proof states | Metamath proofs at scale |
| **LeanDojo** (Yang et al., 2023) | retrieval-augmented tactic generation + Lean verification | introduced ReProver benchmarks |
| **AlphaGeometry** (*Nature*, 2024) | language model + **symbolic deduction engine**; LM supplies auxiliary constructions | 25/30 on IMO-AG-30 |
| **AlphaProof** (DeepMind, 2024) | Gemini autoformalisation + AlphaZero-style RL over **Lean** | silver-medal level at IMO 2024 (28/42) |
| **DeepSeek-Prover V1/V2** (2024–2025) | LLM + Lean, RL with compiler feedback | SOTA on Lean benchmarks |
| **Goedel-Prover**, **Kimina**, **Lean Copilot** | open models / tooling for Lean | open-source ecosystem |

Two things follow from this table.

1. **The "generate then verify" loop is not novel.** It is the mainstream
   architecture of formal mathematical AI, and it has produced medal-level
   results that ARIA cannot approach.
2. **ARIA differs in *where* verification applies, not *whether* it applies.**
   These systems verify whole proofs in an external proof assistant. ARIA
   verifies each internal rewrite step against a declared invariant, inside its
   own algebra — a weaker guarantee in a much smaller, dependency-free system.

Recent evaluations also show how hard the real problem is: on research-level
Lean projects the best models achieve roughly a 10% pass rate (RLMEval, 2025),
and surveys of autoformalisation describe formalisation quality, not search, as
the bottleneck. ARIA does not touch this frontier.

---

## 4. Lineage D — neural-guided symbolic search

The "learned policy orders symbolic operations" idea predates ARIA by years:

* **DreamCoder** (Ellis et al., 2021) — a neural recognition model guides search
  over a symbolic program language, and the library of abstractions grows by
  refactoring solved programs. This is the canonical statement of ARIA's
  neural-core design: *search is guided, never authored, by the network.*
* **AlphaTensor / AlphaDev** (2022–2023) — learned search over algebraic
  rewrites and assembly programs, with exact verification of outputs.
* **AI Feynman** (Udrescu & Tegmark, 2020) — symbolic regression of physical
  laws with dimensional constraints; the physics-side analogue of ARIA's
  science layer.

ARIA's `StepPolicy` is a very small instance of exactly this pattern. Its
measured result — **slower than the hand-ordered baseline at this scale** — is
itself consistent with the literature: neural guidance pays off when the search
space is large and the corpus is big, neither of which is true here.

---

## 5. Lineage E — step-level supervision of LLM reasoning (adjacent, different)

A parallel body of work asks the same question ARIA asks — *is a step correct?* —
but answers it with a learned scorer rather than a sound procedure:

* **Let's Verify Step by Step** (OpenAI, 2023) — process supervision beats
  outcome supervision; released PRM800K step labels.
* **Math-Shepherd** (2023) — automatically mined step rewards; the process
  reward model reranks solutions and improves GSM8K/MATH.

The distinction matters and should be stated precisely:

| | process reward model | ARIA step verifier |
|---|---|---|
| mechanism | learned scalar score | exact symbolic procedure |
| failure mode | confidently wrong | `unknown` |
| can it be fooled? | yes, adversarially | only by a bug in the backend |
| cost | training corpus + inference | polynomial arithmetic |

ARIA's contribution to this axis is the *discipline* of a three-valued verdict
and a measured **invalid-derivation rate**, not a new verification technique.

---

## 6. Lineage F — dimensional analysis and qualitative physics

Checking units is older than computing: it is standard practice in physics
education and in engineering tooling (Modelica, Simulink, FMI validators all
enforce dimensional consistency). Qualitative physics (de Kleer & Brown, 1984;
Forbus's QP theory) and intelligent tutoring systems such as ANDES built
constraint-based physical reasoners decades ago.

ARIA's dimensional algebra is a clean, exact re-implementation of a textbook
technique. It is not novel; what is comparatively unusual is running it in the
*same* verification pipeline as the algebra.

---

## 7. Layer-by-layer: who did it first

| ARIA layer | Precedent | ARIA's delta |
|---|---|---|
| expression trees, canonical forms | Macsyma, Mathematica, SymPy | exact `Fraction` everywhere; dependency-free |
| notation parser | every CAS | small instruction grammar mapped to typed goals |
| rewrite rules | all CAS rewrite engines | each rule returns its own justification string |
| **step verification** | proof assistants verify whole proofs | per-step *invariants* checked inside the algebra (`equation_scaled`, `both_sides_*`, `modular_equivalent`, `root`) |
| three-valued verdicts | SMT solvers (`sat`/`unsat`/`unknown`) | applied to *derivation steps*, surfaced in the CLI and benchmark |
| learned policy over rewrites | DreamCoder, AlphaGeometry, AlphaZero-style provers | tiny, dependency-free, trained on mined verified traces |
| dimensional analysis | engineering tools, qualitative physics | exact rational exponent vectors, in-pipeline |
| benchmark metrics | process-supervision literature | headline metric is **invalid derivation rate**, not pass rate |

---

## 8. What might be genuinely (if modestly) distinctive

1. **Step invariants as an explicit vocabulary.** Most systems either check the
   final answer or hand the whole proof to Lean. ARIA declares, per step, *which
   algebraic invariant* makes the step admissible (`equation_scaled` = residual
   preserved up to a non-zero factor) and checks exactly that. This catches
   squaring both sides automatically and is cheap to audit.
2. **Honesty as a measurable property.** `unknown` is a first-class verdict and
   `invalid_derivation_rate` is a headline benchmark number. Headline systems
   report pass rates; reported failure *modes* are rarer.
3. **A verification boundary the neural component cannot cross.** The policy
   ranks rule outputs; it cannot emit expressions. Enforced by API shape, not by
   convention.
4. **Reproducibility.** The core has no mandatory runtime dependency, the
   benchmark is deterministic for a fixed configuration, and the current suite
   has 282 tests. Useful as a teaching and auditing artifact — not as a
   competitive solver.

None of these is a research result. Items 1–3 are design choices already present
in pieces elsewhere; item 4 is engineering.

---

## 9. What ARIA is not, compared with prior art

* **Not a competitor to AlphaProof or DeepSeek-Prover.** No Lean/Coq backend,
  no general proof search, and no model weights bundled with this repository.
* **Not a general theorem prover.** It verifies identities and solves equations;
  it does not discharge multi-premise proofs.
* **Not evidence against LLMs.** The `MathChannel` places an actual supplied LLM
  at the natural-language edges and ARIA in the symbolic middle; it is a bounded
  neuro-symbolic integration, not a general LLM replacement.
* **Not benchmarked against any external baseline**, which is why its own
  numbers are labelled as a regression harness rather than a capability claim.

---

## 10. Bibliography

**Symbolic / formal**
- Macsyma (1968); Mathematica (1988); Maple; SymPy.
- Coq (1989), Isabelle (1986), Lean (2013) and mathlib.
- de Moura & Bjørner, *Z3: An Efficient SMT Solver* (2008).

**Neuro-symbolic**
- Mao et al., *The Neuro-Symbolic Concept Learner* (2019).
- Manhaeve et al., *DeepProbLog* (2018); Badreddine et al., *Logic Tensor Networks* (2022).
- *The Future Is Neuro-Symbolic: Where Has It Been, and Where Is It Going?* AAAI retrospective.

**Neural theorem proving**
- Polu & Sutskever, *Generative Language Modeling for Automated Theorem Proving* (GPT-f, 2020).
- Yang et al., *LeanDojo: Theorem Proving with Retrieval-Augmented Language Models* (2023).
- Trinh et al., *Solving Olympiad Geometry without Human Demonstrations* (AlphaGeometry), *Nature* 2024.
- DeepMind, *AI achieves silver-medal standard solving IMO problems* (AlphaProof / AlphaGeometry 2, 2024).
- Xin et al., *DeepSeek-Prover* (2024–2025); Lin et al., *Goedel-Prover* (2025).
- *Autoformalization in the Era of Large Language Models: A Survey* (2025).
- RLMEval: *Evaluating Research-Level Neural Theorem Proving* (EMNLP Findings 2025).

**Neural-guided symbolic search**
- Ellis et al., *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning* (PLDI 2021).
- Fawzi et al., *AlphaTensor* (2022); Mankowitz et al., *AlphaDev* (2023).
- Udrescu & Tegmark, *AI Feynman: A Physics-Inspired Method for Symbolic Regression* (2020).

**Step-level verification of reasoning**
- Lightman et al., *Let's Verify Step by Step* (2023).
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations* (2023).

**Qualitative physics**
- de Kleer & Brown, *A Qualitative Physics Based on Confluences* (1984).
- Forbus, *Qualitative Process Theory* (1984); VanLehn et al., *ANDES* (2005).

---

## 11. Practical consequence for this repository

Three habits follow from the literature:

1. **Never claim novelty for a layer.** Cite the lineage in any write-up.
2. **Compete on auditability, not accuracy.** ARIA cannot win a pass-rate
   comparison; it can win "show me why that step is legal".
3. **Move toward an external checker.** The single highest-value engineering
   change is a Lean/Coq export for the verified summary — that converts ARIA's
   internal verdicts into artifacts an independent system can check. It is the
   first item in `docs/ROADMAP.md` for exactly this reason.
