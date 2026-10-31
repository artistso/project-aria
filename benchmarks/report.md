# ARIA benchmark report

| metric | value |
|---|---|
| tasks | 46.0 |
| exact_correctness | 1.0 |
| invalid_derivation_rate | 0.0 |
| verified_trace_rate | 1.0 |
| mean_trace_accuracy | 1.0 |
| unit_consistency | 1.0 |
| total_steps | 72.0 |

## By suite

| suite | tasks | correct | correctness | mean steps | verified traces |
|---|---|---|---|---|---|
| calculus | 5 | 5 | 100% | 1.00 | 100% |
| factor | 5 | 5 | 100% | 1.20 | 100% |
| log-exp | 6 | 6 | 100% | 1.00 | 100% |
| modular | 5 | 5 | 100% | 0.60 | 100% |
| simplify | 7 | 7 | 100% | 1.71 | 100% |
| solve | 12 | 12 | 100% | 3.33 | 100% |
| verify | 6 | 6 | 100% | 0.00 | 100% |

## Task detail

| task | correct | status | steps | trace | result |
|---|---|---|---|---|---|
| solve-01 | ✓ | solved | 4 | 100% | `3` |
| solve-02 | ✓ | solved | 5 | 100% | `12` |
| solve-03 | ✓ | solved | 3 | 100% | `-3` |
| solve-04 | ✓ | solved | 3 | 100% | `-3` |
| solve-05 | ✓ | solved | 3 | 100% | `-1` |
| solve-06 | ✓ | solved | 2 | 100% | `2^(1/2)` |
| solve-07 | ✓ | solved | 3 | 100% | `3/5` |
| solve-08 | ✓ | solved | 4 | 100% | `1` |
| solve-09 | ✓ | solved | 4 | 100% | `` |
| solve-10 | ✓ | refuted | 4 | 100% | `` |
| solve-11 | ✓ | solved | 1 | 100% | `3` |
| solve-12 | ✓ | solved | 4 | 100% | `3` |
| simp-01 | ✓ | solved | 1 | 100% | `4 x` |
| simp-02 | ✓ | solved | 2 | 100% | `1 + 2 x` |
| simp-03 | ✓ | solved | 2 | 100% | `1 + 3 x + 3 x^2 + x^3` |
| simp-04 | ✓ | solved | 1 | 100% | `1 + x` |
| simp-05 | ✓ | solved | 1 | 100% | `1 + x` |
| simp-06 | ✓ | solved | 3 | 100% | `4 + x` |
| simp-07 | ✓ | solved | 2 | 100% | `4 a b` |
| fact-01 | ✓ | solved | 1 | 100% | `(-2 + x) (2 + x)` |
| fact-02 | ✓ | solved | 1 | 100% | `(2 + x) (3 + x)` |
| fact-03 | ✓ | solved | 1 | 100% | `2 x (2 + x)` |
| fact-04 | ✓ | solved | 2 | 100% | `9 (-4/3 + x) (4/3 + x)` |
| fact-05 | ✓ | solved | 1 | 100% | `(-3 + x) (-2 + x)` |
| log-01 | ✓ | solved | 2 | 100% | `log(b)` |
| log-02 | ✓ | solved | 1 | 100% | `3 log(a)` |
| log-03 | ✓ | solved | 3 | 100% | `log(1) + log(a)` |
| log-04 | ✓ | solved | 0 | 100% | `exp(log(x)) = x` |
| log-05 | ✓ | solved | 0 | 100% | `log(a b) = log(a) + log(b)` |
| log-06 | ✓ | refuted | 0 | 100% | `log(a + b) = log(a) + log(b)` |
| mod-01 | ✓ | solved | 1 | 100% | `2` |
| mod-02 | ✓ | solved | 1 | 100% | `5` |
| mod-03 | ✓ | solved | 0 | 100% | `` |
| mod-04 | ✓ | solved | 1 | 100% | `3` |
| mod-05 | ✓ | solved | 0 | 100% | `17 ≡ 2 (mod 5)` |
| calc-01 | ✓ | solved | 1 | 100% | `3 x^2 + 4 x` |
| calc-02 | ✓ | solved | 1 | 100% | `cos(x)` |
| calc-03 | ✓ | solved | 1 | 100% | `x^(-1)` |
| calc-04 | ✓ | solved | 1 | 100% | `2 x sin(x) + x^2 cos(x)` |
| calc-05 | ✓ | solved | 1 | 100% | `2 exp(2 x)` |
| ver-01 | ✓ | solved | 0 | 100% | `(a + b)^2 = 2 a b + a^2 + b^2` |
| ver-02 | ✓ | refuted | 0 | 100% | `(a + b)^2 = a^2 + b^2` |
| ver-03 | ✓ | solved | 0 | 100% | `cos(x)^2 + sin(x)^2 = 1` |
| ver-04 | ✓ | solved | 0 | 100% | `4 = 4` |
| ver-05 | ✓ | refuted | 0 | 100% | `4 = 5` |
| ver-06 | ✓ | solved | 0 | 100% | `(-1 + x^2) (-1 + x)^(-1) = 1 + x` |

## Dimensional analysis

| law | consistent | detail |
|---|---|---|
| coulomb | ✓ | both sides have the same dimension |
| gravitation | ✓ | both sides have the same dimension |
| ideal_gas | ✓ | both sides have the same dimension |
| kinetic_energy | ✓ | both sides have the same dimension |
| mass_energy | ✓ | both sides have the same dimension |
| newton_second | ✓ | both sides have the same dimension |
| ohms_law | ✓ | both sides have the same dimension |
| planck_energy | ✓ | both sides have the same dimension |
| wave_speed | ✓ | both sides have the same dimension |
| control_broken_law | ✓ | F = m a^2 must be rejected |