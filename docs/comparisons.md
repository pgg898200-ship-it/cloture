# How `cloture` Compares to Existing Tools

This page provides a factual, benchmark-verified comparison between `cloture` and existing Python, C++, and Rust statistical libraries — including **when to choose `cloture`** and **when another library is a better fit**.

## Feature & Architecture Matrix

| Capability | `cloture` (`0.1.0`) | `confseq` (`0.0.11`) | `ppi-python` (`0.2.3`) | `expectation` (`0.6.1`) | `online-cp` (`0.4.0`) | `bayesbench` (`0.4.0.dev0`) |
|---|---|---|---|---|---|---|
| **Grey-box distribution mode stopping (`ASC-D`)** | Yes (`2.2 µs` batch) | No | No | No | No | No |
| **Unbounded-category mode stopping (`CITE` + Kraft-IUT + budget lock)** | Yes (`247 ns` batch) | No | No | No | No | No |
| **Realized vs. Population correlated rollout cert (`PopulationCert`)** | Yes | No | No | No | No | No |
| **Correlated judge-panel audit ($n_{\text{eff}}$, Condorcet gap, LOO pruning)** | Yes (`318 ms` for 9x1k) | No | No | No | No | No |
| **Anytime-valid Confidence Sequences (`EB-CS`, `BettingCS`)** | Yes (`89 ns` / `6.3 µs`) | Yes (`442 ns` / `179 µs`) | No | Yes (`22.8 ns` final-only / `1.13 ms` stream) | No | No |
| **Multi-arm sequential racing (paired + LUCB + `e-BH`)** | Yes (`2.75 ms`, `0.0%` tie FWER) | No | No | Pairwise only | No | Bayesian (`30.5 s`, `40%` tie FPR) |
| **Online anytime-valid Active-PPI++ (safe under peeking & inverted judges)** | Yes (`99%` anytime cov) | No | Fixed-horizon only (`71%` under peeking) | No | No | No |
| **Poisoning-resistant multi-agent fusion (`SafeFuse`)** | Yes (`1.24 µs`) | No | No | No | No | No |
| **Recall-controlled trajectory abort (`AbortCascade`) & 4-rate `AgACI`** | Yes (`64.5 ns` Rust) | No | No | No | Yes (`189–313 µs`) | No |
| **Fixed `<1 KB` associative distributed monoid (`EvidenceSketch`)** | Yes (`712 B`, `76 ns` merge) | No | No | No | No | No |
| **Single static CLI binary + JSON-RPC 2.0 MCP server** | Yes (`2.47 MiB`, `3.1 ms` cold) | No | No | No | No | No |
| **Zero external Rust / Python runtime dependencies** | Yes (`15`-line `Cargo.lock`) | Needs NumPy/SciPy/Boost | Needs NumPy/SciPy/statsmodels | Needs NumPy/PyO3 | Needs NumPy/scikit-learn | Needs NumPy/SciPy |

---

## Detailed Library-by-Library Guidance

### `cloture` vs. `confseq`
- **Where `cloture` wins:** `4.0x` faster batch Empirical-Bernstein CS (`110 ns` vs. `442 ns/step`), `28.4x` faster and tighter Hedged ONS Betting CS (`6.32 µs` vs. `179 µs/step`, width `0.095` vs. `0.119` at `N = 1,000`), zero Boost/NumPy build issues on Python 3.12+, plus multi-agent stopping (`ASC-D`, `CITE`, `PopulationCert`), panel auditing, racing, Active-PPI++, and CLI/MCP.
- **When to use `confseq`:** If you specifically need `confseq`'s specialized asymptotic conjugate-mixture boundaries (`conjmix_empbern_cs`, `normal_mixture_bound`, sub-exponential/sub-gamma research boundaries) for offline notebook experiments.

### `cloture` vs. `expectation`
- **Where `cloture` wins:** True `O(1)` streaming updates (`5.16 µs` Python / `88.9 ns` Rust vs. `expectation`'s `1.13 ms/step` 1-step streaming — `219x` faster in Python streaming), no `v = 0` warmup crash on step 1, tighter EB-CS width at `N = 200` (`0.218` vs. `0.284`) and `N = 1,000` (`0.110` vs. `0.120`), plus `BettingCS`, `ASC-D`, `CITE`, `PanelAudit`, `ActivePPI`, `SafeFuse`, `AbortCascade`, `EvidenceSketch`, and standalone CLI/MCP.
- **When to use `expectation`:** If you only need a single final-horizon EB-CS bound over an already-materialized NumPy array (`22.8 ns/step` single-call final-only) or `expectation`'s rich e-process research API for classical two-sample/martingale hypothesis testing.

### `cloture` vs. `ppi-python` (`ppi_py`)
- **Where `cloture` wins:** Anytime-valid online Active-PPI++ that stays valid under continuous peeking (`99.0%` coverage across all `4,000` steps vs. `ppi_py` dropping to `71.0%` when peeked every `200` items), online affine debiasing, automatic adaptive audit probability scheduling (`suggest_prob`), and guaranteed safety under adversarially inverted judges (`99%–100%` coverage with `lambda_hat = 0.0000`), with `80x` faster cold import (`26.5 ms` vs. `2,131 ms`).
- **When to use `ppi-python`:** If you have a **fixed, static dataset**, commit in advance to inspect the confidence interval **strictly once** at final `N`, or need regression/logistic/quantile PPI estimators (`ppi_ols_ci`, `ppi_logistic_ci`, `ppi_quantile_ci`). Because `ppi_py` uses asymptotic normal fixed-horizon intervals, it produces tighter intervals (`0.054` vs. `0.146`) when peeking is strictly prohibited.

### `cloture` vs. `online-cp`
- **Where `cloture` wins:** `93x–154x` faster in Python (`2.02 µs` vs. `189–313 µs/step`) and `2,930x–4,848x` faster in Rust (`64.5 ns/step`), with superior post-shift coverage recovery (`94.4%` vs. `68.0%` for `ConformalRidgeRegressor` after an abrupt shift) and integrated multi-stage Conformal Risk Control trajectory abort (`AbortCascade`).
- **When to use `online-cp`:** If you need `online-cp`'s full online conformal regression wrappers (e.g., `ConformalRidgeRegressor`, `SimpleJumper` exchangeability martingales) integrated with scikit-learn regression pipelines.

### `cloture` vs. `bayesbench`
- **Where `cloture` wins:** Exact frequentist anytime-valid FWER control (`0.0%` false positives on exact-tie nulls vs. `bayesbench.compare` `40.0%` at minimal skip threshold), `959x` faster pairwise certification (`0.205 ms` vs. `196.6 ms`), and `11,112x` faster 6-arm racing (`2.75 ms` and `306` evals vs. `30,514 ms` and `1,200` evals).
- **When to use `bayesbench`:** If you explicitly want Bayesian posterior rank probabilities (`P(arm_i is best)`) under Beta-Binomial priors rather than distribution-free frequentist error guarantees.

### `cloture` vs. Fixed-`N` / Naive Consensus Loops
- **Where `cloture` wins:** Cuts consensus rollouts by `39.5%–91.8%` on clear-to-moderate prompts while preserving `100%` empirical accuracy and `<= 5%` boundary-tie Type-I error (whereas naive `p < 0.05` peeking suffers `28.5%–30.5%` false certifications and `53.8%` CS miscoverage).
- **When Fixed-`N` is fine:** If `N <= 3` or latency/token cost is negligible and you need a fixed deterministic wall-clock latency envelope.
