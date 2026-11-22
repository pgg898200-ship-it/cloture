# `cloture` Competitive Benchmark Report

Every number in this document comes from a single deterministic execution of [`benchmarks/run_benchmarks.py`](../benchmarks/run_benchmarks.py) against the hardened release build (`RUSTFLAGS="-D warnings" cargo build --workspace --release`), saved verbatim in [`benchmarks/results.json`](../benchmarks/results.json).

## Hardware & Software Environment

| Field | Value |
|---|---|
| Timestamp (UTC) | `2026-11-22T16:56:46Z` |
| CPU | `Intel(R) Xeon(R) CPU @ 2.30GHz` (`2` vCPUs, `x86_64`) |
| OS | `Linux 6.18.57` (`glibc 2.39`) |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Python | `3.12.16` |
| `cloture` | `0.1.0` (`0` external Rust crates, `0` Python runtime deps) |
| PyPI Competitors | `confseq==0.0.11`, `ppi-python==0.2.3`, `expectation==0.6.1`, `online-cp==0.4.0`, `bayesbench==0.4.0.dev0`, `numpy==1.26.4`, `scipy==1.15.3` |

### Reproduction

```bash
cargo build --release --workspace
PYTHONPATH=python .venv/bin/python benchmarks/run_benchmarks.py
```

---

## Suite 1: Consensus Stopping (ASC-D Table 2 + Open-Ended Zipf Swarm)

Protocol: `delta = 0.05`, `200` paired trials per regime, `max_n = 256`.

### 1A. Fixed-`K` Categorical (`K = 4`, grey-box distributions vs. black-box labels)

| Regime | Gap $\Delta$ | `AscdStopper` Mean `n` (Acc) | `AscdStopper` (`max_budget=64`) | `CiteStopper` Mean `n` (Acc) | Fixed-32 / Fixed-64 / Fixed-256 Acc | Naive Peeking (`p < 0.05`) |
|---|---|---|---|---|---|---|
| Easy | `0.45` | **`21.0`** (`100.0%`, **`67.2%` saved vs 64**, **`91.8%` vs 256**) | `21.0` (`100.0%`) | `58.7` (`100.0%`) | `100.0%` / `100.0%` / `100.0%` | `21.0` (`100.0%`) |
| Medium | `0.25` | **`38.7`** (`100.0%`, **`39.5%` saved vs 64**, **`84.9%` vs 256**) | `37.6` (`100.0%`) | `141.0` (`100.0%`) | `98.5%` / `99.5%` / `100.0%` | `46.1` (`100.0%`) |
| Hard | `0.12` | **`127.5`** (`100.0%`, **`50.2%` saved vs 256**) | `55.4` (`99.0%`, **`+12.5 pp` over Fixed-64**) | `245.3` (`98.5%`) | `77.5%` / `86.5%` / `98.5%` | `118.1` (`96.0%`) |
| Exact-Tie Null | `0.00` | **`0.0%` Type-I error** | — | **`0.0%` Type-I error** | — | **`28.5%` Type-I error** |

### 1B. Open-Ended Swarm (`K_true = 200`, Zipf `s = 1.25`, unknown `K`)

| Regime | Gap | `CiteStopper` Mean `n` (Acc / Null FPR) | Fixed-32 / Fixed-64 / Fixed-256 Acc | Naive Peeking (`p < 0.05`) |
|---|---|---|---|---|
| Easy (`gap = 0.30`, `tail = 48.0%`) | `0.30` | **`70.9`** (`100.0%` acc, **`72.3%` saved vs 256**) | `100.0%` / `100.0%` / `100.0%` | `26.0` (`100.0%`) |
| Medium (`gap = 0.18`, `tail = 56.2%`) | `0.18` | **`168.6`** (`100.0%` acc, **`34.1%` saved vs 256**) | `93.5%` / `98.5%` / `100.0%` | `55.5` (`100.0%`) |
| Exact-Tie Null (`gap = 0.00`) | `0.00` | **`0.5%` Type-I error** (`<= 5%`) | — | **`30.5%` Type-I error** |

### 1C. Consensus Stopping Latency & Parity

- `AscdStopper(K=4)` Rust CLI: **`2,315 ns/update`** | Python batch: **`2,207 ns/update`** | Faithful NumPy reference: **`63,833 ns/update`** (**`28.9x` speedup**, **`25/25`** exact stopping-time parity).
- `CiteStopper` Rust CLI: **`270.8 ns/update`** | Python batch: **`247.0 ns/update`** | Python streaming: `7,572 ns/update`.

---

## Suite 2: Correlated-Panel Audit (`K = 9`, `N = 1,000`, 2k perms / 2k Condorcet MC / 1k boot)

- **Wall-Clock:** `cloture.audit_panel` median **`317.7 ms`** (p95 `353.0 ms`) vs. vectorized NumPy/SciPy reference **`1,539.7 ms`** (**`4.8x` speedup**).
- **Exact Numerical Parity:** maximum absolute difference across all deterministic statistics (`panel_accuracy`, `mean_phi`, `n_eff_kish`, `n_eff_eigen`, `effective_rank`, `krippendorff_alpha`, `phi_matrix`, `loo_delta_acc`, `loo_delta_neff`) = **`2.66e-15`** (`krippendorff_alpha` diff = `0.0`).
- **Diagnostic Findings:**
  - `mean_phi = 0.385`, `n_eff_kish = 2.21` (`95% CI [2.10, 2.34]`), `n_eff_eigen = 2.13`, `effective_rank = 5.71`, `krippendorff_alpha = 0.510`.
  - Condorcet independence prediction: `94.91%` vs. actual panel accuracy `74.48%` (**`20.43 pp` Condorcet gap**, stratified permutation `p = 0.0005`).
  - **Leave-One-Out Pruning:** dropping 1 redundant judge (index `6`) increases accuracy by **`+1.52 pp`** while cutting judge API calls by **`11.1%`**; dropping 2 redundant judges (`6`, `8`) increases accuracy by **`+2.72 pp`** (`74.48% -> 77.20%`) while cutting judge calls by **`22.2%`**.

---

## Suite 3: Realized vs. Population Correlated Rollouts (`rho = 0.60`, `delta = 0.05`)

| Scenario | Naive Pooled Z | Naive Pooled EB-CS | Single-Unit Prompt 0 (Any Prompt) | `cloture.PopulationCert` | Diagnostics |
|---|---|---|---|---|---|
| Null (`theta = 0.0`, `30 x 16`, 300 reps) | **`33.0%`** | **`40.0%`** | **`17.3%`** (**`99.3%`** any prompt) | **`0.7%`** *(valid `<= 5%`)* | `rho_hat = 0.600`, `n_eff_within = 1.62`, ceiling `1.69`, `optimal_rollouts = 1` (1:1) / `3` (10:1) |
| Signal (`theta = +0.25`, `30 x 16`, 480 rollouts) | `94.7%` | `89.3%` | `37.3%` (`100%`) | `27.3%` | Over-allocated rollouts/prompt |
| Signal (`theta = +0.25`, `120 x 4`, **same 480 rollouts**) | `100.0%` | `97.3%` | `0.0%` (`0.0%`) | **`88.0%`** (**`3.2x` higher power!**) | Guided by `optimal_rollouts` |
| Signal (`theta = +0.25`, `120 x 16`, 1,920 rollouts) | `100.0%` | `100.0%` | `28.0%` (`100%`) | **`90.7%`** | Saturates near `1 / rho` ceiling |

---

## Suite 4: Anytime-Valid CS & Multi-Arm Racing vs. `confseq`, `expectation`, `bayesbench`, Naive Peeking

### 4A. Confidence Sequences (`500` trials x `N = 1,000`, `p = 0.70`, `alpha = 0.05`; throughput at `N = 100,000`)

| Implementation | Anytime Miscov | Width `@200` | Width `@1000` | Batch `100k` (`ns/step`) | Streaming (`ns/step`) |
|---|---|---|---|---|---|
| **`cloture.EmpiricalBernsteinCS`** | **`1.0%`** | `0.218` | `0.110` | **`110.3`** (`88.9` Rust CLI) | **`5,164`** |
| **`cloture.BettingCS(G=201)`** | **`3.6%`** | **`0.190`** | **`0.095`** | **`6,320`** (`6,561` Rust CLI) | **`11,842`** |
| `confseq.predmix_empbern_cs` | `1.0%` | `0.222` | `0.112` | `441.6` | — |
| `confseq.hedged_cs(201)` *(30 trials)* | `0.0%` | `0.214` | `0.119` | `179,326` *(at 1k)* | — |
| `confseq.conjmix_empbern_cs` *(30 trials)* | `0.0%` | `0.239` | `0.096` | `124,705` *(at 1k)* | — |
| `confseq.boundaries.normal_mixture_bound` (C++) | — | — | — | `14,901` | — |
| `expectation.EmpiricalBernsteinCS` (Rust) | `0.0%` *(chunk-50)* | `0.284` | `0.120` | **`22.8`** *(final bound only)* | `1,132,769` *(1-step)* / `10,151` *(ParSeqTest)* |
| Naive CLT / Wald Peeking | **`53.8%`** | `0.127` | `0.057` | `83.6` *(at 1k)* | — |

### 4B. Multi-Arm Racing (`6` arms, `200` tasks, `200` trials) & `bayesbench`

- Signal (`[0.88, 0.72, 0.62, 0.54, 0.45, 0.35]`, shared task correlation `0.75`):
  - `cloture.AgentRace(paired=True)`: **`299.9` evals (`75.0%` saved vs. `1,200`)**, `97.5%` certified, **`100.0%` accuracy**.
  - `cloture.AgentRace(paired=False)` (LUCB): `709.0` evals (`40.9%` saved), `100.0%` accuracy.
- Global-tie Null (`[0.50] * 6`): `cloture` paired FWER = **`0.0%`**, unpaired FWER = **`0.0%`**.
- `bayesbench==0.4.0.dev0` comparison:
  - Pairwise tie null (`0.50 vs 0.50`, 30 trials): `bayesbench.compare` Type-I = **`40.0%`** (`196.6 ms/trial`) vs. `cloture.PairedEProcess` **`0.0%`** (`0.205 ms/trial`, **`959x` faster**).
  - 6-arm 200-task single trial: `bayesbench.BayesianRanker` **`30,514 ms`** (`1,200` evals, did not converge) vs. `cloture.AgentRace(paired=True)` **`2.75 ms`** (`306` evals, **`11,112x` faster**).

---

## Suite 5: Active-PPI++ vs. `ppi-python`, Human-Only CS, & Naive Judges

| Regime | Estimator | Coverage | Mean Final Width | Median Time / Trial |
|---|---|---|---|---|
| **A. Accurate Biased Judge** (`r=0.92`, `+0.15` bias, `N=4,000`, `10%` audit, 100 trials) | **`cloture.ActivePPI`** | **`99.0%` anytime (all 4,000 steps)** | **`0.146`** (`6.29x` var red, `1.46x` eff mult) | **`0.98 ms` batch** / `48.7 ms` stream |
| | `ppi_py.ppi_mean_ci` (peeking every 200) | **`71.0%`** *(invalid under peeking)* | `0.054` | `19.8 ms` |
| | `ppi_py.ppi_mean_ci` (single fixed `N=4,000`) | `96.0%` *(fixed-horizon only)* | **`0.054`** | `1.03 ms` |
| | Unbiased Human-Only EB-CS (`10%`) | `99.0%` anytime | `0.177` | `2.10 ms` |
| | Naive Biased LLM Judge | **`0.0%`** (`+0.150` bias) | `0.021` | — |
| **B1. Exact Inverted Judge** (`f = 1 - y`, `N=2,000`, 100 trials) | **`cloture.ActivePPI`** | **`99.0%` anytime** (`lambda_hat = 0.0000`) | `0.267` | — |
| | Naive Inverted LLM Judge | **`0.0%`** | — | — |
| **B2. Noisy Inverted Judge** (`N=2,000`, 100 trials) | **`cloture.ActivePPI`** | **`100.0%` anytime** (`lambda_hat = 0.0000`) | `0.277` | — |
| | Naive Inverted LLM Judge | **`0.0%`** | — | — |

---

## Suite 6: SafeFuse, AbortCascade, OnlineConformal vs. `online-cp`, & 712-Byte EvidenceSketch

- **`SafeFuse` (`5` agents, `2` poisoned at `t = 200`, `50` trials):** `100.0%` dual-poison quarantine (median `94` steps post-poison, `0.0%` clean false quarantines, `0.0%` clean-null FWER), cumulative loss `171.3` vs. uniform mean `371.3` (`-200.0` regret vs. uniform; `-2.74` vs. oracle clean mean `174.0`).
- **`AbortCascade` (`500` cal / `500` test, `50` trials, target recall `0.92`):** `92.6%` mean test recall (`93.3%` median), `38.2%` failed-episode compute saved (`27.8%` total compute saved), `2.79 ms` median calibration; naive threshold `0.35` collapses to `66.1%` recall.
- **`OnlineConformal` vs. `online-cp` (abrupt shift at `t = 1,000`, target `90%`):**
  - `cloture.OnlineConformal`: **`64.5 ns/step` Rust** / **`2,025 ns/step` Python**; stationary `89.5%`, post-shift recovered **`94.4%`** (overall `91.8%`).
  - Faithful Python AgACI ref: `14,496 ns/step` (`7.2x` slower than `cloture` Python, `225x` slower than Rust).
  - `online-cp` `SimpleJumper`: `188,942 ns/step` (**`93.3x`** slower than `cloture` Python / **`2,930x`** slower than Rust; alarm delay `25` steps).
  - `online-cp` `ConformalRidgeRegressor`: `312,623 ns/step` (**`154x`** slower than `cloture` Python); post-shift coverage drops to **`68.0%`** (fixed unadapted: **`33.5%`**).
- **`EvidenceSketch`:** exact **`712` bytes** wire payload, **`76.3 ns`** Rust merge (`852 ns` Python ctypes); `1,000`-worker fleet merge (`100,000` obs) vs. single-node diff = **`0.0`** mean, **`0.0`** p50/p90 quantiles, `6.32e-15` rel e-value diff; exact Leave-One-Out subtraction diff = **`0.0`** across all metrics.

---

## Suite 7: Systems Footprint & Cold-Start

| Metric | `cloture` | Reference Context |
|---|---|---|
| `import cloture` cold-start (median, 10 runs) | **`26.5 ms`** (p95 `32.3 ms`) | `numpy` `253.3 ms` (`9.6x`), `scipy.stats` `859.7 ms` (`32.4x`), `ppi_py` `2,131.4 ms` (`80.4x`) |
| `cloture --version` CLI cold-start (median, 10 runs) | **`3.05 ms`** (p95 `3.75 ms`) | Zero-dependency static binary (`2.47 MiB`) |
| `cloture mcp` JSON-RPC 2.0 `tools/call` roundtrip | **`76.7 µs`** median (`151.4 µs` p95, `316.6 µs` p99) | In-process stdio server |
| Shared library `libcloture_core.so` | **`4.60 MiB`** | `0` external Rust crate dependencies |
