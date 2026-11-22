# `cloture` Competitive Benchmark Report (`v0.2.0`)

Every number in this document comes from a single deterministic execution of [`benchmarks/run_benchmarks.py`](../benchmarks/run_benchmarks.py) against the hardened `v0.2.0` release build (`RUSTFLAGS="-D warnings" cargo build --workspace --release`), saved verbatim in [`benchmarks/results.json`](../benchmarks/results.json). For the companion real-world study on `2,985` AllenAI RewardBench human-verified items across `11` frontier models, see [`docs/REAL_WORLD_EVAL.md`](REAL_WORLD_EVAL.md) and [`benchmarks/rewardbench_receipts.json`](../benchmarks/rewardbench_receipts.json).

## Hardware & Software Environment

| Field | Value |
|---|---|
| Timestamp (UTC) | `2026-11-22T19:52:02Z` |
| CPU | `Intel(R) Xeon(R) CPU @ 2.30GHz` (`2` vCPUs, `x86_64`) |
| OS | `Linux 6.18.57` (`glibc 2.39`) |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Python | `3.12.16` |
| `cloture` | `0.2.0` (`0` external Rust crates, `0` Python runtime deps, `243 KB` zero-import WASM) |
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

- `AscdStopper(K=4)` Rust CLI: **`683.9 ns/update`** | Python batch: **`667.8 ns/update`** | Faithful NumPy reference: **`57,074 ns/update`** (**`85.5x` speedup**, **`25/25`** exact stopping-time parity).
- `CiteStopper` Rust CLI: **`208.8 ns/update`** | Python batch: **`209.6 ns/update`** | Python streaming: `5,979 ns/update`.

---

## Suite 2: Correlated-Panel Audit (`K = 9`, `N = 1,000`, 2k perms / 2k Condorcet MC / 1k boot)

- **Wall-Clock:** `cloture.audit_panel` median **`26.0 ms`** (p95 `49.5 ms`) vs. vectorized NumPy/SciPy reference **`1,446.7 ms`** (**`55.7x` speedup**; `14.7 ms` on binary panels via exact $O(K^2)$ Poisson-Binomial DP; `67.2 ms` on real 7-LLM × `2,985`-item RewardBench).
- **Exact Numerical Parity:** maximum absolute difference across all deterministic statistics (`panel_accuracy`, `mean_phi`, `n_eff_kish`, `n_eff_eigen`, `effective_rank`, `krippendorff_alpha`, `phi_matrix`, `loo_delta_acc`, `loo_delta_neff`) = **`2.66e-15`** (`krippendorff_alpha` diff = `0.0`).
- **Diagnostic Findings:**
  - `mean_phi = 0.385`, `n_eff_kish = 2.21` (`95% CI [2.09, 2.33]`), `n_eff_eigen = 2.13`, `effective_rank = 5.71`, `krippendorff_alpha = 0.510`.
  - Condorcet independence prediction: `94.92%` vs. actual panel accuracy `74.48%` (**`20.43 pp` Condorcet gap**, stratified permutation `p = 0.0005`).
  - **Leave-One-Out Pruning:** dropping 1 redundant judge (index `6`) increases accuracy by **`+1.52 pp`** while cutting judge API calls by **`11.1%`**; dropping 2 redundant judges (`6`, `8`) increases accuracy by **`+2.72 pp`** (`74.48% -> 77.20%`) while cutting judge calls by **`22.2%`**.
  - **Real-World AllenAI RewardBench (`2,985` items, `11` frontier models):** greedy LOO pruning (`11 -> 3`) raises accuracy **`+3.85 pp` (`92.03% -> 95.88%`)** — beating the best single model (`Skywork-Gemma-27B-RM`, `94.97%`) by **`+0.90 pp`** — at **`-72.7%`** lower API cost.

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
| **`cloture.EmpiricalBernsteinCS`** | **`1.0%`** | `0.218` | `0.110` | **`16.0`** (`29.7` Rust 1-step) | **`3,682`** |
| **`cloture.BettingCS(G=201)`** | **`3.6%`** | **`0.190`** | **`0.095`** | **`330.2`** (`281.4` Rust CLI) | **`5,122`** |
| `confseq.predmix_empbern_cs` | `1.0%` | `0.222` | `0.112` | `410.2` | — |
| `confseq.hedged_cs(201)` *(30 trials)* | `0.0%` | `0.214` | `0.119` | `137,003` *(at 1k)* | — |
| `confseq.conjmix_empbern_cs` *(30 trials)* | `0.0%` | `0.239` | `0.096` | `111,008` *(at 1k)* | — |
| `confseq.boundaries.normal_mixture_bound` (C++) | — | — | — | `14,655` | — |
| `expectation.EmpiricalBernsteinCS` (Rust) | `0.0%` *(chunk-50)* | `0.284` | `0.120` | `17.6` *(final bound only)* | `1,093,127` *(1-step)* / `9,962` *(ParSeqTest)* |
| Naive CLT / Wald Peeking | **`53.8%`** | `0.127` | `0.057` | `75.8` *(at 1k)* | — |

### 4B. Multi-Arm Racing (`6` arms, `200` tasks, `200` trials) & `bayesbench`

- Signal (`[0.88, 0.72, 0.62, 0.54, 0.45, 0.35]`, shared task correlation `0.75`):
  - `cloture.AgentRace(paired=True)`: **`299.9` evals (`75.0%` saved vs. `1,200`)**, `97.5%` certified, **`100.0%` accuracy** (and **`80.4%` saved** across `50/50` shuffled 6-model RewardBench streams).
  - `cloture.AgentRace(paired=False)` (LUCB): `709.1` evals (`40.9%` saved), `100.0%` accuracy.
- Global-tie Null (`[0.50] * 6`): `cloture` paired FWER = **`0.0%`**, unpaired FWER = **`0.0%`**.
- `bayesbench==0.4.0.dev0` comparison:
  - Pairwise tie null (`0.50 vs 0.50`, 30 trials): `bayesbench.compare` Type-I = **`40.0%`** (`194.8 ms/trial`) vs. `cloture.PairedEProcess` **`0.0%`** (`0.334 ms/trial`, **`583x` faster**).
  - 6-arm 200-task single trial: `bayesbench.BayesianRanker` **`29,274 ms`** (`1,200` evals, did not converge) vs. `cloture.AgentRace(paired=True)` **`2.60 ms`** (`306` evals, **`11,276x` faster**).

---

## Suite 5: Active-PPI++ vs. `ppi-python`, Human-Only CS, & Naive Judges

| Regime | Estimator | Coverage | Mean Final Width | Median Time / Trial |
|---|---|---|---|---|
| **A. Accurate Biased Judge** (`r=0.92`, `+0.15` bias, `N=4,000`, `10%` audit, 100 trials) | **`cloture.ActivePPI`** | **`99.0%` anytime (all 4,000 steps)** | **`0.146`** (`6.29x` var red, `1.46x` eff mult) | **`0.94 ms` batch** / `40.5 ms` stream |
| | `ppi_py.ppi_mean_ci` (peeking every 200) | **`71.0%`** *(invalid under peeking)* | `0.054` | `17.2 ms` |
| | `ppi_py.ppi_mean_ci` (single fixed `N=4,000`) | `96.0%` *(fixed-horizon only)* | **`0.054`** | `0.89 ms` |
| | Unbiased Human-Only EB-CS (`10%`) | `99.0%` anytime | `0.177` | `1.50 ms` |
| | Naive Biased LLM Judge | **`0.0%`** (`+0.150` bias) | `0.021` | — |
| **B1. Exact Inverted Judge** (`f = 1 - y`, `N=2,000`, 100 trials) | **`cloture.ActivePPI`** | **`99.0%` anytime** (`lambda_hat = 0.0000`) | `0.267` | — |
| | Naive Inverted LLM Judge | **`0.0%`** | — | — |
| **B2. Noisy Inverted Judge** (`N=2,000`, 100 trials) | **`cloture.ActivePPI`** | **`100.0%` anytime** (`lambda_hat = 0.0000`) | `0.277` | — |
| | Naive Inverted LLM Judge | **`0.0%`** | — | — |

---

## Suite 6: SafeFuse, AbortCascade, OnlineConformal vs. `online-cp`, & 712-Byte EvidenceSketch

- **`SafeFuse` (`5` agents, `2` poisoned at `t = 200`, `50` trials):** `100.0%` dual-poison quarantine (median `94` steps post-poison, `0.0%` clean false quarantines, `0.0%` clean-null FWER), cumulative loss `171.3` vs. uniform mean `371.3` (`-200.0` regret vs. uniform; `-2.74` vs. oracle clean mean `174.0`).
- **`AbortCascade` (`500` cal / `500` test, `50` trials, target recall `0.92`):** `92.6%` mean test recall (`93.3%` median), `38.2%` failed-episode compute saved (`27.8%` total compute saved), `2.14 ms` median calibration; naive threshold `0.35` collapses to `66.1%` recall.
- **`OnlineConformal` vs. `online-cp` (abrupt shift at `t = 1,000`, target `90%`):**
  - `cloture.OnlineConformal`: **`59.7 ns/step` Rust** / **`1,974 ns/step` Python**; stationary `89.5%`, post-shift recovered **`94.4%`** (overall `91.8%`).
  - Faithful Python AgACI ref: `14,795 ns/step` (`7.5x` slower than `cloture` Python, `248x` slower than Rust).
  - `online-cp` `SimpleJumper`: `196,795 ns/step` (**`99.7x`** slower than `cloture` Python / **`3,294x`** slower than Rust; alarm delay `25` steps).
  - `online-cp` `ConformalRidgeRegressor`: `269,383 ns/step` (**`136x`** slower than `cloture` Python); post-shift coverage drops to **`68.0%`** (fixed unadapted: **`33.5%`**).
- **`EvidenceSketch`:** exact **`712` bytes** wire payload, **`70.5 ns`** Rust merge (`847 ns` Python ctypes); `1,000`-worker fleet merge (`100,000` obs) vs. single-node diff = **`0.0`** mean, **`0.0`** p50/p90 quantiles, `6.11e-15` rel e-value diff; exact Leave-One-Out subtraction diff = **`0.0`** across all metrics.

---

## Suite 7: Systems Footprint & Cold-Start

| Metric | `cloture v0.2.0` | Reference Context |
|---|---|---|
| `import cloture` cold-start (median, 10 runs) | **`24.8 ms`** (p95 `32.8 ms`) | `numpy` `201.0 ms` (`8.1x`), `scipy.stats` `848.5 ms` (`34.2x`), `ppi_py` `2,074.8 ms` (`83.7x`) |
| `cloture --version` CLI cold-start (median, 10 runs) | **`3.15 ms`** (p95 `3.27 ms`) | Zero-dependency static binary (`2.91 MiB`) |
| `cloture mcp` JSON-RPC 2.0 `tools/call` roundtrip | **`67.6 µs`** median (`140.8 µs` p95, `255.9 µs` p99) | In-process stdio server |
| Shared library `libcloture_core.so` | **`4.78 MiB`** | `0` external Rust crate dependencies |
| Browser/Edge WebAssembly `cloture_core.wasm` | **`243 KB`** raw (`81.4 KB` gzipped) | `0` host imports (`IMPORTS: []`) |
