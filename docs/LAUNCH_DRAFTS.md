# Organic Launch Drafts (`cloture v0.2.0`)

Authentic, empirical-first drafts designed for technical developer communities. Every number links directly to `benchmarks/results.json` and `benchmarks/rewardbench_receipts.json` and is reproducible locally with `cloture bench`, `benchmarks/run_benchmarks.py`, or in-browser via the `243 KB` WASM playground.

---

## 1. Show HN

**Title:**
`Show HN: Cloture – Zero-dep Rust/WASM engine for anytime-valid multi-agent stopping & judge auditing`

**URL:**
`https://github.com/pgg898200-ship-it/cloture`

**First Comment:**
```
Hi HN,

I built `cloture` after watching multi-agent Best-of-N loops and LLM-as-a-judge panels waste 60%–90% of API spend while quietly violating their own error rates.

Two concrete empirical results drove v0.2.0:

1. We audited two independent real-world benchmarks with zero synthetic data and strict 5-fold out-of-sample CV:
   - AllenAI RewardBench (2,985 items, 11 frontier models): 7 generative LLMs collapse to n_eff = 2.21 effective votes (within-family phi 0.504 vs. cross-family 0.317, p = 0.00050). Greedy LOO pruning 11 -> 3 lifted accuracy +3.85 pp (92.03% -> 95.88%; +3.65 pp 5-fold out-of-sample CV, beating the best single model) at -72.7% API cost.
   - ScalerLab JudgeBench (350 objective MMLU-Pro/LiveBench/LiveCodeBench items, 13 frontier models across OpenAI o1/4o, Anthropic Claude-3.5, Google Gemini-1.5, Meta Llama-3.1, and Skywork): even the Top-7 models from 5 different labs collapse to n_eff = 2.49/7 (+11.29 pp Condorcet gap; +15.36 pp gap on 11 LLMs single-pass). Greedy LOO pruning 13 -> 3 (o1-preview + Claude-3.5-Sonnet + Skywork-RM-Gemma-27B) lifted accuracy +19.29 pp (52.14% -> 71.43%; +18.33 pp 5-fold out-of-sample CV, beating single o1-preview) at -76.9% API cost.
   You can load BOTH real datasets in 1 click (or paste your own CSV privately) inside our 243 KB zero-import browser WASM playground: https://pgg898200-ship-it.github.io/cloture/playground.html — or reproduce everything in 15 seconds in Google Colab: https://colab.research.google.com/github/pgg898200-ship-it/cloture/blob/main/examples/cloture_frontier_eval_colab.ipynb

2. Fixed-N self-consistency wastes most calls on easy prompts, while naive early stopping ("stop when 5 agree" or peeking at p < 0.05) blows up tie-null false certifications to 28.5%–53.8%. Using finite-sample anytime-valid e-processes (grey-box ASC-D over logprobs at 668 ns/update, or category-agnostic CITE at 210 ns/update), `cloture` stops easy prompts in 21 calls (67% saved vs. Fixed-64, 92% vs. Fixed-256) with 0.0%–0.5% tie-null FPR.

Technical choices:
- Zero external Rust crates (`Cargo.lock` is 15 lines; `std` only), zero Python runtime dependencies (`pip install cloture`, `import cloture` takes 24.8 ms via universal `py3-none` ctypes wheels), and zero WASM host imports (`243 KB` raw / `81 KB` gzipped).
- Single 2.91 MiB static CLI binary (`cloture audit --csv judges.csv`, `stop`, `race`, `ppi`, `bench`) + stdio JSON-RPC 2.0 MCP server (`cloture mcp`, 67.6 µs median roundtrip).
- v0.2.0 speedups vs. installed PyPI libraries:
  - PanelAudit (9x1,000, 2k perms / 2k MC / 1k boot): 26.0 ms vs. NumPy/SciPy 1.45 s (55.7x faster, 2.66e-15 max diff)
  - EmpiricalBernsteinCS (100k batch): 16.0 ns/step (25.7x faster than `confseq`, 1.10x faster than `expectation` on single-call batch and 297x faster in Python 1-step streaming)
  - BettingCS (G=201): 281 ns Rust / 330 ns Py batch (415x faster & tighter than `confseq.hedged_cs` 137 µs)
  - AscdStopper (K=4): 668 ns/update (85.5x faster than NumPy, 25/25 exact stop parity)
  - AgentRace (6 arms, paired): 2.60 ms vs. `bayesbench.BayesianRanker` 29.3 s (11,276x faster; saves 80.4% of evals on 50/50 shuffled RewardBench runs)
  - OnlineConformal (4-rate AgACI): 59.7 ns Rust (3,294x faster than `online-cp` SimpleJumper; 94.4% post-shift recovery)
- 282/282 tests including 203 adversarial statistical, FFI, and fuzzing tests.

Honest caveat: if you have a static dataset and strictly commit never to peek sequentially, `ppi-python` produces tighter fixed-horizon asymptotic intervals (0.054 vs. 0.146). Raw JSON receipts are in `benchmarks/results.json` and `benchmarks/rewardbench_receipts.json`.
```

---

## 2. r/LocalLLaMA & r/MachineLearning

**Title:**
`We audited 11 frontier LLM judges on 2,985 AllenAI RewardBench items: pruning 11 -> 3 raised accuracy +3.85 pp at -72.7% cost (zero-dep Rust/Python/WASM tool)`

**Body:**
```
If you run Best-of-N self-consistency, multi-agent debates, LLM-as-a-judge panels, or continuous prompt/model evals, we just shipped `cloture v0.2.0` (MIT): a zero-dependency Rust core + Python SDK (`pip install cloture`) + 2.91 MiB static CLI/MCP server + 243 KB zero-import browser WASM playground:

- GitHub: https://github.com/pgg898200-ship-it/cloture
- Live Browser WASM Playground (1-click real RewardBench dataset or paste your own CSV locally): https://pgg898200-ship-it.github.io/cloture/playground.html

Highlights from real AllenAI RewardBench (2,985 items, 23 strata, 11 frontier models) and our 7-suite benchmark (`benchmarks/results.json`):

- **Correlated judge collapse (`cloture audit --csv judges.csv` / `cloture.audit_csv` / `audit_promptfoo`):** Across 7 generative frontier LLMs (GPT-4o x2, GPT-4o-mini, Gemini-1.5-Flash, Llama-3.1-405B/70B, Gemma-2-27B), Kish `n_eff = 2.21 / 7` (within-family phi 0.504 vs. cross-family 0.317; audited in 67.2 ms). Greedy LOO pruning from 11 judges down to 3 lifted accuracy **+3.85 pp (92.03% -> 95.88%)** — beating the best single model by +0.90 pp — at **-72.7% API cost**. Runs 55.7x faster than vectorized NumPy/SciPy (26.0 ms vs. 1.45 s for 9x1,000, 2.66e-15 max diff).
- **Consensus early stopping (`AscdStopper` / `CiteStopper`):** 67%–92% fewer calls on easy prompts with 0.0%–0.5% tie-null false certifications (vs. 28.5%–30.5% for naive peeking). `await cloture.consensus(...)` cancels remaining in-flight API requests the millisecond consensus is certified (`668 ns/update`, 85.5x faster than NumPy).
- **Sequential racing (`AgentRace`):** Certifies the true RewardBench winner on 50/50 shuffled runs while saving **80.4%** of evals (11,276x faster than `bayesbench.BayesianRanker`).
- **Active-PPI++ (`ActivePPI`):** On real RewardBench (`GPT-4o-mini` proxy -> `GPT-4o`, 10% audit), delivers **100% anytime coverage** across all 2,985 steps with **10.46x variance reduction**, eliminating -12.41 pp proxy bias (and auto-collapses `lambda -> 0` on adversarially inverted judges), while `ppi-python` drops to 71.0% under sequential peeking.
- **Zero deps everywhere:** 0 external Rust crates (15-line `Cargo.lock`), 0 Python runtime deps (`24.8 ms` cold import vs. `2.07 s` for `ppi_py`).

Feedback and real-world judge CSVs very welcome!
```

---

## 3. r/rust

**Title:**
`cloture v0.2.0: Zero-crate Rust + 243 KB zero-import WASM engine (16 ns EB-CS, 281 ns BettingCS, 26 ms parallel panel audit)`

**Body:**
```
I just released `cloture v0.2.0` (MIT): a strictly `std`-only Rust workspace (`cloture-core` + `cloture-cli`) for sequential statistical certification in live multi-agent AI stacks.

Repo: https://github.com/pgg898200-ship-it/cloture
Live 243 KB zero-import WASM playground: https://pgg898200-ship-it.github.io/cloture/playground.html

Rust systems engineering highlights from v0.2.0:
- **Zero external crates (`15`-line `Cargo.lock`):** Built-in RFC-8259 JSON parser, RFC-4180 CSV parser (`cloture audit --csv`), Jacobi symmetric eigensolver, deterministic SplitMix64/Xoshiro256++ PRNG, zero-import `wasm32-unknown-unknown` target (`243 KB` raw / `81 KB` gzipped, `IMPORTS: []`), and stdio JSON-RPC 2.0 MCP server (`67.6 µs` roundtrip).
- **Zero-libm IEEE-754 transcendental kernels (`math.rs`):** Tiered SSE2 Chebyshev `fast_ln` / `fast_ln_1p` / `add_bet_grid_pair` + exact active-cell pruning in `BettingCS(G=201)` (bit-for-bit identical to unpruned grid):
  - `EmpiricalBernsteinCS`: **16.0 ns/step** batch-100k (25.7x faster than `confseq` C++, 1.10x faster than `expectation` Rust on single-call batch, 297x faster in Py streaming)
  - `BettingCS(G=201)`: **281 ns** Rust / **330 ns** Py batch (**415x** faster than `confseq.hedged_cs`)
  - `AscdStopper(K=4)`: **668 ns** batch (**85.5x** faster than NumPy, 25/25 exact stop parity)
- **Zero-dep parallel `PanelAudit` (`std::thread::scope` + exact O(K²) Poisson-Binomial DP):** Cut `9 x 1,000` (2k perms / 2k MC / 1k boot) from 318 ms -> **26.0 ms** (**55.7x** faster than NumPy/SciPy, `2.66e-15` max diff; 14.7 ms on binary panels).
- **Hardened C ABI (`panic = "unwind"` + `catch_unwind`)** powering 5 universal `py3-none` wheels covering Python 3.9–3.14+ with zero CPython ABI coupling.
- **282/282 tests** (68 Rust + 213 Python/CLI/MCP/adversarial breaker + Node.js WASM). Run `cloture bench` to verify on your CPU!
```

---

## 4. X / Twitter Thread (5 Posts)

**1/5:**
```
We audited 11 frontier LLM judges on 2,985 real AllenAI RewardBench items.

7 generative LLMs collapsed to 2.21 effective independent votes.

Pruning 11 judges down to 3 *raised* accuracy +3.85 pp (92.0% -> 95.9%, beating the best single model) while cutting API cost 72.7%.

Meet `cloture v0.2.0` (MIT):
https://github.com/pgg898200-ship-it/cloture
```

**2/5:**
```
Try it on that exact 2,985-item RewardBench panel — or paste your own judge CSV privately — inside a 243 KB zero-import Rust WebAssembly playground (zero backend, zero upload):

https://pgg898200-ship-it.github.io/cloture/playground.html

Or locally in 1 line:
  cloture audit --csv judges.csv --gold-col gold
```

**3/5:**
```
And for Best-of-N / multi-agent debates (`AscdStopper` / `CiteStopper`):
- Easy prompts stop in 21 calls instead of 64 (67% saved) or 256 (92% saved), 100% accuracy
- Exact-tie null: 0.0%–0.5% false cert vs. 28.5%–30.5% naive peeking
- `await cloture.consensus(...)` cancels in-flight API calls the millisecond math certifies a winner
```

**4/5:**
```
v0.2.0 zero-dependency Rust engine speedups vs. installed PyPI packages:
- PanelAudit (9x1,000): 26.0 ms (55.7x vs NumPy/SciPy)
- BettingCS (G=201): 330 ns (415x vs confseq)
- EB-CS (100k batch): 16.0 ns (25.7x vs confseq, 297x Py stream vs expectation)
- AgentRace (6-arm): 2.60 ms (11,276x vs bayesbench)
- OnlineConformal: 59.7 ns (3,294x vs online-cp)
```

**5/5:**
```
0 external Rust crates (15-line Cargo.lock), 0 Python runtime deps (24.8 ms cold import), 2.91 MiB static CLI + 67.6 µs MCP server, 282/282 tests, raw JSON receipts checked into the repo.

Try in 10 seconds:
  pip install cloture
  claude mcp add cloture -- cloture mcp

MIT licensed.
```

---

## 5. Official MCP Registry & `awesome-mcp-servers` Listing

- **Registry manifest:** `server.json` (`io.github.pgg898200-ship-it/cloture`)
- **One-line summary:**
  `cloture` — Zero-dependency Rust MCP server (`2.91 MiB`, `67.6 µs` roundtrip) giving AI agents 6 anytime-valid statistical tools: multi-agent consensus stopping (`ASC-D`/`CITE`), correlated judge-panel `n_eff` auditing, clustered rollout population certification, sequential agent racing, `Active-PPI++`, and poisoning-resistant `SafeFuse` aggregation.
