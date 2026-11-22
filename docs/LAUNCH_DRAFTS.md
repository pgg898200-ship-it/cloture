# Organic Launch Drafts (`cloture v0.1.0`)

Authentic, empirical-first drafts designed for technical developer communities. Every number links directly to `benchmarks/results.json` and is reproducible locally with `cloture bench` or `benchmarks/run_benchmarks.py`.

---

## 1. Show HN

**Title:**
`Show HN: Cloture – Zero-dep Rust engine to stop multi-agent debates when math proves a winner`

**URL:**
`https://github.com/pgg898200-ship-it/cloture`

**First Comment:**
```
Hi HN,

I built `cloture` after watching multi-agent Best-of-N loops and LLM-as-a-judge panels waste 60%–90% of API spend while quietly violating their own error rates.

Two concrete findings drove the design:

1. Fixed-N self-consistency wastes most calls on easy prompts, while naive early stopping ("stop when 5 agree" or peeking at p < 0.05) blows up tie-null false certifications to 28.5%–53.8%. Using finite-sample anytime-valid e-processes (grey-box ASC-D over logprobs or category-agnostic CITE over arbitrary outputs), `cloture` stops easy prompts in 21 calls (67% saved vs. Fixed-64, 92% vs. Fixed-256) with 0.0%–0.5% tie-null FPR.

2. Correlated LLM-as-a-judge panels break Condorcet's jury theorem. On a realistic 9-judge correlated error panel, Kish effective sample size is 2.21 effective votes (20.4 pp below the independence prediction). Leave-one-out pruning showed dropping the 2 most redundant judges *increased* panel accuracy by +2.72 pp while cutting judge API cost 22.2%.

Technical choices:
- Zero external Rust crates (`Cargo.lock` is 15 lines; `std` only) and zero Python runtime dependencies (`import cloture` takes 26.5 ms via ctypes to `libcloture_core`).
- Single 2.47 MiB static binary (`cloture`) with stdin pipelines (`stop`, `audit`, `race`, `ppi`), a built-in CPU microbenchmark (`cloture bench`), and a stdio JSON-RPC 2.0 MCP server (`cloture mcp`, 76.7 µs median roundtrip).
- Also ships `PopulationCert` (preventing 33%–40% false fires from pooled correlated rollouts), `AgentRace` (paired e-process racing, 11,112x faster than `bayesbench.BayesianRanker`), online affine `ActivePPI` (99% anytime coverage even under biased or adversarially inverted judges), `SafeFuse` poisoning quarantine, `OnlineConformal` (64.5 ns Rust AgACI, 2,930x faster than `online-cp`), and a 712-byte associative/commutative wire monoid (`EvidenceSketch`).
- 266/266 tests including 203 adversarial statistical, FFI, and fuzzing tests (which caught 10 real bugs before v0.1.0, including a 96% skewed-null FPR bug in our first SafeFuse draft).

Honest caveats (documented in README & docs/comparisons.md):
- If you never peek sequentially and only compute one fixed-horizon PPI interval at the end of a static dataset, `ppi-python` yields tighter asymptotic intervals (0.054 vs. 0.146).
- If you only run one single-call offline Empirical-Bernstein check over a static 100k array, `expectation` is faster in that single batch call (22.8 ns vs. 110 ns), while `cloture` is 219x faster for 1-step streaming in Python.

Raw benchmark JSON receipts are checked into `benchmarks/results.json`. Happy to discuss the martingale proofs, FFI hardening, or correlated-rollout pitfalls!
```

---

## 2. r/LocalLLaMA & r/MachineLearning

**Title:**
`We measured correlated LLM-as-a-judge collapse (9 judges = 2.2 effective votes) and built cloture: a zero-dependency Rust/Python anytime-valid stopping & eval engine`

**Body:**
```
If you run Best-of-N self-consistency, multi-agent debates, LLM-as-a-judge panels, or continuous prompt/model evals, we built an MIT-licensed, zero-dependency Rust core + Python SDK (`pip install cloture`) + 2.47 MiB CLI/MCP server:

https://github.com/pgg898200-ship-it/cloture

Highlights from our reproducible 7-suite benchmark (`benchmarks/results.json`) against `confseq`, `ppi-python`, `expectation`, `online-cp`, `bayesbench`, and NumPy/SciPy:

- **Consensus early stopping (`AscdStopper` / `CiteStopper`):** 67%–92% fewer calls on easy prompts with 0.0%–0.5% tie-null false certifications (vs. 28.5%–30.5% for naive p < 0.05 peeking). Async `await cloture.consensus(...)` cancels remaining in-flight LLM requests the millisecond consensus is certified.
- **Judge panel audit (`audit_panel`):** On a 9-judge x 1,000-item panel, detects Kish `n_eff = 2.21` (20.4 pp Condorcet gap) in 317.7 ms (4.8x faster than vectorized NumPy/SciPy, 2.66e-15 max diff). Dropping the 2 most correlated judges raised accuracy +2.72 pp at -22.2% judge cost.
- **Correlated rollouts (`PopulationCert`):** Pooling 16 correlated rollouts/prompt across 30 prompts causes 33%–40% false-positive eval gates at alpha=0.05. `PopulationCert` holds 0.7% null FPR and computes optimal rollouts-per-prompt allocation (`30x16 -> 120x4` raises power from 27.3% to 88.0% at the same token budget).
- **Active-PPI++ (`ActivePPI`):** 99.0% anytime coverage with 10% human audits (6.29x variance reduction on +0.15 biased judges; auto-collapses `lambda -> 0` on adversarially inverted judges), while `ppi-python` drops to 71.0% coverage under sequential peeking.
- **Zero deps everywhere:** 0 external Rust crates (15-line `Cargo.lock`), 0 Python runtime dependencies (`26.5 ms` cold import vs. `2.1 s` for `ppi_py`).

We also clearly list where `ppi-python` (single fixed-horizon no-peeking) and `expectation` (single offline batch) are tighter/faster in `docs/comparisons.md`. Feedback and real-world judge-panel traces very welcome!
```

---

## 3. r/rust

**Title:**
`cloture v0.1.0: Zero-crate Rust engine (15-line Cargo.lock) for sub-microsecond anytime-valid multi-agent stopping, panel auditing, and 712B wire monoids`

**Body:**
```
I just released `cloture v0.1.0` (MIT): a zero-external-dependency Rust workspace (`cloture-core` + `cloture-cli`) for sequential statistical certification in live multi-agent AI stacks.

Repo: https://github.com/pgg898200-ship-it/cloture

Rust engineering details that might interest r/rust:
- **Strictly `std`-only:** `Cargo.lock` has 2 entries (`cloture-core`, `cloture-cli`). Built-in RFC 8259 JSON parser/serializer (depth-capped at 128, rejects non-finite numbers and fractional `u64` truncation), Jacobi symmetric eigensolver, deterministic SplitMix64/Xoshiro256++ PRNG, and stdio JSON-RPC 2.0 MCP server — all in ~11k LOC compiling in ~18s to a 2.47 MiB binary.
- **Hardened C ABI (`include/cloture.h`) + universal Python wheels:** Uses `panic = "unwind"` with `std::panic::catch_unwind` at every `extern "C"` boundary (learned the hard way when our adversarial breaker suite showed `panic = "abort"` nullifies `catch_unwind`!). Because Python loads `libcloture_core` via `ctypes`, 5 `py3-none-<platform>` wheels cover Python 3.9–3.14+ with zero CPython ABI coupling.
- **Zero-transcendental Fan-inequality supermartingale (`SafeFuse`):** Replaced `ln_1p` betting with unclamped `lam * x - psi_E(lam) * x^2`, fixing a skewed-null FPR bug from 96.3% -> 0.0–0.75% while speeding up `SafeFuse(K=5)::update` 1.6x to 1.24 µs.
- **712-byte associative/commutative wire monoid (`EvidenceSketch`):** Fixed binary layout (`b"CLTR"` + FNV-1a 64-bit checksum), 76.3 ns merge, 0.0 exact error across 1,000-worker fleet merges and Leave-One-Out subtraction.
- **Microbenchmarks:** `EmpiricalBernsteinCS` 88.9 ns/step, `OnlineConformal` (4-rate AgACI) 64.5 ns/step, `cloture mcp` 76.7 µs roundtrip. Run `cloture bench` to verify on your CPU.

Curious to hear feedback on the FFI surface, no-std feasibility for `cloture-core`, and SIMD vectorization opportunities!
```

---

## 4. X / Twitter Thread (5 Posts)

**1/5:**
```
Multi-agent Best-of-N loops waste 67%–92% of LLM calls on easy prompts — and naive "stop when 5 agree" peeking blows up false certifications to 28%–54%.

And 9 frontier LLM-as-a-judge models often deliver only 2.2 effective independent votes.

Meet `cloture`: a zero-dep Rust + Python engine that ends the debate the millisecond the math proves who won.

https://github.com/pgg898200-ship-it/cloture
```

**2/5:**
```
1) Anytime-valid consensus stopping (`AscdStopper` / `CiteStopper`):
- Easy prompts: 21 calls instead of 64 (67% saved) or 256 (92% saved), 100% accuracy
- Exact-tie null: 0.0%–0.5% false cert vs. 28.5%–30.5% naive peeking
- `await cloture.consensus(...)` cancels remaining in-flight API calls the instant the winner is certified
```

**3/5:**
```
2) Correlated LLM-as-a-judge panel audit (`cloture audit`):
On a 9-judge x 1,000-item panel:
- 9 judges -> 2.21 effective independent votes (20.4 pp Condorcet gap)
- Runs in 317.7 ms (4.8x faster than NumPy/SciPy, 2.66e-15 max diff)
- Dropping the 2 most correlated judges *raised* accuracy +2.72 pp while cutting judge API cost 22.2%
```

**4/5:**
```
3) Built for production systems, not notebooks:
- 0 external Rust crates (15-line Cargo.lock)
- 0 Python runtime deps (`import cloture` in 26.5 ms vs. 2.1s for `ppi_py`)
- Single 2.47 MiB static binary + native JSON-RPC 2.0 MCP server (`cloture mcp`, 76.7 µs)
- 11,112x faster paired agent racing than `bayesbench`, 2,930x faster online conformal than `online-cp`
```

**5/5:**
```
Every number is backed by raw machine-verifiable receipts (`benchmarks/results.json`), 266/266 tests (203 adversarial statistical + FFI breaker tests), and honest caveats on when fixed-horizon tools win.

Try it in 10 seconds:
  pip install cloture
  # or
  claude mcp add cloture -- cloture mcp

MIT licensed.
```

---

## 5. Official MCP Registry & `awesome-mcp-servers` Listing

- **Registry manifest:** `server.json` (`io.github.pgg898200-ship-it/cloture`)
- **One-line summary:**
  `cloture` — Zero-dependency Rust MCP server (`2.47 MiB`, `76.7 µs` roundtrip) giving AI agents 6 anytime-valid statistical tools: multi-agent consensus stopping (`ASC-D`/`CITE`), correlated judge-panel `n_eff` auditing, clustered rollout population certification, sequential agent racing, `Active-PPI++`, and poisoning-resistant `SafeFuse` aggregation.
