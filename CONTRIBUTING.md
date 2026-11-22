# Contributing to `cloture`

Thank you for your interest in contributing! `cloture` enforces strict mathematical and engineering invariants so downstream multi-agent systems can rely on its finite-sample guarantees.

## Core Invariants (Non-Negotiable)

1. **Zero external dependencies:**
   - `crates/cloture-core` and `crates/cloture-cli` must depend on Rust `std` only (`Cargo.lock` contains only `cloture-core` and `cloture-cli`).
   - `python/cloture` must declare `dependencies = []` (NumPy is an optional zero-copy fast path, never required).
   - `js/cloture.mjs` must have zero npm runtime dependencies and zero WASM host imports.
2. **Exact anytime-valid statistical guarantees:**
   - Any PR modifying stopping rules, confidence sequences, e-processes, or conformal updates must include Monte Carlo Type-I error / coverage verification under exact-tie and skewed nulls in both Rust tests and `tests/breaker/`.
3. **Panic-safe C ABI & strict input validation:**
   - Every `extern "C"` function in `crates/cloture-core/src/ffi.rs` must check null pointers/bounds and wrap body execution in `std::panic::catch_unwind`.

## Running the Full Local Verification Suite

```bash
# 1. Rust core & CLI release tests (-D warnings)
RUSTFLAGS="-D warnings" cargo test --release --workspace

# 2. Python wheel build + 211 integration & adversarial breaker tests
python3 scripts/build_wheel.py --out-dir dist
PYTHONPATH=python pytest -v tests/test_python_sdk.py tests/breaker/

# 3. Zero-import WebAssembly SDK test (optional, requires wasm32-unknown-unknown + node)
RUSTFLAGS="-C strip=symbols -C opt-level=s" cargo build --release -p cloture-core --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/cloture_core.wasm site/wasm/cloture_core.wasm
node --test js/test.mjs
```

## Real-World Evaluation Traces Wanted

If you run LLM-as-a-judge panels or Best-of-N self-consistency loops in production and can share anonymized vote matrices (`votes`, `gold`) or logprob distributions, open an issue — real-world correlated-error traces help sharpen our benchmark suite!
