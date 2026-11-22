# Drop-In Ecosystem Integrations

`cloture` has **zero runtime dependencies** in both Rust and Python, so it drops directly into existing OpenAI, Anthropic, vLLM, LiteLLM, LangGraph, Inspect AI, Braintrust, Promptfoo, Claude Code, Cursor, and CI pipelines without pulling in NumPy, SciPy, or PyTorch.

---

## 1. OpenAI / vLLM / OpenRouter (`top_logprobs` -> Grey-Box `AscdStopper`)

Replace fixed `n=32` or `n=64` Best-of-N loops with `cloture.consensus_dist` + `cloture.extract_logprob_distribution`. Stops in **21 calls on easy prompts** (67%–92% token savings) with exact finite-sample Type-I error `<= delta`, cancelling remaining in-flight API requests immediately.

```python
import asyncio
from openai import AsyncOpenAI
import cloture

client = AsyncOpenAI()  # works identically with vLLM / OpenRouter base_url
LABELS = ["A", "B", "C", "D"]


async def sample_mcq_logprobs(prompt: str, call_idx: int) -> dict[str, float]:
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",  # or any vLLM model returning logprobs
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0.7,
        logprobs=True,
        top_logprobs=10,
        seed=call_idx,
    )
    top = resp.choices[0].logprobs.content[0].top_logprobs
    return {entry.token.strip(): entry.logprob for entry in top}


async def solve(prompt: str) -> str:
    res = await cloture.consensus_dist(
        spawn_fn=lambda i: sample_mcq_logprobs(prompt, i),
        extract_probs=lambda lp: cloture.extract_logprob_distribution(lp, LABELS),
        labels=LABELS,
        delta=0.05,
        max_calls=64,
        concurrency=8,
    )
    answer = res.winner if res.winner is not None else res.leader
    print(f"Answer={answer} certified={res.certified} calls={res.n_samples} cancelled={res.n_cancelled}")
    return answer
```

---

## 2. Arbitrary Code / SQL / JSON / Tool Outputs (`CiteStopper` / `consensus`)

When answer spaces are unbounded (canonicalized ASTs, SQL results, extracted final math answers, JSON schemas), use category-agnostic `cloture.consensus`:

```python
import cloture


def canonicalize(text: str) -> str:
    return text.strip().splitlines()[-1].strip()


async def solve_open_ended(prompt: str, call_agent_async):
    res = await cloture.consensus(
        spawn_fn=lambda i: call_agent_async(prompt, seed=i),
        extract_answer=canonicalize,
        delta=0.05,
        max_calls=32,
        concurrency=8,
    )
    return (res.winner if res.winner is not None else res.leader), res
```

---

## 3. Audit an Existing LLM-as-a-Judge Pipeline (CSV / JSONL / Promptfoo / Inspect AI / Braintrust)

Zero-dependency 1-line audit directly from CSV, JSONL, or Promptfoo JSON exports (no pandas or NumPy required):

```python
import cloture

# 1. Direct CSV export (supports arbitrary strings like "pass"/"fail" or integers)
res = cloture.audit_csv("judges.csv", gold_col="gold", strata_col="domain")
print(f"{res.report.k_judges} judges -> {res.report.n_eff_kish:.2f} effective votes | prune: {res.prunable_judges()}")

# 2. Direct Promptfoo eval output (`promptfoo eval -o results.json`)
pf = cloture.audit_promptfoo("results.json")
print(f"Promptfoo providers: {pf.judge_names} -> n_eff={pf.report.n_eff_kish:.2f}")

# 3. Direct integer vote matrices (N x K row-major)
report = cloture.audit_panel(votes=votes, gold=gold, n_perms=2000, n_condorcet_sims=2000, n_boot=1000)
print(f"Condorcet gap: {report.condorcet_gap * 100:.2f} pp (omnibus permutation p={report.permutation_p_value:.5f})")
```

Or zero-code from the static CLI directly on CSV or JSON (including stdin `--csv -`):

```bash
cloture audit --csv judges.csv --gold-col gold --strata-col domain --perms 2000
cloture audit --file panel.json --perms 2000 --json
```

---

## 4. Claude Code, Cursor, Codex & Windsurf (MCP Server)

`cloture mcp` runs a zero-dependency JSON-RPC 2.0 MCP server over stdio (`67.6 µs` median tool latency, `2.91 MiB` static binary).

### Claude Code

```bash
claude mcp add cloture -- cloture mcp
```

### Cursor / Windsurf / Generic `mcp.json`

```json
{
  "mcpServers": {
    "cloture": {
      "command": "cloture",
      "args": ["mcp"]
    }
  }
}
```

### Exposed MCP Tools
1. `cloture_mode_stop` — grey-box `ASC-D` or unbounded `CITE` stopping over streamed samples
2. `cloture_audit_panel` — Kish & eigenvalue `n_eff`, Condorcet gap, permutation p-value, LOO pruning
3. `cloture_population_cert` — clustered unit-level population certification + `optimal_rollouts`
4. `cloture_agent_race` — paired or LUCB multi-arm racing + Wang-Ramdas `e_bh_discoveries`
5. `cloture_active_ppi` — anytime-valid centered-affine Active-PPI++ evaluation
6. `cloture_safe_fuse` — Scale-Free BOA + Fan-inequality Ville e-process poisoning quarantine

---

## 5. GitHub Actions Prompt / Model Regression Gate (`PopulationCert`)

Prevent correlated within-prompt rollouts from triggering 33%–40% false deployment gates:

```python
import cloture

# Each prompt yields R rollout difference scores in [-1.0, 1.0] (new_agent - baseline).
# Feed each prompt's rollouts as one unit into PopulationCert.
cert = cloture.PopulationCert(delta=0.05)
res = None
for prompt_rollout_diffs in per_prompt_diffs:
    res = cert.add_unit(prompt_rollout_diffs)

assert res is not None
print(
    f"Certified improvement={res.certified} theta_hat={res.theta_hat:+.3f} "
    f"rho_hat={res.rho_hat:.2f} ceiling={res.saturation_ceiling:.2f}"
)
# Allocate next eval budget optimally across prompts vs. rollouts-per-prompt:
rec_r = cert.optimal_rollouts(unit_setup_cost=1.0, per_rollout_cost=0.25)
print(f"Recommended rollouts per prompt for next eval sweep: {rec_r}")
```

---

## 6. C / C++ / Go / Node.js FFI (`include/cloture.h`)

`cloture-core` compiles to `libcloture_core.{so,dylib,dll}` and `libcloture_core.a` with a panic-safe C ABI (`panic = "unwind"` + `catch_unwind` on every exported symbol):

```c
#include "cloture.h"
#include <stdio.h>

int main(void) {
    ClotureCite *s = cloture_cite_new(0.05, 64);
    for (int i = 0; i < 12; ++i) {
        ClotureModeResult out = cloture_cite_update(s, 42ULL);
        if (out.certified && out.has_winner) {
            printf("Certified label=%llu in %llu calls\n",
                   (unsigned long long)out.winner,
                   (unsigned long long)out.n_samples);
            break;
        }
    }
    cloture_cite_free(s);
    return 0;
}
```

---

## 7. JavaScript / TypeScript / Node / Deno / Bun / Cloudflare Workers (Zero-Import WASM)

`cloture-core` compiles to a **`243 KB` (`81 KB` gzipped)** standalone WebAssembly module with **`0` host imports** (`js/cloture.mjs` + `js/cloture.d.ts`):

```javascript
import { loadCloture } from "./js/cloture.mjs";

const cloture = await loadCloture("https://pgg898200-ship-it.github.io/cloture/wasm/cloture_core.wasm");

// Cancel remaining in-flight fetch/LLM calls via AbortSignal the millisecond consensus certifies
const res = await cloture.consensus(
  async (callIdx, signal) => callOpenAI(prompt, { seed: callIdx, signal }),
  { delta: 0.05, maxCalls: 32, concurrency: 6 }
);
console.log(res.winner, `certified in ${res.nSamples} calls (cancelled ${res.nCancelled})`);
```
