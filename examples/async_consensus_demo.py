"""Demo: Cut async multi-agent / Best-of-N latency & tokens with anytime-valid stopping.

Run:
    PYTHONPATH=python python3 examples/async_consensus_demo.py
"""
import asyncio
import random
import time
import cloture


async def simulated_agent(prompt_id: int, rollout_idx: int, rng: random.Random) -> dict:
    """Simulates a stochastic LLM agent taking 40-80ms and returning an answer + distribution."""
    await asyncio.sleep(0.04 + 0.04 * rng.random())
    # True modal answer is option 0 ("patch_A") with probability 0.65 vs 0.20, 0.10, 0.05
    base = [0.65, 0.20, 0.10, 0.05]
    # Perturb slightly per trajectory (grey-box distribution feedback)
    z = rng.choices([0, 1, 2, 3], weights=base)[0]
    probs = [0.5 * base[k] + (0.5 if k == z else 0.0) for k in range(4)]
    answer = ["patch_A", "patch_B", "patch_C", "patch_D"][z]
    return {"answer": answer, "probs": probs}


async def main() -> None:
    rng = random.Random(42)

    # 1) Fixed-N baseline (runs all 32 rollouts at matched concurrency=8)
    sem = asyncio.Semaphore(8)

    async def _bounded_agent(i: int) -> dict:
        async with sem:
            return await simulated_agent(0, i, random.Random(1000 + i))

    t0 = time.perf_counter()
    fixed_tasks = [_bounded_agent(i) for i in range(32)]
    fixed_out = await asyncio.gather(*fixed_tasks)
    t_fixed = (time.perf_counter() - t0) * 1000.0

    # 2) Cloture Black-Box Category-Agnostic CITE (cancels remaining tasks immediately!)
    t0 = time.perf_counter()
    cite_res = await cloture.consensus(
        spawn_fn=lambda i: simulated_agent(0, i, random.Random(1000 + i)),
        extract_answer=lambda r: r["answer"],
        delta=0.05,
        max_calls=32,
        concurrency=8,
    )
    t_cite = (time.perf_counter() - t0) * 1000.0

    # 3) Cloture Grey-Box ASC-D (uses full softmax probability vector!)
    t0 = time.perf_counter()
    ascd_res = await cloture.consensus_dist(
        spawn_fn=lambda i: simulated_agent(0, i, random.Random(1000 + i)),
        extract_probs=lambda r: r["probs"],
        labels=["patch_A", "patch_B", "patch_C", "patch_D"],
        delta=0.05,
        max_calls=32,
        concurrency=8,
    )
    t_ascd = (time.perf_counter() - t0) * 1000.0

    print("=== Async Multi-Agent Consensus Comparison (delta=0.05, max_calls=32, concurrency=8) ===")
    print(f"Fixed-32 baseline : calls=32/32 | wall={t_fixed:6.1f} ms")
    print(
        f"Cloture CITE (BB) : calls={cite_res.n_samples:2d}/32 "
        f"({100*(1 - cite_res.n_samples/32):4.1f}% saved) | winner={cite_res.winner} "
        f"| certified={cite_res.certified} | wall={t_cite:6.1f} ms"
    )
    print(
        f"Cloture ASC-D (GB): calls={ascd_res.n_samples:2d}/32 "
        f"({100*(1 - ascd_res.n_samples/32):4.1f}% saved) | winner={ascd_res.winner} "
        f"| certified={ascd_res.certified} | wall={t_ascd:6.1f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
