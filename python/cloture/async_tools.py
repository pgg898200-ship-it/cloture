"""Drop-in async multi-agent consensus stopping & adaptive racing wrappers.

Cancels remaining in-flight coroutines the exact millisecond the anytime-valid
e-process certifies the modal answer or eliminates suboptimal agents.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class AsyncConsensusResult:
    certified: bool
    winner: Any
    leader: Any
    n_samples: int
    n_cancelled: int
    log_evidence: float
    log_threshold: float
    margin_estimate: float
    deterministic_lock: bool
    scope: str
    raw_outputs: List[Any]


async def consensus(
    spawn_fn: Callable[[int], Awaitable[T]],
    *,
    extract_answer: Optional[Callable[[T], Any]] = None,
    extract_weight: Optional[Callable[[T], float]] = None,
    delta: float = 0.05,
    max_calls: int = 32,
    concurrency: int = 4,
    deterministic_lock: bool = True,
) -> AsyncConsensusResult:
    """Run stochastic agents/rollouts concurrently and halt immediately once certified.

    Uses category-agnostic `CiteStopper` (works over arbitrary hashable answers:
    code patches, SQL queries, strings, numbers, JSON dicts). Immediately cancels
    all remaining in-flight tasks once the anytime-valid e-process crosses threshold
    (or once `deterministic_lock` proves no challenger can catch the leader).
    """
    from ._native import CiteStopper

    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    stopper = CiteStopper(delta=delta, max_budget=(max_calls if deterministic_lock else None))
    answer_to_id: Dict[Any, int] = {}
    id_to_answer: Dict[int, Any] = {}
    next_id = 0

    in_flight: Dict[asyncio.Task[T], int] = {}
    launched = 0
    outputs: List[Any] = []
    cancelled = 0
    last_dec = None

    def _launch_up_to_limit() -> None:
        nonlocal launched
        while launched < max_calls and len(in_flight) < concurrency:
            idx = launched
            launched += 1
            task = asyncio.create_task(spawn_fn(idx))
            in_flight[task] = idx

    try:
        _launch_up_to_limit()
        while in_flight:
            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                in_flight.pop(task, None)
                out = task.result()
                outputs.append(out)
                ans = extract_answer(out) if extract_answer is not None else out
                key = ans if isinstance(ans, (str, int, float, bytes, tuple, bool)) else str(ans)
                if key not in answer_to_id:
                    answer_to_id[key] = next_id
                    id_to_answer[next_id] = ans
                    next_id += 1
                lid = answer_to_id[key]
                if extract_weight is not None:
                    w = float(extract_weight(out))
                    last_dec = stopper.update_weighted(lid, w)
                else:
                    last_dec = stopper.update(lid)
                if last_dec.certified:
                    break
            if last_dec is not None and last_dec.certified:
                break
            _launch_up_to_limit()
    finally:
        for task in list(in_flight.keys()):
            if not task.done():
                task.cancel()
                cancelled += 1
        if in_flight:
            await asyncio.gather(*in_flight.keys(), return_exceptions=True)

    assert last_dec is not None
    winner_val = id_to_answer.get(last_dec.winner) if last_dec.winner is not None else None
    leader_val = id_to_answer.get(last_dec.leader) if last_dec.leader is not None else None
    return AsyncConsensusResult(
        certified=last_dec.certified,
        winner=winner_val,
        leader=leader_val,
        n_samples=last_dec.n_samples,
        n_cancelled=cancelled,
        log_evidence=last_dec.log_evidence,
        log_threshold=last_dec.log_threshold,
        margin_estimate=last_dec.margin_estimate,
        deterministic_lock=last_dec.deterministic_lock,
        scope=last_dec.scope,
        raw_outputs=outputs,
    )


async def consensus_dist(
    spawn_fn: Callable[[int], Awaitable[T]],
    *,
    extract_probs: Callable[[T], Sequence[float]],
    labels: Optional[Sequence[Any]] = None,
    k: Optional[int] = None,
    delta: float = 0.05,
    max_calls: int = 32,
    concurrency: int = 4,
) -> AsyncConsensusResult:
    """Grey-box distribution-valued `ASC-D` async consensus stopping.

    Achieves 46%–95.6% fewer rollouts than label-only stopping by consuming each
    trajectory's full probability vector (e.g., from LLM logprobs).
    """
    from ._native import AscdStopper

    if labels is not None:
        n_k = len(labels)
    elif k is not None:
        n_k = k
    else:
        raise ValueError("Provide either `labels` or `k`")

    stopper = AscdStopper(k=n_k, delta=delta, max_budget=max_calls)
    in_flight: Dict[asyncio.Task[T], int] = {}
    launched = 0
    outputs: List[Any] = []
    cancelled = 0
    last_dec = None

    def _launch_up_to_limit() -> None:
        nonlocal launched
        while launched < max_calls and len(in_flight) < concurrency:
            idx = launched
            launched += 1
            task = asyncio.create_task(spawn_fn(idx))
            in_flight[task] = idx

    try:
        _launch_up_to_limit()
        while in_flight:
            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                in_flight.pop(task, None)
                out = task.result()
                outputs.append(out)
                probs = list(extract_probs(out))
                last_dec = stopper.update(probs)
                if last_dec.certified:
                    break
            if last_dec is not None and last_dec.certified:
                break
            _launch_up_to_limit()
    finally:
        for task in list(in_flight.keys()):
            if not task.done():
                task.cancel()
                cancelled += 1
        if in_flight:
            await asyncio.gather(*in_flight.keys(), return_exceptions=True)

    assert last_dec is not None
    def _map_idx(idx: Optional[int]) -> Any:
        if idx is None:
            return None
        return labels[idx] if labels is not None else idx

    return AsyncConsensusResult(
        certified=last_dec.certified,
        winner=_map_idx(last_dec.winner),
        leader=_map_idx(last_dec.leader),
        n_samples=last_dec.n_samples,
        n_cancelled=cancelled,
        log_evidence=last_dec.log_evidence,
        log_threshold=last_dec.log_threshold,
        margin_estimate=last_dec.margin_estimate,
        deterministic_lock=last_dec.deterministic_lock,
        scope=last_dec.scope,
        raw_outputs=outputs,
    )


def extract_logprob_distribution(
    top_logprobs: Dict[str, float],
    candidate_tokens: Sequence[str],
    temperature: float = 1.0,
) -> List[float]:
    """Convert OpenAI / vLLM / OpenRouter `top_logprobs` dict into a normalized simplex vector.

    Drop-in helper for feeding grey-box LLM logprobs directly into `AscdStopper`.
    """
    import math

    if not candidate_tokens:
        raise ValueError("candidate_tokens must be non-empty")
    inv_t = 1.0 / max(temperature, 1e-9)
    # Match exact or stripped uppercase tokens cleanly
    norm_map = {k.strip().upper(): v for k, v in top_logprobs.items()}
    raw: List[float] = []
    min_seen = min(top_logprobs.values()) - 10.0 if top_logprobs else -20.0
    for tok in candidate_tokens:
        if tok in top_logprobs:
            lp = top_logprobs[tok]
        else:
            lp = norm_map.get(tok.strip().upper(), min_seen)
        raw.append(lp * inv_t)
    m = max(raw)
    exps = [math.exp(v - m) for v in raw]
    total = sum(exps)
    if total <= 0.0:
        u = 1.0 / len(candidate_tokens)
        return [u] * len(candidate_tokens)
    return [v / total for v in exps]


async def race_agents(
    eval_fn: Callable[[int, Any], Awaitable[float]],
    n_arms: int,
    tasks: Sequence[Any],
    *,
    delta: float = 0.05,
    paired: bool = True,
    max_tasks: Optional[int] = None,
) -> Tuple[Optional[int], int, List[int]]:
    """Adaptively race `n_arms` agents/prompts across `tasks` and halt losers early.

    Returns `(winner_idx, evals_used, remaining_active_arms)`.
    """
    from ._native import AgentRace

    race = AgentRace(n_arms=n_arms, delta=delta, paired=paired)
    limit = min(len(tasks), max_tasks) if max_tasks is not None else len(tasks)
    evals = 0

    if paired:
        for t_idx in range(limit):
            task = tasks[t_idx]
            active = [a for a in range(n_arms) if race.is_active(a)]
            if len(active) <= 1:
                break
            scores = [float("nan")] * n_arms
            results = await asyncio.gather(*(eval_fn(a, task) for a in active))
            for a, s in zip(active, results):
                scores[a] = float(s)
                evals += 1
            winner = race.observe_paired(scores)
            if winner is not None:
                return winner, evals, [winner]
    else:
        cursors = [0] * n_arms
        while True:
            winner = race.winner()
            if winner is not None:
                return winner, evals, [winner]
            arm = race.next_arm_lucb()
            if cursors[arm] >= limit:
                # Try any active arm that still has remaining tasks
                fallback = next(
                    (a for a in range(n_arms) if race.is_active(a) and cursors[a] < limit),
                    None,
                )
                if fallback is None:
                    break
                arm = fallback
            score = float(await eval_fn(arm, tasks[cursors[arm]]))
            cursors[arm] += 1
            evals += 1
            race.observe_arm(arm, score)

    active = [a for a in range(n_arms) if race.is_active(a)]
    return race.winner(), evals, active
