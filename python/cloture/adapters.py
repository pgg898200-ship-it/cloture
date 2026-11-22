"""Zero-dependency drop-in CSV & JSONL adapters for real-world LLM evaluation exports.

Supports tabular judge-panel exports from Promptfoo, Braintrust, Inspect AI,
LangSmith, or pandas without requiring pandas or NumPy.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ._native import PanelReport, audit_panel


@dataclass(frozen=True)
class NamedPanelAudit:
    judge_names: List[str]
    report: PanelReport

    def prunable_judges(self) -> List[Tuple[str, float, float]]:
        """Return `(judge_name, loo_delta_acc, loo_delta_neff)` for judges with `loo_delta_acc >= 0`."""
        out: List[Tuple[str, float, float]] = []
        for name, d_acc, d_neff in zip(
            self.judge_names, self.report.loo_delta_acc, self.report.loo_delta_neff
        ):
            if d_acc >= 0.0:
                out.append((name, d_acc, d_neff))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def __getattr__(self, name: str):
        return getattr(self.report, name)


def _encode_labels(
    raw_votes: Sequence[Sequence[str]],
    raw_gold: Sequence[str],
    raw_strata: Optional[Sequence[str]] = None,
) -> Tuple[List[List[int]], List[int], Optional[List[int]]]:
    label_map: Dict[str, int] = {}

    def enc(val: str) -> int:
        key = str(val).strip()
        if key not in label_map:
            label_map[key] = len(label_map)
        return label_map[key]

    gold_ids = [enc(g) for g in raw_gold]
    votes_ids = [[enc(v) for v in row] for row in raw_votes]
    strata_ids: Optional[List[int]] = None
    if raw_strata is not None:
        smap: Dict[str, int] = {}
        strata_ids = []
        for s in raw_strata:
            sk = str(s).strip()
            if sk not in smap:
                smap[sk] = len(smap)
            strata_ids.append(smap[sk])
    return votes_ids, gold_ids, strata_ids


def audit_csv(
    path: Union[str, Path],
    *,
    gold_col: str = "gold",
    judge_cols: Optional[Sequence[str]] = None,
    strata_col: Optional[str] = None,
    n_condorcet_sims: int = 2000,
    n_perms: int = 2000,
    n_boot: int = 1000,
    seed: int = 42,
) -> NamedPanelAudit:
    """Audit an LLM-as-a-judge CSV export directly (zero pandas/numpy required).

    Accepts arbitrary string or integer labels (e.g. `"pass"/"fail"`, `"A"/"B"/"C"`,
    or `0/1`) and maps them consistently into integer category IDs before calling
    `cloture.audit_panel`.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV file {p} has no header row")
        if gold_col not in reader.fieldnames:
            raise ValueError(f"gold_col={gold_col!r} not found in columns {reader.fieldnames}")
        if strata_col is not None and strata_col not in reader.fieldnames:
            raise ValueError(f"strata_col={strata_col!r} not found in columns {reader.fieldnames}")

        if judge_cols is not None:
            jcols = list(judge_cols)
            missing = [c for c in jcols if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing judge columns: {missing}")
        else:
            excluded = {gold_col}
            if strata_col is not None:
                excluded.add(strata_col)
            jcols = [c for c in reader.fieldnames if c not in excluded]
        if len(jcols) < 1:
            raise ValueError("CSV must contain at least one judge column")

        raw_votes: List[List[str]] = []
        raw_gold: List[str] = []
        raw_strata: Optional[List[str]] = [] if strata_col is not None else None
        for row in reader:
            raw_gold.append(row[gold_col])
            raw_votes.append([row[c] for c in jcols])
            if raw_strata is not None and strata_col is not None:
                raw_strata.append(row[strata_col])

    votes_ids, gold_ids, strata_ids = _encode_labels(raw_votes, raw_gold, raw_strata)
    rep = audit_panel(
        votes_ids,
        gold_ids,
        strata=strata_ids,
        n_condorcet_sims=n_condorcet_sims,
        n_perms=n_perms,
        n_boot=n_boot,
        seed=seed,
    )
    return NamedPanelAudit(judge_names=jcols, report=rep)


def audit_jsonl(
    path: Union[str, Path],
    *,
    gold_key: str = "gold",
    votes_key: str = "judges",
    strata_key: Optional[str] = None,
    n_condorcet_sims: int = 2000,
    n_perms: int = 2000,
    n_boot: int = 1000,
    seed: int = 42,
) -> NamedPanelAudit:
    """Audit a JSONL evaluation trace where each line has `gold` and a `judges` dict.

    Example line: `{"gold": "pass", "judges": {"gpt4": "pass", "claude": "pass", "llama": "fail"}}`
    """
    p = Path(path)
    jcols: Optional[List[str]] = None
    raw_votes: List[List[str]] = []
    raw_gold: List[str] = []
    raw_strata: Optional[List[str]] = [] if strata_key is not None else None

    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if gold_key not in obj or votes_key not in obj:
                raise ValueError(f"Line {lineno} missing required keys {gold_key!r} / {votes_key!r}")
            jmap = obj[votes_key]
            if not isinstance(jmap, dict) or not jmap:
                raise ValueError(f"Line {lineno}: {votes_key!r} must be a non-empty JSON object")
            if jcols is None:
                jcols = sorted(jmap.keys())
            raw_gold.append(str(obj[gold_key]))
            raw_votes.append([str(jmap[c]) for c in jcols])
            if raw_strata is not None and strata_key is not None:
                raw_strata.append(str(obj.get(strata_key, "0")))

    if not jcols or not raw_gold:
        raise ValueError(f"JSONL file {p} contained no records")

    votes_ids, gold_ids, strata_ids = _encode_labels(raw_votes, raw_gold, raw_strata)
    rep = audit_panel(
        votes_ids,
        gold_ids,
        strata=strata_ids,
        n_condorcet_sims=n_condorcet_sims,
        n_perms=n_perms,
        n_boot=n_boot,
        seed=seed,
    )
    return NamedPanelAudit(judge_names=jcols, report=rep)


def extract_openai_logprobs(
    response,
    candidate_tokens: Sequence[str],
    *,
    choice_index: int = 0,
    token_index: int = 0,
    temperature: float = 1.0,
) -> List[float]:
    """Extract normalized simplex probabilities directly from an OpenAI / vLLM / OpenRouter ChatCompletion.

    Works with both SDK response objects (`response.choices[0].logprobs.content[0].top_logprobs`)
    and raw JSON dicts (`response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]`).
    """
    from .async_tools import extract_logprob_distribution

    if isinstance(response, dict):
        choices = response.get("choices") or []
        choice = choices[choice_index]
        logprobs_obj = choice.get("logprobs") or {}
        content = logprobs_obj.get("content") or []
        top_entries = content[token_index].get("top_logprobs") or []
        lp_map = {str(e["token"]): float(e["logprob"]) for e in top_entries}
    else:
        choice = response.choices[choice_index]
        content = choice.logprobs.content
        top_entries = content[token_index].top_logprobs
        lp_map = {str(e.token): float(e.logprob) for e in top_entries}

    return extract_logprob_distribution(lp_map, candidate_tokens, temperature=temperature)


def audit_promptfoo(
    path: Union[str, Path],
    *,
    gold_var: str = "expected",
    n_condorcet_sims: int = 2000,
    n_perms: int = 2000,
    n_boot: int = 1000,
    seed: int = 42,
) -> NamedPanelAudit:
    """Audit a Promptfoo JSON output file (`promptfoo eval -o results.json`) across multiple providers.

    Groups evaluations by test case (`testIdx`) and treats each distinct provider (`provider.label`
    or `provider.id`) as a judge column.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    results = raw.get("results", raw)
    if isinstance(results, dict):
        results = results.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("Promptfoo JSON does not contain a non-empty results list")

    by_test: Dict[int, Dict[str, str]] = {}
    gold_by_test: Dict[int, str] = {}
    providers_seen: set[str] = set()

    for idx, entry in enumerate(results):
        test_idx = int(entry.get("testIdx", idx))
        prov = entry.get("provider", {})
        if isinstance(prov, dict):
            pname = str(prov.get("label") or prov.get("id") or f"provider_{idx}")
        else:
            pname = str(prov)
        providers_seen.add(pname)
        by_test.setdefault(test_idx, {})[pname] = "1" if entry.get("success") else "0"
        if test_idx not in gold_by_test:
            vars_dict = entry.get("vars") or {}
            if gold_var in vars_dict and entry.get("response") is not None and entry.get("success") is None:
                gold_by_test[test_idx] = str(vars_dict[gold_var])
            else:
                gold_by_test[test_idx] = "1"

    jcols = sorted(providers_seen)
    test_ids = [tid for tid in sorted(by_test.keys()) if all(p in by_test[tid] for p in jcols)]
    if len(test_ids) < 2:
        raise ValueError("Need at least 2 test cases evaluated by all providers")

    raw_votes = [[by_test[tid][p] for p in jcols] for tid in test_ids]
    raw_gold = [gold_by_test[tid] for tid in test_ids]
    votes_ids, gold_ids, _ = _encode_labels(raw_votes, raw_gold)
    rep = audit_panel(
        votes_ids,
        gold_ids,
        n_condorcet_sims=n_condorcet_sims,
        n_perms=n_perms,
        n_boot=n_boot,
        seed=seed,
    )
    return NamedPanelAudit(judge_names=jcols, report=rep)
