#!/usr/bin/env python3
"""Real-world evaluation of cloture on ScalerLab JudgeBench (N=350 objective items).

Fetches official public evaluation logs from:
  https://github.com/ScalerLab/JudgeBench (outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,...)

Covers 13 real frontier models spanning 5 distinct families/architectures:
  - OpenAI: o1-preview, o1-mini, GPT-4o, GPT-4o-mini
  - Anthropic: Claude-3.5-Sonnet, Claude-3-Haiku
  - Google: Gemini-1.5-Pro, Gemini-1.5-Flash
  - Meta: Llama-3.1-405B, Llama-3.1-70B, Llama-3.1-8B
  - Dedicated RM / Critic: Skywork-RM-Gemma-27B, Skywork-Critic-70B

Runs four reproducible, zero-synthetic-data experiments:
  1. Correlated judge-panel audit (11 generative LLMs + Top-7 cross-family frontier panel)
     under both Official Swap-Consistent and Single-Pass (Pass-0) protocols.
  2. Greedy Leave-One-Out (LOO) pruning (13 -> 1) PLUS strict 5-fold out-of-sample CV
     (selecting the pruned 3-judge subset solely on 80% training folds and evaluating
     on held-out 20% test folds).
  3. Paired sequential AgentRace across Top-6 frontier judges (50 shuffled streams).
  4. Real Active-PPI++ (GPT-4o / Claude-3.5 proxy -> o1-preview reasoning judge, 20% audit).

Writes machine-verifiable receipts to benchmarks/judgebench_receipts.json.
"""

from __future__ import annotations

import base64
import json
import os
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import cloture  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/ScalerLab/JudgeBench/main/outputs/"

MODELS: list[tuple[str, str, str, str]] = [
    ("o1-preview", "OpenAI", "arena_hard", "o1-preview-2024-09-12"),
    ("o1-mini", "OpenAI", "arena_hard", "o1-mini-2024-09-12"),
    ("GPT-4o", "OpenAI", "arena_hard", "gpt-4o-2024-05-13"),
    ("GPT-4o-mini", "OpenAI", "arena_hard", "gpt-4o-mini-2024-07-18"),
    ("Claude-3.5-Sonnet", "Anthropic", "arena_hard", "claude-3-5-sonnet-20240620"),
    ("Claude-3-Haiku", "Anthropic", "arena_hard", "claude-3-haiku-20240307"),
    ("Gemini-1.5-Pro", "Google", "arena_hard", "gemini-1.5-pro-001"),
    ("Gemini-1.5-Flash", "Google", "arena_hard", "gemini-1.5-flash-001"),
    ("Llama-3.1-405B", "Meta", "arena_hard", "meta-llama_Meta-Llama-3.1-405B-Instruct"),
    ("Llama-3.1-70B", "Meta", "arena_hard", "meta-llama_Meta-Llama-3.1-70B-Instruct"),
    ("Llama-3.1-8B", "Meta", "arena_hard", "meta-llama_Meta-Llama-3.1-8B-Instruct"),
    ("Skywork-RM-Gemma-27B", "Skywork-RM", "reward_model", "Skywork_Skywork-Reward-Gemma-2-27B"),
    ("Skywork-Critic-70B", "Skywork-Critic", "skywork_critic", "Skywork_Skywork-Critic-Llama-3.1-70B"),
]


def fetch_model_jsonl(jname: str, jmodel: str) -> list[dict]:
    fname = f"dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name={jname},judge_model={jmodel}.jsonl"
    url = BASE_URL + urllib.parse.quote(fname)
    req = urllib.request.Request(url, headers={"User-Agent": "cloture-real-world-eval/0.2"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]


def greedy_loo_trace(
    names: list[str],
    votes_by_model: dict[str, list[int]],
    gold: list[int],
    strata: list[int],
    n_items: int,
) -> list[dict]:
    active = list(range(len(names)))
    trace: list[dict] = []
    while len(active) >= 1:
        sub_names = [names[i] for i in active]
        if len(active) == 1:
            v = votes_by_model[sub_names[0]]
            acc = sum(1 for a, g in zip(v, gold) if a == g) / n_items
            trace.append(
                {
                    "k": 1,
                    "kept": sub_names,
                    "panel_accuracy": acc,
                    "n_eff_kish": 1.0,
                    "mean_phi": 0.0,
                    "dropped_next": None,
                    "cost_savings_pct": (1.0 - 1.0 / len(names)) * 100.0,
                }
            )
            break
        sub_votes = [[votes_by_model[n][i] for n in sub_names] for i in range(n_items)]
        rep = cloture.audit_panel(
            sub_votes, gold, strata=strata, n_perms=200, n_condorcet_sims=200, n_boot=100, seed=42
        )
        best_pos = max(range(len(active)), key=lambda p: (rep.loo_delta_acc[p], rep.loo_delta_neff[p]))
        drop_name = sub_names[best_pos]
        trace.append(
            {
                "k": len(active),
                "kept": sub_names,
                "panel_accuracy": rep.panel_accuracy,
                "n_eff_kish": rep.n_eff_kish,
                "mean_phi": rep.mean_phi,
                "dropped_next": drop_name,
                "dropped_loo_delta_acc": rep.loo_delta_acc[best_pos],
                "cost_savings_pct": (1.0 - len(active) / len(names)) * 100.0,
            }
        )
        active.pop(best_pos)
    return trace


def cross_validated_loo_pruning(
    names: list[str],
    votes_by_model: dict[str, list[int]],
    gold: list[int],
    target_k: int = 3,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    n_items = len(gold)
    idx = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]

    full_accs: list[float] = []
    pruned_accs: list[float] = []
    selected_trios: list[list[str]] = []

    for f in range(n_folds):
        test_set = set(folds[f])
        train_idx = [i for i in range(n_items) if i not in test_set]
        test_idx = sorted(test_set)

        active = list(range(len(names)))
        while len(active) > target_k:
            tr_votes = [[votes_by_model[names[j]][i] for j in active] for i in train_idx]
            tr_gold = [gold[i] for i in train_idx]
            rep = cloture.audit_panel(tr_votes, tr_gold, n_perms=10, n_condorcet_sims=10, n_boot=10, seed=42)
            best_pos = max(range(len(active)), key=lambda p: (rep.loo_delta_acc[p], rep.loo_delta_neff[p]))
            active.pop(best_pos)

        selected_trios.append([names[j] for j in active])
        te_gold = [gold[i] for i in test_idx]
        te_full = cloture.audit_panel(
            [[votes_by_model[n][i] for n in names] for i in test_idx],
            te_gold,
            n_perms=10,
            n_condorcet_sims=10,
            n_boot=10,
            seed=42,
        )
        te_pruned = cloture.audit_panel(
            [[votes_by_model[names[j]][i] for j in active] for i in test_idx],
            te_gold,
            n_perms=10,
            n_condorcet_sims=10,
            n_boot=10,
            seed=42,
        )
        full_accs.append(te_full.panel_accuracy)
        pruned_accs.append(te_pruned.panel_accuracy)

    mean_full = statistics.mean(full_accs)
    mean_pruned = statistics.mean(pruned_accs)
    return {
        "n_folds": n_folds,
        "target_k": target_k,
        "cost_savings_pct": (1.0 - target_k / len(names)) * 100.0,
        "mean_full_test_accuracy": mean_full,
        "mean_pruned_test_accuracy": mean_pruned,
        "out_of_sample_gain_pp": (mean_pruned - mean_full) * 100.0,
        "fold_full_accuracies": full_accs,
        "fold_pruned_accuracies": pruned_accs,
        "selected_subsets_per_fold": selected_trios,
    }


def main() -> None:
    print("Fetching official ScalerLab/JudgeBench outputs for 13 frontier models...")
    gold: list[int] | None = None
    strata: list[int] | None = None
    strata_names: list[str] = []
    pair_ids: list[str] | None = None

    votes_swap: dict[str, list[int]] = {}
    votes_pass0: dict[str, list[int]] = {}
    model_meta: list[dict] = []

    for label, family, jname, jmodel in MODELS:
        recs = fetch_model_jsonl(jname, jmodel)
        if pair_ids is None:
            pair_ids = [r["pair_id"] for r in recs]
            gold = [0 if r["label"] == "A>B" else 1 for r in recs]
            src_map: dict[str, int] = {}
            strata = []
            for r in recs:
                bucket = r["source"].split("-")[0]
                if bucket not in src_map:
                    src_map[bucket] = len(strata_names)
                    strata_names.append(bucket)
                strata.append(src_map[bucket])
        else:
            assert [r["pair_id"] for r in recs] == pair_ids, f"pair_id mismatch for {label}"

        vc_list: list[int] = []
        v0_list: list[int] = []
        for r in recs:
            j = r["judgments"]
            d0 = j[0].get("decision") if len(j) > 0 and isinstance(j[0], dict) else None
            d1 = j[1].get("decision") if len(j) > 1 and isinstance(j[1], dict) else None
            v0 = 0 if d0 == "A>B" else (1 if d0 == "B>A" else 2)
            v0_list.append(v0)
            if len(j) == 1:
                vc = v0
            elif d0 == "A>B" and d1 == "B>A":
                vc = 0
            elif d0 == "B>A" and d1 == "A>B":
                vc = 1
            else:
                vc = 2
            vc_list.append(vc)

        votes_swap[label] = vc_list
        votes_pass0[label] = v0_list
        assert gold is not None
        acc_swap = sum(1 for v, g in zip(vc_list, gold) if v == g) / len(gold)
        acc_p0 = sum(1 for v, g in zip(v0_list, gold) if v == g) / len(gold)
        model_meta.append(
            {
                "name": label,
                "family": family,
                "judge_name": jname,
                "judge_model": jmodel,
                "swap_consistent_accuracy": acc_swap,
                "pass0_accuracy": acc_p0,
            }
        )
        print(f"  {label:22s} ({family:14s}): swap={acc_swap*100:5.2f}%  pass0={acc_p0*100:5.2f}%")

    assert gold is not None and strata is not None
    n_items = len(gold)
    all_names = [m[0] for m in MODELS]
    gen_names = [m[0] for m in MODELS[:11]]
    top7_names = [
        "o1-preview",
        "o1-mini",
        "GPT-4o",
        "Claude-3.5-Sonnet",
        "Gemini-1.5-Pro",
        "Llama-3.1-405B",
        "Skywork-RM-Gemma-27B",
    ]
    fam_by_name = {m[0]: m[1] for m in MODELS}

    # 1. Audits: 11 Generative LLMs + Top-7 Cross-Family Panel (both Swap-Consistent & Pass-0)
    def run_panel_summary(names_sub: list[str], vmap: dict[str, list[int]]) -> dict:
        mat = [[vmap[n][i] for n in names_sub] for i in range(n_items)]
        t0 = time.perf_counter()
        rep = cloture.audit_panel(
            mat, gold, strata=strata, n_perms=2000, n_condorcet_sims=2000, n_boot=1000, seed=42
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        within, cross = [], []
        for i in range(len(names_sub)):
            for j in range(i + 1, len(names_sub)):
                if fam_by_name[names_sub[i]] == fam_by_name[names_sub[j]]:
                    within.append(rep.phi_matrix[i][j])
                else:
                    cross.append(rep.phi_matrix[i][j])
        return {
            "models": names_sub,
            "k_judges": rep.k_judges,
            "audit_ms": dt_ms,
            "panel_accuracy": rep.panel_accuracy,
            "best_single_accuracy": rep.best_single_accuracy,
            "condorcet_predicted_accuracy": rep.condorcet_predicted_accuracy,
            "condorcet_gap_pp": rep.condorcet_gap * 100.0,
            "permutation_p_value": rep.permutation_p_value,
            "n_eff_kish": rep.n_eff_kish,
            "n_eff_ci95": [rep.n_eff_ci_low, rep.n_eff_ci_high],
            "n_eff_eigen": rep.n_eff_eigen,
            "effective_rank": rep.effective_rank,
            "krippendorff_alpha": rep.krippendorff_alpha,
            "mean_phi": rep.mean_phi,
            "within_family_mean_phi": statistics.mean(within) if within else None,
            "cross_family_mean_phi": statistics.mean(cross) if cross else None,
            "phi_matrix": rep.phi_matrix,
            "loo_delta_acc": rep.loo_delta_acc,
            "loo_delta_neff": rep.loo_delta_neff,
        }

    gen11_swap = run_panel_summary(gen_names, votes_swap)
    gen11_pass0 = run_panel_summary(gen_names, votes_pass0)
    top7_swap = run_panel_summary(top7_names, votes_swap)
    top7_pass0 = run_panel_summary(top7_names, votes_pass0)

    # 2. Greedy LOO pruning + strict 5-fold out-of-sample CV
    prune_swap = greedy_loo_trace(all_names, votes_swap, gold, strata, n_items)
    prune_pass0 = greedy_loo_trace(all_names, votes_pass0, gold, strata, n_items)
    cv5_swap = cross_validated_loo_pruning(all_names, votes_swap, gold, target_k=3, n_folds=5, seed=42)
    cv5_pass0 = cross_validated_loo_pruning(all_names, votes_pass0, gold, target_k=3, n_folds=5, seed=42)

    # 3. Paired AgentRace across 6 top models (50 shuffled streams, Swap-Consistent)
    race_models = [
        "o1-preview",
        "Skywork-RM-Gemma-27B",
        "Claude-3.5-Sonnet",
        "o1-mini",
        "Skywork-Critic-70B",
        "Llama-3.1-405B",
    ]
    race_correct = {
        m: [1.0 if v == g else 0.0 for v, g in zip(votes_swap[m], gold)] for m in race_models
    }
    race_evals: list[int] = []
    race_cert = 0
    race_right = 0
    for trial in range(50):
        rng = random.Random(1000 + trial)
        order = list(range(n_items))
        rng.shuffle(order)
        race = cloture.AgentRace(len(race_models), delta=0.05, paired=True)
        evals_used = 0
        for idx in order:
            active_arms = [a for a in range(len(race_models)) if race.is_active(a)]
            scores = [race_correct[race_models[a]][idx] for a in range(len(race_models))]
            race.observe_paired(scores)
            evals_used += len(active_arms)
            w = race.winner()
            if w is not None:
                race_cert += 1
                if w == 0:
                    race_right += 1
                break
        else:
            if race.is_active(0):
                race_right += 1
        race_evals.append(evals_used)

    full_race_budget = len(race_models) * n_items
    race_summary = {
        "models": race_models,
        "true_winner": race_models[0],
        "n_trials": 50,
        "full_budget_evals": full_race_budget,
        "mean_evals": statistics.mean(race_evals),
        "median_evals": statistics.median(race_evals),
        "pct_evals_saved": (1.0 - statistics.mean(race_evals) / full_race_budget) * 100.0,
        "certified_rate": race_cert / 50.0,
        "winner_accuracy": race_right / 50.0,
    }

    # 4. Real Active-PPI++: GPT-4o (cheap proxy) -> o1-preview (expensive reasoning judge), 20% audit
    y_true_vec = [1.0 if v == g else 0.0 for v, g in zip(votes_swap["o1-preview"], gold)]
    f_proxy_vec = [1.0 if v == g else 0.0 for v, g in zip(votes_swap["GPT-4o"], gold)]
    true_o1_acc = statistics.mean(y_true_vec)
    proxy_4o_acc = statistics.mean(f_proxy_vec)

    ppi_anytime_ok = 0
    ppi_widths: list[float] = []
    ppi_var_reds: list[float] = []
    human_widths: list[float] = []
    for trial in range(50):
        rng = random.Random(2000 + trial)
        order = list(range(n_items))
        rng.shuffle(order)
        ppi = cloture.ActivePPI(alpha=0.05, target_audit_rate=0.20, min_audit_prob=0.10)
        eb = cloture.EmpiricalBernsteinCS(alpha=0.05, min_val=0.0, max_val=1.0)
        covered_all = True
        last_res = None
        last_eb = None
        for idx in order:
            f = f_proxy_vec[idx]
            y = y_true_vec[idx]
            pi = ppi.suggest_prob(f, 1.0)
            audited = rng.random() < pi
            res = ppi.observe(f, audited, y if audited else 0.0, pi)
            if not (res.lower <= true_o1_acc <= res.upper):
                covered_all = False
            last_res = res
            if audited:
                last_eb = eb.update(y)
        if covered_all:
            ppi_anytime_ok += 1
        assert last_res is not None and last_eb is not None
        ppi_widths.append(last_res.upper - last_res.lower)
        ppi_var_reds.append(last_res.variance_reduction)
        human_widths.append(last_eb.upper - last_eb.lower)

    ppi_summary = {
        "proxy_model": "GPT-4o",
        "target_model": "o1-preview",
        "n_trials": 50,
        "audit_rate": 0.20,
        "true_target_accuracy": true_o1_acc,
        "naive_proxy_accuracy": proxy_4o_acc,
        "naive_proxy_bias_pp": (proxy_4o_acc - true_o1_acc) * 100.0,
        "anytime_coverage_all_350_steps": ppi_anytime_ok / 50.0,
        "mean_final_width_ppi": statistics.mean(ppi_widths),
        "mean_final_width_human_only": statistics.mean(human_widths),
        "mean_variance_reduction": statistics.mean(ppi_var_reds),
    }

    # 5. Compact 2-bit-packed preset for browser WASM playground (0=A>B, 1=B>A, 2=tie/inconsistent + gold bit)
    # Encodes Swap-Consistent votes (13 x 350 = 4,550 values -> 1,138 bytes base64) + gold (44 bytes base64)
    def pack_2bit(vals: list[int]) -> str:
        buf = bytearray((len(vals) + 3) // 4)
        for idx, v in enumerate(vals):
            buf[idx >> 2] |= (v & 3) << ((idx & 3) * 2)
        return base64.b64encode(bytes(buf)).decode("ascii")

    def pack_1bit(vals: list[int]) -> str:
        buf = bytearray((len(vals) + 7) // 8)
        for idx, v in enumerate(vals):
            if v:
                buf[idx >> 3] |= 1 << (idx & 7)
        return base64.b64encode(bytes(buf)).decode("ascii")

    flat_swap = [votes_swap[n][i] for i in range(n_items) for n in all_names]

    out = {
        "source_dataset": "ScalerLab/JudgeBench (response_model=gpt-4o-2024-05-13)",
        "source_url": "https://github.com/ScalerLab/JudgeBench",
        "n_items": n_items,
        "n_strata": len(strata_names),
        "strata_names": strata_names,
        "models": model_meta,
        "generative_11_swap": gen11_swap,
        "generative_11_pass0": gen11_pass0,
        "top7_cross_family_swap": top7_swap,
        "top7_cross_family_pass0": top7_pass0,
        "frontier_13_pruning_swap": prune_swap,
        "frontier_13_pruning_pass0": prune_pass0,
        "cv5_out_of_sample_swap": cv5_swap,
        "cv5_out_of_sample_pass0": cv5_pass0,
        "paired_agent_race_top6": race_summary,
        "active_ppi_gpt4o_to_o1_preview": ppi_summary,
        "playground_preset": {
            "n_items": n_items,
            "k_judges": len(all_names),
            "model_names": all_names,
            "votes_2bit_b64": pack_2bit(flat_swap),
            "gold_1bit_b64": pack_1bit(gold),
            "strata_2bit_b64": pack_2bit(strata),
        },
    }

    dest = ROOT / "benchmarks" / "judgebench_receipts.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {dest} ({os.path.getsize(dest):,} bytes)")
    print(f"  11 Gen LLMs (swap): acc={gen11_swap['panel_accuracy']*100:.2f}% n_eff={gen11_swap['n_eff_kish']:.2f}/11 gap={gen11_swap['condorcet_gap_pp']:+.2f}pp")
    print(f"  11 Gen LLMs (pass0): acc={gen11_pass0['panel_accuracy']*100:.2f}% n_eff={gen11_pass0['n_eff_kish']:.2f}/11 gap={gen11_pass0['condorcet_gap_pp']:+.2f}pp")
    print(f"  Top-7 Cross-Family (swap): acc={top7_swap['panel_accuracy']*100:.2f}% n_eff={top7_swap['n_eff_kish']:.2f}/7 gap={top7_swap['condorcet_gap_pp']:+.2f}pp")
    print(f"  5-fold OOS CV (swap): {cv5_swap['mean_full_test_accuracy']*100:.2f}% -> {cv5_swap['mean_pruned_test_accuracy']*100:.2f}% ({cv5_swap['out_of_sample_gain_pp']:+.2f}pp)")
    print(f"  5-fold OOS CV (pass0): {cv5_pass0['mean_full_test_accuracy']*100:.2f}% -> {cv5_pass0['mean_pruned_test_accuracy']*100:.2f}% ({cv5_pass0['out_of_sample_gain_pp']:+.2f}pp)")
    print(f"  AgentRace Top-6: saved={race_summary['pct_evals_saved']:.1f}% cert={race_summary['certified_rate']*100:.0f}% acc={race_summary['winner_accuracy']*100:.0f}%")
    print(f"  ActivePPI (4o->o1): cov={ppi_summary['anytime_coverage_all_350_steps']*100:.0f}% var_red={ppi_summary['mean_variance_reduction']:.2f}x bias={ppi_summary['naive_proxy_bias_pp']:+.2f}pp")


if __name__ == "__main__":
    main()
