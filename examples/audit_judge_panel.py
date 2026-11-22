"""Demo: Audit a correlated 9-judge LLM panel and find redundant judges to drop.

Run:
    PYTHONPATH=python .venv/bin/python examples/audit_judge_panel.py
"""
import sys
from pathlib import Path
import numpy as np
import cloture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from traces import gen_correlated_judge_panel


def main() -> None:
    trace = gen_correlated_judge_panel(
        seed=42, n_items=1000, k_judges=9, target_phi=0.391
    )
    votes = np.ascontiguousarray(trace.votes, dtype=np.uint32)
    gold = np.ascontiguousarray(trace.true_labels, dtype=np.uint32)
    judge_names = [f"judge_{j}" + (" (redundant)" if j == trace.redundant_judge_idx else "") for j in range(trace.k_judges)]

    report = cloture.audit_panel(
        votes=votes,
        gold=gold,
        n_condorcet_sims=2000,
        n_perms=5000,
        n_boot=1000,
        seed=42,
    )
    print("=== Cloture Correlated-Panel Audit (9 Judges, 1,000 Items) ===")
    print(f"Nominal Judges (k)      : {report.k_judges}")
    print(f"Effective Votes (Kish)  : {report.n_eff_kish:.2f} [95% CI: {report.n_eff_ci_low:.2f}, {report.n_eff_ci_high:.2f}]")
    print(f"Effective Votes (Eigen) : {report.n_eff_eigen:.2f} (lambda_max={report.lambda_max:.2f}, eff_rank={report.effective_rank:.2f})")
    print(f"Independence Ratio      : {100.0 * report.independence_ratio:.1f}% (asymptotic ceiling: {report.asymptotic_ceiling:.2f} votes)")
    print(f"Mean Pairwise Phi       : {report.mean_phi:.3f} +/- {report.std_phi:.3f}")
    print(f"Panel Majority Acc      : {100.0 * report.panel_accuracy:.2f}% (Best single: {100.0 * report.best_single_accuracy:.2f}% [{judge_names[report.best_single_index]}])")
    print(f"Condorcet Null Acc      : {100.0 * report.condorcet_predicted_accuracy:.2f}% -> Condorcet Gap = {100.0 * report.condorcet_gap:.2f} pp")
    print(f"Permutation Omnibus p   : {report.permutation_p_value:.5f}")
    print(f"Krippendorff Alpha      : {report.krippendorff_alpha:.3f}")
    print(f"Unanimous Items         : {100.0 * report.unanimous_rate:.1f}% (acc={100.0 * report.unanimous_accuracy:.1f}%, all-wrong={100.0 * report.all_wrong_rate:.1f}%)")
    print("\nLeave-One-Out Pruning Impact:")
    for j, name in enumerate(judge_names):
        da = report.loo_delta_acc[j] * 100.0
        dn = report.loo_delta_neff[j]
        tag = " <-- REDUNDANT (drop saves API cost!)" if da >= 0.0 else ""
        print(f"  Drop {name:20s}: delta_acc={da:+6.2f} pp | delta_n_eff={dn:+.3f}{tag}")


if __name__ == "__main__":
    main()
