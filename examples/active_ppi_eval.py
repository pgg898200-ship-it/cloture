"""Demo: Anytime-Valid Active-PPI++ hybrid LLM-judge + sparse human audit evaluation.

Run:
    PYTHONPATH=python .venv/bin/python examples/active_ppi_eval.py
"""
import random
import sys
from pathlib import Path
import cloture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from traces import gen_active_ppi_stream


def main() -> None:
    trace = gen_active_ppi_stream(
        seed=7, n_items=3000, judge_corr=0.85, judge_bias=0.12
    )
    rng = random.Random(7)
    rng_human = random.Random(707)
    ppi = cloture.ActivePPI(alpha=0.05, target_audit_rate=0.10, min_audit_prob=0.06)
    naive = cloture.EmpiricalBernsteinCS(alpha=0.05)
    nres = None

    for i, (f_i, y_i, u_i) in enumerate(
        zip(trace.f_judge, trace.y_true, trace.uncertainty), start=1
    ):
        pi_t = ppi.suggest_prob(float(f_i), float(u_i))
        audited = rng.random() < pi_t
        res = ppi.observe(
            float(f_i),
            audited=audited,
            gold_label=(float(y_i) if audited else 0.0),
            audit_prob=pi_t,
        )
        # Matched-budget uniform 10% human-only baseline (unbiased)
        if rng_human.random() < 0.10:
            nres = naive.update(float(y_i))
        if i in (300, 1000, 3000) and nres is not None:
            print(
                f"t={i:4d} (audited={res.n_audited:3d}, {100*res.n_audited/i:4.1f}%) | "
                f"Active-PPI++ [{res.lower:.3f}, {res.upper:.3f}] (w={res.width:.3f}, lam={res.lambda_hat:.2f}) | "
                f"Human-only CS [{nres.lower:.3f}, {nres.upper:.3f}] (w={nres.width:.3f}) | "
                f"true={trace.true_mean:.3f}"
            )


if __name__ == "__main__":
    main()
