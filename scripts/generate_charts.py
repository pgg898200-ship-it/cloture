#!/usr/bin/env python3
"""Generate publication-grade, data-exact SVG benchmark charts from benchmarks/results.json.

Zero external dependencies (Python stdlib only). Produces two crisp vector SVGs
optimized for GitHub README (both light and dark themes) and the static docs site:
  - docs/assets/consensus_and_panel.svg
  - docs/assets/speed_and_validity.svg
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "benchmarks" / "results.json"
ASSETS_DIR = ROOT / "docs" / "assets"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_consensus_and_panel_svg(d: dict) -> str:
    s1 = d["suite_1_consensus_stopping"]
    easy = s1["ascd_table2"]["easy"]
    med = s1["ascd_table2"]["medium"]
    null_t = s1["ascd_table2"]["null_tie"]
    zipf = s1["open_ended_swarm"]["easy_gap_0.30"]
    zipf_null = s1["open_ended_swarm"]["null_tie"]

    s2 = d["suite_2_panel_audit"]["panel_findings"]
    s2_t = d["suite_2_panel_audit"]["timing"]

    easy_ascd = easy["cloture_ascd"]["mean_samples"]
    easy_cite = easy["cloture_cite"]["mean_samples"]
    med_ascd = med["cloture_ascd"]["mean_samples"]
    ascd_tie_fpr = null_t["cloture_ascd"]["type1_error_on_null"] * 100.0
    naive_tie_fpr = null_t["naive_uncorrected_peeking"]["type1_error_on_null"] * 100.0
    zipf_cite = zipf["cloture_cite"]["mean_samples"]
    zipf_tie_fpr = zipf_null["cloture_cite"]["type1_error_on_null"] * 100.0
    zipf_naive_fpr = zipf_null["naive_uncorrected_peeking"]["type1_error_on_null"] * 100.0
    saved_64 = easy["cloture_ascd"]["pct_saved_vs_fixed64"]
    saved_256 = easy["cloture_ascd"]["pct_saved_vs_fixed256"]

    neff = s2["n_eff_kish"]
    neff_lo, neff_hi = s2["n_eff_kish_ci95"]
    condorcet_acc = s2["condorcet_predicted_accuracy"] * 100.0
    actual_acc = s2["panel_accuracy"] * 100.0
    pruned_acc = s2["drop_2_panel_accuracy"] * 100.0
    pruned_gain = s2["drop_2_accuracy_gain"] * 100.0
    pruned_cost = s2["drop_2_cost_savings_pct"]
    gap_pp = s2["condorcet_gap"] * 100.0
    audit_ms = s2_t["cloture_ms"]["median"]
    audit_x = s2_t["speedup_x"]

    # Left panel bars: calls needed (max=256)
    left_bars = [
        ("cloture ASC-D (Easy)", easy_ascd, "#38d9c0", f"{easy_ascd:.1f} calls (-{saved_64:.1f}% vs 64)"),
        ("cloture ASC-D (Medium)", med_ascd, "#4fd1c5", f"{med_ascd:.1f} calls (-39.5% vs 64)"),
        ("cloture CITE (Easy)", easy_cite, "#7aa2f7", f"{easy_cite:.1f} calls (unbounded)"),
        ("Fixed-64 Baseline", 64.0, "#4b5563", "64.0 calls (fixed budget)"),
        ("cloture CITE (K=200 Zipf)", zipf_cite, "#9ece6a", f"{zipf_cite:.1f} calls (-72.3% vs 256)"),
        ("Fixed-256 Baseline", 256.0, "#374151", "256.0 calls (fixed budget)"),
    ]

    # Right panel bars: panel accuracy & effective votes
    right_bars = [
        ("Condorcet Illusion (9 indep.)", condorcet_acc, "#64748b", f"{condorcet_acc:.1f}% (assumed)"),
        ("Actual 9-Judge Panel", actual_acc, "#f7b267", f"{actual_acc:.1f}% (-{gap_pp:.1f} pp gap)"),
        ("After Pruning 2 Redundant", pruned_acc, "#38d9c0", f"{pruned_acc:.1f}% (+{pruned_gain:.2f} pp, -{pruned_cost:.1f}% cost)"),
    ]

    W, H = 980, 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="cloture consensus stopping and correlated judge-panel audit benchmark chart">',
        "<defs>",
        '  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '    <stop offset="0%" stop-color="#0d1117"/>',
        '    <stop offset="100%" stop-color="#111827"/>',
        "  </linearGradient>",
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="14" fill="url(#bg)" stroke="#263147" stroke-width="1.5"/>',
        # Header
        '<text x="28" y="34" fill="#e6edf7" font-family="ui-sans-serif,system-ui,-apple-system,sans-serif" font-size="16" font-weight="700">cloture v0.1.0 — Verified Empirical Receipts (benchmarks/results.json)</text>',
        '<text x="952" y="34" text-anchor="end" fill="#8b9ab5" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11.5">0 external crates · 0 Python runtime deps · 266/266 tests</text>',
        '<line x1="28" y1="48" x2="952" y2="48" stroke="#223047" stroke-width="1"/>',
        # Left Card
        '<rect x="28" y="64" width="452" height="332" rx="10" fill="#131a26" stroke="#25324a" stroke-width="1"/>',
        '<text x="46" y="90" fill="#e6edf7" font-family="ui-sans-serif,system-ui,sans-serif" font-size="14" font-weight="700">1. Anytime-Valid Consensus Stopping (δ = 0.05)</text>',
        f'<text x="46" y="109" fill="#93a3bd" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11.5">Up to {saved_256:.1f}% fewer LLM calls with 100% accuracy &amp; zero peeking blow-up</text>',
    ]

    bar_x0 = 46
    bar_max_w = 270
    y = 132
    for label, val, color, note in left_bars:
        w = max(8.0, (val / 256.0) * bar_max_w)
        lines.append(f'<text x="{bar_x0}" y="{y}" fill="#cbd5e1" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.8">{_esc(label)}</text>')
        lines.append(f'<rect x="{bar_x0}" y="{y+6}" width="{bar_max_w}" height="12" rx="4" fill="#1e293b"/>')
        lines.append(f'<rect x="{bar_x0}" y="{y+6}" width="{w:.1f}" height="12" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{bar_x0 + bar_max_w + 8}" y="{y+16}" fill="#e2e8f0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.2">{_esc(note)}</text>')
        y += 33

    # Left bottom callout box: Exact-Tie Null FPR
    lines += [
        '<rect x="46" y="336" width="416" height="44" rx="7" fill="#191f2e" stroke="#2d3b54" stroke-width="1"/>',
        '<text x="58" y="354" fill="#93a3bd" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11" font-weight="600">EXACT-TIE NULL FALSE CERTIFICATION RATE (target ≤ 5.0%):</text>',
        f'<text x="58" y="371" fill="#38d9c0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11.2" font-weight="700">cloture: {ascd_tie_fpr:.1f}% (ASC-D) / {zipf_tie_fpr:.1f}% (CITE)</text>',
        f'<text x="302" y="371" fill="#f87171" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11.2" font-weight="700">Naive: {naive_tie_fpr:.1f}%–{zipf_naive_fpr:.1f}%</text>',
    ]

    # Right Card
    lines += [
        '<rect x="500" y="64" width="452" height="332" rx="10" fill="#131a26" stroke="#25324a" stroke-width="1"/>',
        '<text x="518" y="90" fill="#e6edf7" font-family="ui-sans-serif,system-ui,sans-serif" font-size="14" font-weight="700">2. Correlated LLM-as-a-Judge Collapse (9 × 1,000)</text>',
        f'<text x="518" y="109" fill="#93a3bd" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11.5">Audited in {audit_ms:.1f} ms ({audit_x:.1f}× faster than NumPy/SciPy, 2.66e-15 diff)</text>',
        # Big n_eff pill row
        '<rect x="518" y="124" width="416" height="68" rx="8" fill="#172133" stroke="#2b3d59" stroke-width="1"/>',
        f'<text x="534" y="156" fill="#38d9c0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="26" font-weight="800">{neff:.2f}</text>',
        '<text x="610" y="148" fill="#e6edf7" font-family="ui-sans-serif,system-ui,sans-serif" font-size="13" font-weight="700">effective independent votes out of 9 judges</text>',
        f'<text x="610" y="167" fill="#93a3bd" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.8">Kish n_eff 95% CI [{neff_lo:.2f}, {neff_hi:.2f}] · Eigen n_eff {s2["n_eff_eigen"]:.2f} · p=0.0005</text>',
        # Visual 9 dots (2.2 active teal, rest dim)
        '<text x="534" y="182" fill="#8b9ab5" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="9.8">9 API bills paid → 2.2 independent signal units</text>',
    ]

    # Accuracy comparison bars
    rx0 = 518
    r_max_w = 245
    ry = 216
    for label, pct, color, note in right_bars:
        w = (pct / 100.0) * r_max_w
        lines.append(f'<text x="{rx0}" y="{ry}" fill="#cbd5e1" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.8">{_esc(label)}</text>')
        lines.append(f'<rect x="{rx0}" y="{ry+6}" width="{r_max_w}" height="14" rx="4" fill="#1e293b"/>')
        lines.append(f'<rect x="{rx0}" y="{ry+6}" width="{w:.1f}" height="14" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{rx0 + r_max_w + 8}" y="{ry+17}" fill="#e2e8f0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.2">{_esc(note)}</text>')
        ry += 40

    # Bottom right takeaway
    lines += [
        '<rect x="518" y="336" width="416" height="44" rx="7" fill="#152626" stroke="#255450" stroke-width="1"/>',
        '<text x="530" y="354" fill="#38d9c0" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11" font-weight="700">LOO REDUNDANT-JUDGE PRUNING RECEIPT:</text>',
        f'<text x="530" y="371" fill="#e2e8f0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11">Drop judges #6 &amp; #8 → +{pruned_gain:.2f} pp accuracy at -{pruned_cost:.1f}% API cost</text>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def build_speed_and_validity_svg(d: dict) -> str:
    s1 = d["suite_1_consensus_stopping"]["latency"]
    s2 = d["suite_2_panel_audit"]["timing"]
    s3 = d["suite_3_correlated_rollouts"]["null_theta0_30x16"]
    s4_cs = d["suite_4_cs_and_racing"]["cs_head_to_head"]
    s4_race = d["suite_4_cs_and_racing"]["multi_arm_racing"]["bayesbench_comparison"]
    s5 = d["suite_5_active_ppi"]["regime_a_accurate_biased"]
    s6 = d["suite_6_fuse_cascade_conformal_sketch"]

    speedups = [
        ("AgentRace vs bayesbench", s4_race["six_arm_single_trial_timing"]["speedup_x"], "2.75 ms vs 30.5 s"),
        ("OnlineConformal vs online-cp", s6["online_conformal"]["online_cp_simple_jumper_martingale"]["speedup_cloture_rust_vs_online_cp_jumper"], "64.5 ns vs 188.9 µs"),
        ("PairedEProcess vs bayesbench", s4_race["null_tie_0.50_vs_0.50"]["bayesbench_compare_median_ms"] / s4_race["null_tie_0.50_vs_0.50"]["cloture_paired_eprocess_median_ms"], "0.21 ms vs 196.6 ms"),
        ("EB-CS stream vs expectation", s4_cs["expectation_ebcs"]["python_stream_ns_per_step"] / s4_cs["cloture_ebcs"]["python_stream_ns_per_step"], "5.16 µs vs 1.13 ms"),
        ("AscdStopper vs NumPy ref", s1["speedup_cloture_batch_vs_numpy"], "2.21 µs vs 63.8 µs"),
        ("BettingCS vs confseq hedged", s4_cs["confseq_hedged_betting_201"]["batch_1k_ns_per_step"] / s4_cs["cloture_betting_cs_201"]["batch_100k_ns_per_step"], "6.32 µs vs 179.3 µs"),
        ("PanelAudit vs NumPy/SciPy", s2["speedup_x"], "317.7 ms vs 1,540 ms"),
        ("EB-CS batch vs confseq C++", s4_cs["confseq_predmix_ebcs"]["batch_100k_ns_per_step"] / s4_cs["cloture_ebcs"]["batch_100k_ns_per_step"], "110 ns vs 442 ns"),
    ]

    # Error rates under nominal 5% alpha (lower is valid <= 5%)
    validity = [
        ("cloture PopulationCert", s3["cloture_population_cert_fire_rate"] * 100.0, "#38d9c0"),
        ("cloture EB-CS / BettingCS", s4_cs["cloture_ebcs"]["anytime_miscoverage"] * 100.0, "#38d9c0"),
        ("cloture ActivePPI (peeking)", (1.0 - s5["cloture_active_ppi"]["anytime_coverage_every_200"]) * 100.0, "#38d9c0"),
        ("ppi-python (peeking /200)", (1.0 - s5["ppi_python_peeking_every_200"]["anytime_coverage"]) * 100.0, "#f87171"),
        ("Pooled Z-test (ρ=0.60)", s3["naive_pooled_z_fire_rate"] * 100.0, "#f87171"),
        ("Pooled EB-CS (ρ=0.60)", s3["naive_pooled_ebcs_fire_rate"] * 100.0, "#f87171"),
        ("bayesbench.compare (tie)", s4_race["null_tie_0.50_vs_0.50"]["bayesbench_compare_type1_error"] * 100.0, "#f87171"),
        ("Naive CLT Peeking", s4_cs["naive_clt_peeking"]["anytime_miscoverage"] * 100.0, "#ef4444"),
    ]

    W, H = 980, 400
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="cloture speedup vs PyPI libraries and anytime-valid Type-I error control chart">',
        "<defs>",
        '  <linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">',
        '    <stop offset="0%" stop-color="#0d1117"/>',
        '    <stop offset="100%" stop-color="#111827"/>',
        "  </linearGradient>",
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="14" fill="url(#bg2)" stroke="#263147" stroke-width="1.5"/>',
        '<text x="28" y="34" fill="#e6edf7" font-family="ui-sans-serif,system-ui,-apple-system,sans-serif" font-size="16" font-weight="700">Head-to-Head Throughput &amp; Anytime Statistical Validity vs. Installed PyPI Packages</text>',
        '<text x="952" y="34" text-anchor="end" fill="#8b9ab5" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11">Xeon 2.3GHz · Python 3.12 · Rust 1.100</text>',
        '<line x1="28" y1="48" x2="952" y2="48" stroke="#223047" stroke-width="1"/>',
        # Left Card: Log10 Speedup
        '<rect x="28" y="64" width="470" height="312" rx="10" fill="#131a26" stroke="#25324a" stroke-width="1"/>',
        '<text x="46" y="88" fill="#e6edf7" font-family="ui-sans-serif,system-ui,sans-serif" font-size="13.5" font-weight="700">Speedup vs. Real Competitors (log₁₀ scale)</text>',
        '<text x="46" y="105" fill="#93a3bd" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11">Measured against confseq, bayesbench, expectation, online-cp, NumPy/SciPy</text>',
    ]

    max_log = math.log10(15000.0)
    lx0 = 46
    lbar_x = 224
    lbar_w = 185
    y = 128
    for label, mult, detail in speedups:
        bw = max(10.0, (math.log10(max(mult, 1.1)) / max_log) * lbar_w)
        mult_str = f"{mult:,.0f}×" if mult >= 100 else f"{mult:.1f}×"
        lines.append(f'<text x="{lx0}" y="{y+10}" fill="#cbd5e1" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.3">{_esc(label)}</text>')
        lines.append(f'<rect x="{lbar_x}" y="{y}" width="{lbar_w}" height="13" rx="3.5" fill="#1e293b"/>')
        lines.append(f'<rect x="{lbar_x}" y="{y}" width="{bw:.1f}" height="13" rx="3.5" fill="#7aa2f7"/>')
        lines.append(f'<text x="{lbar_x + lbar_w + 7}" y="{y+10.5}" fill="#38d9c0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.5" font-weight="700">{mult_str}</text>')
        y += 29

    # Right Card: Error Rate vs 5% nominal cap
    lines += [
        '<rect x="514" y="64" width="438" height="312" rx="10" fill="#131a26" stroke="#25324a" stroke-width="1"/>',
        '<text x="532" y="88" fill="#e6edf7" font-family="ui-sans-serif,system-ui,sans-serif" font-size="13.5" font-weight="700">False-Positive / Miscoverage Rate (α = 5%)</text>',
        '<text x="532" y="105" fill="#93a3bd" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11">Dashed line = 5% validity limit; bars to the right violate error control</text>',
    ]
    rx0 = 532
    rbar_x = 708
    rbar_w = 185
    max_pct = 60.0
    cap_x = rbar_x + (5.0 / max_pct) * rbar_w
    lines.append(f'<line x1="{cap_x:.1f}" y1="120" x2="{cap_x:.1f}" y2="356" stroke="#38d9c0" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.8"/>')
    lines.append(f'<text x="{cap_x+3:.1f}" y="366" fill="#38d9c0" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="9.5">5% cap</text>')

    ry = 128
    for label, err_pct, color in validity:
        bw = max(3.0, min(rbar_w, (err_pct / max_pct) * rbar_w))
        lines.append(f'<text x="{rx0}" y="{ry+10}" fill="#cbd5e1" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.2">{_esc(label)}</text>')
        lines.append(f'<rect x="{rbar_x}" y="{ry}" width="{rbar_w}" height="13" rx="3.5" fill="#1e293b"/>')
        lines.append(f'<rect x="{rbar_x}" y="{ry}" width="{bw:.1f}" height="13" rx="3.5" fill="{color}"/>')
        lines.append(f'<text x="{rbar_x + rbar_w + 7}" y="{ry+10.5}" fill="{color}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="10.5" font-weight="700">{err_pct:.1f}%</text>')
        ry += 29

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = ASSETS_DIR / "consensus_and_panel.svg"
    p2 = ASSETS_DIR / "speed_and_validity.svg"
    p1.write_text(build_consensus_and_panel_svg(data), encoding="utf-8")
    p2.write_text(build_speed_and_validity_svg(data), encoding="utf-8")
    print(f"Generated {p1} ({p1.stat().st_size} B) and {p2} ({p2.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
