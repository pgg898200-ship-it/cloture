//! Recall-Controlled Multi-Stage Agent Abort & Escalation Cascade (`AbortCascade`)
//! and Multi-Rate Crossing-Free Online Conformal Calibration (`OnlineConformal`).
//!
//! 1. [`AbortCascade`] calibrates per-stage abort thresholds `tau_0, ..., tau_{S-1}` with
//!    an exact finite-sample Conformal Risk Control (CRC) recall guarantee:
//!    - On moderate/large multi-stage datasets, splits positives into a design fold and a
//!      strictly held-out calibration fold, runs joint coordinate descent with pairwise
//!      budget-transfer moves to minimize failed-episode compute cost, and then performs
//!      exact 1-D monotone Conformal Risk Control recalibration on the held-out positives.
//!    - On single-stage or small-sample datasets, uses exact order-statistic / union-bound
//!      conformal calibration.
//! 2. [`OnlineConformal`] implements a 4-rate `AgACI` / `DtACI` expert bank
//!    (`gamma in [0.005, 0.02, 0.08, 0.20]`) aggregated with Fixed-Share Hedge weights over
//!    pinball loss, keeping empirical coverage tightly glued to `1 - alpha` under sudden
//!    distribution shift with O(1) zero-allocation updates.

use crate::math::Rng;

/// Evaluation metrics returned by [`AbortCascade::evaluate`].
#[derive(Clone, Debug)]
pub struct CascadeMetrics {
    /// Fraction of truly successful episodes that survived all stages without aborting.
    pub empirical_recall: f64,
    /// Fraction of total compute/tokens saved across all evaluated episodes.
    pub compute_savings_fraction: f64,
    /// Fraction of compute/tokens saved specifically on failed episodes.
    pub failed_compute_savings_fraction: f64,
    /// Fraction of all episodes aborted at each stage `s in 0..S`.
    pub abort_rate_per_stage: Vec<f64>,
    /// Total episodes evaluated.
    pub total_episodes: usize,
    /// Total successful episodes evaluated.
    pub positive_episodes: usize,
    /// Total failed episodes evaluated.
    pub negative_episodes: usize,
}

/// Recall-controlled multi-stage agent abort & escalation cascade.
#[derive(Clone, Debug)]
pub struct AbortCascade {
    thresholds: Vec<f64>,
    target_recall: f64,
    n_stages: usize,
    calibration_positives: usize,
    calibration_negatives: usize,
}

impl AbortCascade {
    /// Calibrate per-stage abort thresholds from labeled trajectories `(stage_scores, succeeded)`
    /// where higher `stage_scores[s]` indicates higher estimated probability of eventual success.
    ///
    /// Uses default stage savings `S - s` (earlier stages save strictly more remaining compute).
    /// For explicit per-stage compute costs, see [`AbortCascade::calibrate_with_costs`].
    pub fn calibrate<S: AsRef<[f64]>>(
        calibration: &[(S, bool)],
        n_stages: usize,
        target_recall: f64,
    ) -> Result<Self, &'static str> {
        let default_costs = vec![1.0; n_stages + 1];
        Self::calibrate_with_costs(calibration, n_stages, target_recall, &default_costs)
    }

    /// Calibrate per-stage abort thresholds with explicit per-stage compute/token costs.
    ///
    /// Cost interpretation:
    /// - If `stage_costs.len() == n_stages + 1`: checkpoint `s` sits after stage `s`, so
    ///   aborting at stage `s` incurs `sum(stage_costs[0..=s])` and saves `sum(stage_costs[s+1..])`.
    /// - If `stage_costs.len() == n_stages` and `n_stages > 1`: aborting at stage `s` incurs
    ///   `sum(stage_costs[0..=s])` and saves `sum(stage_costs[s+1..n_stages])`.
    /// - If `stage_costs.len() == 1` and `n_stages == 1`: single pre-execution gate saving `stage_costs[0]`.
    pub fn calibrate_with_costs<S: AsRef<[f64]>>(
        calibration: &[(S, bool)],
        n_stages: usize,
        target_recall: f64,
        stage_costs: &[f64],
    ) -> Result<Self, &'static str> {
        if n_stages == 0 {
            return Err("n_stages must be >= 1");
        }
        if !target_recall.is_finite() || target_recall <= 0.0 || target_recall > 1.0 {
            return Err("target_recall must be finite and in (0, 1]");
        }
        if stage_costs.len() < n_stages {
            return Err("stage_costs length must be >= n_stages");
        }
        if stage_costs.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err("stage_costs must be finite and non-negative");
        }

        let mut positives: Vec<Vec<f64>> = Vec::new();
        let mut negatives: Vec<Vec<f64>> = Vec::new();

        for (scores_ref, succeeded) in calibration {
            let scores = scores_ref.as_ref();
            if scores.len() < n_stages {
                continue;
            }
            if scores[..n_stages].iter().any(|v| !v.is_finite()) {
                continue;
            }
            if *succeeded {
                positives.push(scores[..n_stages].to_vec());
            } else {
                negatives.push(scores[..n_stages].to_vec());
            }
        }

        let n_pos = positives.len();
        let n_neg = negatives.len();
        if n_pos == 0 {
            return Err("calibration data must contain at least one valid positive episode");
        }

        let required_full = ((n_pos + 1) as f64 * target_recall).ceil() as usize;
        if required_full > n_pos {
            // Insufficient calibration positives to certify target_recall with any finite cutoff:
            // guarantee 100% recall by never aborting.
            return Ok(Self {
                thresholds: vec![f64::NEG_INFINITY; n_stages],
                target_recall,
                n_stages,
                calibration_positives: n_pos,
                calibration_negatives: n_neg,
            });
        }

        // Exact single-stage conformal order statistic.
        if n_stages == 1 {
            let mut s0: Vec<f64> = positives.iter().map(|p| p[0]).collect();
            s0.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = n_pos - required_full;
            return Ok(Self {
                thresholds: vec![s0[idx]],
                target_recall,
                n_stages,
                calibration_positives: n_pos,
                calibration_negatives: n_neg,
            });
        }

        // Compute incurred cost for aborting at each stage s in 0..n_stages, plus full run cost.
        let abort_cost = compute_abort_costs(n_stages, stage_costs);
        let full_cost = *abort_cost.last().unwrap_or(&1.0);
        let stage_savings: Vec<f64> = (0..n_stages)
            .map(|s| (full_cost - abort_cost[s]).max(0.0))
            .collect();

        // Check whether we can do exact split-conformal joint optimization:
        // hold out >= 75% of positives for exact 1-D CRC recalibration (low conditional
        // coverage variance) while using up to 200 design positives + all negatives for
        // joint coordinate descent.
        let n_design = (n_pos / 4).clamp(12, 200);
        let can_split = if !negatives.is_empty() && n_pos >= 40 && n_pos > n_design {
            let n_cal = n_pos - n_design;
            let req_cal = ((n_cal + 1) as f64 * target_recall).ceil() as usize;
            req_cal <= n_cal && n_cal >= 20
        } else {
            false
        };

        let thresholds = if can_split {
            // Deterministic shuffle of positives to prevent order bias in the split.
            let mut perm: Vec<usize> = (0..n_pos).collect();
            let mut rng = Rng::from_seed(0xC107_05E0_CAFE_BABEu64 ^ (n_pos as u64));
            rng.shuffle(&mut perm);

            let pos_design: Vec<&[f64]> = perm[..n_design]
                .iter()
                .map(|&i| positives[i].as_slice())
                .collect();
            let pos_cal: Vec<&[f64]> = perm[n_design..]
                .iter()
                .map(|&i| positives[i].as_slice())
                .collect();
            let neg_slices: Vec<&[f64]> = negatives.iter().map(|v| v.as_slice()).collect();

            // Phase 1: Joint coordinate descent + pairwise budget transfer on (pos_design, neg).
            let tau_design = optimize_thresholds_coordinate_descent(
                &pos_design,
                &neg_slices,
                n_stages,
                target_recall,
                &abort_cost,
                &stage_savings,
            );

            // Compute stage score scales using ONLY design positives + negatives (independent of pos_cal).
            let stage_scales = compute_stage_scales(&pos_design, &neg_slices, n_stages);

            // Phase 2: Exact 1-D monotone Conformal Risk Control recalibration on held-out pos_cal.
            recalibrate_crc_1d(&tau_design, &stage_scales, &pos_cal, target_recall)
        } else {
            // Exact finite-sample union-bound conformal allocation over stages proportional to savings.
            calibrate_union_bound(&positives, n_stages, target_recall, &stage_savings)
        };

        Ok(Self {
            thresholds,
            target_recall,
            n_stages,
            calibration_positives: n_pos,
            calibration_negatives: n_neg,
        })
    }

    /// Number of stages in the cascade.
    #[inline]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Configured target recall `1 - alpha`.
    #[inline]
    pub fn target_recall(&self) -> f64 {
        self.target_recall
    }

    /// Calibrated per-stage abort thresholds `[tau_0, ..., tau_{S-1}]`.
    #[inline]
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    /// Number of positive episodes used during calibration.
    #[inline]
    pub fn calibration_positives(&self) -> usize {
        self.calibration_positives
    }

    /// Number of negative episodes used during calibration.
    #[inline]
    pub fn calibration_negatives(&self) -> usize {
        self.calibration_negatives
    }

    /// Runtime O(1) decision: returns `true` if the episode should abort at `stage`.
    ///
    /// - Returns `false` if `stage >= n_stages` or if `score` is non-finite (conservative
    ///   preservation of recall).
    #[inline]
    pub fn should_abort(&self, stage: usize, score: f64) -> bool {
        if stage >= self.n_stages || !score.is_finite() {
            return false;
        }
        score < self.thresholds[stage]
    }

    /// Evaluate calibrated cascade on labeled test trajectories `(stage_scores, succeeded, stage_costs)`.
    pub fn evaluate(&self, trajectories: &[(Vec<f64>, bool, Vec<f64>)]) -> CascadeMetrics {
        let mut total_episodes = 0usize;
        let mut positive_episodes = 0usize;
        let mut negative_episodes = 0usize;
        let mut survived_positives = 0usize;

        let mut total_full_cost = 0.0_f64;
        let mut total_saved_cost = 0.0_f64;
        let mut neg_full_cost = 0.0_f64;
        let mut neg_saved_cost = 0.0_f64;

        let mut aborts_per_stage = vec![0usize; self.n_stages];

        for (scores, succeeded, costs) in trajectories {
            total_episodes += 1;
            let stages = self.n_stages.min(scores.len());

            let mut aborted_stage: Option<usize> = None;
            for s in 0..stages {
                if self.should_abort(s, scores[s]) {
                    aborted_stage = Some(s);
                    aborts_per_stage[s] += 1;
                    break;
                }
            }

            if *succeeded {
                positive_episodes += 1;
                if aborted_stage.is_none() {
                    survived_positives += 1;
                }
            } else {
                negative_episodes += 1;
            }

            let (full_c, saved_c) = episode_cost_and_savings(self.n_stages, costs, aborted_stage);
            total_full_cost += full_c;
            total_saved_cost += saved_c;
            if !*succeeded {
                neg_full_cost += full_c;
                neg_saved_cost += saved_c;
            }
        }

        let empirical_recall = if positive_episodes > 0 {
            survived_positives as f64 / positive_episodes as f64
        } else {
            1.0
        };
        let compute_savings_fraction = if total_full_cost > 0.0 {
            (total_saved_cost / total_full_cost).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let failed_compute_savings_fraction = if neg_full_cost > 0.0 {
            (neg_saved_cost / neg_full_cost).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let abort_rate_per_stage = if total_episodes > 0 {
            aborts_per_stage
                .into_iter()
                .map(|c| c as f64 / total_episodes as f64)
                .collect()
        } else {
            vec![0.0; self.n_stages]
        };

        CascadeMetrics {
            empirical_recall,
            compute_savings_fraction,
            failed_compute_savings_fraction,
            abort_rate_per_stage,
            total_episodes,
            positive_episodes,
            negative_episodes,
        }
    }
}

/// Returns a vector of length `n_stages + 1`:
/// - `out[s]` for `s < n_stages` = cost incurred if aborted at stage `s`.
/// - `out[n_stages]` = full cost if the episode survives all stages.
fn compute_abort_costs(n_stages: usize, stage_costs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_stages + 1];
    if n_stages == 1 && stage_costs.len() == 1 {
        out[0] = 0.0;
        out[1] = stage_costs[0];
        return out;
    }
    let mut cum = 0.0_f64;
    for s in 0..n_stages {
        cum += stage_costs[s];
        out[s] = cum;
    }
    let mut total = cum;
    for &c in stage_costs.iter().skip(n_stages) {
        total += c;
    }
    out[n_stages] = total;
    out
}

/// Compute `(full_cost, saved_cost)` for a single evaluated trajectory.
fn episode_cost_and_savings(
    n_stages: usize,
    costs: &[f64],
    aborted_stage: Option<usize>,
) -> (f64, f64) {
    if costs.is_empty() {
        let full = n_stages as f64;
        let saved = match aborted_stage {
            Some(s) if n_stages == 1 && s == 0 => 1.0,
            Some(s) if s + 1 < n_stages => (n_stages - 1 - s) as f64,
            _ => 0.0,
        };
        return (full, saved);
    }

    let sanitize = |v: f64| if v.is_finite() && v >= 0.0 { v } else { 0.0 };
    let full: f64 = costs.iter().copied().map(sanitize).sum();

    let saved = match aborted_stage {
        None => 0.0,
        Some(s) => {
            if n_stages == 1 && costs.len() == 1 && s == 0 {
                sanitize(costs[0])
            } else if s + 1 < costs.len() {
                costs[(s + 1)..].iter().copied().map(sanitize).sum()
            } else {
                0.0
            }
        }
    };
    (full, saved)
}

/// Evaluate `(surviving_positives, total_negative_cost)` under thresholds `tau`.
#[inline]
fn eval_thresholds(
    pos: &[&[f64]],
    neg: &[&[f64]],
    tau: &[f64],
    abort_cost: &[f64],
) -> (usize, f64) {
    let s_len = tau.len();
    let mut surv_pos = 0usize;
    for p in pos {
        let mut ok = true;
        for m in 0..s_len {
            if p[m] < tau[m] {
                ok = false;
                break;
            }
        }
        if ok {
            surv_pos += 1;
        }
    }

    let full_cost = abort_cost[s_len];
    let mut neg_cost = 0.0_f64;
    for n in neg {
        let mut c = full_cost;
        for m in 0..s_len {
            if n[m] < tau[m] {
                c = abort_cost[m];
                break;
            }
        }
        neg_cost += c;
    }

    (surv_pos, neg_cost)
}

/// Build a deduplicated, sorted list of candidate thresholds for stage `m`.
fn stage_candidates(pos: &[&[f64]], neg: &[&[f64]], stage: usize) -> Vec<f64> {
    let mut all: Vec<f64> = Vec::with_capacity(pos.len() + neg.len());
    for p in pos {
        all.push(p[stage]);
    }
    for n in neg {
        all.push(n[stage]);
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut cands = Vec::with_capacity(80);
    cands.push(f64::NEG_INFINITY);

    if all.is_empty() {
        return cands;
    }

    // Include up to 64 evenly spaced order statistics across positives + negatives,
    // plus slightly-above-max to allow aggressive abort if stage is informative.
    let max_grid = 64usize;
    if all.len() <= max_grid {
        for &v in &all {
            if cands.last().copied().unwrap_or(f64::NEG_INFINITY) < v {
                cands.push(v);
            }
        }
    } else {
        for g in 0..max_grid {
            let idx = g * (all.len() - 1) / (max_grid - 1);
            let v = all[idx];
            if cands.last().copied().unwrap_or(f64::NEG_INFINITY) < v {
                cands.push(v);
            }
        }
    }
    // Also add exact low positive order statistics so recall budget steps are finely resolved.
    let mut pos_s: Vec<f64> = pos.iter().map(|p| p[stage]).collect();
    pos_s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tail = pos_s.len().min(32);
    for &v in &pos_s[..tail] {
        cands.push(v);
    }
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cands.dedup_by(|a, b| (*a == *b) || (a.is_finite() && b.is_finite() && (*a - *b).abs() <= 1e-12));
    cands
}

/// Phase 1: Joint coordinate descent + pairwise recall-budget transfer minimizing
/// expected failed-episode compute cost subject to design recall >= `target_recall`.
fn optimize_thresholds_coordinate_descent(
    pos: &[&[f64]],
    neg: &[&[f64]],
    n_stages: usize,
    target_recall: f64,
    abort_cost: &[f64],
    stage_savings: &[f64],
) -> Vec<f64> {
    let min_surv = ((pos.len() as f64) * target_recall).ceil() as usize;
    let cands: Vec<Vec<f64>> = (0..n_stages)
        .map(|m| {
            if stage_savings[m] <= 0.0 {
                vec![f64::NEG_INFINITY]
            } else {
                stage_candidates(pos, neg, m)
            }
        })
        .collect();

    let mut tau = vec![f64::NEG_INFINITY; n_stages];
    let mut tau_idx = vec![0usize; n_stages];
    let (mut best_surv, mut best_cost) = eval_thresholds(pos, neg, &tau, abort_cost);

    // Order stages by descending potential savings so earlier high-value stages get priority.
    let mut stage_order: Vec<usize> = (0..n_stages).collect();
    stage_order.sort_by(|&a, &b| {
        stage_savings[b]
            .partial_cmp(&stage_savings[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Coordinate descent sweeps + pairwise budget transfers.
    for _sweep in 0..8 {
        let mut improved = false;

        // Single-stage coordinate sweeps.
        for &m in &stage_order {
            if cands[m].len() <= 1 {
                continue;
            }
            let orig_val = tau[m];
            let orig_idx = tau_idx[m];
            let mut local_best_idx = orig_idx;
            let mut local_best_cost = best_cost;
            let mut local_best_surv = best_surv;

            for (ci, &cand) in cands[m].iter().enumerate() {
                tau[m] = cand;
                let (surv, cost) = eval_thresholds(pos, neg, &tau, abort_cost);
                if surv >= min_surv {
                    let strictly_better = cost + 1e-12 < local_best_cost
                        || ((cost - local_best_cost).abs() <= 1e-12 && surv > local_best_surv);
                    if strictly_better {
                        local_best_cost = cost;
                        local_best_surv = surv;
                        local_best_idx = ci;
                    }
                }
            }

            if local_best_idx != orig_idx {
                tau[m] = cands[m][local_best_idx];
                tau_idx[m] = local_best_idx;
                best_cost = local_best_cost;
                best_surv = local_best_surv;
                improved = true;
            } else {
                tau[m] = orig_val;
            }
        }

        // Pairwise budget transfer: loosen stage `a` slightly to unlock tightening stage `b`.
        for a in 0..n_stages {
            if tau_idx[a] == 0 {
                continue;
            }
            for b in 0..n_stages {
                if a == b || cands[b].len() <= 1 {
                    continue;
                }
                let orig_a_idx = tau_idx[a];
                let orig_b_idx = tau_idx[b];
                let orig_a = tau[a];
                let orig_b = tau[b];

                let mut pair_best: Option<(usize, usize, f64, usize)> = None;
                let max_step_down = orig_a_idx.min(6);
                for down in 1..=max_step_down {
                    let new_a_idx = orig_a_idx - down;
                    tau[a] = cands[a][new_a_idx];
                    // Try tightening stage `b` above its current index.
                    for new_b_idx in (orig_b_idx + 1)..cands[b].len() {
                        tau[b] = cands[b][new_b_idx];
                        let (surv, cost) = eval_thresholds(pos, neg, &tau, abort_cost);
                        if surv >= min_surv {
                            let cur_best_cost =
                                pair_best.map(|x| x.2).unwrap_or(best_cost);
                            if cost + 1e-12 < cur_best_cost {
                                pair_best = Some((new_a_idx, new_b_idx, cost, surv));
                            }
                        } else {
                            // Since candidates are sorted ascending, higher tau[b] only reduces survival.
                            break;
                        }
                    }
                }

                if let Some((na, nb, nc, ns)) = pair_best {
                    tau_idx[a] = na;
                    tau_idx[b] = nb;
                    tau[a] = cands[a][na];
                    tau[b] = cands[b][nb];
                    best_cost = nc;
                    best_surv = ns;
                    improved = true;
                } else {
                    tau[a] = orig_a;
                    tau[b] = orig_b;
                }
            }
        }

        if !improved {
            break;
        }
    }

    tau
}

/// Compute per-stage robust score scale (std dev, fallback 1.0) strictly from design + negative data.
fn compute_stage_scales(pos_design: &[&[f64]], neg: &[&[f64]], n_stages: usize) -> Vec<f64> {
    let mut scales = vec![1.0_f64; n_stages];
    let total = pos_design.len() + neg.len();
    if total < 2 {
        return scales;
    }
    for m in 0..n_stages {
        let mut sum = 0.0_f64;
        for p in pos_design {
            sum += p[m];
        }
        for n in neg {
            sum += n[m];
        }
        let mean = sum / (total as f64);
        let mut var = 0.0_f64;
        for p in pos_design {
            let d = p[m] - mean;
            var += d * d;
        }
        for n in neg {
            let d = n[m] - mean;
            var += d * d;
        }
        let sd = (var / ((total - 1) as f64)).sqrt();
        if sd.is_finite() && sd > 1e-8 {
            scales[m] = sd;
        }
    }
    scales
}

/// Phase 2: Exact finite-sample 1-D monotone Conformal Risk Control recalibration on
/// strictly held-out positive episodes `pos_cal`.
///
/// Uses the 1-parameter monotone threshold family:
/// `tau_m(lambda) = tau_design[m] - lambda * scale[m]` for active stages (`tau_design[m] > -inf`).
/// An episode survives iff `max_m (tau_design[m] - s_m) / scale[m] <= lambda`.
fn recalibrate_crc_1d(
    tau_design: &[f64],
    stage_scales: &[f64],
    pos_cal: &[&[f64]],
    target_recall: f64,
) -> Vec<f64> {
    let n_stages = tau_design.len();
    let active_stages: Vec<usize> = (0..n_stages)
        .filter(|&m| tau_design[m].is_finite())
        .collect();
    if active_stages.is_empty() {
        return vec![f64::NEG_INFINITY; n_stages];
    }

    let n_cal = pos_cal.len();
    let k = ((n_cal + 1) as f64 * target_recall).ceil() as usize;
    if k > n_cal {
        return vec![f64::NEG_INFINITY; n_stages];
    }

    let mut nonconf: Vec<f64> = Vec::with_capacity(n_cal);
    for p in pos_cal {
        let mut r = f64::NEG_INFINITY;
        for &m in &active_stages {
            let v = (tau_design[m] - p[m]) / stage_scales[m];
            if v > r {
                r = v;
            }
        }
        nonconf.push(r);
    }

    nonconf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lambda_hat = nonconf[k - 1];

    let mut out = vec![f64::NEG_INFINITY; n_stages];
    for &m in &active_stages {
        // Subtract a microscopic 1e-12 * scale guard so IEEE-754 rounding never flips equality.
        out[m] = tau_design[m] - lambda_hat * stage_scales[m] - 1e-12 * stage_scales[m];
    }
    out
}

/// Exact finite-sample union-bound conformal allocation over stages for small-sample regimes.
fn calibrate_union_bound(
    positives: &[Vec<f64>],
    n_stages: usize,
    target_recall: f64,
    stage_savings: &[f64],
) -> Vec<f64> {
    let n_pos = positives.len();
    let alpha = 1.0 - target_recall;
    let total_savings: f64 = stage_savings.iter().sum();

    let mut thresholds = vec![f64::NEG_INFINITY; n_stages];
    let mut col = Vec::with_capacity(n_pos);

    for m in 0..n_stages {
        let share = if total_savings > 0.0 {
            stage_savings[m] / total_savings
        } else {
            1.0 / (n_stages as f64)
        };
        if share <= 0.0 {
            continue;
        }
        let alpha_m = alpha * share;
        let k_m = ((n_pos + 1) as f64 * (1.0 - alpha_m)).ceil() as usize;
        if k_m > n_pos {
            continue;
        }
        col.clear();
        for p in positives {
            col.push(p[m]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds[m] = col[n_pos - k_m];
    }

    thresholds
}

/// Default 4-rate step-size bank for [`OnlineConformal`] (`AgACI` / `DtACI`).
pub const ONLINE_CONFORMAL_GAMMAS: [f64; 4] = [0.005, 0.02, 0.08, 0.20];

/// Multi-Rate Crossing-Free Online Conformal Calibrator (`AgACI` / `DtACI`).
///
/// Maintains a 4-rate expert bank `gamma in [0.005, 0.02, 0.08, 0.20]` aggregated via
/// Fixed-Share Hedge weights over pinball loss. Adapts `alpha_t` online in `O(1)` time
/// with zero heap allocations so empirical coverage remains locked to `1 - alpha` even
/// across abrupt distribution shifts.
#[derive(Clone, Debug)]
pub struct OnlineConformal {
    target_alpha: f64,
    gammas: [f64; 4],
    expert_alphas: [f64; 4],
    weights: [f64; 4],
    eta: f64,
    fixed_share: f64,
    steps: u64,
    covered_count: u64,
}

impl OnlineConformal {
    /// Create a new `OnlineConformal` tracker for target miscoverage rate `target_alpha` in `(0, 1)`.
    pub fn new(target_alpha: f64) -> Result<Self, &'static str> {
        if !target_alpha.is_finite() || target_alpha <= 0.0 || target_alpha >= 1.0 {
            return Err("target_alpha must be finite and in (0, 1)");
        }
        Ok(Self {
            target_alpha,
            gammas: ONLINE_CONFORMAL_GAMMAS,
            expert_alphas: [target_alpha; 4],
            weights: [0.25; 4],
            eta: 4.0,
            fixed_share: 0.02,
            steps: 0,
            covered_count: 0,
        })
    }

    /// Target miscoverage rate `alpha`.
    #[inline]
    pub fn target_alpha(&self) -> f64 {
        self.target_alpha
    }

    /// Current crossing-free aggregated miscoverage level `alpha_t` clamped to `[0.0, 1.0]`.
    #[inline]
    pub fn alpha_t(&self) -> f64 {
        self.raw_alpha_t().clamp(0.0, 1.0)
    }

    /// Unclamped aggregated miscoverage level `sum_k w_k * alpha_t^{(k)}`.
    #[inline]
    pub fn raw_alpha_t(&self) -> f64 {
        self.weights[0] * self.expert_alphas[0]
            + self.weights[1] * self.expert_alphas[1]
            + self.weights[2] * self.expert_alphas[2]
            + self.weights[3] * self.expert_alphas[3]
    }

    /// Current adapted coverage target `1 - alpha_t` in `[0.0, 1.0]`.
    #[inline]
    pub fn target_coverage(&self) -> f64 {
        1.0 - self.alpha_t()
    }

    /// Current Hedge weights over the 4 rate experts.
    #[inline]
    pub fn weights(&self) -> &[f64; 4] {
        &self.weights
    }

    /// Current per-expert `alpha_t^{(k)}` values.
    #[inline]
    pub fn expert_alphas(&self) -> &[f64; 4] {
        &self.expert_alphas
    }

    /// Fixed step-size bank `[0.005, 0.02, 0.08, 0.20]`.
    #[inline]
    pub fn gammas(&self) -> &[f64; 4] {
        &self.gammas
    }

    /// Total online updates processed.
    #[inline]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Running empirical coverage fraction across all updates so far.
    #[inline]
    pub fn empirical_coverage(&self) -> f64 {
        if self.steps == 0 {
            1.0 - self.target_alpha
        } else {
            self.covered_count as f64 / self.steps as f64
        }
    }

    /// Exact `DtACI` / `AgACI` update given the realized conformal p-value `conformal_p` in `[0, 1]`.
    ///
    /// An observation is covered at level `alpha'` iff `conformal_p >= alpha'`.
    /// Evaluates exact counterfactual pinball loss for all 4 experts, updates Fixed-Share
    /// Hedge weights, steps each expert by `gamma_k * (alpha - err_k)`, and returns the
    /// newly adapted `alpha_{t+1}` in `[0, 1]`.
    #[inline]
    pub fn update_pvalue(&mut self, conformal_p: f64) -> f64 {
        if !conformal_p.is_finite() {
            return self.alpha_t();
        }
        let beta = conformal_p.clamp(0.0, 1.0);
        let alpha = self.target_alpha;

        // Track empirical coverage of the pre-update aggregated threshold.
        let agg_alpha = self.raw_alpha_t();
        self.steps += 1;
        if beta >= agg_alpha {
            self.covered_count += 1;
        }

        // 1. Evaluate pinball loss for each expert and update Fixed-Share Hedge weights.
        let mut losses = [0.0_f64; 4];
        let mut min_loss = f64::INFINITY;
        for k in 0..4 {
            let a_k = self.expert_alphas[k].clamp(0.0, 1.0);
            let l = pinball_loss(alpha, beta, a_k);
            losses[k] = l;
            if l < min_loss {
                min_loss = l;
            }
        }

        let mut w_sum = 0.0_f64;
        let mut tmp_w = [0.0_f64; 4];
        for k in 0..4 {
            let delta = losses[k] - min_loss;
            let w = if delta <= 0.0 {
                self.weights[k]
            } else {
                self.weights[k] * (-self.eta * delta).exp()
            };
            tmp_w[k] = w;
            w_sum += w;
        }
        if w_sum > 0.0 && w_sum.is_finite() {
            let inv = 1.0 / w_sum;
            let keep = 1.0 - self.fixed_share;
            let share = self.fixed_share * 0.25;
            for k in 0..4 {
                self.weights[k] = keep * (tmp_w[k] * inv) + share;
            }
        } else {
            self.weights = [0.25; 4];
        }

        // 2. Per-expert ACI gradient step on counterfactual miscoverage.
        for k in 0..4 {
            let err_k = if beta < self.expert_alphas[k] {
                1.0
            } else {
                0.0
            };
            let next_a = self.expert_alphas[k] + self.gammas[k] * (alpha - err_k);
            // Allow modest negative/above-1 excursion so exact ACI telescoping holds
            // while remaining bounded against pathological streams.
            self.expert_alphas[k] = next_a.clamp(-0.5, 1.5);
        }

        self.alpha_t()
    }

    /// Binary update when only the realized miscoverage indicator `miscovered` is observed.
    ///
    /// Maps `miscovered = true` to `beta = 0.0` and `miscovered = false` to `beta = 1.0`
    /// for pinball-loss Hedge weighting while stepping every expert with the realized error.
    #[inline]
    pub fn update_miscovered(&mut self, miscovered: bool) -> f64 {
        let alpha = self.target_alpha;
        let err = if miscovered { 1.0 } else { 0.0 };
        let beta = if miscovered { 0.0 } else { 1.0 };

        self.steps += 1;
        if !miscovered {
            self.covered_count += 1;
        }

        let mut losses = [0.0_f64; 4];
        let mut min_loss = f64::INFINITY;
        for k in 0..4 {
            let a_k = self.expert_alphas[k].clamp(0.0, 1.0);
            let l = pinball_loss(alpha, beta, a_k);
            losses[k] = l;
            if l < min_loss {
                min_loss = l;
            }
        }

        let mut w_sum = 0.0_f64;
        let mut tmp_w = [0.0_f64; 4];
        for k in 0..4 {
            let delta = losses[k] - min_loss;
            let w = if delta <= 0.0 {
                self.weights[k]
            } else {
                self.weights[k] * (-self.eta * delta).exp()
            };
            tmp_w[k] = w;
            w_sum += w;
        }
        if w_sum > 0.0 && w_sum.is_finite() {
            let inv = 1.0 / w_sum;
            let keep = 1.0 - self.fixed_share;
            let share = self.fixed_share * 0.25;
            for k in 0..4 {
                self.weights[k] = keep * (tmp_w[k] * inv) + share;
            }
        } else {
            self.weights = [0.25; 4];
        }

        for k in 0..4 {
            let next_a = self.expert_alphas[k] + self.gammas[k] * (alpha - err);
            self.expert_alphas[k] = next_a.clamp(-0.5, 1.5);
        }

        self.alpha_t()
    }
}

/// Pinball (check) loss `rho_alpha(beta, a)` bounded in `[0, 1]`.
#[inline]
fn pinball_loss(alpha: f64, beta: f64, a: f64) -> f64 {
    let diff = beta - a;
    if diff >= 0.0 {
        alpha * diff
    } else {
        (1.0 - alpha) * (-diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal(rng: &mut Rng) -> f64 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Generate synthetic 3-stage agent trajectories where score informativeness grows over stages.
    fn gen_trajectories(
        rng: &mut Rng,
        n: usize,
        p_success: f64,
    ) -> Vec<(Vec<f64>, bool, Vec<f64>)> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let succ = rng.next_f64() < p_success;
            let (s0, s1, s2) = if succ {
                (
                    sigmoid(1.6 + 0.8 * normal(rng)),
                    sigmoid(2.2 + 0.7 * normal(rng)),
                    sigmoid(2.8 + 0.6 * normal(rng)),
                )
            } else {
                (
                    sigmoid(-1.4 + 0.9 * normal(rng)),
                    sigmoid(-2.0 + 0.8 * normal(rng)),
                    sigmoid(-2.5 + 0.7 * normal(rng)),
                )
            };
            out.push((vec![s0, s1, s2], succ, vec![100.0, 200.0, 400.0]));
        }
        out
    }

    #[test]
    fn abort_cascade_validation_and_edge_cases() {
        let empty: Vec<(Vec<f64>, bool)> = Vec::new();
        assert!(AbortCascade::calibrate(&empty, 3, 0.90).is_err());
        assert!(AbortCascade::calibrate(&[(vec![0.5], true)], 0, 0.90).is_err());
        assert!(AbortCascade::calibrate(&[(vec![0.5], true)], 1, 0.0).is_err());
        assert!(AbortCascade::calibrate(&[(vec![0.5], true)], 1, 1.5).is_err());
        assert!(AbortCascade::calibrate(&[(vec![0.5], false)], 1, 0.90).is_err());

        // Insufficient positive samples to certify 0.95 recall (only 5 positives => ceil(6*0.95)=6 > 5):
        // must safely fall back to NEG_INFINITY (never abort => 100% recall).
        let tiny: Vec<(Vec<f64>, bool)> = (0..5).map(|i| (vec![i as f64, i as f64], true)).collect();
        let c = AbortCascade::calibrate(&tiny, 2, 0.95).unwrap();
        assert!(c.thresholds().iter().all(|&t| t == f64::NEG_INFINITY));
        assert!(!c.should_abort(0, -1e9));
        assert!(!c.should_abort(0, f64::NAN));
        assert!(!c.should_abort(99, 0.0));
    }

    #[test]
    fn abort_cascade_achieves_high_savings_and_preserves_recall() {
        let mut rng = Rng::from_seed(12345);
        let cal_full = gen_trajectories(&mut rng, 2000, 0.45);
        let cal: Vec<(Vec<f64>, bool)> = cal_full
            .iter()
            .map(|(s, y, _)| (s.clone(), *y))
            .collect();
        let test = gen_trajectories(&mut rng, 2000, 0.45);

        let target_recall = 0.92;
        let cascade =
            AbortCascade::calibrate_with_costs(&cal, 3, target_recall, &[100.0, 200.0, 400.0])
                .unwrap();

        let metrics = cascade.evaluate(&test);
        assert!(
            metrics.empirical_recall >= target_recall - 0.025,
            "empirical recall {:.4} dropped below target {:.4}",
            metrics.empirical_recall,
            target_recall
        );
        assert!(
            metrics.failed_compute_savings_fraction >= 0.70,
            "expected >= 70% compute savings on failed trajectories, got {:.4}",
            metrics.failed_compute_savings_fraction
        );
        assert!(
            metrics.compute_savings_fraction >= 0.35,
            "expected >= 35% overall compute savings, got {:.4}",
            metrics.compute_savings_fraction
        );
    }

    #[test]
    fn abort_cascade_exact_finite_sample_conformal_recall_monte_carlo() {
        // Verify exact finite-sample conformal guarantee across 100 independent calibration/test splits!
        let mut rng = Rng::from_seed(20261122);
        let target_recall = 0.90;
        let trials = 100;
        let mut recall_sum = 0.0;

        for _ in 0..trials {
            let cal_full = gen_trajectories(&mut rng, 250, 0.5);
            let cal: Vec<(Vec<f64>, bool)> = cal_full
                .iter()
                .map(|(s, y, _)| (s.clone(), *y))
                .collect();
            let test = gen_trajectories(&mut rng, 200, 0.5);

            let cascade = AbortCascade::calibrate(&cal, 3, target_recall).unwrap();
            let m = cascade.evaluate(&test);
            recall_sum += m.empirical_recall;
        }

        let avg_recall = recall_sum / (trials as f64);
        assert!(
            avg_recall >= target_recall - 0.005,
            "Monte Carlo expected test recall {:.4} violated conformal guarantee {:.4}",
            avg_recall,
            target_recall
        );
    }

    #[test]
    fn abort_cascade_single_stage_and_asymmetric_stage_optimization() {
        let mut rng = Rng::from_seed(9999);

        // Single-stage exact conformal quantile.
        let mut cal1 = Vec::new();
        for _ in 0..300 {
            let y = rng.next_f64() < 0.5;
            let s = if y {
                0.7 + 0.1 * normal(&mut rng)
            } else {
                0.2 + 0.1 * normal(&mut rng)
            };
            cal1.push((vec![s], y));
        }
        let c1 = AbortCascade::calibrate(&cal1, 1, 0.90).unwrap();
        let test1: Vec<(Vec<f64>, bool, Vec<f64>)> = (0..500)
            .map(|_| {
                let y = rng.next_f64() < 0.5;
                let s = if y {
                    0.7 + 0.1 * normal(&mut rng)
                } else {
                    0.2 + 0.1 * normal(&mut rng)
                };
                (vec![s], y, vec![50.0])
            })
            .collect();
        let m1 = c1.evaluate(&test1);
        assert!(m1.empirical_recall >= 0.87);
        assert!(m1.failed_compute_savings_fraction >= 0.80);

        // Asymmetric stages: stage 0 is pure noise, stage 1 strongly separates failures!
        // Joint coordinate descent should allocate budget to stage 1 rather than wasting it on stage 0.
        let mut cal_asym = Vec::new();
        for _ in 0..500 {
            let y = rng.next_f64() < 0.5;
            let s0 = normal(&mut rng); // identical noise for pos & neg!
            let s1 = if y {
                3.0 + 0.5 * normal(&mut rng)
            } else {
                -3.0 + 0.5 * normal(&mut rng)
            };
            let s2 = normal(&mut rng);
            cal_asym.push((vec![s0, s1, s2], y));
        }
        let c_asym =
            AbortCascade::calibrate_with_costs(&cal_asym, 3, 0.90, &[10.0, 20.0, 500.0]).unwrap();
        let test_asym: Vec<(Vec<f64>, bool, Vec<f64>)> = (0..500)
            .map(|_| {
                let y = rng.next_f64() < 0.5;
                let s0 = normal(&mut rng);
                let s1 = if y {
                    3.0 + 0.5 * normal(&mut rng)
                } else {
                    -3.0 + 0.5 * normal(&mut rng)
                };
                let s2 = normal(&mut rng);
                (vec![s0, s1, s2], y, vec![10.0, 20.0, 500.0])
            })
            .collect();
        let m_asym = c_asym.evaluate(&test_asym);
        assert!(
            m_asym.empirical_recall >= 0.88,
            "recall {:.4} too low",
            m_asym.empirical_recall
        );
        assert!(
            m_asym.failed_compute_savings_fraction >= 0.85,
            "joint optimization should abort at informative stage 1 and save stage 2 cost, got {:.4}",
            m_asym.failed_compute_savings_fraction
        );
    }

    #[test]
    fn online_conformal_stationary_and_sudden_shift_recovery() {
        assert!(OnlineConformal::new(0.0).is_err());
        assert!(OnlineConformal::new(1.0).is_err());
        assert!(OnlineConformal::new(f64::NAN).is_err());

        let target_alpha = 0.10;
        let mut rng = Rng::from_seed(314159);
        let mut oc = OnlineConformal::new(target_alpha).unwrap();

        // Phase 1 (stationary): calibrated scores produce Uniform(0, 1) conformal p-values.
        for _ in 0..2000 {
            let p = rng.next_f64();
            let a = oc.update_pvalue(p);
            assert!((0.0..=1.0).contains(&a));
        }
        assert!(
            (oc.empirical_coverage() - (1.0 - target_alpha)).abs() < 0.025,
            "stationary empirical coverage {:.4} drifted from {:.4}",
            oc.empirical_coverage(),
            1.0 - target_alpha
        );

        // Phase 2 (sudden distribution shift!): scores suddenly become much harder,
        // so conformal p-values shrink sharply by 5x (beta ~ U(0, 0.2) instead of U(0, 1)).
        let mut post_shift_covered = 0usize;
        let shift_steps = 2000;
        for _ in 0..shift_steps {
            let a_before = oc.raw_alpha_t();
            let shifted_p = rng.next_f64() * 0.2;
            if shifted_p >= a_before {
                post_shift_covered += 1;
            }
            oc.update_pvalue(shifted_p);
        }
        let shift_cov = post_shift_covered as f64 / shift_steps as f64;
        assert!(
            (shift_cov - (1.0 - target_alpha)).abs() < 0.03,
            "post-shift coverage {:.4} failed to track {:.4}",
            shift_cov,
            1.0 - target_alpha
        );
        // Adapted alpha_t should have dropped roughly 5x (~0.02 instead of 0.10)!
        assert!(
            oc.alpha_t() < 0.05,
            "adapted alpha_t {:.4} should tighten under severe shift",
            oc.alpha_t()
        );

        // Verify NaN p-values are safely ignored and binary mode tracks coverage too.
        let before_steps = oc.steps();
        oc.update_pvalue(f64::NAN);
        assert_eq!(oc.steps(), before_steps);

        let mut oc_bin = OnlineConformal::new(0.10).unwrap();
        let mut covered_bin = 0usize;
        for _ in 0..2000 {
            let a = oc_bin.alpha_t();
            // Simulate miscoverage probability proportional to current alpha_t.
            let miscov = rng.next_f64() < a;
            if !miscov {
                covered_bin += 1;
            }
            oc_bin.update_miscovered(miscov);
        }
        let cov_bin = covered_bin as f64 / 2000.0;
        assert!(
            (cov_bin - 0.90).abs() < 0.04,
            "binary OnlineConformal coverage {:.4} drifted from 0.90",
            cov_bin
        );
    }
}
