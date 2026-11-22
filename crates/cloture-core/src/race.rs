//! Anytime-valid multi-agent / model / prompt racing with FWER control,
//! LUCB adaptive allocation, paired variance cancellation, and arbitrary-dependence e-BH FDR control.
//!
//! - [`AgentRace`]: Eliminates provably suboptimal arms while controlling family-wise error rate
//!   (FWER <= `delta`) uniformly over all sample sizes and adaptive stopping/allocation rules.
//!   Supports both unpaired LUCB evaluation and paired-task evaluation (where shared prompt
//!   difficulty variance cancels via [`PairedEProcess`]).
//! - [`e_bh_discoveries`]: Exact e-BH multiple-testing procedure (Wang & Ramdas, JRSS-B 2022)
//!   guaranteeing FDR <= `alpha` under arbitrary dependence among e-values.

use crate::betting::{EmpiricalBernsteinCS, PairedEProcess};

/// Per-arm summary statistics inside an [`AgentRace`].
#[derive(Clone, Debug)]
pub struct ArmStat {
    pub arm: usize,
    pub n: u64,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    pub active: bool,
}

/// Snapshot of the current state of an [`AgentRace`].
#[derive(Clone, Debug)]
pub struct RaceStatus {
    pub finished: bool,
    pub winner: Option<usize>,
    pub active_arms: Vec<usize>,
    pub eliminated_arms: Vec<usize>,
    pub arm_stats: Vec<ArmStat>,
    pub total_evals: u64,
}

/// Flat upper-triangular pair index for `0 <= i < j < m`.
#[inline]
fn pair_index(i: usize, j: usize, m: usize) -> usize {
    debug_assert!(i < j && j < m);
    i * m - (i * (i + 1)) / 2 + (j - i - 1)
}

/// Anytime-valid multi-arm race comparing `M >= 2` candidate policies, models, or prompts
/// with exact finite-sample FWER <= `delta`.
#[derive(Clone, Debug)]
pub struct AgentRace {
    n_arms: usize,
    delta: f64,
    paired: bool,
    arms: Vec<EmpiricalBernsteinCS>,
    active: Vec<bool>,
    elim_scratch: Vec<bool>,
    pairs: Vec<PairedEProcess>,
    pair_log_thresh: f64,
    total_evals: u64,
    n_paired_rounds: u64,
}

impl AgentRace {
    /// Creates a new anytime-valid race over `n_arms >= 2` candidates with FWER <= `delta`.
    ///
    /// - When `paired == false`, each arm maintains an [`EmpiricalBernsteinCS`] at level
    ///   `delta / n_arms`, yielding simultaneous `(1 - delta)` coverage across all arms.
    /// - When `paired == true`, error budget is split cleanly between pairwise [`PairedEProcess`]
    ///   eliminations (`0.8 * delta`, threshold `2 * (M - 1) / (0.8 * delta)`) and marginal CS
    ///   eliminations (`0.2 * delta`, per-arm level `0.2 * delta / M`), proving FWER <= `delta`
    ///   both under unique-best-arm and global-tie nulls.
    pub fn new(n_arms: usize, delta: f64, paired: bool) -> Result<Self, &'static str> {
        if n_arms < 2 {
            return Err("n_arms must be at least 2");
        }
        let max_arms = if paired { 2_048 } else { 100_000 };
        if n_arms > max_arms {
            return Err("n_arms exceeds supported maximum");
        }
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return Err("delta must be finite and strictly in (0, 1)");
        }

        let (delta_cs, pair_log_thresh, pairs) = if paired {
            let delta_cs = 0.2 * delta;
            let delta_pair = 0.8 * delta;
            let thresh = (2.0 * ((n_arms - 1) as f64) / delta_pair).ln();
            let n_pairs = n_arms
                .checked_mul(n_arms - 1)
                .map(|v| v / 2)
                .ok_or("n_pairs overflow")?;
            let mut ps = Vec::with_capacity(n_pairs);
            // Individual pair alpha is unused for elimination (we compare against pair_log_thresh).
            let pair_alpha = (delta_pair / (2.0 * ((n_arms - 1) as f64))).clamp(1e-15, 0.5);
            for _ in 0..n_pairs {
                ps.push(PairedEProcess::new(pair_alpha, 1.0, true)?);
            }
            (delta_cs, thresh, ps)
        } else {
            (delta, f64::INFINITY, Vec::new())
        };

        let alpha_arm = delta_cs / (n_arms as f64);
        let mut arms = Vec::with_capacity(n_arms);
        for _ in 0..n_arms {
            arms.push(EmpiricalBernsteinCS::new(alpha_arm)?);
        }

        Ok(Self {
            n_arms,
            delta,
            paired,
            arms,
            active: vec![true; n_arms],
            elim_scratch: vec![false; n_arms],
            pairs,
            pair_log_thresh,
            total_evals: 0,
            n_paired_rounds: 0,
        })
    }

    /// Eliminates any active arm `j` whose upper CS bound is strictly below `max_{k active} L_k`.
    fn run_cs_elimination(&mut self) {
        let mut max_lower = f64::NEG_INFINITY;
        for i in 0..self.n_arms {
            if self.active[i] {
                let lo = self.arms[i].lower();
                if lo > max_lower {
                    max_lower = lo;
                }
            }
        }
        if !max_lower.is_finite() {
            return;
        }
        for i in 0..self.n_arms {
            if self.active[i] && self.arms[i].upper() < max_lower {
                self.active[i] = false;
            }
        }
    }

    /// Eliminates any active arm `j` that is strictly dominated in paired e-process evidence
    /// by another active arm `i` at `log E(i > j) >= pair_log_thresh`.
    fn run_paired_elimination(&mut self) {
        if !self.paired {
            return;
        }
        let m = self.n_arms;
        let thresh = self.pair_log_thresh;

        // Collect eliminations into preallocated scratch before mutating `active`.
        self.elim_scratch.fill(false);
        let mut any_elim = false;
        for i in 0..m {
            if !self.active[i] {
                continue;
            }
            for j in (i + 1)..m {
                if !self.active[j] {
                    continue;
                }
                let p = &self.pairs[pair_index(i, j, m)];
                if p.log_e_a_beats_b() >= thresh {
                    self.elim_scratch[j] = true;
                    any_elim = true;
                }
                if p.log_e_b_beats_a() >= thresh {
                    self.elim_scratch[i] = true;
                    any_elim = true;
                }
            }
        }

        if !any_elim {
            return;
        }

        let mut remaining = 0usize;
        for i in 0..m {
            if self.active[i] && !self.elim_scratch[i] {
                remaining += 1;
            }
        }

        if remaining >= 1 {
            for i in 0..m {
                if self.elim_scratch[i] {
                    self.active[i] = false;
                }
            }
        } else {
            // Only reachable on the <= delta failure event: retain highest empirical mean arm.
            let mut best = 0usize;
            let mut best_mean = f64::NEG_INFINITY;
            for i in 0..m {
                if self.active[i] && self.arms[i].mean() > best_mean {
                    best_mean = self.arms[i].mean();
                    best = i;
                }
            }
            for i in 0..m {
                self.active[i] = i == best;
            }
        }
    }

    /// Observes a single score in `[0, 1]` for `arm` and updates eliminations.
    /// Out-of-bounds arm indices or non-finite scores are safely ignored.
    pub fn observe_arm(&mut self, arm: usize, score_01: f64) -> RaceStatus {
        if arm >= self.n_arms || !score_01.is_finite() {
            return self.status();
        }
        self.arms[arm].update(score_01);
        self.total_evals += 1;
        self.run_cs_elimination();
        self.status()
    }

    /// Observes a paired evaluation across all `M` arms on the same task/prompt.
    ///
    /// When `paired == true`, updates all active pairwise [`PairedEProcess`] trackers in addition
    /// to per-arm [`EmpiricalBernsteinCS`] trackers, canceling shared task-difficulty variance.
    /// Slice length mismatches are safely ignored; individual non-finite entries are skipped.
    pub fn observe_paired(&mut self, scores_01: &[f64]) -> RaceStatus {
        if scores_01.len() != self.n_arms {
            return self.status();
        }
        let m = self.n_arms;
        let mut any_valid = false;
        for i in 0..m {
            let s = scores_01[i];
            if s.is_finite() {
                self.arms[i].update(s);
                self.total_evals += 1;
                any_valid = true;
            }
        }
        if !any_valid {
            return self.status();
        }

        if self.paired {
            for i in 0..m {
                if !self.active[i] || !scores_01[i].is_finite() {
                    continue;
                }
                let si = scores_01[i];
                for j in (i + 1)..m {
                    if !self.active[j] || !scores_01[j].is_finite() {
                        continue;
                    }
                    let idx = pair_index(i, j, m);
                    self.pairs[idx].push(si, scores_01[j]);
                }
            }
            self.n_paired_rounds += 1;
            self.run_paired_elimination();
        }

        self.run_cs_elimination();
        self.status()
    }

    /// LUCB adaptive allocation (Kalyanakrishnan et al. 2012):
    ///
    /// 1. Pulls any active arm with zero observations first.
    /// 2. Otherwise alternates between the empirical leader `arg max_{i active} mean_i`
    ///    and the highest-upper-bound challenger `arg max_{j active, j != leader} upper_j`.
    ///
    /// Strictly predictable (depends only on past observations), preserving exact FWER <= `delta`.
    pub fn next_arm_lucb(&self) -> usize {
        // Initialize every active arm with at least one pull.
        for i in 0..self.n_arms {
            if self.active[i] && self.arms[i].n() == 0 {
                return i;
            }
        }

        // Find empirical leader among active arms.
        let mut leader: Option<usize> = None;
        for i in 0..self.n_arms {
            if !self.active[i] {
                continue;
            }
            leader = match leader {
                None => Some(i),
                Some(best) => {
                    let mi = self.arms[i].mean();
                    let mb = self.arms[best].mean();
                    if mi > mb || (mi == mb && self.arms[i].n() < self.arms[best].n()) {
                        Some(i)
                    } else {
                        Some(best)
                    }
                }
            };
        }
        let leader = leader.unwrap_or(0);

        // Find highest-upper-bound challenger among remaining active arms.
        let mut challenger: Option<usize> = None;
        for j in 0..self.n_arms {
            if !self.active[j] || j == leader {
                continue;
            }
            challenger = match challenger {
                None => Some(j),
                Some(best) => {
                    let uj = self.arms[j].upper();
                    let ub = self.arms[best].upper();
                    if uj > ub
                        || (uj == ub && self.arms[j].mean() > self.arms[best].mean())
                        || (uj == ub
                            && self.arms[j].mean() == self.arms[best].mean()
                            && self.arms[j].n() < self.arms[best].n())
                    {
                        Some(j)
                    } else {
                        Some(best)
                    }
                }
            };
        }

        match challenger {
            None => leader,
            Some(c) => {
                if self.total_evals % 2 == 0 {
                    leader
                } else {
                    c
                }
            }
        }
    }

    /// Returns the pairwise e-value certifying `arm_a > arm_b` if `paired == true`
    /// and `arm_a != arm_b` are valid indices.
    pub fn paired_e_value(&self, arm_a: usize, arm_b: usize) -> Option<f64> {
        if !self.paired || arm_a >= self.n_arms || arm_b >= self.n_arms || arm_a == arm_b {
            return None;
        }
        if arm_a < arm_b {
            Some(self.pairs[pair_index(arm_a, arm_b, self.n_arms)].e_value_a_beats_b())
        } else {
            Some(self.pairs[pair_index(arm_b, arm_a, self.n_arms)].e_value_b_beats_a())
        }
    }

    #[inline]
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.active_count() <= 1
    }

    #[inline]
    pub fn winner(&self) -> Option<usize> {
        let mut found: Option<usize> = None;
        for (i, &a) in self.active.iter().enumerate() {
            if a {
                if found.is_some() {
                    return None;
                }
                found = Some(i);
            }
        }
        found
    }

    #[inline]
    pub fn total_evals(&self) -> u64 {
        self.total_evals
    }

    #[inline]
    pub fn n_arms(&self) -> usize {
        self.n_arms
    }

    #[inline]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    #[inline]
    pub fn is_paired(&self) -> bool {
        self.paired
    }

    /// Returns the complete race status snapshot.
    pub fn status(&self) -> RaceStatus {
        let mut active_arms = Vec::with_capacity(self.n_arms);
        let mut eliminated_arms = Vec::with_capacity(self.n_arms);
        let mut arm_stats = Vec::with_capacity(self.n_arms);

        for i in 0..self.n_arms {
            let cs = &self.arms[i];
            let is_active = self.active[i];
            if is_active {
                active_arms.push(i);
            } else {
                eliminated_arms.push(i);
            }
            arm_stats.push(ArmStat {
                arm: i,
                n: cs.n(),
                mean: cs.mean(),
                lower: cs.lower(),
                upper: cs.upper(),
                active: is_active,
            });
        }

        let winner = if active_arms.len() == 1 {
            Some(active_arms[0])
        } else {
            None
        };

        RaceStatus {
            finished: active_arms.len() <= 1,
            winner,
            active_arms,
            eliminated_arms,
            arm_stats,
            total_evals: self.total_evals,
        }
    }
}

/// Exact e-BH multiple-testing procedure (Wang & Ramdas, JRSS-B 2022).
///
/// Given nonnegative e-values `e_values[0..K]` (non-finite or negative entries are sanitized to
/// `0.0`) and target FDR level `alpha in (0, 1]`, finds the largest `k* in 1..=K` such that
/// `e_(k*) >= K / (alpha * k*)` (where `e_(1) >= ... >= e_(K)` are sorted descending) and
/// returns the sorted original 0-based indices of the top `k*` discoveries.
///
/// Guarantees `FDR <= (|H_0| / K) * alpha <= alpha` under **arbitrary dependence** among e-values.
pub fn e_bh_discoveries(e_values: &[f64], alpha: f64) -> Vec<usize> {
    let k_total = e_values.len();
    if k_total == 0 || !alpha.is_finite() || alpha <= 0.0 {
        return Vec::new();
    }
    let alpha_eff = alpha.min(1.0);

    let mut indexed: Vec<(usize, f64)> = e_values
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let clean = if e.is_nan() || e < 0.0 { 0.0 } else { e };
            (i, clean)
        })
        .collect();

    // Sort descending by e-value (ties broken deterministically by original index).
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let k_f = k_total as f64;
    let mut k_star = 0usize;
    for (rank_zero, &(_, e)) in indexed.iter().enumerate() {
        let rank = (rank_zero + 1) as f64;
        let threshold = k_f / (alpha_eff * rank);
        if e >= threshold {
            k_star = rank_zero + 1;
        }
    }

    if k_star == 0 {
        return Vec::new();
    }

    let mut discoveries: Vec<usize> = indexed[..k_star].iter().map(|&(idx, _)| idx).collect();
    discoveries.sort_unstable();
    discoveries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    #[test]
    fn test_race_validation_and_nan_safety() {
        assert!(AgentRace::new(1, 0.05, false).is_err());
        assert!(AgentRace::new(0, 0.05, true).is_err());
        assert!(AgentRace::new(3, 0.0, false).is_err());
        assert!(AgentRace::new(3, 1.0, false).is_err());
        assert!(AgentRace::new(3, f64::NAN, true).is_err());

        let mut race = AgentRace::new(3, 0.05, true).unwrap();
        race.observe_arm(99, 0.5);
        race.observe_arm(0, f64::NAN);
        race.observe_paired(&[0.5, 0.5]); // wrong length ignored
        assert_eq!(race.total_evals(), 0);
        assert!(!race.is_finished());
        assert_eq!(race.winner(), None);
    }

    #[test]
    fn test_lucb_unpaired_finds_best_arm_efficiently() {
        let means = [0.40, 0.55, 0.80, 0.30];
        let mut race = AgentRace::new(means.len(), 0.05, false).unwrap();
        let mut rng = Rng::from_seed(101);

        let mut status = race.status();
        while !status.finished && status.total_evals < 10_000 {
            let arm = race.next_arm_lucb();
            let x = if rng.next_f64() < means[arm] { 1.0 } else { 0.0 };
            status = race.observe_arm(arm, x);
        }
        assert!(status.finished, "LUCB did not finish in 10k evals");
        assert_eq!(status.winner, Some(2));
        // Weakest arm (0.30) should be eliminated early and receive far fewer pulls than top two.
        assert!(
            status.arm_stats[3].n < status.arm_stats[2].n,
            "LUCB failed to focus pulls: arm3={} vs arm2={}",
            status.arm_stats[3].n,
            status.arm_stats[2].n
        );
    }

    #[test]
    fn test_paired_race_cancels_prompt_difficulty_much_faster() {
        // Shared prompt difficulty U ~ Uniform(0, 0.75) dominates marginal variance,
        // while arm 1 consistently beats arms 0 and 2 by +0.15 on every task.
        let offsets = [0.05, 0.22, 0.00];
        let delta = 0.05;
        let mut rng_paired = Rng::from_seed(5555);
        let mut rng_unpaired = Rng::from_seed(5555);

        let mut race_paired = AgentRace::new(3, delta, true).unwrap();
        let mut rounds_paired = 0u64;
        while !race_paired.is_finished() && rounds_paired < 5000 {
            let diff = rng_paired.next_f64() * 0.75;
            let scores = [diff + offsets[0], diff + offsets[1], diff + offsets[2]];
            race_paired.observe_paired(&scores);
            rounds_paired += 1;
        }

        let mut race_unpaired = AgentRace::new(3, delta, false).unwrap();
        let mut rounds_unpaired = 0u64;
        while !race_unpaired.is_finished() && rounds_unpaired < 5000 {
            let diff = rng_unpaired.next_f64() * 0.75;
            let scores = [diff + offsets[0], diff + offsets[1], diff + offsets[2]];
            race_unpaired.observe_paired(&scores);
            rounds_unpaired += 1;
        }

        assert_eq!(race_paired.winner(), Some(1));
        assert_eq!(race_unpaired.winner(), Some(1));
        assert!(
            rounds_paired * 3 < rounds_unpaired,
            "expected paired race to be >=3x faster: paired={} rounds vs unpaired={} rounds",
            rounds_paired,
            rounds_unpaired
        );
    }

    #[test]
    fn test_race_fwer_under_global_tie_null_paired_and_unpaired() {
        let delta = 0.05;
        let trials = 300;
        let horizon = 250;
        let mut rng = Rng::from_seed(424242);

        for &paired in &[false, true] {
            let mut false_winners = 0usize;
            for _ in 0..trials {
                let mut race = AgentRace::new(3, delta, paired).unwrap();
                for _ in 0..horizon {
                    // All 3 arms have identical mean 0.5 with +/-1 pairwise swings.
                    let s0 = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
                    let s1 = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
                    let s2 = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
                    let st = race.observe_paired(&[s0, s1, s2]);
                    if st.winner.is_some() {
                        false_winners += 1;
                        break;
                    }
                }
            }
            let fwer = (false_winners as f64) / (trials as f64);
            assert!(
                fwer <= delta + 0.025,
                "FWER violated under tie (paired={}): fwer={} > delta={}",
                paired,
                fwer,
                delta
            );
        }
    }

    #[test]
    fn test_e_bh_discoveries_exact_and_fdr_control() {
        // K = 5, alpha = 0.1 => thresholds K / (alpha * k) = 50, 25, 16.67, 12.5, 10.
        // Sorted values: 60 (k=1 >= 50), 20 (k=2 < 25), 18 (k=3 >= 16.67!), 5 (k=4 < 12.5), 1 (k=5 < 10).
        // Largest k* is 3 -> rejects indices with values 60, 20, 18.
        let es = [1.0, 20.0, 5.0, 60.0, 18.0];
        let disc = e_bh_discoveries(&es, 0.1);
        assert_eq!(disc, vec![1, 3, 4]);

        // Edge cases: empty, NaN, inf, alpha <= 0, ties at boundary.
        assert!(e_bh_discoveries(&[], 0.05).is_empty());
        assert!(e_bh_discoveries(&[100.0], 0.0).is_empty());
        assert_eq!(e_bh_discoveries(&[f64::NAN, -5.0, f64::INFINITY], 0.05), vec![2]);
        // Ties at boundary: K=4, alpha=0.2 => k=2 thresh=10, k=3 thresh=6.67.
        // Values [7.0, 7.0, 7.0, 0.5] => k=3 satisfies 7.0 >= 6.67, all three 7.0s rejected!
        assert_eq!(e_bh_discoveries(&[7.0, 0.5, 7.0, 7.0], 0.2), vec![0, 2, 3]);

        // Monte Carlo FDR check under strong positive dependence (shared uniform U!).
        let alpha = 0.10;
        let k_hyp = 20;
        let n_null = 15;
        let trials = 2000;
        let mut rng = Rng::from_seed(7777);
        let mut sum_fdp = 0.0;

        for _ in 0..trials {
            let mut e_vals = vec![0.0; k_hyp];
            // Shared factor induces strong dependence across null e-values, each with E[e] = 1:
            // E = 0.5 * U^(-0.5) has E[E] = 1.0 on (0, 1].
            let u_shared = rng.next_f64().max(1e-12);
            for i in 0..n_null {
                let u = if i % 2 == 0 {
                    u_shared
                } else {
                    rng.next_f64().max(1e-12)
                };
                e_vals[i] = 0.5 / u.sqrt();
            }
            for i in n_null..k_hyp {
                e_vals[i] = 40.0; // true alternatives
            }
            let r = e_bh_discoveries(&e_vals, alpha);
            if !r.is_empty() {
                let false_disc = r.iter().filter(|&&idx| idx < n_null).count();
                sum_fdp += (false_disc as f64) / (r.len() as f64);
            }
        }
        let fdr = sum_fdp / (trials as f64);
        assert!(
            fdr <= alpha + 0.02,
            "e-BH FDR={} exceeded alpha={} under dependence",
            fdr,
            alpha
        );
    }
}
