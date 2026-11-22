//! Anytime-valid consensus stopping and correlated-rollout certification.
//!
//! Provides three complementary sequential certification primitives:
//!
//! - [`AscdStopper`]: Grey-box distribution-valued anytime-valid consensus
//!   stopping (`ASC-D`) for known fixed category count `K >= 2`, supporting
//!   probability simplex vectors, one-hot votes, and confidence-weighted votes,
//!   with exact additive shard merging.
//! - [`CiteStopper`]: Category-agnostic unbounded-category anytime-valid mode
//!   stopping using discovery-order Kraft error allocation
//!   `w_r = 1 / (r * (r + 1))`, exact post-discovery intersection-union
//!   supermartingales, and exact deterministic finite-budget locking.
//! - [`PopulationCert`]: Realized-unit vs. population-level certification under
//!   within-unit correlated rollouts, combining exact online unbalanced one-way
//!   random-effects ANOVA, a prompt-level anytime-valid e-process, and an
//!   optimal two-stage rollout budget allocator.

use std::collections::HashMap;

use crate::math::{
    add_bet_grid_pair, fast_ln_1p_bounded, fnv1a_64, log_mean_exp_grid, log_mean_exp_grid_with_max,
    max_grid, CompSum, DEFAULT_BET_GRID, GRID_LEN, LN_GRID_LEN, LOG1M_GRID, LOG1P_GRID,
};

/// Scope of a sequential mode or population certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertScope {
    /// Certified for the realized unit (e.g., the rollout distribution of a single prompt).
    RealizedUnit,
    /// Certified for the population distribution across independent units.
    Population,
    /// Deterministically locked for a finite maximum budget (remaining samples cannot overturn leader).
    DeterministicBudget,
}

/// Decision snapshot returned after each update to a mode stopper.
#[derive(Clone, Debug)]
pub struct ModeDecision {
    /// True once either the anytime-valid statistical threshold or deterministic budget lock is reached.
    pub certified: bool,
    /// Certified winning category index (or discovered slot in [`CiteStopper`]), if `certified` is true.
    pub winner: Option<usize>,
    /// Current empirical leader index (or discovered slot), if at least one valid sample was observed.
    pub leader: Option<usize>,
    /// Total number of valid observations ingested so far.
    pub n_samples: u64,
    /// Log-evidence (minimum log-wealth across challengers) for the current leader.
    pub log_evidence: f64,
    /// Required log-threshold for statistical certification of the current leader.
    pub log_threshold: f64,
    /// Empirical per-sample margin `(score[leader] - score[runner_up]) / n_samples`.
    pub margin_estimate: f64,
    /// Certification scope of the decision.
    pub scope: CertScope,
    /// True when remaining samples in `max_budget` cannot overturn the current leader.
    pub deterministic_lock: bool,
}

// ============================================================================
// 1. AscdStopper — Grey-Box Distribution-Valued Anytime-Valid Consensus (ASC-D)
// ============================================================================

/// Anytime-valid consensus stopper for a known fixed set of `K >= 2` categories.
///
/// Tracks pairwise betting supermartingales in log-space across the fixed
/// 16-point geometric betting grid [`DEFAULT_BET_GRID`]. All per-observation
/// update paths perform zero heap allocations.
#[derive(Clone, Debug)]
pub struct AscdStopper {
    k: usize,
    delta: f64,
    max_budget: Option<u64>,
    log_threshold: f64,
    n_samples: u64,
    scores: Vec<f64>,
    scores_comp: Vec<f64>,
    log_wealth: Vec<[f64; GRID_LEN]>,
    scratch: Vec<f64>,
}

impl AscdStopper {
    /// Creates a new `AscdStopper` for `2 <= k <= 2048` categories at error level `delta in (0, 1)`.
    pub fn new(k: usize, delta: f64, max_budget: Option<u64>) -> Result<Self, &'static str> {
        if k < 2 || k > 2048 {
            return Err("k must be in 2..=2048 (use CiteStopper for unbounded categories)");
        }
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return Err("delta must be finite and strictly inside (0, 1)");
        }
        if let Some(0) = max_budget {
            return Err("max_budget must be >= 1 when specified");
        }
        // Union bound over all K arms guarantees FWER <= delta even under exact boundary ties (e.g. p = [0.5, 0.5]).
        let log_threshold = ((k as f64) / delta).ln();
        let kk = k
            .checked_mul(k)
            .ok_or("k * k overflows usize")?;
        Ok(Self {
            k,
            delta,
            max_budget,
            log_threshold,
            n_samples: 0,
            scores: vec![0.0; k],
            scores_comp: vec![0.0; k],
            log_wealth: vec![[0.0; GRID_LEN]; kk],
            scratch: vec![0.0; k],
        })
    }

    /// Number of categories `K`.
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Configured Type-I misidentification error tolerance `delta`.
    #[inline]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Statistical log-threshold `ln((K - 1) / delta)`.
    #[inline]
    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    /// Number of valid observations ingested.
    #[inline]
    pub fn n_samples(&self) -> u64 {
        self.n_samples
    }

    /// Compensated cumulative score for category `idx`.
    #[inline]
    pub fn score(&self, idx: usize) -> Option<f64> {
        if idx < self.k {
            Some(self.scores[idx] + self.scores_comp[idx])
        } else {
            None
        }
    }

    #[inline(always)]
    fn add_score(&mut self, idx: usize, x: f64) {
        let s = self.scores[idx];
        let t = s + x;
        if s >= x {
            self.scores_comp[idx] += (s - t) + x;
        } else {
            self.scores_comp[idx] += (x - t) + s;
        }
        self.scores[idx] = t;
    }

    /// Updates with a probability/score vector of length `k`.
    ///
    /// Invalid inputs (wrong length, any NaN/Inf entry, or non-positive total mass
    /// after clamping negative entries to zero) are safely ignored without mutating state.
    pub fn update(&mut self, probs: &[f64]) -> ModeDecision {
        let k = self.k;
        if probs.len() != k {
            return self.decision();
        }
        let mut sum = 0.0_f64;
        for (slot, &v) in self.scratch[..k].iter_mut().zip(probs.iter()) {
            if !v.is_finite() {
                return self.decision();
            }
            let c = if v > 0.0 { v } else { 0.0 };
            *slot = c;
            sum += c;
        }
        if !(sum > 0.0 && sum < f64::INFINITY) {
            return self.decision();
        }
        if sum == 1.0 {
            self.n_samples += 1;
            for a in 0..k {
                let pa = self.scratch[a];
                if pa > 0.0 {
                    self.add_score(a, pa);
                }
            }
        } else {
            let inv_sum = 1.0 / sum;
            if inv_sum.is_finite() {
                for slot in &mut self.scratch[..k] {
                    *slot = (*slot * inv_sum).clamp(0.0, 1.0);
                }
            } else {
                for slot in &mut self.scratch[..k] {
                    *slot = (*slot / sum).clamp(0.0, 1.0);
                }
            }
            self.n_samples += 1;
            for a in 0..k {
                let pa = self.scratch[a];
                if pa > 0.0 {
                    self.add_score(a, pa);
                }
            }
        }

        let scratch = &self.scratch[..k];
        let mut tail = &mut self.log_wealth[..k * k];
        for a in 0..k {
            let (row_a, rest) = tail.split_at_mut(k);
            tail = rest;
            let pa = scratch[a];
            for b in (a + 1)..k {
                // Both pa and scratch[b] are in [0.0, 1.0], so diff is guaranteed in [-1.0, 1.0].
                let diff = pa - scratch[b];
                if diff == 0.0 {
                    continue;
                }
                let cell_ab = &mut row_a[b];
                let cell_ba = &mut tail[(b - a - 1) * k + a];
                add_bet_grid_pair(cell_ab, cell_ba, diff);
            }
        }
        self.decision()
    }

    /// Updates with a matched black-box one-hot label `label in 0..k`.
    ///
    /// Uses precomputed `LOG1P_GRID` / `LOG1M_GRID` in `O(K * GRID_LEN)` time
    /// with zero transcendental calls and zero heap allocations.
    pub fn update_one_hot(&mut self, label: usize) -> ModeDecision {
        if label >= self.k {
            return self.decision();
        }
        self.n_samples += 1;
        self.add_score(label, 1.0);
        let k = self.k;
        let label_row = label * k;
        for j in 0..k {
            if j == label {
                continue;
            }
            let pos = &mut self.log_wealth[label_row + j];
            for g in 0..GRID_LEN {
                pos[g] += LOG1P_GRID[g];
            }
            let neg = &mut self.log_wealth[j * k + label];
            for g in 0..GRID_LEN {
                neg[g] += LOG1M_GRID[g];
            }
        }
        self.decision()
    }

    /// Updates with a confidence-weighted observation `theta = confidence * e_label`,
    /// where finite `confidence` is clamped to `[0.0, 1.0]`.
    ///
    /// Out-of-range labels or non-finite confidences are safely ignored without mutating state.
    pub fn update_confidence_weighted(&mut self, label: usize, confidence: f64) -> ModeDecision {
        if label >= self.k || !confidence.is_finite() {
            return self.decision();
        }
        let c = confidence.clamp(0.0, 1.0);
        self.n_samples += 1;
        if c == 0.0 {
            return self.decision();
        }
        self.add_score(label, c);
        let mut pos_inc = [0.0_f64; GRID_LEN];
        let mut neg_inc = [0.0_f64; GRID_LEN];
        if c == 1.0 {
            pos_inc = LOG1P_GRID;
            neg_inc = LOG1M_GRID;
        } else {
            for g in 0..GRID_LEN {
                let lc = DEFAULT_BET_GRID[g] * c;
                pos_inc[g] = fast_ln_1p_bounded(lc);
                neg_inc[g] = fast_ln_1p_bounded(-lc);
            }
        }
        let k = self.k;
        let label_row = label * k;
        for j in 0..k {
            if j == label {
                continue;
            }
            let pos = &mut self.log_wealth[label_row + j];
            for g in 0..GRID_LEN {
                pos[g] += pos_inc[g];
            }
            let neg = &mut self.log_wealth[j * k + label];
            for g in 0..GRID_LEN {
                neg[g] += neg_inc[g];
            }
        }
        self.decision()
    }

    /// Evaluates the current stopping decision without mutating state.
    pub fn decision(&self) -> ModeDecision {
        if self.n_samples == 0 {
            return ModeDecision {
                certified: false,
                winner: None,
                leader: None,
                n_samples: 0,
                log_evidence: 0.0,
                log_threshold: self.log_threshold,
                margin_estimate: 0.0,
                scope: CertScope::RealizedUnit,
                deterministic_lock: false,
            };
        }
        let k = self.k;
        let mut leader = 0usize;
        let mut runner_up = 1usize;
        let mut leader_score = self.scores[0] + self.scores_comp[0];
        let mut second_score = f64::NEG_INFINITY;
        for a in 1..k {
            let s = self.scores[a] + self.scores_comp[a];
            if s > leader_score {
                second_score = leader_score;
                runner_up = leader;
                leader_score = s;
                leader = a;
            } else if s > second_score {
                second_score = s;
                runner_up = a;
            }
        }
        if !second_score.is_finite() {
            second_score = 0.0;
        }

        let row_offset = leader * k;
        // Evaluate the closest challenger (runner_up) first so the exact lower bound
        // `max_grid(lw) - LN_GRID_LEN <= log_mean_exp_grid(lw)` prunes dominated challengers.
        let mut min_log_w = log_mean_exp_grid(&self.log_wealth[row_offset + runner_up]);
        for j in 0..k {
            if j == leader || j == runner_up {
                continue;
            }
            let arr = &self.log_wealth[row_offset + j];
            let m = max_grid(arr);
            if m.is_finite() && (m - LN_GRID_LEN) >= min_log_w {
                continue;
            }
            let lw = log_mean_exp_grid_with_max(arr, m);
            if lw < min_log_w {
                min_log_w = lw;
            }
        }

        let stat_certified = min_log_w >= self.log_threshold;
        let det_lock = if let Some(n_max) = self.max_budget {
            let rem = n_max.saturating_sub(self.n_samples) as f64;
            let eps = 1e-12 * (n_max.max(self.n_samples) as f64).max(1.0);
            (leader_score - second_score) > rem + eps
        } else {
            false
        };

        let (certified, winner, scope) = if stat_certified {
            (true, Some(leader), CertScope::RealizedUnit)
        } else if det_lock {
            (true, Some(leader), CertScope::DeterministicBudget)
        } else {
            (false, None, CertScope::RealizedUnit)
        };

        ModeDecision {
            certified,
            winner,
            leader: Some(leader),
            n_samples: self.n_samples,
            log_evidence: min_log_w,
            log_threshold: self.log_threshold,
            margin_estimate: (leader_score - second_score) / ((self.n_samples as i64) as f64),
            scope,
            deterministic_lock: det_lock,
        }
    }

    /// Resets all accumulated counts, scores, and log-wealth to initial state without reallocating.
    pub fn reset(&mut self) {
        self.n_samples = 0;
        for s in &mut self.scores {
            *s = 0.0;
        }
        for c in &mut self.scores_comp {
            *c = 0.0;
        }
        for cell in &mut self.log_wealth {
            *cell = [0.0; GRID_LEN];
        }
    }

    /// Exact additive merge of another `AscdStopper` with the same `k`.
    ///
    /// Valid because the fixed geometric betting grid produces order-invariant
    /// additive log-wealth factors across independent rollout shards.
    ///
    /// # Panics
    /// Panics if `self.k != other.k`.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.k, other.k, "AscdStopper::merge requires identical k");
        self.n_samples = self.n_samples.saturating_add(other.n_samples);
        for a in 0..self.k {
            self.add_score(a, other.scores[a]);
            self.add_score(a, other.scores_comp[a]);
        }
        for (dst, src) in self.log_wealth.iter_mut().zip(other.log_wealth.iter()) {
            for g in 0..GRID_LEN {
                dst[g] += src[g];
            }
        }
    }
}

// ============================================================================
// 2. CiteStopper — Unbounded-Category Anytime-Valid Mode Stopper (CITE)
// ============================================================================

#[derive(Clone, Debug)]
struct CiteCategory {
    label_id: u64,
    total_count: u64,
    total_weight: CompSum,
    /// Snapshot of `total_count[b]` for `b in 0..idx` taken at the moment `idx` was discovered.
    snap_counts: Vec<u64>,
    /// Post-discovery positive log-wealth (populated in weighted mode).
    pos_lw: [f64; GRID_LEN],
    /// Cumulative negative log-wealth including discovery step (populated in weighted mode).
    neg_lw: [f64; GRID_LEN],
    /// Snapshot of `neg_lw[b]` for `b in 0..idx` at discovery time (populated in weighted mode).
    neg_snap: Vec<[f64; GRID_LEN]>,
}

/// Category-agnostic anytime-valid mode stopper for unknown or unbounded label spaces.
///
/// Allocates Kraft error weights `w_r = 1 / (r * (r + 1))` by discovery rank `r = 1, 2, ...`,
/// tracks exact strictly-post-discovery supermartingales against every seen and unseen
/// challenger simultaneously, and optionally certifies deterministic finite-budget locks.
#[derive(Clone, Debug)]
pub struct CiteStopper {
    delta: f64,
    max_budget: Option<u64>,
    n_samples: u64,
    has_fractional_weights: bool,
    categories: Vec<CiteCategory>,
    id_to_slot: HashMap<u64, usize>,
    str_to_id: HashMap<String, u64>,
    slot_strings: Vec<Option<String>>,
}

impl CiteStopper {
    /// Creates a new unbounded-category `CiteStopper` at error tolerance `delta in (0, 1)`.
    pub fn new(delta: f64, max_budget: Option<u64>) -> Result<Self, &'static str> {
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return Err("delta must be finite and strictly inside (0, 1)");
        }
        if let Some(0) = max_budget {
            return Err("max_budget must be >= 1 when specified");
        }
        Ok(Self {
            delta,
            max_budget,
            n_samples: 0,
            has_fractional_weights: false,
            categories: Vec::new(),
            id_to_slot: HashMap::new(),
            str_to_id: HashMap::new(),
            slot_strings: Vec::new(),
        })
    }

    /// Configured error tolerance `delta`.
    #[inline]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Total number of valid observations ingested.
    #[inline]
    pub fn n_samples(&self) -> u64 {
        self.n_samples
    }

    /// Number of distinct categories discovered so far.
    #[inline]
    pub fn num_discovered(&self) -> usize {
        self.categories.len()
    }

    /// Returns the `u64` label ID assigned to discovered slot `slot`.
    #[inline]
    pub fn label_of(&self, slot: usize) -> Option<u64> {
        self.categories.get(slot).map(|c| c.label_id)
    }

    /// Returns the interned string label for discovered slot `slot`, if registered via `update_str`.
    #[inline]
    pub fn str_of(&self, slot: usize) -> Option<&str> {
        self.slot_strings.get(slot).and_then(|o| o.as_deref())
    }

    /// Returns the unweighted observation count for discovered slot `slot`.
    #[inline]
    pub fn count_of(&self, slot: usize) -> Option<u64> {
        self.categories.get(slot).map(|c| c.total_count)
    }

    /// Returns the cumulative weight for discovered slot `slot`.
    #[inline]
    pub fn weight_of(&self, slot: usize) -> Option<f64> {
        self.categories.get(slot).map(|c| c.total_weight.value())
    }

    /// Returns the `u64` label ID of the certified winner, if certified.
    pub fn winner_label(&self) -> Option<u64> {
        self.decision().winner.and_then(|s| self.label_of(s))
    }

    /// Returns the `u64` label ID of the current leader, if any samples were observed.
    pub fn leader_label(&self) -> Option<u64> {
        self.decision().leader.and_then(|s| self.label_of(s))
    }

    /// Returns the interned string of the certified winner, if certified and string-keyed.
    pub fn winner_str(&self) -> Option<&str> {
        self.decision().winner.and_then(|s| self.str_of(s))
    }

    /// Returns the interned string of the current leader, if string-keyed.
    pub fn leader_str(&self) -> Option<&str> {
        self.decision().leader.and_then(|s| self.str_of(s))
    }

    /// Kraft log-threshold `ln(r * (r + 1) / delta)` for 0-indexed slot `idx` (rank `r = idx + 1`).
    #[inline]
    pub fn slot_log_threshold(&self, idx: usize) -> f64 {
        let r = (idx + 1) as f64;
        (r * (r + 1.0) / self.delta).ln()
    }

    /// Promotes existing unweighted categories to exact weighted log-wealth storage.
    fn promote_to_weighted(&mut self) {
        if self.has_fractional_weights {
            return;
        }
        self.has_fractional_weights = true;
        for cat in &mut self.categories {
            let post_self = cat.total_count.saturating_sub(1) as f64;
            let total_c = cat.total_count as f64;
            for g in 0..GRID_LEN {
                cat.pos_lw[g] = post_self * LOG1P_GRID[g];
                cat.neg_lw[g] = total_c * LOG1M_GRID[g];
            }
            cat.neg_snap = cat
                .snap_counts
                .iter()
                .map(|&sc| {
                    let sc_f = sc as f64;
                    let mut arr = [0.0_f64; GRID_LEN];
                    for g in 0..GRID_LEN {
                        arr[g] = sc_f * LOG1M_GRID[g];
                    }
                    arr
                })
                .collect();
        }
    }

    /// Updates with an unweighted observation `label_id` (weight = 1.0).
    pub fn update(&mut self, label_id: u64) -> ModeDecision {
        self.n_samples += 1;
        if let Some(&idx) = self.id_to_slot.get(&label_id) {
            let cat = &mut self.categories[idx];
            cat.total_count += 1;
            cat.total_weight.add(1.0);
            if self.has_fractional_weights {
                for g in 0..GRID_LEN {
                    cat.pos_lw[g] += LOG1P_GRID[g];
                    cat.neg_lw[g] += LOG1M_GRID[g];
                }
            }
        } else {
            let idx = self.categories.len();
            let snap_counts: Vec<u64> = self.categories.iter().map(|c| c.total_count).collect();
            let (neg_lw, neg_snap) = if self.has_fractional_weights {
                let ns: Vec<[f64; GRID_LEN]> = self.categories.iter().map(|c| c.neg_lw).collect();
                (LOG1M_GRID, ns)
            } else {
                ([0.0; GRID_LEN], Vec::new())
            };
            let mut total_weight = CompSum::default();
            total_weight.add(1.0);
            self.categories.push(CiteCategory {
                label_id,
                total_count: 1,
                total_weight,
                snap_counts,
                pos_lw: [0.0; GRID_LEN],
                neg_lw,
                neg_snap,
            });
            self.id_to_slot.insert(label_id, idx);
            self.slot_strings.push(None);
        }
        self.decision()
    }

    /// Updates with a weighted observation `label_id` and `weight` clamped to `(0, 1]`.
    ///
    /// Non-finite or non-positive weights (`weight <= 0.0` or NaN/Inf) are safely ignored
    /// without mutating state.
    pub fn update_weighted(&mut self, label_id: u64, weight: f64) -> ModeDecision {
        if !weight.is_finite() || weight <= 0.0 {
            return self.decision();
        }
        let w = weight.min(1.0);
        if w == 1.0 {
            return self.update(label_id);
        }
        if !self.has_fractional_weights {
            self.promote_to_weighted();
        }

        let mut pos_inc = [0.0_f64; GRID_LEN];
        let mut neg_inc = [0.0_f64; GRID_LEN];
        for g in 0..GRID_LEN {
            let lw = DEFAULT_BET_GRID[g] * w;
            pos_inc[g] = fast_ln_1p_bounded(lw);
            neg_inc[g] = fast_ln_1p_bounded(-lw);
        }

        self.n_samples += 1;
        if let Some(&idx) = self.id_to_slot.get(&label_id) {
            let cat = &mut self.categories[idx];
            cat.total_count += 1;
            cat.total_weight.add(w);
            for g in 0..GRID_LEN {
                cat.pos_lw[g] += pos_inc[g];
                cat.neg_lw[g] += neg_inc[g];
            }
        } else {
            let idx = self.categories.len();
            let snap_counts: Vec<u64> = self.categories.iter().map(|c| c.total_count).collect();
            let neg_snap: Vec<[f64; GRID_LEN]> = self.categories.iter().map(|c| c.neg_lw).collect();
            let mut total_weight = CompSum::default();
            total_weight.add(w);
            self.categories.push(CiteCategory {
                label_id,
                total_count: 1,
                total_weight,
                snap_counts,
                pos_lw: [0.0; GRID_LEN],
                neg_lw: neg_inc,
                neg_snap,
            });
            self.id_to_slot.insert(label_id, idx);
            self.slot_strings.push(None);
        }
        self.decision()
    }

    fn intern_str(&mut self, answer: &str) -> u64 {
        if let Some(&id) = self.str_to_id.get(answer) {
            return id;
        }
        let mut id = fnv1a_64(answer.as_bytes());
        while self.id_to_slot.contains_key(&id) {
            id = id.wrapping_add(1);
        }
        self.str_to_id.insert(answer.to_owned(), id);
        id
    }

    /// Updates with an unweighted string answer, collision-safely interned to a `u64` ID.
    pub fn update_str(&mut self, answer: &str) -> ModeDecision {
        let id = self.intern_str(answer);
        let had_slot = self.id_to_slot.contains_key(&id);
        let decision = self.update(id);
        if !had_slot {
            if let Some(&slot) = self.id_to_slot.get(&id) {
                self.slot_strings[slot] = Some(answer.to_owned());
            }
        }
        decision
    }

    /// Updates with a weighted string answer (`weight` clamped to `(0, 1]`).
    pub fn update_str_weighted(&mut self, answer: &str, weight: f64) -> ModeDecision {
        if !weight.is_finite() || weight <= 0.0 {
            return self.decision();
        }
        let id = self.intern_str(answer);
        let had_slot = self.id_to_slot.contains_key(&id);
        let decision = self.update_weighted(id, weight);
        if !had_slot {
            if let Some(&slot) = self.id_to_slot.get(&id) {
                self.slot_strings[slot] = Some(answer.to_owned());
            }
        }
        decision
    }

    /// Evaluates the current stopping decision without mutating state.
    pub fn decision(&self) -> ModeDecision {
        if self.n_samples == 0 || self.categories.is_empty() {
            return ModeDecision {
                certified: false,
                winner: None,
                leader: None,
                n_samples: 0,
                log_evidence: 0.0,
                log_threshold: (2.0 / self.delta).ln(),
                margin_estimate: 0.0,
                scope: CertScope::RealizedUnit,
                deterministic_lock: false,
            };
        }

        let m = self.categories.len();
        let mut leader = 0usize;
        let mut leader_w = self.categories[0].total_weight.value();
        let mut second_w = 0.0_f64;
        let mut leader_c = self.categories[0].total_count;
        let mut second_c = 0u64;

        for idx in 1..m {
            let w = self.categories[idx].total_weight.value();
            let c = self.categories[idx].total_count;
            let better = if self.has_fractional_weights {
                w > leader_w
            } else {
                c > leader_c
            };
            if better {
                second_w = leader_w;
                second_c = leader_c;
                leader_w = w;
                leader_c = c;
                leader = idx;
            } else if self.has_fractional_weights {
                if w > second_w {
                    second_w = w;
                    second_c = c;
                }
            } else if c > second_c {
                second_c = c;
                second_w = w;
            }
        }

        let log_threshold = self.slot_log_threshold(leader);
        let log_evidence = self.compute_post_discovery_log_evidence(leader);
        let stat_certified = log_evidence >= log_threshold;

        let det_lock = if let Some(n_max) = self.max_budget {
            let rem = n_max.saturating_sub(self.n_samples);
            if !self.has_fractional_weights {
                leader_c > second_c.saturating_add(rem)
            } else {
                let eps = 1e-12 * (n_max.max(self.n_samples) as f64).max(1.0);
                (leader_w - second_w) > (rem as f64) + eps
            }
        } else {
            false
        };

        let (certified, winner, scope) = if stat_certified {
            (true, Some(leader), CertScope::RealizedUnit)
        } else if det_lock {
            (true, Some(leader), CertScope::DeterministicBudget)
        } else {
            (false, None, CertScope::RealizedUnit)
        };

        ModeDecision {
            certified,
            winner,
            leader: Some(leader),
            n_samples: self.n_samples,
            log_evidence,
            log_threshold,
            margin_estimate: (leader_w - second_w) / (self.n_samples as f64),
            scope,
            deterministic_lock: det_lock,
        }
    }

    fn compute_post_discovery_log_evidence(&self, idx: usize) -> f64 {
        let m = self.categories.len();
        let cand = &self.categories[idx];
        if !self.has_fractional_weights {
            // Exact post-discovery count for candidate `idx` (excluding discovery step).
            let c_a = cand.total_count.saturating_sub(1);
            // Worst-case post-discovery count across all other seen and unseen categories.
            let mut c_max_other = 0u64;
            for b in 0..m {
                if b == idx {
                    continue;
                }
                let snap = if b < idx { cand.snap_counts[b] } else { 0 };
                let c_b = self.categories[b].total_count.saturating_sub(snap);
                if c_b > c_max_other {
                    c_max_other = c_b;
                }
            }
            let c_a_f = c_a as f64;
            let c_o_f = c_max_other as f64;
            let mut ell = [0.0_f64; GRID_LEN];
            for g in 0..GRID_LEN {
                ell[g] = c_a_f * LOG1P_GRID[g] + c_o_f * LOG1M_GRID[g];
            }
            log_mean_exp_grid(&ell)
        } else {
            // Against all infinitely many unseen categories (neg_lw = 0).
            let mut min_lw = log_mean_exp_grid(&cand.pos_lw);
            // Against every seen challenger b != idx using exact post-discovery neg_lw.
            let mut ell = [0.0_f64; GRID_LEN];
            for b in 0..m {
                if b == idx {
                    continue;
                }
                let other_neg = &self.categories[b].neg_lw;
                if b < idx {
                    let snap = &cand.neg_snap[b];
                    for g in 0..GRID_LEN {
                        let neg_post = (other_neg[g] - snap[g]).min(0.0);
                        ell[g] = cand.pos_lw[g] + neg_post;
                    }
                } else {
                    for g in 0..GRID_LEN {
                        ell[g] = cand.pos_lw[g] + other_neg[g];
                    }
                }
                let lw = log_mean_exp_grid(&ell);
                if lw < min_lw {
                    min_lw = lw;
                }
            }
            min_lw
        }
    }

    /// Clears all observed categories and resets the stopper to its initial state.
    pub fn reset(&mut self) {
        self.n_samples = 0;
        self.has_fractional_weights = false;
        self.categories.clear();
        self.id_to_slot.clear();
        self.str_to_id.clear();
        self.slot_strings.clear();
    }
}

// ============================================================================
// 3. PopulationCert — Realized-vs-Population Correlated Rollout Remedy
// ============================================================================

/// Decision and variance-component summary returned by [`PopulationCert`].
#[derive(Clone, Debug)]
pub struct PopulationDecision {
    /// True once the unit-level anytime-valid e-process crosses `ln(1 / delta)`.
    pub certified: bool,
    /// Number of independent units (e.g., prompts) observed.
    pub n_units: u64,
    /// Total number of rollouts across all units.
    pub total_rollouts: u64,
    /// Unweighted mean of per-unit sample means (unbiased estimate of population parameter `theta`).
    pub mean_unit_score: f64,
    /// Current log-evidence of the unit-level e-process.
    pub log_evidence: f64,
    /// Required log-threshold `ln(1 / delta)`.
    pub log_threshold: f64,
    /// Estimated within-unit variance `sigma_w^2` from online one-way random-effects ANOVA.
    pub sigma_w2: f64,
    /// Estimated between-unit variance `sigma_b^2` from online one-way random-effects ANOVA.
    pub sigma_b2: f64,
    /// Estimated intraclass correlation `rho_hat in [0, 0.999999]`.
    pub rho_hat: f64,
    /// Within-unit effective sample size `mean_n / (1 + (mean_n - 1) * rho_hat)`.
    pub n_eff_within: f64,
    /// Asymptotic saturation ceiling `1 / rho_hat` for rollouts within a single unit.
    pub within_unit_ceiling: f64,
    /// Always [`CertScope::Population`].
    pub scope: CertScope,
}

/// Multi-unit anytime-valid population certifier and variance-component estimator.
///
/// Prevents the correlated-rollout fallacy (mistaking within-unit rollout replication
/// for population-level evidence) by:
/// 1. Running a strictly unit-level anytime-valid betting e-process over unit means `u_q in [-1, 1]`.
/// 2. Maintaining an exact online unbalanced one-way random-effects ANOVA decomposition
///    (`sigma_w2`, `sigma_b2`, `rho_hat`, within-unit saturation `n_eff_within`, ceiling `1/rho_hat`).
/// 3. Providing the optimal two-stage rollout count per unit given setup and rollout costs.
#[derive(Clone, Debug)]
pub struct PopulationCert {
    delta: f64,
    log_threshold: f64,
    n_units: u64,
    total_rollouts: u64,
    sum_n_sq: f64,
    grand_mean: f64,
    ss_between: f64,
    ss_within: f64,
    unit_mean_sum: CompSum,
    log_wealth: [f64; GRID_LEN],
}

impl PopulationCert {
    /// Creates a new `PopulationCert` testing `H_0^pop: theta <= 0` at level `delta in (0, 1)`.
    pub fn new(delta: f64) -> Result<Self, &'static str> {
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return Err("delta must be finite and strictly inside (0, 1)");
        }
        Ok(Self {
            delta,
            log_threshold: (1.0 / delta).ln(),
            n_units: 0,
            total_rollouts: 0,
            sum_n_sq: 0.0,
            grand_mean: 0.0,
            ss_between: 0.0,
            ss_within: 0.0,
            unit_mean_sum: CompSum::default(),
            log_wealth: [0.0; GRID_LEN],
        })
    }

    /// Configured Type-I error tolerance `delta`.
    #[inline]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Ingests all rollouts belonging to one independent unit (e.g., one prompt).
    ///
    /// Non-finite entries are skipped; finite entries are clamped to `[-1.0, 1.0]`.
    /// Empty or all-non-finite slices are ignored without mutating state.
    pub fn add_unit_rollouts(&mut self, scores_in_minus1_to_1: &[f64]) -> PopulationDecision {
        let mut m: u64 = 0;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for &raw in scores_in_minus1_to_1 {
            if !raw.is_finite() {
                continue;
            }
            let x = raw.clamp(-1.0, 1.0);
            m += 1;
            let d = x - mean;
            mean += d / (m as f64);
            let d2 = x - mean;
            m2 += d * d2;
        }
        if m == 0 {
            return self.decision();
        }
        let u_q = mean.clamp(-1.0, 1.0);

        // Online exact unbalanced one-way ANOVA update (Chan's parallel SS_B identity).
        let n_old = self.total_rollouts as f64;
        let m_f = m as f64;
        let n_new = n_old + m_f;
        if self.total_rollouts == 0 {
            self.grand_mean = u_q;
            self.ss_between = 0.0;
        } else {
            let diff = u_q - self.grand_mean;
            self.grand_mean += (m_f / n_new) * diff;
            self.ss_between += (n_old * m_f / n_new) * diff * diff;
        }

        self.n_units += 1;
        self.total_rollouts += m;
        self.sum_n_sq += m_f * m_f;
        self.ss_within += m2;
        self.unit_mean_sum.add(u_q);

        // Unit-level anytime-valid betting e-process.
        for g in 0..GRID_LEN {
            self.log_wealth[g] += fast_ln_1p_bounded(DEFAULT_BET_GRID[g] * u_q);
        }

        self.decision()
    }

    fn variance_components(&self) -> (f64, f64, f64) {
        let q = self.n_units;
        let n = self.total_rollouts;
        if q == 0 || n == 0 {
            return (0.0, 0.0, 0.0);
        }
        let df_w = n.saturating_sub(q);
        let sigma_w2 = if df_w > 0 {
            (self.ss_within / (df_w as f64)).max(0.0)
        } else {
            0.0
        };
        let sigma_b2 = if q >= 2 {
            let df_b = (q - 1) as f64;
            let ms_b = self.ss_between / df_b;
            let n_f = n as f64;
            let n0 = (n_f - self.sum_n_sq / n_f) / df_b;
            if n0 > 1e-15 {
                ((ms_b - sigma_w2) / n0).max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        let total_var = sigma_b2 + sigma_w2;
        let rho_hat = if total_var > 1e-15 {
            (sigma_b2 / total_var).clamp(0.0, 0.999999)
        } else {
            0.0
        };
        (sigma_w2, sigma_b2, rho_hat)
    }

    /// Evaluates the current population certification decision and ANOVA summary.
    pub fn decision(&self) -> PopulationDecision {
        let (sigma_w2, sigma_b2, rho_hat) = self.variance_components();
        let mean_n = if self.n_units > 0 {
            (self.total_rollouts as f64) / (self.n_units as f64)
        } else {
            0.0
        };
        let n_eff_within = if mean_n > 0.0 {
            mean_n / (1.0 + (mean_n - 1.0) * rho_hat)
        } else {
            0.0
        };
        let within_unit_ceiling = if rho_hat > 1e-12 {
            1.0 / rho_hat
        } else {
            f64::INFINITY
        };
        let log_evidence = if self.n_units > 0 {
            log_mean_exp_grid(&self.log_wealth)
        } else {
            0.0
        };
        let mean_unit_score = if self.n_units > 0 {
            self.unit_mean_sum.value() / (self.n_units as f64)
        } else {
            0.0
        };
        PopulationDecision {
            certified: self.n_units > 0 && log_evidence >= self.log_threshold,
            n_units: self.n_units,
            total_rollouts: self.total_rollouts,
            mean_unit_score,
            log_evidence,
            log_threshold: self.log_threshold,
            sigma_w2,
            sigma_b2,
            rho_hat,
            n_eff_within,
            within_unit_ceiling,
            scope: CertScope::Population,
        }
    }

    /// Computes the optimal number of rollouts per unit minimizing variance per unit cost:
    /// `max(1, round(sqrt(unit_setup_cost * sigma_w2 / (per_rollout_cost * sigma_b2))))`.
    ///
    /// Returns `1` whenever `unit_setup_cost <= 0`, `per_rollout_cost <= 0`, inputs are non-finite,
    /// `sigma_w2 <= 0`, or fewer than 2 units have been observed.
    pub fn optimal_rollouts_per_unit(&self, unit_setup_cost: f64, per_rollout_cost: f64) -> usize {
        if self.n_units < 2
            || !unit_setup_cost.is_finite()
            || !per_rollout_cost.is_finite()
            || unit_setup_cost <= 0.0
            || per_rollout_cost <= 0.0
        {
            return 1;
        }
        let (sigma_w2, sigma_b2, _) = self.variance_components();
        if sigma_w2 <= 0.0 {
            return 1;
        }
        if sigma_b2 <= 1e-15 {
            return 1_000_000;
        }
        let m_star = ((unit_setup_cost * sigma_w2) / (per_rollout_cost * sigma_b2)).sqrt();
        if !m_star.is_finite() {
            return 1_000_000;
        }
        m_star.round().clamp(1.0, 1_000_000.0) as usize
    }

    /// Resets all accumulated unit statistics and log-wealth to initial state.
    pub fn reset(&mut self) {
        self.n_units = 0;
        self.total_rollouts = 0;
        self.sum_n_sq = 0.0;
        self.grand_mean = 0.0;
        self.ss_between = 0.0;
        self.ss_within = 0.0;
        self.unit_mean_sum = CompSum::default();
        self.log_wealth = [0.0; GRID_LEN];
    }

    /// Exact additive merge of two independent `PopulationCert` shards.
    pub fn merge(&mut self, other: &Self) {
        if other.n_units == 0 || other.total_rollouts == 0 {
            return;
        }
        if self.total_rollouts == 0 {
            self.n_units = other.n_units;
            self.total_rollouts = other.total_rollouts;
            self.sum_n_sq = other.sum_n_sq;
            self.grand_mean = other.grand_mean;
            self.ss_between = other.ss_between;
            self.ss_within = other.ss_within;
            self.unit_mean_sum = other.unit_mean_sum;
            self.log_wealth = other.log_wealth;
            return;
        }
        let n1 = self.total_rollouts as f64;
        let n2 = other.total_rollouts as f64;
        let n_new = n1 + n2;
        let diff = other.grand_mean - self.grand_mean;
        self.grand_mean += (n2 / n_new) * diff;
        self.ss_between += other.ss_between + (n1 * n2 / n_new) * diff * diff;
        self.ss_within += other.ss_within;
        self.sum_n_sq += other.sum_n_sq;
        self.n_units = self.n_units.saturating_add(other.n_units);
        self.total_rollouts = self.total_rollouts.saturating_add(other.total_rollouts);
        self.unit_mean_sum.merge(other.unit_mean_sum);
        for g in 0..GRID_LEN {
            self.log_wealth[g] += other.log_wealth[g];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    fn sample_categorical(rng: &mut Rng, probs: &[f64]) -> usize {
        let u = rng.next_f64();
        let mut cdf = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cdf += p;
            if u < cdf {
                return i;
            }
        }
        probs.len() - 1
    }

    #[test]
    fn ascd_validation_and_sanitization() {
        assert!(AscdStopper::new(1, 0.05, None).is_err());
        assert!(AscdStopper::new(3, 0.0, None).is_err());
        assert!(AscdStopper::new(3, 1.0, None).is_err());
        assert!(AscdStopper::new(3, f64::NAN, None).is_err());
        assert!(AscdStopper::new(3, 0.05, Some(0)).is_err());

        let mut s = AscdStopper::new(3, 0.05, None).unwrap();
        // Wrong length ignored
        let d = s.update(&[0.5, 0.5]);
        assert_eq!(d.n_samples, 0);
        // NaN / Inf ignored
        let d = s.update(&[0.5, f64::NAN, 0.5]);
        assert_eq!(d.n_samples, 0);
        let d = s.update(&[f64::INFINITY, 0.1, 0.1]);
        assert_eq!(d.n_samples, 0);
        // All zero / negative ignored
        let d = s.update(&[0.0, 0.0, 0.0]);
        assert_eq!(d.n_samples, 0);
        let d = s.update(&[-1.0, -2.0, -3.0]);
        assert_eq!(d.n_samples, 0);
        // Negative tiny drift clamped and renormalized
        let d = s.update(&[-1e-16, 3.0, 1.0]);
        assert_eq!(d.n_samples, 1);
        assert_eq!(d.leader, Some(1));
        assert!((s.score(1).unwrap() - 0.75).abs() < 1e-12);
        assert!((s.score(2).unwrap() - 0.25).abs() < 1e-12);
        // Invalid one-hot & confidence ignored
        let d = s.update_one_hot(3);
        assert_eq!(d.n_samples, 1);
        let d = s.update_confidence_weighted(0, f64::NAN);
        assert_eq!(d.n_samples, 1);
        let d = s.update_confidence_weighted(1, 5.0); // clamped to 1.0
        assert_eq!(d.n_samples, 2);
        assert!((s.score(1).unwrap() - 1.75).abs() < 1e-12);
    }

    #[test]
    fn ascd_type1_error_under_null() {
        let delta = 0.05;
        let k = 4;
        let n_trials = 400;
        let max_steps = 400;
        let mut rng = Rng::from_seed(0xA5CD_0001);
        // True mode is 0 with tiny gap over 1,2,3 (hard near-null): misidentifying != 0 must be <= delta.
        let probs = [0.253, 0.249, 0.249, 0.249];
        let mut wrong_certs = 0usize;
        let mut stopper = AscdStopper::new(k, delta, None).unwrap();

        for _ in 0..n_trials {
            stopper.reset();
            for _ in 0..max_steps {
                let y = sample_categorical(&mut rng, &probs);
                let d = stopper.update_one_hot(y);
                if d.certified {
                    if d.winner != Some(0) {
                        wrong_certs += 1;
                    }
                    break;
                }
            }
        }
        let emp_err = (wrong_certs as f64) / (n_trials as f64);
        assert!(
            emp_err <= delta,
            "ASC-D misidentification rate {emp_err:.4} exceeded delta={delta}"
        );
    }

    #[test]
    fn ascd_greybox_faster_than_blackbox_and_merge_exact() {
        let delta = 0.01;
        let mut grey = AscdStopper::new(4, delta, None).unwrap();
        let mut black = AscdStopper::new(4, delta, None).unwrap();
        let mut rng = Rng::from_seed(0xA5CD_0002);

        // Grey-box gets the smooth distribution [0.7, 0.1, 0.1, 0.1]; black-box gets noisy one-hot draws.
        let p = [0.7, 0.1, 0.1, 0.1];
        let mut grey_stop = None;
        let mut black_stop = None;
        for step in 1..=200 {
            if grey_stop.is_none() {
                let d = grey.update(&p);
                if d.certified {
                    assert_eq!(d.winner, Some(0));
                    grey_stop = Some(step);
                }
            }
            if black_stop.is_none() {
                let y = sample_categorical(&mut rng, &p);
                let d = black.update_one_hot(y);
                if d.certified {
                    assert_eq!(d.winner, Some(0));
                    black_stop = Some(step);
                }
            }
        }
        let gs = grey_stop.expect("grey-box should certify");
        let bs = black_stop.expect("black-box should certify");
        assert!(gs <= bs, "expected grey-box ({gs}) <= black-box ({bs})");

        // Exact additive merge test
        let mut whole = AscdStopper::new(3, 0.05, None).unwrap();
        let mut shard_a = AscdStopper::new(3, 0.05, None).unwrap();
        let mut shard_b = AscdStopper::new(3, 0.05, None).unwrap();
        for i in 0..50 {
            let v = if i % 3 == 0 {
                [0.8, 0.15, 0.05]
            } else {
                [0.2, 0.7, 0.1]
            };
            whole.update(&v);
            if i < 25 {
                shard_a.update(&v);
            } else {
                shard_b.update(&v);
            }
        }
        shard_a.merge(&shard_b);
        let dw = whole.decision();
        let dm = shard_a.decision();
        assert_eq!(dw.n_samples, dm.n_samples);
        assert_eq!(dw.leader, dm.leader);
        assert!((dw.log_evidence - dm.log_evidence).abs() < 1e-11);
        assert!((dw.margin_estimate - dm.margin_estimate).abs() < 1e-12);
    }

    #[test]
    fn ascd_large_n_stability_and_deterministic_budget() {
        let mut s = AscdStopper::new(3, 0.001, None).unwrap();
        for i in 0..100_000 {
            let y = if i % 10 < 6 { 0 } else { 1 };
            s.update_one_hot(y);
        }
        let d = s.decision();
        assert!(d.certified);
        assert_eq!(d.winner, Some(0));
        assert!(d.log_evidence.is_finite());

        // Deterministic lock with very small delta so budget lock triggers first
        let mut b = AscdStopper::new(3, 1e-12, Some(10)).unwrap();
        for step in 1..=5 {
            let d = b.update_one_hot(2);
            assert!(!d.deterministic_lock, "step {step} should not lock yet");
        }
        let d6 = b.update_one_hot(2);
        assert!(d6.deterministic_lock);
        assert!(d6.certified);
        assert_eq!(d6.winner, Some(2));
        assert_eq!(d6.scope, CertScope::DeterministicBudget);
    }

    #[test]
    fn cite_post_discovery_type1_error_and_unbounded_strings() {
        let delta = 0.05;
        let n_trials = 400;
        let max_steps = 300;
        let mut rng = Rng::from_seed(0xC17E_0001);

        // Unbounded-like tail: 1 true mode (id 0, p=0.20) + 4 close challengers (p=0.195 each) + unique tail
        let mut wrong_certs = 0usize;
        let mut stopper = CiteStopper::new(delta, None).unwrap();
        let mut unique_id = 1000u64;

        for _ in 0..n_trials {
            stopper.reset();
            for _ in 0..max_steps {
                let u = rng.next_f64();
                let label = if u < 0.20 {
                    0
                } else if u < 0.395 {
                    1
                } else if u < 0.590 {
                    2
                } else if u < 0.785 {
                    3
                } else if u < 0.980 {
                    4
                } else {
                    unique_id += 1;
                    unique_id
                };
                let d = stopper.update(label);
                if d.certified {
                    if stopper.label_of(d.winner.unwrap()) != Some(0) {
                        wrong_certs += 1;
                    }
                    break;
                }
            }
        }
        let emp_err = (wrong_certs as f64) / (n_trials as f64);
        assert!(
            emp_err <= delta,
            "CITE misidentification rate {emp_err:.4} exceeded delta={delta}"
        );

        // String interning & certification on clear consensus
        let mut cs = CiteStopper::new(0.01, None).unwrap();
        let mut certified_at = None;
        for step in 1..=60 {
            let ans = if step % 6 == 0 { "wrong_patch" } else { "fix_null_ptr" };
            let d = cs.update_str(ans);
            if d.certified {
                certified_at = Some(step);
                assert_eq!(cs.winner_str(), Some("fix_null_ptr"));
                assert_eq!(d.scope, CertScope::RealizedUnit);
                break;
            }
        }
        assert!(certified_at.is_some());
    }

    #[test]
    fn cite_deterministic_lock_and_weighted_exactness() {
        // Exact deterministic lock on budget N=10
        let mut cs = CiteStopper::new(1e-12, Some(10)).unwrap();
        for _ in 0..5 {
            let d = cs.update(42);
            assert!(!d.deterministic_lock);
            assert!(!d.certified);
        }
        let d6 = cs.update(42); // 6 vs 0 with 4 remaining -> 6 > 0 + 4 -> locked!
        assert!(d6.deterministic_lock);
        assert!(d6.certified);
        assert_eq!(cs.winner_label(), Some(42));
        assert_eq!(d6.scope, CertScope::DeterministicBudget);

        // Exact tie at budget exhaustion does NOT lock
        let mut tie = CiteStopper::new(1e-6, Some(4)).unwrap();
        tie.update(1);
        tie.update(2);
        tie.update(1);
        let d4 = tie.update(2);
        assert!(!d4.deterministic_lock);
        assert!(!d4.certified);

        // Weighted updates + mixed promotion exactness vs pure weighted
        let mut mixed = CiteStopper::new(0.05, None).unwrap();
        let mut pure_w = CiteStopper::new(0.05, None).unwrap();
        pure_w.promote_to_weighted();

        let seq = [
            (10u64, 1.0),
            (20, 1.0),
            (10, 1.0),
            (30, 0.8),
            (10, 0.9),
            (20, 0.5),
            (10, 1.0),
            (10, 0.75),
        ];
        for &(id, w) in &seq {
            if w == 1.0 {
                mixed.update(id);
            } else {
                mixed.update_weighted(id, w);
            }
            pure_w.update_weighted(id, w);
        }
        let dm = mixed.decision();
        let dp = pure_w.decision();
        assert_eq!(dm.leader, dp.leader);
        assert_eq!(dm.n_samples, dp.n_samples);
        assert!((dm.log_evidence - dp.log_evidence).abs() < 1e-12);
        // Invalid weights ignored
        let before = mixed.n_samples();
        mixed.update_weighted(999, f64::NAN);
        mixed.update_weighted(999, 0.0);
        mixed.update_weighted(999, -0.5);
        assert_eq!(mixed.n_samples(), before);
    }

    #[test]
    fn population_cert_anova_and_correlated_rollout_validity() {
        let delta = 0.05;
        let mut pc = PopulationCert::new(delta).unwrap();

        // Single prompt with 500 identical +1 rollouts CANNOT certify population null!
        let single_prompt = vec![1.0; 500];
        let d1 = pc.add_unit_rollouts(&single_prompt);
        assert!(!d1.certified, "single unit must never certify population null");
        assert_eq!(d1.n_units, 1);
        assert_eq!(d1.total_rollouts, 500);
        assert_eq!(d1.scope, CertScope::Population);

        // Verify ANOVA accuracy on balanced clusters with known sigma_b^2 and sigma_w^2
        let mut anova = PopulationCert::new(0.05).unwrap();
        let mut rng = Rng::from_seed(0x9090_0001);
        // Unit means +/- 0.6 (sigma_b^2 = 0.36), within-unit noise +/- 0.2 (sigma_w^2 = 0.04) -> rho = 0.9
        for q in 0..200 {
            let mu = if q % 2 == 0 { 0.6 } else { -0.6 };
            let mut rollouts = [0.0; 10];
            for r in &mut rollouts {
                let eps = if rng.next_f64() < 0.5 { 0.2 } else { -0.2 };
                *r = mu + eps;
            }
            anova.add_unit_rollouts(&rollouts);
        }
        let da = anova.decision();
        assert!((da.sigma_w2 - 0.04).abs() < 0.01, "sigma_w2={}", da.sigma_w2);
        assert!((da.sigma_b2 - 0.36).abs() < 0.03, "sigma_b2={}", da.sigma_b2);
        assert!((da.rho_hat - 0.90).abs() < 0.03, "rho_hat={}", da.rho_hat);
        assert!((da.within_unit_ceiling - (1.0 / da.rho_hat)).abs() < 1e-9);
        assert!(da.n_eff_within < da.within_unit_ceiling);

        // Optimal rollouts per unit: setup=9, rollout=1, sigma_w2=0.04, sigma_b2=0.36 -> sqrt(9*0.04/0.36) = 1
        assert_eq!(anova.optimal_rollouts_per_unit(9.0, 1.0), 1);
        // setup=36, rollout=1 -> sqrt(36 * 0.04 / 0.36) = sqrt(4) = 2
        assert_eq!(anova.optimal_rollouts_per_unit(36.0, 1.0), 2);
        // setup=0 -> always 1
        assert_eq!(anova.optimal_rollouts_per_unit(0.0, 1.0), 1);

        // Monte Carlo Type-I error under strongly correlated population null (theta = 0, rho ~ 0.9)
        let n_trials = 300;
        let mut false_certs = 0usize;
        let mut cert = PopulationCert::new(delta).unwrap();
        for _ in 0..n_trials {
            cert.reset();
            for _ in 0..80 {
                let unit_bias = if rng.next_f64() < 0.5 { 0.8 } else { -0.8 };
                let mut scores = [0.0; 20];
                for s in &mut scores {
                    let noise = (rng.next_f64() - 0.5) * 0.4;
                    *s = (unit_bias + noise).clamp(-1.0, 1.0);
                }
                if cert.add_unit_rollouts(&scores).certified {
                    false_certs += 1;
                    break;
                }
            }
        }
        let fpr = (false_certs as f64) / (n_trials as f64);
        assert!(
            fpr <= delta,
            "PopulationCert FPR {fpr:.4} exceeded delta={delta}"
        );

        // Exact merge check on unbalanced units
        let mut full = PopulationCert::new(0.05).unwrap();
        let mut s1 = PopulationCert::new(0.05).unwrap();
        let mut s2 = PopulationCert::new(0.05).unwrap();
        let units: [&[f64]; 4] = [
            &[0.5, 0.7, 0.6],
            &[-0.2, 0.1],
            &[0.9, 0.8, 0.85, 0.95],
            &[0.3],
        ];
        for (i, u) in units.iter().enumerate() {
            full.add_unit_rollouts(u);
            if i < 2 {
                s1.add_unit_rollouts(u);
            } else {
                s2.add_unit_rollouts(u);
            }
        }
        s1.merge(&s2);
        let df = full.decision();
        let dm = s1.decision();
        assert_eq!(df.n_units, dm.n_units);
        assert_eq!(df.total_rollouts, dm.total_rollouts);
        assert!((df.sigma_w2 - dm.sigma_w2).abs() < 1e-12);
        assert!((df.sigma_b2 - dm.sigma_b2).abs() < 1e-12);
        assert!((df.rho_hat - dm.rho_hat).abs() < 1e-12);
        assert!((df.log_evidence - dm.log_evidence).abs() < 1e-12);
    }

    #[test]
    fn ascd_subnormal_and_weighted_cite_type1_and_exact_unbalanced_anova() {
        // Subnormal probabilities must never introduce NaNs during normalization
        let mut s = AscdStopper::new(3, 0.05, None).unwrap();
        let d = s.update(&[1e-320, 1e-320, 0.0]);
        assert_eq!(d.n_samples, 1);
        assert!(d.log_evidence.is_finite());
        assert!((s.score(0).unwrap() - 0.5).abs() < 1e-12);
        assert!((s.score(1).unwrap() - 0.5).abs() < 1e-12);

        // Weighted CiteStopper Type-I error check under near-null with varying confidence weights
        let delta = 0.05;
        let mut rng = Rng::from_seed(0xC17E_9999);
        let mut cs = CiteStopper::new(delta, None).unwrap();
        let mut wrong = 0usize;
        let trials = 300;
        for _ in 0..trials {
            cs.reset();
            for _ in 0..250 {
                let u = rng.next_f64();
                let label = if u < 0.34 {
                    100
                } else if u < 0.67 {
                    200
                } else {
                    300
                };
                let w = 0.25 + 0.75 * rng.next_f64();
                let d = cs.update_weighted(label, w);
                if d.certified {
                    if cs.label_of(d.winner.unwrap()) != Some(100) {
                        wrong += 1;
                    }
                    break;
                }
            }
        }
        let emp_err = (wrong as f64) / (trials as f64);
        assert!(
            emp_err <= delta,
            "Weighted CITE misidentification rate {emp_err:.4} exceeded delta={delta}"
        );

        // Exact textbook unbalanced one-way random-effects ANOVA verification:
        // Cluster 0: [0.2, 0.4, 0.6] -> n=3, mean=0.4, SSw=0.08
        // Cluster 1: [-0.8, -0.6] -> n=2, mean=-0.7, SSw=0.02
        // Cluster 2: [0.1, 0.3, 0.5, 0.7, 0.9] -> n=5, mean=0.5, SSw=0.40
        // Grand mean = 0.23, SSw = 0.50, dfW = 7 -> sigma_w2 = 0.50 / 7.0
        // SSb = 3*(0.17)^2 + 2*(-0.93)^2 + 5*(0.27)^2 = 2.181, MSb = 1.0905
        // n0 = (10 - (9 + 4 + 25)/10) / 2 = 3.1 -> sigma_b2 = (1.0905 - 0.50/7.0) / 3.1
        let mut pc = PopulationCert::new(0.05).unwrap();
        pc.add_unit_rollouts(&[0.2, 0.4, 0.6]);
        pc.add_unit_rollouts(&[-0.8, -0.6]);
        let d = pc.add_unit_rollouts(&[0.1, 0.3, 0.5, 0.7, 0.9]);
        let expected_sw2 = 0.50 / 7.0;
        let expected_sb2 = (1.0905 - expected_sw2) / 3.1;
        let expected_rho = expected_sb2 / (expected_sb2 + expected_sw2);
        assert!((d.sigma_w2 - expected_sw2).abs() < 1e-12);
        assert!((d.sigma_b2 - expected_sb2).abs() < 1e-12);
        assert!((d.rho_hat - expected_rho).abs() < 1e-12);

        // All-NaN / empty rollouts ignored safely
        let before = d.n_units;
        let d2 = pc.add_unit_rollouts(&[f64::NAN, f64::INFINITY]);
        assert_eq!(d2.n_units, before);
        let d3 = pc.add_unit_rollouts(&[]);
        assert_eq!(d3.n_units, before);
    }

    #[test]
    fn ascd_antisymmetric_25_stream_exact_parity_and_tie_null() {
        let mut rng = Rng::from_seed(0xA5CD_2525);
        let ks = [2usize, 3, 4, 5, 8];
        let mut parity_matches = 0usize;

        for case_idx in 0..25 {
            let k = ks[case_idx % ks.len()];
            let delta = if case_idx % 2 == 0 { 0.05 } else { 0.01 };
            let mut stopper = AscdStopper::new(k, delta, None).unwrap();
            let log_thresh = ((k as f64) / delta).ln();
            let mut ref_scores = vec![0.0_f64; k];
            let mut ref_lw = vec![[0.0_f64; GRID_LEN]; k * k];

            let mut fast_stop = None;
            let mut ref_stop = None;
            let winner_bias = 0.25 + 0.05 * ((case_idx % 5) as f64);

            for step in 1..=200 {
                let mut probs = vec![0.0_f64; k];
                if case_idx % 7 == 0 && step % 5 == 0 {
                    // Exact one-hot step
                    probs[0] = 1.0;
                } else if case_idx % 7 == 1 && step % 3 == 0 {
                    // Near-tie step (exercises |diff| < 0.02 path)
                    let base = 1.0 / (k as f64);
                    for a in 0..k {
                        probs[a] = base + if a == 0 { 1e-3 } else { -1e-3 / ((k - 1) as f64) };
                    }
                } else {
                    let mut s = 0.0;
                    for a in 0..k {
                        let w = rng.next_f64() + if a == 0 { winner_bias } else { 0.0 };
                        probs[a] = w;
                        s += w;
                    }
                    for p in &mut probs {
                        *p /= s;
                    }
                }

                let d = stopper.update(&probs);
                for a in 0..k {
                    ref_scores[a] += probs[a];
                    for b in 0..k {
                        if a == b {
                            continue;
                        }
                        let diff = (probs[a] - probs[b]).clamp(-1.0, 1.0);
                        for g in 0..GRID_LEN {
                            ref_lw[a * k + b][g] += (DEFAULT_BET_GRID[g] * diff).ln_1p();
                        }
                    }
                }
                let mut ref_leader = 0usize;
                for a in 1..k {
                    if ref_scores[a] > ref_scores[ref_leader] {
                        ref_leader = a;
                    }
                }
                let mut ref_min_lw = f64::INFINITY;
                for b in 0..k {
                    if b == ref_leader {
                        continue;
                    }
                    let arr = &ref_lw[ref_leader * k + b];
                    let m = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let lw = m
                        + arr.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
                        - (GRID_LEN as f64).ln();
                    if lw < ref_min_lw {
                        ref_min_lw = lw;
                    }
                }
                assert!(
                    (d.log_evidence - ref_min_lw).abs() < 1e-11,
                    "case {case_idx} step {step}: log_evidence {} vs {}",
                    d.log_evidence,
                    ref_min_lw
                );
                if fast_stop.is_none() && d.certified {
                    fast_stop = Some((step, d.winner));
                }
                if ref_stop.is_none() && ref_min_lw >= log_thresh {
                    ref_stop = Some((step, Some(ref_leader)));
                }
                if fast_stop.is_some() && ref_stop.is_some() {
                    break;
                }
            }
            assert_eq!(fast_stop, ref_stop, "stopping mismatch on case {case_idx}");
            parity_matches += 1;
        }
        assert_eq!(parity_matches, 25);

        // Exact tie-null: deterministic tied probability vectors must produce 0.0% Type-I error.
        for &k in &[2usize, 4] {
            let tie_p = vec![1.0 / (k as f64); k];
            let mut stopper = AscdStopper::new(k, 0.05, None).unwrap();
            for _ in 0..200 {
                let d = stopper.update(&tie_p);
                assert!(!d.certified, "tie-null falsely certified for k={k}");
                assert_eq!(d.log_evidence, 0.0);
            }
        }
    }
}

