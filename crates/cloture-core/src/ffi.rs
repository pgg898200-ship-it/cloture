//! Zero-dependency C ABI (`extern "C"`) matching `include/cloture.h`.
//!
//! Safety guarantees across the FFI boundary:
//! - Every entry point checks for null pointers and invalid lengths.
//! - Every entry point catches panics via `std::panic::catch_unwind` so Rust
//!   can never unwind across the C ABI into Python, C, C++, or Go callers.
//! - Struct layouts are `#[repr(C)]` and match `include/cloture.h` and
//!   `python/cloture/_native.py` field-for-field.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::betting::{BettingCS, CsBounds, EmpiricalBernsteinCS, PairedEProcess, PairedStatus};
use crate::cascade::{AbortCascade, OnlineConformal};
use crate::fuse::SafeFuse;
use crate::mode::{AscdStopper, CertScope, CiteStopper, ModeDecision, PopulationCert, PopulationDecision};
use crate::panel::{OnlinePanelTracker, PanelAudit, PanelAuditConfig};
use crate::ppi::{ActivePPI, PpiStatus};
use crate::race::{e_bh_discoveries, AgentRace};
use crate::sketch::{EvidenceSketch, WIRE_SIZE};

pub const CLOTURE_SCOPE_REALIZED_UNIT: i32 = 0;
pub const CLOTURE_SCOPE_POPULATION: i32 = 1;
pub const CLOTURE_SCOPE_DETERMINISTIC_LOCK: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ClotureModeResult {
    pub certified: i32,
    pub deterministic_lock: i32,
    pub has_winner: i32,
    pub has_leader: i32,
    pub winner: u64,
    pub leader: u64,
    pub n_samples: u64,
    pub log_evidence: f64,
    pub log_threshold: f64,
    pub margin_estimate: f64,
    pub scope: i32,
}

impl Default for ClotureModeResult {
    fn default() -> Self {
        Self {
            certified: 0,
            deterministic_lock: 0,
            has_winner: 0,
            has_leader: 0,
            winner: 0,
            leader: 0,
            n_samples: 0,
            log_evidence: 0.0,
            log_threshold: 0.0,
            margin_estimate: 0.0,
            scope: CLOTURE_SCOPE_REALIZED_UNIT,
        }
    }
}

fn scope_to_i32(scope: CertScope) -> i32 {
    match scope {
        CertScope::RealizedUnit => CLOTURE_SCOPE_REALIZED_UNIT,
        CertScope::Population => CLOTURE_SCOPE_POPULATION,
        CertScope::DeterministicBudget => CLOTURE_SCOPE_DETERMINISTIC_LOCK,
    }
}

fn mode_decision_to_c(d: ModeDecision) -> ClotureModeResult {
    ClotureModeResult {
        certified: if d.certified { 1 } else { 0 },
        deterministic_lock: if d.deterministic_lock { 1 } else { 0 },
        has_winner: if d.winner.is_some() { 1 } else { 0 },
        has_leader: if d.leader.is_some() { 1 } else { 0 },
        winner: d.winner.map(|v| v as u64).unwrap_or(0),
        leader: d.leader.map(|v| v as u64).unwrap_or(0),
        n_samples: d.n_samples,
        log_evidence: d.log_evidence,
        log_threshold: d.log_threshold,
        margin_estimate: d.margin_estimate,
        scope: scope_to_i32(d.scope),
    }
}

fn cite_decision_to_c(stopper: &CiteStopper, d: ModeDecision) -> ClotureModeResult {
    let winner_opt = d.winner.and_then(|slot| stopper.label_of(slot));
    let leader_opt = d.leader.and_then(|slot| stopper.label_of(slot));
    ClotureModeResult {
        certified: if d.certified { 1 } else { 0 },
        deterministic_lock: if d.deterministic_lock { 1 } else { 0 },
        has_winner: if winner_opt.is_some() { 1 } else { 0 },
        has_leader: if leader_opt.is_some() { 1 } else { 0 },
        winner: winner_opt.unwrap_or(0),
        leader: leader_opt.unwrap_or(0),
        n_samples: d.n_samples,
        log_evidence: d.log_evidence,
        log_threshold: d.log_threshold,
        margin_estimate: d.margin_estimate,
        scope: scope_to_i32(d.scope),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CloturePopResult {
    pub n_units: u64,
    pub total_rollouts: u64,
    pub certified: i32,
    pub theta_hat: f64,
    pub sigma_b2: f64,
    pub sigma_w2: f64,
    pub rho_hat: f64,
    pub n_eff_within: f64,
    pub saturation_ceiling: f64,
    pub log_evidence: f64,
    pub log_threshold: f64,
}

fn pop_decision_to_c(d: PopulationDecision) -> CloturePopResult {
    CloturePopResult {
        n_units: d.n_units,
        total_rollouts: d.total_rollouts,
        certified: if d.certified { 1 } else { 0 },
        theta_hat: d.mean_unit_score,
        sigma_b2: d.sigma_b2,
        sigma_w2: d.sigma_w2,
        rho_hat: d.rho_hat,
        n_eff_within: d.n_eff_within,
        saturation_ceiling: d.within_unit_ceiling,
        log_evidence: d.log_evidence,
        log_threshold: d.log_threshold,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ClotureCsResult {
    pub n: u64,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    pub width: f64,
}

impl Default for ClotureCsResult {
    fn default() -> Self {
        Self {
            n: 0,
            mean: 0.5,
            lower: 0.0,
            upper: 1.0,
            width: 1.0,
        }
    }
}

fn cs_bounds_to_c(b: CsBounds) -> ClotureCsResult {
    ClotureCsResult {
        n: b.n,
        mean: b.mean,
        lower: b.lower,
        upper: b.upper,
        width: b.width(),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CloturePairedResult {
    pub n_pairs: u64,
    pub mean_diff: f64,
    pub e_value_a_beats_b: f64,
    pub e_value_b_beats_a: f64,
    pub anytime_p_value: f64,
    pub winner: i32,
}

impl Default for CloturePairedResult {
    fn default() -> Self {
        Self {
            n_pairs: 0,
            mean_diff: 0.0,
            e_value_a_beats_b: 1.0,
            e_value_b_beats_a: 1.0,
            anytime_p_value: 1.0,
            winner: -1,
        }
    }
}

fn paired_status_to_c(s: PairedStatus) -> CloturePairedResult {
    CloturePairedResult {
        n_pairs: s.n_pairs,
        mean_diff: s.mean_diff,
        e_value_a_beats_b: s.e_value_a_beats_b,
        e_value_b_beats_a: s.e_value_b_beats_a,
        anytime_p_value: s.anytime_p_value,
        winner: s.winner.map(|w| w as i32).unwrap_or(-1),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CloturePpiResult {
    pub n_total: u64,
    pub n_audited: u64,
    pub mean_estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub width: f64,
    pub lambda_hat: f64,
    pub variance_reduction: f64,
}

impl Default for CloturePpiResult {
    fn default() -> Self {
        Self {
            n_total: 0,
            n_audited: 0,
            mean_estimate: 0.5,
            lower: 0.0,
            upper: 1.0,
            width: 1.0,
            lambda_hat: 0.0,
            variance_reduction: 1.0,
        }
    }
}

fn ppi_status_to_c(s: PpiStatus) -> CloturePpiResult {
    CloturePpiResult {
        n_total: s.n_total,
        n_audited: s.n_audited,
        mean_estimate: s.mean_estimate,
        lower: s.lower,
        upper: s.upper,
        width: s.width,
        lambda_hat: s.lambda_hat,
        variance_reduction: s.variance_reduction_factor,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CloturePanelSummary {
    pub k_judges: u32,
    pub n_items: u32,
    pub panel_accuracy: f64,
    pub best_single_accuracy: f64,
    pub best_single_index: u32,
    pub panel_lift: f64,
    pub mean_phi: f64,
    pub std_phi: f64,
    pub n_eff_kish: f64,
    pub n_eff_eigen: f64,
    pub lambda_max: f64,
    pub effective_rank: f64,
    pub independence_ratio: f64,
    pub asymptotic_ceiling: f64,
    pub krippendorff_alpha: f64,
    pub unanimous_rate: f64,
    pub unanimous_accuracy: f64,
    pub all_wrong_rate: f64,
    pub condorcet_predicted_accuracy: f64,
    pub condorcet_gap: f64,
    pub permutation_p_value: f64,
    pub n_eff_ci_low: f64,
    pub n_eff_ci_high: f64,
}

// ============================================================================
// Engine 1a: ASC-D (Grey-box distribution-valued mode stopping)
// ============================================================================

#[no_mangle]
pub extern "C" fn cloture_ascd_new(k: usize, delta: f64, max_budget: u64) -> *mut AscdStopper {
    catch_unwind(|| {
        let budget = if max_budget == 0 { None } else { Some(max_budget) };
        match AscdStopper::new(k, delta, budget) {
            Ok(s) => Box::into_raw(Box::new(s)),
            Err(_) => std::ptr::null_mut(),
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ascd_free(ptr: *mut AscdStopper) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ascd_update(
    ptr: *mut AscdStopper,
    probs: *const f64,
    len: usize,
) -> ClotureModeResult {
    if ptr.is_null() || probs.is_null() || len == 0 {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        let sl = slice::from_raw_parts(probs, len);
        mode_decision_to_c(stopper.update(sl))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ascd_update_batch(
    ptr: *mut AscdStopper,
    probs_flat: *const f64,
    n_rows: usize,
    k: usize,
) -> ClotureModeResult {
    if ptr.is_null() {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        if probs_flat.is_null() || n_rows == 0 || k != stopper.k() {
            return mode_decision_to_c(stopper.decision());
        }
        let total = match n_rows.checked_mul(k) {
            Some(v) => v,
            None => return mode_decision_to_c(stopper.decision()),
        };
        let flat = slice::from_raw_parts(probs_flat, total);
        let mut dec = stopper.decision();
        for row in flat.chunks_exact(k) {
            dec = stopper.update(row);
            if dec.certified {
                break;
            }
        }
        mode_decision_to_c(dec)
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ascd_update_one_hot(
    ptr: *mut AscdStopper,
    label: usize,
) -> ClotureModeResult {
    if ptr.is_null() {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        mode_decision_to_c(stopper.update_one_hot(label))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ascd_update_weighted(
    ptr: *mut AscdStopper,
    label: usize,
    confidence: f64,
) -> ClotureModeResult {
    if ptr.is_null() {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        mode_decision_to_c(stopper.update_confidence_weighted(label, confidence))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ascd_reset(ptr: *mut AscdStopper) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            (*ptr).reset();
        }));
    }
}

// ============================================================================
// Engine 1b: CITE (Category-agnostic unbounded-category stopping)
// ============================================================================

#[no_mangle]
pub extern "C" fn cloture_cite_new(delta: f64, max_budget: u64) -> *mut CiteStopper {
    catch_unwind(|| {
        let budget = if max_budget == 0 { None } else { Some(max_budget) };
        match CiteStopper::new(delta, budget) {
            Ok(s) => Box::into_raw(Box::new(s)),
            Err(_) => std::ptr::null_mut(),
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cite_free(ptr: *mut CiteStopper) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cite_update(
    ptr: *mut CiteStopper,
    label_id: u64,
) -> ClotureModeResult {
    if ptr.is_null() {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        let dec = stopper.update(label_id);
        cite_decision_to_c(stopper, dec)
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cite_update_batch(
    ptr: *mut CiteStopper,
    labels: *const u64,
    n: usize,
) -> ClotureModeResult {
    if ptr.is_null() {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        if labels.is_null() || n == 0 {
            let dec = stopper.decision();
            return cite_decision_to_c(stopper, dec);
        }
        let sl = slice::from_raw_parts(labels, n);
        let mut dec = stopper.decision();
        for &lbl in sl {
            dec = stopper.update(lbl);
            if dec.certified {
                break;
            }
        }
        cite_decision_to_c(stopper, dec)
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cite_update_weighted(
    ptr: *mut CiteStopper,
    label_id: u64,
    weight: f64,
) -> ClotureModeResult {
    if ptr.is_null() {
        return ClotureModeResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stopper = &mut *ptr;
        let dec = stopper.update_weighted(label_id, weight);
        cite_decision_to_c(stopper, dec)
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cite_reset(ptr: *mut CiteStopper) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            (*ptr).reset();
        }));
    }
}

// ============================================================================
// Engine 1c: PopulationCert (Realized vs Population correlated rollouts)
// ============================================================================

#[no_mangle]
pub extern "C" fn cloture_pop_new(delta: f64) -> *mut PopulationCert {
    catch_unwind(|| match PopulationCert::new(delta) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_pop_free(ptr: *mut PopulationCert) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_pop_add_unit(
    ptr: *mut PopulationCert,
    scores: *const f64,
    len: usize,
) -> CloturePopResult {
    if ptr.is_null() {
        return CloturePopResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let pop = &mut *ptr;
        if scores.is_null() || len == 0 {
            return pop_decision_to_c(pop.decision());
        }
        let sl = slice::from_raw_parts(scores, len);
        pop_decision_to_c(pop.add_unit_rollouts(sl))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_pop_optimal_rollouts(
    ptr: *const PopulationCert,
    setup_cost: f64,
    rollout_cost: f64,
) -> usize {
    if ptr.is_null() {
        return 1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        (*ptr).optimal_rollouts_per_unit(setup_cost, rollout_cost)
    }))
    .unwrap_or(1)
}

// ============================================================================
// Engine 2: PanelAudit & OnlinePanelTracker
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn cloture_panel_audit(
    votes_flat: *const u32,
    gold: *const u32,
    strata: *const u32,
    n_items: usize,
    k_judges: usize,
    n_condorcet_sims: u32,
    n_perms: u32,
    n_boot: u32,
    seed: u64,
    out_summary: *mut CloturePanelSummary,
    out_phi_matrix: *mut f64,
    out_judge_acc: *mut f64,
    out_loo_delta_acc: *mut f64,
    out_loo_delta_neff: *mut f64,
) -> i32 {
    if votes_flat.is_null() || gold.is_null() || out_summary.is_null() || n_items == 0 || k_judges == 0 {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let total = match n_items.checked_mul(k_judges) {
            Some(v) => v,
            None => return -2,
        };
        let votes = slice::from_raw_parts(votes_flat, total);
        let gold_sl = slice::from_raw_parts(gold, n_items);
        let strata_vec: Option<Vec<usize>> = if strata.is_null() {
            None
        } else {
            let s = slice::from_raw_parts(strata, n_items);
            Some(s.iter().map(|&v| v as usize).collect())
        };
        let audit = match PanelAudit::from_votes(
            votes,
            gold_sl,
            n_items,
            k_judges,
            strata_vec.as_deref(),
        ) {
            Ok(a) => a,
            Err(_) => return -3,
        };
        if n_condorcet_sims > 1_000_000 || n_perms > 1_000_000 || n_boot > 1_000_000 {
            return -4;
        }
        let cfg = PanelAuditConfig {
            n_condorcet_sims: n_condorcet_sims as usize,
            n_permutations: n_perms as usize,
            n_bootstrap: n_boot as usize,
            subset_target_ratio: 0.90,
            seed,
        };
        let rep = audit.run_with_config(&cfg);
        *out_summary = CloturePanelSummary {
            k_judges: rep.k_judges as u32,
            n_items: rep.n_items as u32,
            panel_accuracy: rep.panel_accuracy,
            best_single_accuracy: rep.best_single_accuracy,
            best_single_index: rep.best_single_index as u32,
            panel_lift: rep.panel_lift,
            mean_phi: rep.mean_phi,
            std_phi: rep.std_phi,
            n_eff_kish: rep.n_eff_kish,
            n_eff_eigen: rep.n_eff_eigen,
            lambda_max: rep.lambda_max,
            effective_rank: rep.effective_rank,
            independence_ratio: rep.independence_ratio,
            asymptotic_ceiling: rep.asymptotic_ceiling,
            krippendorff_alpha: rep.krippendorff_alpha,
            unanimous_rate: rep.unanimous_rate,
            unanimous_accuracy: rep.unanimous_accuracy,
            all_wrong_rate: rep.all_wrong_rate,
            condorcet_predicted_accuracy: rep.condorcet_predicted_accuracy,
            condorcet_gap: rep.condorcet_gap,
            permutation_p_value: rep.permutation_p_value,
            n_eff_ci_low: rep.n_eff_ci_low,
            n_eff_ci_high: rep.n_eff_ci_high,
        };
        if !out_phi_matrix.is_null() {
            let dst = slice::from_raw_parts_mut(out_phi_matrix, k_judges * k_judges);
            dst.copy_from_slice(&rep.phi_matrix[..k_judges * k_judges]);
        }
        if !out_judge_acc.is_null() {
            let dst = slice::from_raw_parts_mut(out_judge_acc, k_judges);
            dst.copy_from_slice(&rep.judge_accuracies[..k_judges]);
        }
        if !out_loo_delta_acc.is_null() {
            let dst = slice::from_raw_parts_mut(out_loo_delta_acc, k_judges);
            for (j, item) in rep.loo_analysis.iter().enumerate().take(k_judges) {
                dst[j] = item.delta_accuracy;
            }
        }
        if !out_loo_delta_neff.is_null() {
            let dst = slice::from_raw_parts_mut(out_loo_delta_neff, k_judges);
            for (j, item) in rep.loo_analysis.iter().enumerate().take(k_judges) {
                dst[j] = item.delta_n_eff;
            }
        }
        0
    }))
    .unwrap_or(-99)
}

#[no_mangle]
pub extern "C" fn cloture_online_panel_new(k_judges: usize) -> *mut OnlinePanelTracker {
    catch_unwind(|| match OnlinePanelTracker::try_new(k_judges) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_online_panel_free(ptr: *mut OnlinePanelTracker) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_online_panel_observe(
    ptr: *mut OnlinePanelTracker,
    errors: *const u8,
    len: usize,
) {
    if ptr.is_null() || errors.is_null() || len == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let sl = slice::from_raw_parts(errors, len);
        (*ptr).observe(sl);
    }));
}

#[no_mangle]
pub unsafe extern "C" fn cloture_online_panel_neff_kish(ptr: *const OnlinePanelTracker) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).n_eff_kish())).unwrap_or(0.0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_online_panel_neff_eigen(ptr: *const OnlinePanelTracker) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).n_eff_eigen())).unwrap_or(0.0)
}

// ============================================================================
// Engine 3: EmpiricalBernsteinCS, BettingCS, PairedEProcess, AgentRace, ActivePPI
// ============================================================================

#[no_mangle]
pub extern "C" fn cloture_ebcs_new(
    alpha: f64,
    min_val: f64,
    max_val: f64,
) -> *mut EmpiricalBernsteinCS {
    catch_unwind(|| match EmpiricalBernsteinCS::with_bounds(alpha, min_val, max_val) {
        Ok(cs) => Box::into_raw(Box::new(cs)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ebcs_free(ptr: *mut EmpiricalBernsteinCS) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ebcs_update(
    ptr: *mut EmpiricalBernsteinCS,
    x: f64,
) -> ClotureCsResult {
    if ptr.is_null() {
        return ClotureCsResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| cs_bounds_to_c((*ptr).update(x)))).unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ebcs_update_batch(
    ptr: *mut EmpiricalBernsteinCS,
    xs: *const f64,
    n: usize,
) -> ClotureCsResult {
    if ptr.is_null() {
        return ClotureCsResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        if xs.is_null() || n == 0 {
            return cs_bounds_to_c((*ptr).bounds());
        }
        let sl = slice::from_raw_parts(xs, n);
        cs_bounds_to_c((*ptr).update_batch(sl))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn cloture_betcs_new(alpha: f64, grid_points: usize) -> *mut BettingCS {
    catch_unwind(|| match BettingCS::with_grid(alpha, grid_points) {
        Ok(cs) => Box::into_raw(Box::new(cs)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_betcs_free(ptr: *mut BettingCS) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_betcs_update(ptr: *mut BettingCS, x: f64) -> ClotureCsResult {
    if ptr.is_null() {
        return ClotureCsResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| cs_bounds_to_c((*ptr).update(x)))).unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_betcs_update_batch(
    ptr: *mut BettingCS,
    xs: *const f64,
    n: usize,
) -> ClotureCsResult {
    if ptr.is_null() {
        return ClotureCsResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        if xs.is_null() || n == 0 {
            return cs_bounds_to_c((*ptr).bounds());
        }
        let sl = slice::from_raw_parts(xs, n);
        cs_bounds_to_c((*ptr).update_batch(sl))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn cloture_paired_new(alpha: f64, scale: f64) -> *mut PairedEProcess {
    catch_unwind(|| match PairedEProcess::new(alpha, scale, true) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_paired_free(ptr: *mut PairedEProcess) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_paired_update(
    ptr: *mut PairedEProcess,
    a: f64,
    b: f64,
) -> CloturePairedResult {
    if ptr.is_null() {
        return CloturePairedResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| paired_status_to_c((*ptr).update(a, b)))).unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_paired_update_batch(
    ptr: *mut PairedEProcess,
    a: *const f64,
    b: *const f64,
    n: usize,
) -> CloturePairedResult {
    if ptr.is_null() {
        return CloturePairedResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let pe = &mut *ptr;
        if a.is_null() || b.is_null() || n == 0 {
            return paired_status_to_c(pe.status());
        }
        let sa = slice::from_raw_parts(a, n);
        let sb = slice::from_raw_parts(b, n);
        for i in 0..n {
            pe.update(sa[i], sb[i]);
        }
        paired_status_to_c(pe.status())
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn cloture_race_new(n_arms: usize, delta: f64, paired: i32) -> *mut AgentRace {
    catch_unwind(|| match AgentRace::new(n_arms, delta, paired != 0) {
        Ok(r) => Box::into_raw(Box::new(r)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_free(ptr: *mut AgentRace) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_observe_arm(
    ptr: *mut AgentRace,
    arm: usize,
    score: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let st = (*ptr).observe_arm(arm, score);
        st.winner.map(|w| w as i32).unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_observe_paired(
    ptr: *mut AgentRace,
    scores: *const f64,
    n_arms: usize,
) -> i32 {
    if ptr.is_null() || scores.is_null() || n_arms == 0 {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let sl = slice::from_raw_parts(scores, n_arms);
        let st = (*ptr).observe_paired(sl);
        st.winner.map(|w| w as i32).unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_next_arm_lucb(ptr: *const AgentRace) -> usize {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).next_arm_lucb())).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_winner(ptr: *const AgentRace) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        (*ptr).winner().map(|w| w as i32).unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_active_count(ptr: *const AgentRace) -> usize {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).active_count())).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_race_arm_active(ptr: *const AgentRace, arm: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let st = (*ptr).status();
        match st.arm_stats.get(arm) {
            Some(a) if a.active => 1,
            _ => 0,
        }
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_e_bh_discoveries(
    e_values: *const f64,
    n: usize,
    alpha: f64,
    out_indices: *mut usize,
    cap: usize,
) -> usize {
    if e_values.is_null() || n == 0 {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let sl = slice::from_raw_parts(e_values, n);
        let disc = e_bh_discoveries(sl, alpha);
        if !out_indices.is_null() && cap > 0 {
            let count = disc.len().min(cap);
            let dst = slice::from_raw_parts_mut(out_indices, count);
            dst.copy_from_slice(&disc[..count]);
        }
        disc.len()
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn cloture_ppi_new(
    alpha: f64,
    target_audit_rate: f64,
    min_audit_prob: f64,
) -> *mut ActivePPI {
    catch_unwind(|| match ActivePPI::new(alpha, target_audit_rate, min_audit_prob) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ppi_free(ptr: *mut ActivePPI) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ppi_suggest_prob(
    ptr: *const ActivePPI,
    judge_pred: f64,
    uncertainty_hint: f64,
) -> f64 {
    if ptr.is_null() {
        return 1.0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let hint = if uncertainty_hint.is_finite() && uncertainty_hint >= 0.0 {
            Some(uncertainty_hint)
        } else {
            None
        };
        (*ptr).suggest_audit_prob(judge_pred, hint)
    }))
    .unwrap_or(1.0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ppi_observe(
    ptr: *mut ActivePPI,
    judge_pred: f64,
    audited: i32,
    gold_label: f64,
    audit_prob: f64,
) -> CloturePpiResult {
    if ptr.is_null() {
        return CloturePpiResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let gold = if audited != 0 { Some(gold_label) } else { None };
        ppi_status_to_c((*ptr).observe(judge_pred, audited != 0, gold, audit_prob))
    }))
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn cloture_ppi_observe_batch(
    ptr: *mut ActivePPI,
    judge_preds: *const f64,
    audited: *const u8,
    gold_labels: *const f64,
    audit_probs: *const f64,
    n: usize,
) -> CloturePpiResult {
    if ptr.is_null() {
        return CloturePpiResult::default();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let ppi = &mut *ptr;
        if judge_preds.is_null()
            || audited.is_null()
            || gold_labels.is_null()
            || audit_probs.is_null()
            || n == 0
        {
            return ppi_status_to_c(ppi.status());
        }
        let fs = slice::from_raw_parts(judge_preds, n);
        let au = slice::from_raw_parts(audited, n);
        let ys = slice::from_raw_parts(gold_labels, n);
        let ps = slice::from_raw_parts(audit_probs, n);
        for i in 0..n {
            let is_aud = au[i] != 0;
            let gold = if is_aud { Some(ys[i]) } else { None };
            ppi.observe(fs[i], is_aud, gold, ps[i]);
        }
        ppi_status_to_c(ppi.status())
    }))
    .unwrap_or_default()
}

// ============================================================================
// Engine 4: SafeFuse, AbortCascade, OnlineConformal
// ============================================================================

#[no_mangle]
pub extern "C" fn cloture_fuse_new(
    n_agents: usize,
    quarantine_alpha: f64,
    fallback_alpha: f64,
) -> *mut SafeFuse {
    catch_unwind(|| match SafeFuse::new(n_agents, quarantine_alpha, fallback_alpha) {
        Ok(f) => Box::into_raw(Box::new(f)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_fuse_free(ptr: *mut SafeFuse) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_fuse_predict(
    ptr: *const SafeFuse,
    preds: *const f64,
    n: usize,
    out_fallback: *mut i32,
) -> f64 {
    if ptr.is_null() || preds.is_null() || n == 0 {
        if !out_fallback.is_null() {
            *out_fallback = 0;
        }
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let sl = slice::from_raw_parts(preds, n);
        let pred = (*ptr).predict(sl);
        if !out_fallback.is_null() {
            *out_fallback = if pred.used_fallback { 1 } else { 0 };
        }
        pred.fused
    }))
    .unwrap_or(f64::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_fuse_update(
    ptr: *mut SafeFuse,
    losses: *const f64,
    n: usize,
    baseline_loss: f64,
    has_baseline: i32,
) -> i32 {
    if ptr.is_null() || losses.is_null() || n == 0 {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let sl = slice::from_raw_parts(losses, n);
        let base = if has_baseline != 0 {
            Some(baseline_loss)
        } else {
            None
        };
        let upd = (*ptr).update(sl, base);
        upd.newly_quarantined.len() as i32
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_fuse_quarantined_count(ptr: *const SafeFuse) -> usize {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        (*ptr).quarantined().iter().filter(|&&q| q).count()
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_fuse_is_quarantined(ptr: *const SafeFuse, agent: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        match (*ptr).quarantined().get(agent) {
            Some(&true) => 1,
            _ => 0,
        }
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_fuse_weights(
    ptr: *const SafeFuse,
    out: *mut f64,
    cap: usize,
) -> usize {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let w = (*ptr).weights();
        if !out.is_null() && cap > 0 {
            let count = w.len().min(cap);
            let dst = slice::from_raw_parts_mut(out, count);
            dst.copy_from_slice(&w[..count]);
        }
        w.len()
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cascade_calibrate(
    scores_flat: *const f64,
    succeeded: *const u8,
    n_episodes: usize,
    n_stages: usize,
    target_recall: f64,
    stage_costs: *const f64,
    n_costs: usize,
) -> *mut AbortCascade {
    if scores_flat.is_null() || succeeded.is_null() || n_episodes == 0 || n_stages == 0 {
        return std::ptr::null_mut();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let total = match n_episodes.checked_mul(n_stages) {
            Some(v) => v,
            None => return std::ptr::null_mut(),
        };
        let flat = slice::from_raw_parts(scores_flat, total);
        let succ = slice::from_raw_parts(succeeded, n_episodes);
        let data: Vec<(&[f64], bool)> = flat
            .chunks_exact(n_stages)
            .zip(succ.iter())
            .map(|(row, &s)| (row, s != 0))
            .collect();
        let res = if !stage_costs.is_null() && n_costs >= n_stages {
            let costs = slice::from_raw_parts(stage_costs, n_costs);
            AbortCascade::calibrate_with_costs(&data, n_stages, target_recall, costs)
        } else {
            AbortCascade::calibrate(&data, n_stages, target_recall)
        };
        match res {
            Ok(c) => Box::into_raw(Box::new(c)),
            Err(_) => std::ptr::null_mut(),
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cascade_free(ptr: *mut AbortCascade) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cascade_should_abort(
    ptr: *const AbortCascade,
    stage: usize,
    score: f64,
) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        if (*ptr).should_abort(stage, score) {
            1
        } else {
            0
        }
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_cascade_thresholds(
    ptr: *const AbortCascade,
    out: *mut f64,
    cap: usize,
) -> usize {
    if ptr.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let t = (*ptr).thresholds();
        if !out.is_null() && cap > 0 {
            let count = t.len().min(cap);
            let dst = slice::from_raw_parts_mut(out, count);
            dst.copy_from_slice(&t[..count]);
        }
        t.len()
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn cloture_conformal_new(target_alpha: f64) -> *mut OnlineConformal {
    catch_unwind(|| match OnlineConformal::new(target_alpha) {
        Ok(c) => Box::into_raw(Box::new(c)),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_conformal_free(ptr: *mut OnlineConformal) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_conformal_update_pvalue(
    ptr: *mut OnlineConformal,
    p: f64,
) -> f64 {
    if ptr.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).update_pvalue(p))).unwrap_or(f64::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_conformal_update_miscovered(
    ptr: *mut OnlineConformal,
    miscovered: i32,
) -> f64 {
    if ptr.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).update_miscovered(miscovered != 0))).unwrap_or(f64::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_conformal_alpha_t(ptr: *const OnlineConformal) -> f64 {
    if ptr.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).alpha_t())).unwrap_or(f64::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_conformal_empirical_coverage(ptr: *const OnlineConformal) -> f64 {
    if ptr.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).empirical_coverage())).unwrap_or(f64::NAN)
}

// ============================================================================
// Engine 5: EvidenceSketch (<1 KB Associative Distributed Monoid)
// ============================================================================

#[no_mangle]
pub extern "C" fn cloture_sketch_new() -> *mut EvidenceSketch {
    catch_unwind(|| Box::into_raw(Box::new(EvidenceSketch::new()))).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_free(ptr: *mut EvidenceSketch) {
    if !ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_observe(
    ptr: *mut EvidenceSketch,
    score: f64,
    bet_increment: f64,
) {
    if ptr.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let bet = if bet_increment.is_finite() && bet_increment != 0.0 {
            Some(bet_increment)
        } else {
            None
        };
        (*ptr).observe(score, bet);
    }));
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_merge(
    dst: *mut EvidenceSketch,
    src: *const EvidenceSketch,
) -> i32 {
    if dst.is_null() || src.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        if std::ptr::eq(dst, src) {
            let cloned = (*src).clone();
            (*dst).merge(&cloned);
        } else {
            (*dst).merge(&*src);
        }
        0
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_subtract(
    dst: *mut EvidenceSketch,
    local: *const EvidenceSketch,
) -> i32 {
    if dst.is_null() || local.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let res = if std::ptr::eq(dst, local) {
            let cloned = (*local).clone();
            (*dst).subtract(&cloned)
        } else {
            (*dst).subtract(&*local)
        };
        match res {
            Ok(updated) => {
                *dst = updated;
                0
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_quantile(ptr: *const EvidenceSketch, q: f64) -> f64 {
    if ptr.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).quantile(q))).unwrap_or(f64::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_e_value(ptr: *const EvidenceSketch) -> f64 {
    if ptr.is_null() {
        return 1.0;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).e_value())).unwrap_or(1.0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_mean(ptr: *const EvidenceSketch) -> f64 {
    if ptr.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| (*ptr).mean())).unwrap_or(f64::NAN)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_serialized_len(ptr: *const EvidenceSketch) -> usize {
    if ptr.is_null() {
        return 0;
    }
    WIRE_SIZE
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_to_bytes(
    ptr: *const EvidenceSketch,
    buf: *mut u8,
    cap: usize,
) -> usize {
    if ptr.is_null() || buf.is_null() || cap < WIRE_SIZE {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let bytes = (*ptr).to_fixed_bytes();
        let dst = slice::from_raw_parts_mut(buf, WIRE_SIZE);
        dst.copy_from_slice(&bytes);
        WIRE_SIZE
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn cloture_sketch_from_bytes(
    buf: *const u8,
    len: usize,
) -> *mut EvidenceSketch {
    if buf.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let sl = slice::from_raw_parts(buf, len);
        match EvidenceSketch::from_bytes(sl) {
            Ok(s) => Box::into_raw(Box::new(s)),
            Err(_) => std::ptr::null_mut(),
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}
