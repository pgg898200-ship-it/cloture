/*
 * cloture.h — Zero-dependency C ABI for cloture-core
 *
 * Anytime-valid multi-agent consensus stopping, correlated-panel auditing,
 * sequential agent racing, active prediction-powered inference, safe fusion,
 * recall-controlled abort cascades, and <2 KB distributed evidence monoids.
 *
 * License: MIT OR Apache-2.0
 */

#ifndef CLOTURE_H
#define CLOTURE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct ClotureAscd ClotureAscd;
typedef struct ClotureCite ClotureCite;
typedef struct CloturePopCert CloturePopCert;
typedef struct ClotureEbCs ClotureEbCs;
typedef struct ClotureBettingCs ClotureBettingCs;
typedef struct CloturePairedE CloturePairedE;
typedef struct ClotureRace ClotureRace;
typedef struct ClotureActivePpi ClotureActivePpi;
typedef struct ClotureSafeFuse ClotureSafeFuse;
typedef struct ClotureSketch ClotureSketch;
typedef struct ClotureOnlinePanel ClotureOnlinePanel;
typedef struct ClotureCascade ClotureCascade;
typedef struct ClotureOnlineConformal ClotureOnlineConformal;

/* CertScope values */
#define CLOTURE_SCOPE_REALIZED_UNIT       0
#define CLOTURE_SCOPE_POPULATION          1
#define CLOTURE_SCOPE_DETERMINISTIC_LOCK  2

typedef struct {
    int32_t certified;          /* 1 if certified, 0 otherwise */
    int32_t deterministic_lock; /* 1 if locked by finite max_budget */
    int32_t has_winner;         /* 1 if winner is valid, 0 otherwise */
    int32_t has_leader;         /* 1 if leader is valid, 0 otherwise */
    uint64_t winner;            /* winner index / u64 label_id (valid iff has_winner == 1) */
    uint64_t leader;            /* current empirical leader / u64 label_id (valid iff has_leader == 1) */
    uint64_t n_samples;
    double log_evidence;
    double log_threshold;
    double margin_estimate;
    int32_t scope;
} ClotureModeResult;

typedef struct {
    uint64_t n_units;
    uint64_t total_rollouts;
    int32_t certified;
    double theta_hat;
    double sigma_b2;
    double sigma_w2;
    double rho_hat;
    double n_eff_within;
    double saturation_ceiling;
    double log_evidence;
    double log_threshold;
} CloturePopResult;

typedef struct {
    uint64_t n;
    double mean;
    double lower;
    double upper;
    double width;
} ClotureCsResult;

typedef struct {
    uint64_t n_pairs;
    double mean_diff;
    double e_value_a_beats_b;
    double e_value_b_beats_a;
    double anytime_p_value;
    int32_t winner; /* -1 none, 0 A wins, 1 B wins */
} CloturePairedResult;

typedef struct {
    uint64_t n_total;
    uint64_t n_audited;
    double mean_estimate;
    double lower;
    double upper;
    double width;
    double lambda_hat;
    double variance_reduction;
} CloturePpiResult;

typedef struct {
    uint32_t k_judges;
    uint32_t n_items;
    double panel_accuracy;
    double best_single_accuracy;
    uint32_t best_single_index;
    double panel_lift;
    double mean_phi;
    double std_phi;
    double n_eff_kish;
    double n_eff_eigen;
    double lambda_max;
    double effective_rank;
    double independence_ratio;
    double asymptotic_ceiling;
    double krippendorff_alpha;
    double unanimous_rate;
    double unanimous_accuracy;
    double all_wrong_rate;
    double condorcet_predicted_accuracy;
    double condorcet_gap;
    double permutation_p_value;
    double n_eff_ci_low;
    double n_eff_ci_high;
} CloturePanelSummary;

/* ---- Engine 1: ASC-D (Grey-box distribution-valued mode stopping) ---- */
ClotureAscd *cloture_ascd_new(size_t k, double delta, uint64_t max_budget);
void cloture_ascd_free(ClotureAscd *ptr);
ClotureModeResult cloture_ascd_update(ClotureAscd *ptr, const double *probs, size_t len);
ClotureModeResult cloture_ascd_update_batch(ClotureAscd *ptr, const double *probs_flat, size_t n_rows, size_t k);
ClotureModeResult cloture_ascd_update_one_hot(ClotureAscd *ptr, size_t label);
ClotureModeResult cloture_ascd_update_weighted(ClotureAscd *ptr, size_t label, double confidence);
void cloture_ascd_reset(ClotureAscd *ptr);

/* ---- Engine 1b: CITE (Category-agnostic unbounded-category stopping) ---- */
ClotureCite *cloture_cite_new(double delta, uint64_t max_budget);
void cloture_cite_free(ClotureCite *ptr);
ClotureModeResult cloture_cite_update(ClotureCite *ptr, uint64_t label_id);
ClotureModeResult cloture_cite_update_batch(ClotureCite *ptr, const uint64_t *labels, size_t n);
ClotureModeResult cloture_cite_update_weighted(ClotureCite *ptr, uint64_t label_id, double weight);
void cloture_cite_reset(ClotureCite *ptr);

/* ---- Engine 1c: PopulationCert (Realized vs Population correlated rollouts) ---- */
CloturePopCert *cloture_pop_new(double delta);
void cloture_pop_free(CloturePopCert *ptr);
CloturePopResult cloture_pop_add_unit(CloturePopCert *ptr, const double *scores, size_t len);
size_t cloture_pop_optimal_rollouts(const CloturePopCert *ptr, double setup_cost, double rollout_cost);

/* ---- Engine 2: PanelAudit & OnlinePanelTracker ---- */
int32_t cloture_panel_audit(
    const uint32_t *votes_flat,   /* n_items * k_judges row-major */
    const uint32_t *gold,         /* n_items */
    const uint32_t *strata,       /* optional n_items (NULL for single stratum) */
    size_t n_items,
    size_t k_judges,
    uint32_t n_condorcet_sims,
    uint32_t n_perms,
    uint32_t n_boot,
    uint64_t seed,
    CloturePanelSummary *out_summary,
    double *out_phi_matrix,       /* optional k*k */
    double *out_judge_acc,        /* optional k */
    double *out_loo_delta_acc,    /* optional k */
    double *out_loo_delta_neff    /* optional k */
);

ClotureOnlinePanel *cloture_online_panel_new(size_t k_judges);
void cloture_online_panel_free(ClotureOnlinePanel *ptr);
void cloture_online_panel_observe(ClotureOnlinePanel *ptr, const uint8_t *errors, size_t len);
double cloture_online_panel_neff_kish(const ClotureOnlinePanel *ptr);
double cloture_online_panel_neff_eigen(const ClotureOnlinePanel *ptr);

/* ---- Engine 3: Confidence Sequences, Paired E-Process, Race, Active-PPI++ ---- */
ClotureEbCs *cloture_ebcs_new(double alpha, double min_val, double max_val);
void cloture_ebcs_free(ClotureEbCs *ptr);
ClotureCsResult cloture_ebcs_update(ClotureEbCs *ptr, double x);
ClotureCsResult cloture_ebcs_update_batch(ClotureEbCs *ptr, const double *xs, size_t n);

ClotureBettingCs *cloture_betcs_new(double alpha, size_t grid_points);
void cloture_betcs_free(ClotureBettingCs *ptr);
ClotureCsResult cloture_betcs_update(ClotureBettingCs *ptr, double x);
ClotureCsResult cloture_betcs_update_batch(ClotureBettingCs *ptr, const double *xs, size_t n);

CloturePairedE *cloture_paired_new(double alpha, double scale);
void cloture_paired_free(CloturePairedE *ptr);
CloturePairedResult cloture_paired_update(CloturePairedE *ptr, double a, double b);
CloturePairedResult cloture_paired_update_batch(CloturePairedE *ptr, const double *a, const double *b, size_t n);

ClotureRace *cloture_race_new(size_t n_arms, double delta, int32_t paired);
void cloture_race_free(ClotureRace *ptr);
int32_t cloture_race_observe_arm(ClotureRace *ptr, size_t arm, double score);
int32_t cloture_race_observe_paired(ClotureRace *ptr, const double *scores, size_t n_arms);
size_t cloture_race_next_arm_lucb(const ClotureRace *ptr);
int32_t cloture_race_winner(const ClotureRace *ptr); /* -1 if undecided */
size_t cloture_race_active_count(const ClotureRace *ptr);
int32_t cloture_race_arm_active(const ClotureRace *ptr, size_t arm);

ClotureActivePpi *cloture_ppi_new(double alpha, double target_audit_rate, double min_audit_prob);
void cloture_ppi_free(ClotureActivePpi *ptr);
double cloture_ppi_suggest_prob(const ClotureActivePpi *ptr, double judge_pred, double uncertainty_hint);
CloturePpiResult cloture_ppi_observe(ClotureActivePpi *ptr, double judge_pred, int32_t audited, double gold_label, double audit_prob);
CloturePpiResult cloture_ppi_observe_batch(
    ClotureActivePpi *ptr,
    const double *judge_preds,
    const uint8_t *audited,
    const double *gold_labels,
    const double *audit_probs,
    size_t n
);

/* ---- Engine 4: SafeFuse (Scale-Free BOA + Ville E-Process Quarantine) ---- */
ClotureSafeFuse *cloture_fuse_new(size_t n_agents, double quarantine_alpha, double fallback_alpha);
void cloture_fuse_free(ClotureSafeFuse *ptr);
double cloture_fuse_predict(const ClotureSafeFuse *ptr, const double *preds, size_t n, int32_t *out_fallback);
int32_t cloture_fuse_update(ClotureSafeFuse *ptr, const double *losses, size_t n, double baseline_loss, int32_t has_baseline);
size_t cloture_fuse_quarantined_count(const ClotureSafeFuse *ptr);
int32_t cloture_fuse_is_quarantined(const ClotureSafeFuse *ptr, size_t agent);
size_t cloture_fuse_weights(const ClotureSafeFuse *ptr, double *out, size_t cap);

/* ---- Engine 4b: AbortCascade & OnlineConformal (AgACI) ---- */
ClotureCascade *cloture_cascade_calibrate(
    const double *scores_flat,
    const uint8_t *succeeded,
    size_t n_episodes,
    size_t n_stages,
    double target_recall,
    const double *stage_costs,
    size_t n_costs
);
void cloture_cascade_free(ClotureCascade *ptr);
int32_t cloture_cascade_should_abort(const ClotureCascade *ptr, size_t stage, double score);
size_t cloture_cascade_thresholds(const ClotureCascade *ptr, double *out, size_t cap);

ClotureOnlineConformal *cloture_conformal_new(double target_alpha);
void cloture_conformal_free(ClotureOnlineConformal *ptr);
double cloture_conformal_update_pvalue(ClotureOnlineConformal *ptr, double p);
double cloture_conformal_update_miscovered(ClotureOnlineConformal *ptr, int32_t miscovered);
double cloture_conformal_alpha_t(const ClotureOnlineConformal *ptr);
double cloture_conformal_empirical_coverage(const ClotureOnlineConformal *ptr);

/* ---- Multiple Testing: Wang-Ramdas e-BH ---- */
size_t cloture_e_bh_discoveries(
    const double *e_values,
    size_t n,
    double alpha,
    size_t *out_indices,
    size_t cap
);

/* ---- Engine 5: EvidenceSketch (<1 KB Associative Distributed Monoid) ---- */
ClotureSketch *cloture_sketch_new(void);
void cloture_sketch_free(ClotureSketch *ptr);
void cloture_sketch_observe(ClotureSketch *ptr, double score, double bet_increment);
int32_t cloture_sketch_merge(ClotureSketch *dst, const ClotureSketch *src);
int32_t cloture_sketch_subtract(ClotureSketch *dst, const ClotureSketch *local);
double cloture_sketch_quantile(const ClotureSketch *ptr, double q);
double cloture_sketch_e_value(const ClotureSketch *ptr);
double cloture_sketch_mean(const ClotureSketch *ptr);
size_t cloture_sketch_serialized_len(const ClotureSketch *ptr);
size_t cloture_sketch_to_bytes(const ClotureSketch *ptr, uint8_t *buf, size_t cap);
ClotureSketch *cloture_sketch_from_bytes(const uint8_t *buf, size_t len);

/* ---- Portable Memory & Pointer-Out Helpers (WASM / Go / Node / Browser) ---- */
uint8_t *cloture_alloc(size_t bytes);
void cloture_dealloc(uint8_t *ptr, size_t bytes);
int32_t cloture_ascd_update_into(ClotureAscd *ptr, const double *probs, size_t len, ClotureModeResult *out);
int32_t cloture_cite_update_into(ClotureCite *ptr, uint64_t label_id, ClotureModeResult *out);
int32_t cloture_pop_add_unit_into(CloturePopCert *ptr, const double *scores, size_t len, CloturePopResult *out);
int32_t cloture_ebcs_update_into(ClotureEbCs *ptr, double x, ClotureCsResult *out);
int32_t cloture_betcs_update_into(ClotureBetCs *ptr, double x, ClotureCsResult *out);
int32_t cloture_ppi_observe_into(CloturePpi *ptr, double judge_pred, int32_t audited, double gold_label, double audit_prob, CloturePpiResult *out);

#ifdef __cplusplus
}
#endif

#endif /* CLOTURE_H */
