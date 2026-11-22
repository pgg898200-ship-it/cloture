"""Zero-dependency ctypes bindings to libcloture_core (C ABI in include/cloture.h)."""
from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _find_library() -> Path:
    env_path = os.environ.get("CLOTURE_LIB")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    sys_name = platform.system()
    if sys_name == "Darwin":
        names = ["libcloture_core.dylib"]
    elif sys_name == "Windows":
        names = ["cloture_core.dll", "libcloture_core.dll"]
    else:
        names = ["libcloture_core.so"]

    here = Path(__file__).resolve().parent
    root = here.parents[1]
    candidates: List[Path] = []
    for name in names:
        candidates.append(here / name)
        candidates.append(root / "target" / "release" / name)
        candidates.append(root / "target" / "debug" / name)
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        "Could not locate libcloture_core shared library. "
        "Run `cargo build --release -p cloture-core` or set CLOTURE_LIB."
    )


class _CModeResult(ctypes.Structure):
    _fields_ = [
        ("certified", ctypes.c_int32),
        ("deterministic_lock", ctypes.c_int32),
        ("has_winner", ctypes.c_int32),
        ("has_leader", ctypes.c_int32),
        ("winner", ctypes.c_uint64),
        ("leader", ctypes.c_uint64),
        ("n_samples", ctypes.c_uint64),
        ("log_evidence", ctypes.c_double),
        ("log_threshold", ctypes.c_double),
        ("margin_estimate", ctypes.c_double),
        ("scope", ctypes.c_int32),
    ]


class _CPopResult(ctypes.Structure):
    _fields_ = [
        ("n_units", ctypes.c_uint64),
        ("total_rollouts", ctypes.c_uint64),
        ("certified", ctypes.c_int32),
        ("theta_hat", ctypes.c_double),
        ("sigma_b2", ctypes.c_double),
        ("sigma_w2", ctypes.c_double),
        ("rho_hat", ctypes.c_double),
        ("n_eff_within", ctypes.c_double),
        ("saturation_ceiling", ctypes.c_double),
        ("log_evidence", ctypes.c_double),
        ("log_threshold", ctypes.c_double),
    ]


class _CCsResult(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_uint64),
        ("mean", ctypes.c_double),
        ("lower", ctypes.c_double),
        ("upper", ctypes.c_double),
        ("width", ctypes.c_double),
    ]


class _CPairedResult(ctypes.Structure):
    _fields_ = [
        ("n_pairs", ctypes.c_uint64),
        ("mean_diff", ctypes.c_double),
        ("e_value_a_beats_b", ctypes.c_double),
        ("e_value_b_beats_a", ctypes.c_double),
        ("anytime_p_value", ctypes.c_double),
        ("winner", ctypes.c_int32),
    ]


class _CPpiResult(ctypes.Structure):
    _fields_ = [
        ("n_total", ctypes.c_uint64),
        ("n_audited", ctypes.c_uint64),
        ("mean_estimate", ctypes.c_double),
        ("lower", ctypes.c_double),
        ("upper", ctypes.c_double),
        ("width", ctypes.c_double),
        ("lambda_hat", ctypes.c_double),
        ("variance_reduction", ctypes.c_double),
    ]


class _CPanelSummary(ctypes.Structure):
    _fields_ = [
        ("k_judges", ctypes.c_uint32),
        ("n_items", ctypes.c_uint32),
        ("panel_accuracy", ctypes.c_double),
        ("best_single_accuracy", ctypes.c_double),
        ("best_single_index", ctypes.c_uint32),
        ("panel_lift", ctypes.c_double),
        ("mean_phi", ctypes.c_double),
        ("std_phi", ctypes.c_double),
        ("n_eff_kish", ctypes.c_double),
        ("n_eff_eigen", ctypes.c_double),
        ("lambda_max", ctypes.c_double),
        ("effective_rank", ctypes.c_double),
        ("independence_ratio", ctypes.c_double),
        ("asymptotic_ceiling", ctypes.c_double),
        ("krippendorff_alpha", ctypes.c_double),
        ("unanimous_rate", ctypes.c_double),
        ("unanimous_accuracy", ctypes.c_double),
        ("all_wrong_rate", ctypes.c_double),
        ("condorcet_predicted_accuracy", ctypes.c_double),
        ("condorcet_gap", ctypes.c_double),
        ("permutation_p_value", ctypes.c_double),
        ("n_eff_ci_low", ctypes.c_double),
        ("n_eff_ci_high", ctypes.c_double),
    ]


_LIB: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    lib = ctypes.CDLL(str(_find_library()))
    vp = ctypes.c_void_p
    dp = ctypes.POINTER(ctypes.c_double)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i32p = ctypes.POINTER(ctypes.c_int32)

    # ASC-D
    lib.cloture_ascd_new.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_uint64]
    lib.cloture_ascd_new.restype = vp
    lib.cloture_ascd_free.argtypes = [vp]
    lib.cloture_ascd_free.restype = None
    lib.cloture_ascd_update.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_ascd_update.restype = _CModeResult
    lib.cloture_ascd_update_batch.argtypes = [vp, dp, ctypes.c_size_t, ctypes.c_size_t]
    lib.cloture_ascd_update_batch.restype = _CModeResult
    lib.cloture_ascd_update_one_hot.argtypes = [vp, ctypes.c_size_t]
    lib.cloture_ascd_update_one_hot.restype = _CModeResult
    lib.cloture_ascd_update_weighted.argtypes = [vp, ctypes.c_size_t, ctypes.c_double]
    lib.cloture_ascd_update_weighted.restype = _CModeResult
    lib.cloture_ascd_reset.argtypes = [vp]
    lib.cloture_ascd_reset.restype = None

    # CITE
    lib.cloture_cite_new.argtypes = [ctypes.c_double, ctypes.c_uint64]
    lib.cloture_cite_new.restype = vp
    lib.cloture_cite_free.argtypes = [vp]
    lib.cloture_cite_free.restype = None
    lib.cloture_cite_update.argtypes = [vp, ctypes.c_uint64]
    lib.cloture_cite_update.restype = _CModeResult
    lib.cloture_cite_update_batch.argtypes = [vp, u64p, ctypes.c_size_t]
    lib.cloture_cite_update_batch.restype = _CModeResult
    lib.cloture_cite_update_weighted.argtypes = [vp, ctypes.c_uint64, ctypes.c_double]
    lib.cloture_cite_update_weighted.restype = _CModeResult
    lib.cloture_cite_reset.argtypes = [vp]
    lib.cloture_cite_reset.restype = None

    # PopulationCert
    lib.cloture_pop_new.argtypes = [ctypes.c_double]
    lib.cloture_pop_new.restype = vp
    lib.cloture_pop_free.argtypes = [vp]
    lib.cloture_pop_free.restype = None
    lib.cloture_pop_add_unit.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_pop_add_unit.restype = _CPopResult
    lib.cloture_pop_optimal_rollouts.argtypes = [vp, ctypes.c_double, ctypes.c_double]
    lib.cloture_pop_optimal_rollouts.restype = ctypes.c_size_t

    # PanelAudit
    lib.cloture_panel_audit.argtypes = [
        u32p, u32p, u32p,
        ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
        ctypes.POINTER(_CPanelSummary), dp, dp, dp, dp,
    ]
    lib.cloture_panel_audit.restype = ctypes.c_int32
    lib.cloture_online_panel_new.argtypes = [ctypes.c_size_t]
    lib.cloture_online_panel_new.restype = vp
    lib.cloture_online_panel_free.argtypes = [vp]
    lib.cloture_online_panel_free.restype = None
    lib.cloture_online_panel_observe.argtypes = [vp, u8p, ctypes.c_size_t]
    lib.cloture_online_panel_observe.restype = None
    lib.cloture_online_panel_neff_kish.argtypes = [vp]
    lib.cloture_online_panel_neff_kish.restype = ctypes.c_double
    lib.cloture_online_panel_neff_eigen.argtypes = [vp]
    lib.cloture_online_panel_neff_eigen.restype = ctypes.c_double

    # EBCS & BettingCS & PairedE
    lib.cloture_ebcs_new.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.cloture_ebcs_new.restype = vp
    lib.cloture_ebcs_free.argtypes = [vp]
    lib.cloture_ebcs_free.restype = None
    lib.cloture_ebcs_update.argtypes = [vp, ctypes.c_double]
    lib.cloture_ebcs_update.restype = _CCsResult
    lib.cloture_ebcs_update_batch.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_ebcs_update_batch.restype = _CCsResult

    lib.cloture_betcs_new.argtypes = [ctypes.c_double, ctypes.c_size_t]
    lib.cloture_betcs_new.restype = vp
    lib.cloture_betcs_free.argtypes = [vp]
    lib.cloture_betcs_free.restype = None
    lib.cloture_betcs_update.argtypes = [vp, ctypes.c_double]
    lib.cloture_betcs_update.restype = _CCsResult
    lib.cloture_betcs_update_batch.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_betcs_update_batch.restype = _CCsResult

    lib.cloture_paired_new.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.cloture_paired_new.restype = vp
    lib.cloture_paired_free.argtypes = [vp]
    lib.cloture_paired_free.restype = None
    lib.cloture_paired_update.argtypes = [vp, ctypes.c_double, ctypes.c_double]
    lib.cloture_paired_update.restype = _CPairedResult
    lib.cloture_paired_update_batch.argtypes = [vp, dp, dp, ctypes.c_size_t]
    lib.cloture_paired_update_batch.restype = _CPairedResult

    # Race
    lib.cloture_race_new.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_int32]
    lib.cloture_race_new.restype = vp
    lib.cloture_race_free.argtypes = [vp]
    lib.cloture_race_free.restype = None
    lib.cloture_race_observe_arm.argtypes = [vp, ctypes.c_size_t, ctypes.c_double]
    lib.cloture_race_observe_arm.restype = ctypes.c_int32
    lib.cloture_race_observe_paired.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_race_observe_paired.restype = ctypes.c_int32
    lib.cloture_race_next_arm_lucb.argtypes = [vp]
    lib.cloture_race_next_arm_lucb.restype = ctypes.c_size_t
    lib.cloture_race_winner.argtypes = [vp]
    lib.cloture_race_winner.restype = ctypes.c_int32
    lib.cloture_race_active_count.argtypes = [vp]
    lib.cloture_race_active_count.restype = ctypes.c_size_t
    lib.cloture_race_arm_active.argtypes = [vp, ctypes.c_size_t]
    lib.cloture_race_arm_active.restype = ctypes.c_int32

    # ActivePPI
    lib.cloture_ppi_new.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.cloture_ppi_new.restype = vp
    lib.cloture_ppi_free.argtypes = [vp]
    lib.cloture_ppi_free.restype = None
    lib.cloture_ppi_suggest_prob.argtypes = [vp, ctypes.c_double, ctypes.c_double]
    lib.cloture_ppi_suggest_prob.restype = ctypes.c_double
    lib.cloture_ppi_observe.argtypes = [vp, ctypes.c_double, ctypes.c_int32, ctypes.c_double, ctypes.c_double]
    lib.cloture_ppi_observe.restype = _CPpiResult
    lib.cloture_ppi_observe_batch.argtypes = [vp, dp, u8p, dp, dp, ctypes.c_size_t]
    lib.cloture_ppi_observe_batch.restype = _CPpiResult

    # SafeFuse
    lib.cloture_fuse_new.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_double]
    lib.cloture_fuse_new.restype = vp
    lib.cloture_fuse_free.argtypes = [vp]
    lib.cloture_fuse_free.restype = None
    lib.cloture_fuse_predict.argtypes = [vp, dp, ctypes.c_size_t, i32p]
    lib.cloture_fuse_predict.restype = ctypes.c_double
    lib.cloture_fuse_update.argtypes = [vp, dp, ctypes.c_size_t, ctypes.c_double, ctypes.c_int32]
    lib.cloture_fuse_update.restype = ctypes.c_int32
    lib.cloture_fuse_quarantined_count.argtypes = [vp]
    lib.cloture_fuse_quarantined_count.restype = ctypes.c_size_t
    lib.cloture_fuse_is_quarantined.argtypes = [vp, ctypes.c_size_t]
    lib.cloture_fuse_is_quarantined.restype = ctypes.c_int32
    lib.cloture_fuse_weights.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_fuse_weights.restype = ctypes.c_size_t

    # AbortCascade
    lib.cloture_cascade_calibrate.argtypes = [
        dp, u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, dp, ctypes.c_size_t
    ]
    lib.cloture_cascade_calibrate.restype = vp
    lib.cloture_cascade_free.argtypes = [vp]
    lib.cloture_cascade_free.restype = None
    lib.cloture_cascade_should_abort.argtypes = [vp, ctypes.c_size_t, ctypes.c_double]
    lib.cloture_cascade_should_abort.restype = ctypes.c_int32
    lib.cloture_cascade_thresholds.argtypes = [vp, dp, ctypes.c_size_t]
    lib.cloture_cascade_thresholds.restype = ctypes.c_size_t

    # OnlineConformal
    lib.cloture_conformal_new.argtypes = [ctypes.c_double]
    lib.cloture_conformal_new.restype = vp
    lib.cloture_conformal_free.argtypes = [vp]
    lib.cloture_conformal_free.restype = None
    lib.cloture_conformal_update_pvalue.argtypes = [vp, ctypes.c_double]
    lib.cloture_conformal_update_pvalue.restype = ctypes.c_double
    lib.cloture_conformal_update_miscovered.argtypes = [vp, ctypes.c_int32]
    lib.cloture_conformal_update_miscovered.restype = ctypes.c_double
    lib.cloture_conformal_alpha_t.argtypes = [vp]
    lib.cloture_conformal_alpha_t.restype = ctypes.c_double
    lib.cloture_conformal_empirical_coverage.argtypes = [vp]
    lib.cloture_conformal_empirical_coverage.restype = ctypes.c_double

    # e-BH
    lib.cloture_e_bh_discoveries.argtypes = [
        dp, ctypes.c_size_t, ctypes.c_double, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t
    ]
    lib.cloture_e_bh_discoveries.restype = ctypes.c_size_t

    # EvidenceSketch
    lib.cloture_sketch_new.argtypes = []
    lib.cloture_sketch_new.restype = vp
    lib.cloture_sketch_free.argtypes = [vp]
    lib.cloture_sketch_free.restype = None
    lib.cloture_sketch_observe.argtypes = [vp, ctypes.c_double, ctypes.c_double]
    lib.cloture_sketch_observe.restype = None
    lib.cloture_sketch_merge.argtypes = [vp, vp]
    lib.cloture_sketch_merge.restype = ctypes.c_int32
    lib.cloture_sketch_subtract.argtypes = [vp, vp]
    lib.cloture_sketch_subtract.restype = ctypes.c_int32
    lib.cloture_sketch_quantile.argtypes = [vp, ctypes.c_double]
    lib.cloture_sketch_quantile.restype = ctypes.c_double
    lib.cloture_sketch_e_value.argtypes = [vp]
    lib.cloture_sketch_e_value.restype = ctypes.c_double
    lib.cloture_sketch_mean.argtypes = [vp]
    lib.cloture_sketch_mean.restype = ctypes.c_double
    lib.cloture_sketch_serialized_len.argtypes = [vp]
    lib.cloture_sketch_serialized_len.restype = ctypes.c_size_t
    lib.cloture_sketch_to_bytes.argtypes = [vp, u8p, ctypes.c_size_t]
    lib.cloture_sketch_to_bytes.restype = ctypes.c_size_t
    lib.cloture_sketch_from_bytes.argtypes = [u8p, ctypes.c_size_t]
    lib.cloture_sketch_from_bytes.restype = vp

    _LIB = lib
    return lib


_SCOPE_NAMES = {
    0: "realized_unit",
    1: "population",
    2: "deterministic_budget",
}


def _as_f64_ptr(seq: Sequence[float]) -> Tuple[object, ctypes.POINTER(ctypes.c_double), int]:
    """Zero-copy pointer for contiguous float64 NumPy arrays; fast ctypes fallback otherwise."""
    if hasattr(seq, "__array_interface__") and getattr(seq, "dtype", None) is not None:
        if str(seq.dtype) == "float64" and seq.flags["C_CONTIGUOUS"]:
            ptr = seq.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return seq, ptr, int(seq.size)
    arr = (ctypes.c_double * len(seq))(*seq)
    return arr, arr, len(seq)


@dataclass(frozen=True)
class ModeDecision:
    certified: bool
    deterministic_lock: bool
    winner: Optional[int]
    leader: Optional[int]
    n_samples: int
    log_evidence: float
    log_threshold: float
    margin_estimate: float
    scope: str

    @classmethod
    def _from_c(cls, r: _CModeResult) -> "ModeDecision":
        return cls(
            certified=bool(r.certified),
            deterministic_lock=bool(r.deterministic_lock),
            winner=int(r.winner) if r.has_winner else None,
            leader=int(r.leader) if r.has_leader else None,
            n_samples=int(r.n_samples),
            log_evidence=float(r.log_evidence),
            log_threshold=float(r.log_threshold),
            margin_estimate=float(r.margin_estimate),
            scope=_SCOPE_NAMES.get(int(r.scope), "realized_unit"),
        )


@dataclass(frozen=True)
class PopDecision:
    n_units: int
    total_rollouts: int
    certified: bool
    theta_hat: float
    sigma_b2: float
    sigma_w2: float
    rho_hat: float
    n_eff_within: float
    saturation_ceiling: float
    log_evidence: float
    log_threshold: float

    @classmethod
    def _from_c(cls, r: _CPopResult) -> "PopDecision":
        return cls(
            n_units=int(r.n_units),
            total_rollouts=int(r.total_rollouts),
            certified=bool(r.certified),
            theta_hat=float(r.theta_hat),
            sigma_b2=float(r.sigma_b2),
            sigma_w2=float(r.sigma_w2),
            rho_hat=float(r.rho_hat),
            n_eff_within=float(r.n_eff_within),
            saturation_ceiling=float(r.saturation_ceiling),
            log_evidence=float(r.log_evidence),
            log_threshold=float(r.log_threshold),
        )


@dataclass(frozen=True)
class CsResult:
    n: int
    mean: float
    lower: float
    upper: float
    width: float

    @classmethod
    def _from_c(cls, r: _CCsResult) -> "CsResult":
        return cls(
            n=int(r.n),
            mean=float(r.mean),
            lower=float(r.lower),
            upper=float(r.upper),
            width=float(r.width),
        )


@dataclass(frozen=True)
class PairedResult:
    n_pairs: int
    mean_diff: float
    e_value_a_beats_b: float
    e_value_b_beats_a: float
    anytime_p_value: float
    winner: Optional[int]

    @classmethod
    def _from_c(cls, r: _CPairedResult) -> "PairedResult":
        return cls(
            n_pairs=int(r.n_pairs),
            mean_diff=float(r.mean_diff),
            e_value_a_beats_b=float(r.e_value_a_beats_b),
            e_value_b_beats_a=float(r.e_value_b_beats_a),
            anytime_p_value=float(r.anytime_p_value),
            winner=None if r.winner < 0 else int(r.winner),
        )


@dataclass(frozen=True)
class PpiResult:
    n_total: int
    n_audited: int
    mean_estimate: float
    lower: float
    upper: float
    width: float
    lambda_hat: float
    variance_reduction: float

    @classmethod
    def _from_c(cls, r: _CPpiResult) -> "PpiResult":
        return cls(
            n_total=int(r.n_total),
            n_audited=int(r.n_audited),
            mean_estimate=float(r.mean_estimate),
            lower=float(r.lower),
            upper=float(r.upper),
            width=float(r.width),
            lambda_hat=float(r.lambda_hat),
            variance_reduction=float(r.variance_reduction),
        )


@dataclass(frozen=True)
class PanelReport:
    k_judges: int
    n_items: int
    panel_accuracy: float
    best_single_accuracy: float
    best_single_index: int
    panel_lift: float
    mean_phi: float
    std_phi: float
    n_eff_kish: float
    n_eff_eigen: float
    lambda_max: float
    effective_rank: float
    independence_ratio: float
    asymptotic_ceiling: float
    krippendorff_alpha: float
    unanimous_rate: float
    unanimous_accuracy: float
    all_wrong_rate: float
    condorcet_predicted_accuracy: float
    condorcet_gap: float
    permutation_p_value: float
    n_eff_ci_low: float
    n_eff_ci_high: float
    phi_matrix: List[List[float]]
    judge_accuracies: List[float]
    loo_delta_acc: List[float]
    loo_delta_neff: List[float]


class AscdStopper:
    """Grey-box distribution-valued anytime-valid consensus stopper (ASC-D)."""

    def __init__(self, k: int, delta: float = 0.05, max_budget: Optional[int] = None) -> None:
        self._lib = _get_lib()
        self._k = int(k)
        self._ptr = self._lib.cloture_ascd_new(self._k, float(delta), int(max_budget or 0))
        if not self._ptr:
            raise ValueError(f"Invalid AscdStopper parameters: k={k}, delta={delta}")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_ascd_free(ptr)
            self._ptr = None

    def update(self, probs: Sequence[float]) -> ModeDecision:
        _keep, ptr, n = _as_f64_ptr(probs)
        return ModeDecision._from_c(self._lib.cloture_ascd_update(self._ptr, ptr, n))

    def update_batch(self, probs_2d: Sequence[Sequence[float]]) -> ModeDecision:
        if hasattr(probs_2d, "__array_interface__"):
            if getattr(probs_2d, "ndim", None) != 2 or probs_2d.shape[1] != self._k:
                raise ValueError(
                    f"probs_2d must have shape (n_rows, {self._k}), got {getattr(probs_2d, 'shape', None)}"
                )
            if str(probs_2d.dtype) == "float64" and probs_2d.flags["C_CONTIGUOUS"]:
                ptr = probs_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                return ModeDecision._from_c(
                    self._lib.cloture_ascd_update_batch(self._ptr, ptr, probs_2d.shape[0], probs_2d.shape[1])
                )
        n_rows = len(probs_2d)
        flat: List[float] = []
        for idx, row in enumerate(probs_2d):
            if len(row) != self._k:
                raise ValueError(
                    f"Row {idx} of probs_2d has length {len(row)}, expected k={self._k}"
                )
            flat.extend(float(x) for x in row)
        arr = (ctypes.c_double * len(flat))(*flat)
        return ModeDecision._from_c(self._lib.cloture_ascd_update_batch(self._ptr, arr, n_rows, self._k))

    def update_one_hot(self, label: int) -> ModeDecision:
        return ModeDecision._from_c(self._lib.cloture_ascd_update_one_hot(self._ptr, int(label)))

    def update_weighted(self, label: int, confidence: float) -> ModeDecision:
        return ModeDecision._from_c(self._lib.cloture_ascd_update_weighted(self._ptr, int(label), float(confidence)))

    def reset(self) -> None:
        self._lib.cloture_ascd_reset(self._ptr)


class CiteStopper:
    """Category-agnostic unbounded-category anytime-valid mode stopper (CITE + Kraft-IUT)."""

    def __init__(self, delta: float = 0.05, max_budget: Optional[int] = None) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_cite_new(float(delta), int(max_budget or 0))
        if not self._ptr:
            raise ValueError(f"Invalid CiteStopper parameters: delta={delta}")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_cite_free(ptr)
            self._ptr = None

    def update(self, label_id: int) -> ModeDecision:
        return ModeDecision._from_c(self._lib.cloture_cite_update(self._ptr, int(label_id)))

    def update_batch(self, labels: Sequence[int]) -> ModeDecision:
        if hasattr(labels, "__array_interface__") and str(labels.dtype) == "uint64" and labels.flags["C_CONTIGUOUS"]:
            ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            return ModeDecision._from_c(self._lib.cloture_cite_update_batch(self._ptr, ptr, int(labels.size)))
        arr = (ctypes.c_uint64 * len(labels))(*labels)
        return ModeDecision._from_c(self._lib.cloture_cite_update_batch(self._ptr, arr, len(labels)))

    def update_weighted(self, label_id: int, weight: float) -> ModeDecision:
        return ModeDecision._from_c(self._lib.cloture_cite_update_weighted(self._ptr, int(label_id), float(weight)))

    def reset(self) -> None:
        self._lib.cloture_cite_reset(self._ptr)


class PopulationCert:
    """Prompt-level anytime-valid e-process & correlated-rollout ICC diagnostics."""

    def __init__(self, delta: float = 0.05) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_pop_new(float(delta))
        if not self._ptr:
            raise ValueError(f"Invalid PopulationCert delta={delta}")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_pop_free(ptr)
            self._ptr = None

    def add_unit(self, scores: Sequence[float]) -> PopDecision:
        _keep, ptr, n = _as_f64_ptr(scores)
        return PopDecision._from_c(self._lib.cloture_pop_add_unit(self._ptr, ptr, n))

    def optimal_rollouts(self, unit_setup_cost: float, per_rollout_cost: float) -> int:
        return int(self._lib.cloture_pop_optimal_rollouts(self._ptr, float(unit_setup_cost), float(per_rollout_cost)))


def audit_panel(
    votes: Sequence[Sequence[int]],
    gold: Sequence[int],
    *,
    strata: Optional[Sequence[int]] = None,
    n_condorcet_sims: int = 2000,
    n_perms: int = 2000,
    n_boot: int = 1000,
    seed: int = 42,
) -> PanelReport:
    """Run full correlated-panel audit (Kish & Eigen n_eff, Condorcet gap, permutation p, LOO pruning)."""
    lib = _get_lib()
    n_items = len(gold)
    if n_items == 0:
        raise ValueError("Empty panel")
    if len(votes) != n_items:
        raise ValueError(f"votes length ({len(votes)}) must match gold length ({n_items})")
    if hasattr(votes, "__array_interface__"):
        if getattr(votes, "ndim", None) != 2 or votes.shape[0] != n_items:
            raise ValueError(f"votes must be 2D with {n_items} rows, got shape {getattr(votes, 'shape', None)}")
        k_judges = int(votes.shape[1])
    else:
        k_judges = len(votes[0])
    if k_judges == 0:
        raise ValueError("k_judges must be >= 1")

    if hasattr(votes, "__array_interface__") and str(votes.dtype) == "uint32" and votes.flags["C_CONTIGUOUS"]:
        v_arr = votes
        v_ptr = votes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    else:
        flat_v: List[int] = []
        for idx, row in enumerate(votes):
            if len(row) != k_judges:
                raise ValueError(f"Row {idx} of votes has length {len(row)}, expected {k_judges}")
            flat_v.extend(int(x) for x in row)
        v_arr = (ctypes.c_uint32 * len(flat_v))(*flat_v)
        v_ptr = v_arr

    if hasattr(gold, "__array_interface__"):
        if getattr(gold, "ndim", None) != 1 or gold.shape[0] != n_items:
            raise ValueError(f"gold must be 1D of length {n_items}")
    if hasattr(gold, "__array_interface__") and str(gold.dtype) == "uint32" and gold.flags["C_CONTIGUOUS"]:
        g_arr = gold
        g_ptr = gold.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    else:
        g_arr = (ctypes.c_uint32 * n_items)(*[int(x) for x in gold])
        g_ptr = g_arr

    if strata is not None:
        if len(strata) != n_items:
            raise ValueError(f"strata length ({len(strata)}) must match gold length ({n_items})")
        if hasattr(strata, "__array_interface__"):
            if getattr(strata, "ndim", None) != 1 or strata.shape[0] != n_items:
                raise ValueError(f"strata must be 1D of length {n_items}")
        if hasattr(strata, "__array_interface__") and str(strata.dtype) == "uint32" and strata.flags["C_CONTIGUOUS"]:
            s_arr = strata
            s_ptr = strata.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        else:
            s_arr = (ctypes.c_uint32 * n_items)(*[int(x) for x in strata])
            s_ptr = s_arr
    else:
        s_arr = None
        s_ptr = ctypes.POINTER(ctypes.c_uint32)()

    summary = _CPanelSummary()
    phi_flat = (ctypes.c_double * (k_judges * k_judges))()
    j_acc = (ctypes.c_double * k_judges)()
    loo_acc = (ctypes.c_double * k_judges)()
    loo_neff = (ctypes.c_double * k_judges)()

    rc = lib.cloture_panel_audit(
        v_ptr, g_ptr, s_ptr,
        n_items, k_judges,
        int(n_condorcet_sims), int(n_perms), int(n_boot), int(seed),
        ctypes.byref(summary), phi_flat, j_acc, loo_acc, loo_neff,
    )
    if rc != 0:
        raise ValueError(f"cloture_panel_audit failed with status {rc}")

    phi_mat = [
        [float(phi_flat[i * k_judges + j]) for j in range(k_judges)]
        for i in range(k_judges)
    ]
    return PanelReport(
        k_judges=int(summary.k_judges),
        n_items=int(summary.n_items),
        panel_accuracy=float(summary.panel_accuracy),
        best_single_accuracy=float(summary.best_single_accuracy),
        best_single_index=int(summary.best_single_index),
        panel_lift=float(summary.panel_lift),
        mean_phi=float(summary.mean_phi),
        std_phi=float(summary.std_phi),
        n_eff_kish=float(summary.n_eff_kish),
        n_eff_eigen=float(summary.n_eff_eigen),
        lambda_max=float(summary.lambda_max),
        effective_rank=float(summary.effective_rank),
        independence_ratio=float(summary.independence_ratio),
        asymptotic_ceiling=float(summary.asymptotic_ceiling),
        krippendorff_alpha=float(summary.krippendorff_alpha),
        unanimous_rate=float(summary.unanimous_rate),
        unanimous_accuracy=float(summary.unanimous_accuracy),
        all_wrong_rate=float(summary.all_wrong_rate),
        condorcet_predicted_accuracy=float(summary.condorcet_predicted_accuracy),
        condorcet_gap=float(summary.condorcet_gap),
        permutation_p_value=float(summary.permutation_p_value),
        n_eff_ci_low=float(summary.n_eff_ci_low),
        n_eff_ci_high=float(summary.n_eff_ci_high),
        phi_matrix=phi_mat,
        judge_accuracies=[float(j_acc[j]) for j in range(k_judges)],
        loo_delta_acc=[float(loo_acc[j]) for j in range(k_judges)],
        loo_delta_neff=[float(loo_neff[j]) for j in range(k_judges)],
    )


class OnlinePanelTracker:
    """Streaming O(k^2) effective-votes tracker for live multi-agent panels."""

    def __init__(self, k_judges: int) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_online_panel_new(int(k_judges))
        if not self._ptr:
            raise ValueError(f"Invalid k_judges={k_judges}")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_online_panel_free(ptr)
            self._ptr = None

    def observe(self, errors: Sequence[int]) -> None:
        arr = (ctypes.c_uint8 * len(errors))(*[1 if x else 0 for x in errors])
        self._lib.cloture_online_panel_observe(self._ptr, arr, len(errors))

    def n_eff_kish(self) -> float:
        return float(self._lib.cloture_online_panel_neff_kish(self._ptr))

    def n_eff_eigen(self) -> float:
        return float(self._lib.cloture_online_panel_neff_eigen(self._ptr))


class EmpiricalBernsteinCS:
    """O(1) sub-microsecond anytime-valid Predictable Plug-In Empirical-Bernstein CS."""

    def __init__(self, alpha: float = 0.05, min_val: float = 0.0, max_val: float = 1.0) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_ebcs_new(float(alpha), float(min_val), float(max_val))
        if not self._ptr:
            raise ValueError("Invalid EmpiricalBernsteinCS parameters")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_ebcs_free(ptr)
            self._ptr = None

    def update(self, x: float) -> CsResult:
        return CsResult._from_c(self._lib.cloture_ebcs_update(self._ptr, float(x)))

    def update_batch(self, xs: Sequence[float]) -> CsResult:
        _keep, ptr, n = _as_f64_ptr(xs)
        return CsResult._from_c(self._lib.cloture_ebcs_update_batch(self._ptr, ptr, n))


class BettingCS:
    """Tightest finite-sample Hedged ONS Betting Confidence Sequence on [0, 1]."""

    def __init__(self, alpha: float = 0.05, grid_points: int = 201) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_betcs_new(float(alpha), int(grid_points))
        if not self._ptr:
            raise ValueError("Invalid BettingCS parameters")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_betcs_free(ptr)
            self._ptr = None

    def update(self, x: float) -> CsResult:
        return CsResult._from_c(self._lib.cloture_betcs_update(self._ptr, float(x)))

    def update_batch(self, xs: Sequence[float]) -> CsResult:
        _keep, ptr, n = _as_f64_ptr(xs)
        return CsResult._from_c(self._lib.cloture_betcs_update_batch(self._ptr, ptr, n))


class PairedEProcess:
    """Exact paired anytime-valid e-process comparing Policy A vs Policy B."""

    def __init__(self, alpha: float = 0.05, scale: float = 1.0) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_paired_new(float(alpha), float(scale))
        if not self._ptr:
            raise ValueError("Invalid PairedEProcess parameters")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_paired_free(ptr)
            self._ptr = None

    def update(self, a: float, b: float) -> PairedResult:
        return PairedResult._from_c(self._lib.cloture_paired_update(self._ptr, float(a), float(b)))

    def update_batch(self, a: Sequence[float], b: Sequence[float]) -> PairedResult:
        _ka, pa, na = _as_f64_ptr(a)
        _kb, pb, nb = _as_f64_ptr(b)
        n = min(na, nb)
        return PairedResult._from_c(self._lib.cloture_paired_update_batch(self._ptr, pa, pb, n))


class AgentRace:
    """Anytime-valid multi-arm/multi-agent sequential racing with LUCB scheduling."""

    def __init__(self, n_arms: int, delta: float = 0.05, paired: bool = True) -> None:
        self._lib = _get_lib()
        self._n_arms = int(n_arms)
        self._ptr = self._lib.cloture_race_new(self._n_arms, float(delta), 1 if paired else 0)
        if not self._ptr:
            raise ValueError("Invalid AgentRace parameters")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_race_free(ptr)
            self._ptr = None

    def observe_arm(self, arm: int, score: float) -> Optional[int]:
        w = int(self._lib.cloture_race_observe_arm(self._ptr, int(arm), float(score)))
        return None if w < 0 else w

    def observe_paired(self, scores: Sequence[float]) -> Optional[int]:
        _keep, ptr, n = _as_f64_ptr(scores)
        w = int(self._lib.cloture_race_observe_paired(self._ptr, ptr, n))
        return None if w < 0 else w

    def next_arm_lucb(self) -> int:
        return int(self._lib.cloture_race_next_arm_lucb(self._ptr))

    def winner(self) -> Optional[int]:
        w = int(self._lib.cloture_race_winner(self._ptr))
        return None if w < 0 else w

    def active_count(self) -> int:
        return int(self._lib.cloture_race_active_count(self._ptr))

    def is_active(self, arm: int) -> bool:
        return bool(self._lib.cloture_race_arm_active(self._ptr, int(arm)))


class ActivePPI:
    """Anytime-valid online Active Prediction-Powered Inference (Active-PPI++ / AAELI)."""

    def __init__(self, alpha: float = 0.05, target_audit_rate: float = 0.10, min_audit_prob: float = 0.05) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_ppi_new(float(alpha), float(target_audit_rate), float(min_audit_prob))
        if not self._ptr:
            raise ValueError("Invalid ActivePPI parameters")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_ppi_free(ptr)
            self._ptr = None

    def suggest_prob(self, judge_pred: float, uncertainty_hint: float = -1.0) -> float:
        return float(self._lib.cloture_ppi_suggest_prob(self._ptr, float(judge_pred), float(uncertainty_hint)))

    def observe(self, judge_pred: float, audited: bool, gold_label: float, audit_prob: float) -> PpiResult:
        return PpiResult._from_c(
            self._lib.cloture_ppi_observe(
                self._ptr, float(judge_pred), 1 if audited else 0, float(gold_label), float(audit_prob)
            )
        )

    def observe_batch(
        self,
        judge_preds: Sequence[float],
        audited: Sequence[int],
        gold_labels: Sequence[float],
        audit_probs: Sequence[float],
    ) -> PpiResult:
        _kf, pf, n = _as_f64_ptr(judge_preds)
        _kg, pg, n_g = _as_f64_ptr(gold_labels)
        _kp, pp, n_p = _as_f64_ptr(audit_probs)
        n_a = len(audited)
        if not (n == n_a == n_g == n_p):
            raise ValueError(
                f"Batch lengths must match: judge_preds={n}, audited={n_a}, "
                f"gold_labels={n_g}, audit_probs={n_p}"
            )
        if hasattr(audited, "__array_interface__") and str(audited.dtype) == "uint8" and audited.flags["C_CONTIGUOUS"]:
            if getattr(audited, "ndim", None) != 1 or audited.shape[0] != n:
                raise ValueError(f"audited must be 1D of length {n}")
            a_keep = audited
            pa = audited.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        else:
            a_keep = (ctypes.c_uint8 * n)(*[1 if x else 0 for x in audited])
            pa = a_keep
        return PpiResult._from_c(self._lib.cloture_ppi_observe_batch(self._ptr, pf, pa, pg, pp, n))


class SafeFuse:
    """Scale-Free BOA multi-agent fusion with anytime-valid Ville E-process poisoning quarantine."""

    def __init__(self, n_agents: int, quarantine_alpha: float = 0.01, fallback_alpha: float = 0.05) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_fuse_new(int(n_agents), float(quarantine_alpha), float(fallback_alpha))
        if not self._ptr:
            raise ValueError("Invalid SafeFuse parameters")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_fuse_free(ptr)
            self._ptr = None

    def predict(self, preds: Sequence[float]) -> Tuple[float, bool]:
        _keep, ptr, n = _as_f64_ptr(preds)
        fb = ctypes.c_int32(0)
        val = float(self._lib.cloture_fuse_predict(self._ptr, ptr, n, ctypes.byref(fb)))
        return val, bool(fb.value)

    def update(self, losses: Sequence[float], baseline_loss: Optional[float] = None) -> int:
        _keep, ptr, n = _as_f64_ptr(losses)
        has_b = 1 if baseline_loss is not None else 0
        bval = float(baseline_loss) if baseline_loss is not None else 0.0
        return int(self._lib.cloture_fuse_update(self._ptr, ptr, n, bval, has_b))

    def quarantined_count(self) -> int:
        return int(self._lib.cloture_fuse_quarantined_count(self._ptr))

    def is_quarantined(self, agent: int) -> bool:
        return bool(self._lib.cloture_fuse_is_quarantined(self._ptr, int(agent)))

    def weights(self) -> List[float]:
        n = int(self._lib.cloture_fuse_weights(self._ptr, None, 0))
        if n == 0:
            return []
        buf = (ctypes.c_double * n)()
        written = int(self._lib.cloture_fuse_weights(self._ptr, buf, n))
        return [float(buf[i]) for i in range(written)]


class AbortCascade:
    """Recall-controlled multi-stage trajectory abort & escalation cascade (CRC)."""

    def __init__(self, _ptr: ctypes.c_void_p, n_stages: int) -> None:
        self._lib = _get_lib()
        self._ptr = _ptr
        self._n_stages = n_stages

    @classmethod
    def calibrate(
        cls,
        scores: Sequence[Sequence[float]],
        succeeded: Sequence[int],
        target_recall: float = 0.92,
        stage_costs: Optional[Sequence[float]] = None,
    ) -> "AbortCascade":
        lib = _get_lib()
        n_ep = len(succeeded)
        if n_ep == 0:
            raise ValueError("Empty calibration data")
        if len(scores) != n_ep:
            raise ValueError(f"scores length ({len(scores)}) must match succeeded length ({n_ep})")
        if hasattr(scores, "__array_interface__"):
            if getattr(scores, "ndim", None) != 2 or scores.shape[0] != n_ep:
                raise ValueError(f"scores must be 2D with {n_ep} rows, got shape {getattr(scores, 'shape', None)}")
            n_stages = int(scores.shape[1])
        else:
            n_stages = len(scores[0])
        if n_stages == 0:
            raise ValueError("n_stages must be >= 1")
        if hasattr(scores, "__array_interface__") and str(scores.dtype) == "float64" and scores.flags["C_CONTIGUOUS"]:
            s_keep = scores
            s_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            flat_s: List[float] = []
            for idx, row in enumerate(scores):
                if len(row) != n_stages:
                    raise ValueError(f"Row {idx} of scores has length {len(row)}, expected {n_stages}")
                flat_s.extend(float(x) for x in row)
            s_keep = (ctypes.c_double * len(flat_s))(*flat_s)
            s_ptr = s_keep
        if hasattr(succeeded, "__array_interface__"):
            if getattr(succeeded, "ndim", None) != 1 or succeeded.shape[0] != n_ep:
                raise ValueError(f"succeeded must be 1D of length {n_ep}")
        if hasattr(succeeded, "__array_interface__") and str(succeeded.dtype) == "uint8" and succeeded.flags["C_CONTIGUOUS"]:
            y_keep = succeeded
            y_ptr = succeeded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        else:
            y_keep = (ctypes.c_uint8 * n_ep)(*[1 if x else 0 for x in succeeded])
            y_ptr = y_keep
        if stage_costs is not None:
            _ck, c_ptr, n_costs = _as_f64_ptr(stage_costs)
        else:
            _ck = None
            c_ptr = ctypes.POINTER(ctypes.c_double)()
            n_costs = 0
        ptr = lib.cloture_cascade_calibrate(
            s_ptr, y_ptr, n_ep, n_stages, float(target_recall), c_ptr, n_costs
        )
        if not ptr:
            raise ValueError("Invalid AbortCascade calibration inputs")
        return cls(_ptr=ptr, n_stages=n_stages)

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_cascade_free(ptr)
            self._ptr = None

    def should_abort(self, stage: int, score: float) -> bool:
        return bool(self._lib.cloture_cascade_should_abort(self._ptr, int(stage), float(score)))

    def thresholds(self) -> List[float]:
        n = int(self._lib.cloture_cascade_thresholds(self._ptr, None, 0))
        if n == 0:
            return []
        buf = (ctypes.c_double * n)()
        written = int(self._lib.cloture_cascade_thresholds(self._ptr, buf, n))
        return [float(buf[i]) for i in range(written)]


class OnlineConformal:
    """4-rate crossing-free AgACI / DtACI online conformal controller."""

    def __init__(self, target_alpha: float = 0.10) -> None:
        self._lib = _get_lib()
        self._ptr = self._lib.cloture_conformal_new(float(target_alpha))
        if not self._ptr:
            raise ValueError(f"Invalid target_alpha={target_alpha}")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_conformal_free(ptr)
            self._ptr = None

    def update_pvalue(self, p: float) -> float:
        return float(self._lib.cloture_conformal_update_pvalue(self._ptr, float(p)))

    def update_miscovered(self, miscovered: bool) -> float:
        return float(self._lib.cloture_conformal_update_miscovered(self._ptr, 1 if miscovered else 0))

    @property
    def alpha_t(self) -> float:
        return float(self._lib.cloture_conformal_alpha_t(self._ptr))

    @property
    def empirical_coverage(self) -> float:
        return float(self._lib.cloture_conformal_empirical_coverage(self._ptr))


def e_bh_discoveries(e_values: Sequence[float], alpha: float = 0.05) -> List[int]:
    """Wang-Ramdas e-BH multiple-testing discovery filter (FDR <= alpha under arbitrary dependence)."""
    lib = _get_lib()
    if len(e_values) == 0:
        return []
    _keep, ptr, n = _as_f64_ptr(e_values)
    out = (ctypes.c_size_t * n)()
    count = int(lib.cloture_e_bh_discoveries(ptr, n, float(alpha), out, n))
    return [int(out[i]) for i in range(min(count, n))]


class EvidenceSketch:
    """<1 KB associative, commutative distributed calibration & e-process monoid."""

    def __init__(self, _ptr: Optional[ctypes.c_void_p] = None) -> None:
        self._lib = _get_lib()
        self._ptr = _ptr if _ptr is not None else self._lib.cloture_sketch_new()
        if not self._ptr:
            raise RuntimeError("Failed to allocate EvidenceSketch")

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.cloture_sketch_free(ptr)
            self._ptr = None

    def observe(self, score: float, bet_increment: float = 0.0) -> None:
        self._lib.cloture_sketch_observe(self._ptr, float(score), float(bet_increment))

    def merge(self, other: "EvidenceSketch") -> None:
        if self._lib.cloture_sketch_merge(self._ptr, other._ptr) != 0:
            raise ValueError("Failed to merge EvidenceSketch")

    def subtract(self, local: "EvidenceSketch") -> None:
        if self._lib.cloture_sketch_subtract(self._ptr, local._ptr) != 0:
            raise ValueError("Cannot subtract local sketch (counts underflow)")

    def quantile(self, q: float) -> float:
        return float(self._lib.cloture_sketch_quantile(self._ptr, float(q)))

    def e_value(self) -> float:
        return float(self._lib.cloture_sketch_e_value(self._ptr))

    def mean(self) -> float:
        return float(self._lib.cloture_sketch_mean(self._ptr))

    def to_bytes(self) -> bytes:
        n = int(self._lib.cloture_sketch_serialized_len(self._ptr))
        buf = (ctypes.c_uint8 * n)()
        written = int(self._lib.cloture_sketch_to_bytes(self._ptr, buf, n))
        return bytes(buf[:written])

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvidenceSketch":
        lib = _get_lib()
        arr = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ptr = lib.cloture_sketch_from_bytes(arr, len(data))
        if not ptr:
            raise ValueError("Invalid EvidenceSketch binary payload")
        return cls(_ptr=ptr)
