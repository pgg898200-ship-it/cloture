//! Zero-dependency JSON-RPC 2.0 Model Context Protocol (MCP) server over stdio.
//!
//! Exposes `cloture`'s six anytime-valid multi-agent verification engines as
//! deterministic MCP tools so AI coding agents, orchestrators, and evaluation
//! harnesses can invoke exact statistical stopping and panel auditing directly.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

use cloture_core::fuse::SafeFuse;
use cloture_core::mode::{AscdStopper, CertScope, CiteStopper, PopulationCert};
use cloture_core::panel::{PanelAudit, PanelAuditConfig};
use cloture_core::ppi::ActivePPI;
use cloture_core::race::{e_bh_discoveries, AgentRace};

use crate::json::{parse_json, Json};

const PROTOCOL_VERSION: &str = "2024-11-05";

pub fn run_mcp_server() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line_res in stdin.lock().lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req = match parse_json(trimmed) {
            Ok(v) => v,
            Err(err) => {
                let resp = jsonrpc_error(Json::Null, -32700, &format!("Parse error: {}", err));
                writeln!(out, "{}", resp.to_json_string())?;
                out.flush()?;
                continue;
            }
        };

        if req.as_obj().is_none() {
            let resp = jsonrpc_error(Json::Null, -32600, "Invalid Request: expected JSON object");
            writeln!(out, "{}", resp.to_json_string())?;
            out.flush()?;
            continue;
        }

        let id = req.get("id").cloned();
        let method = match req.get("method").and_then(|m| m.as_str()) {
            Some(m) => m,
            None => {
                let req_id = id.unwrap_or(Json::Null);
                let resp = jsonrpc_error(req_id, -32600, "Invalid Request: missing method");
                writeln!(out, "{}", resp.to_json_string())?;
                out.flush()?;
                continue;
            }
        };

        // JSON-RPC notifications (no `id`) receive no response.
        let req_id = match id {
            Some(v) => v,
            None => continue,
        };

        let params = req.get("params");
        let resp = match method {
            "initialize" => handle_initialize(req_id),
            "ping" => jsonrpc_ok(req_id, Json::Obj(BTreeMap::new())),
            "tools/list" => handle_tools_list(req_id),
            "tools/call" => handle_tools_call(req_id, params),
            _ => jsonrpc_error(req_id, -32601, &format!("Method not found: {}", method)),
        };

        writeln!(out, "{}", resp.to_json_string())?;
        out.flush()?;
    }
    Ok(())
}

fn json_obj(entries: Vec<(&str, Json)>) -> Json {
    let mut map = BTreeMap::new();
    for (k, v) in entries {
        map.insert(k.to_string(), v);
    }
    Json::Obj(map)
}

fn jsonrpc_ok(id: Json, result: Json) -> Json {
    json_obj(vec![
        ("jsonrpc", Json::Str("2.0".to_string())),
        ("id", id),
        ("result", result),
    ])
}

fn jsonrpc_error(id: Json, code: i64, message: &str) -> Json {
    json_obj(vec![
        ("jsonrpc", Json::Str("2.0".to_string())),
        ("id", id),
        (
            "error",
            json_obj(vec![
                ("code", Json::Num(code as f64)),
                ("message", Json::Str(message.to_string())),
            ]),
        ),
    ])
}

fn handle_initialize(id: Json) -> Json {
    let result = json_obj(vec![
        ("protocolVersion", Json::Str(PROTOCOL_VERSION.to_string())),
        (
            "capabilities",
            json_obj(vec![("tools", Json::Obj(BTreeMap::new()))]),
        ),
        (
            "serverInfo",
            json_obj(vec![
                ("name", Json::Str("cloture".to_string())),
                ("version", Json::Str(env!("CARGO_PKG_VERSION").to_string())),
            ]),
        ),
    ]);
    jsonrpc_ok(id, result)
}

fn schema_prop(ty: &str, desc: &str) -> Json {
    json_obj(vec![
        ("type", Json::Str(ty.to_string())),
        ("description", Json::Str(desc.to_string())),
    ])
}

fn tool_def(name: &str, desc: &str, props: Vec<(&str, Json)>, required: Vec<&str>) -> Json {
    json_obj(vec![
        ("name", Json::Str(name.to_string())),
        ("description", Json::Str(desc.to_string())),
        (
            "inputSchema",
            json_obj(vec![
                ("type", Json::Str("object".to_string())),
                ("properties", json_obj(props)),
                (
                    "required",
                    Json::Arr(required.into_iter().map(|s| Json::Str(s.to_string())).collect()),
                ),
            ]),
        ),
    ])
}

fn handle_tools_list(id: Json) -> Json {
    let tools = vec![
        tool_def(
            "cloture_mode_stop",
            "Certify multi-agent consensus mode stopping via grey-box ASC-D (probability vectors) or category-agnostic CITE (arbitrary strings/labels) with exact anytime-valid error <= delta and optional deterministic budget lock.",
            vec![
                ("mode", schema_prop("string", "\"ascd\" for probability vectors or \"cite\" for arbitrary string/integer answers")),
                ("samples", schema_prop("array", "List of probability vectors (for ascd) or answer labels/strings (for cite)")),
                ("delta", schema_prop("number", "Anytime-valid error level in (0, 1), default 0.05")),
                ("max_budget", schema_prop("integer", "Optional finite max rollout budget for deterministic lock")),
            ],
            vec!["samples"],
        ),
        tool_def(
            "cloture_audit_panel",
            "Audit a multi-agent LLM judge panel for correlated error collapse: computes Kish & Eigenvalue effective votes (n_eff), Condorcet independence gap, stratified permutation p-value, Krippendorff's alpha, leave-one-out pruning deltas, and optimal diverse subset.",
            vec![
                ("votes", schema_prop("array", "2D integer array [n_items][k_judges] of judge votes")),
                ("gold", schema_prop("array", "1D integer array [n_items] of ground-truth labels")),
                ("strata", schema_prop("array", "Optional 1D integer array [n_items] of difficulty/task strata")),
                ("n_perms", schema_prop("integer", "Number of permutations (default 2000)")),
                ("seed", schema_prop("integer", "Deterministic PRNG seed (default 42)")),
            ],
            vec!["votes", "gold"],
        ),
        tool_def(
            "cloture_population_cert",
            "Distinguish realized-unit consensus from population-level superiority across correlated rollouts per prompt; returns unit-level anytime-valid certificate, ICC rho_hat, within-unit saturation ceiling, and optimal rollouts per unit.",
            vec![
                ("units", schema_prop("array", "2D array [n_units][n_rollouts_q] of bounded scores in [-1, 1]")),
                ("delta", schema_prop("number", "Anytime-valid error level in (0, 1), default 0.05")),
                ("setup_cost", schema_prop("number", "Per-unit setup cost for optimal rollout allocation (default 1.0)")),
                ("rollout_cost", schema_prop("number", "Per-rollout cost (default 1.0)")),
            ],
            vec!["units"],
        ),
        tool_def(
            "cloture_active_ppi",
            "Run anytime-valid Active Prediction-Powered Inference (Active-PPI++) combining cheap LLM-judge scores with sparse ground-truth audits, guaranteed unbiased even if the judge is biased or adversarial.",
            vec![
                ("observations", schema_prop("array", "List of objects {pred, audited, gold, prob}")),
                ("alpha", schema_prop("number", "Miscoverage level in (0, 1), default 0.05")),
                ("target_audit_rate", schema_prop("number", "Target audit fraction in (0, 1], default 0.10")),
                ("min_audit_prob", schema_prop("number", "Strict positivity floor in (0, target_audit_rate], default 0.05")),
            ],
            vec!["observations"],
        ),
        tool_def(
            "cloture_agent_race",
            "Run anytime-valid multi-arm sequential racing over paired score rounds (or e-BH multiple-testing discovery filtering) with exact FWER <= delta.",
            vec![
                ("rounds", schema_prop("array", "2D array [n_rounds][n_arms] of paired scores in [0, 1]")),
                ("delta", schema_prop("number", "Family-wise error level in (0, 1), default 0.05")),
                ("paired", schema_prop("boolean", "Use variance-canceling paired e-processes (default true)")),
                ("e_values", schema_prop("array", "Optional list of e-values for Wang-Ramdas e-BH FDR discovery filtering")),
            ],
            vec![],
        ),
        tool_def(
            "cloture_safe_fuse",
            "Run Scale-Free BOA multi-agent fusion with Ville E-process poisoning quarantine across rounds of agent losses and predictions.",
            vec![
                ("losses", schema_prop("array", "2D array [n_rounds][n_agents] of agent losses")),
                ("preds", schema_prop("array", "Optional 2D array [n_rounds][n_agents] of agent predictions")),
                ("baseline_losses", schema_prop("array", "Optional 1D array [n_rounds] of baseline losses for global safety fallback")),
                ("quarantine_alpha", schema_prop("number", "Per-agent quarantine FWER in (0, 1), default 0.01")),
                ("fallback_alpha", schema_prop("number", "Global fallback alpha in (0, 1), default 0.05")),
            ],
            vec!["losses"],
        ),
    ];

    jsonrpc_ok(id, json_obj(vec![("tools", Json::Arr(tools))]))
}

fn tool_ok(id: Json, payload: Json) -> Json {
    let text = payload.to_json_string();
    jsonrpc_ok(
        id,
        json_obj(vec![
            (
                "content",
                Json::Arr(vec![json_obj(vec![
                    ("type", Json::Str("text".to_string())),
                    ("text", Json::Str(text)),
                ])]),
            ),
            ("isError", Json::Bool(false)),
        ]),
    )
}

fn tool_err(id: Json, msg: &str) -> Json {
    jsonrpc_ok(
        id,
        json_obj(vec![
            (
                "content",
                Json::Arr(vec![json_obj(vec![
                    ("type", Json::Str("text".to_string())),
                    ("text", Json::Str(msg.to_string())),
                ])]),
            ),
            ("isError", Json::Bool(true)),
        ]),
    )
}

fn handle_tools_call(id: Json, params: Option<&Json>) -> Json {
    let params = match params {
        Some(p) => p,
        None => return jsonrpc_error(id, -32602, "Missing params"),
    };
    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => return jsonrpc_error(id, -32602, "Missing tool name"),
    };
    let empty_obj = Json::Obj(BTreeMap::new());
    let args = params.get("arguments").unwrap_or(&empty_obj);

    let res = match name {
        "cloture_mode_stop" => call_mode_stop(args),
        "cloture_audit_panel" => call_audit_panel(args),
        "cloture_population_cert" => call_population_cert(args),
        "cloture_active_ppi" => call_active_ppi(args),
        "cloture_agent_race" => call_agent_race(args),
        "cloture_safe_fuse" => call_safe_fuse(args),
        _ => return jsonrpc_error(id, -32601, &format!("Unknown tool: {}", name)),
    };

    match res {
        Ok(payload) => tool_ok(id, payload),
        Err(msg) => tool_err(id, &msg),
    }
}

fn scope_str(s: CertScope) -> &'static str {
    match s {
        CertScope::RealizedUnit => "realized_unit",
        CertScope::Population => "population",
        CertScope::DeterministicBudget => "deterministic_budget",
    }
}

fn call_mode_stop(args: &Json) -> Result<Json, String> {
    let samples = args
        .get("samples")
        .and_then(|v| v.as_arr())
        .ok_or("Missing required array 'samples'")?;
    if samples.is_empty() {
        return Err("'samples' must be non-empty".to_string());
    }
    let delta = args.get("delta").and_then(|v| v.as_f64()).unwrap_or(0.05);
    let max_budget = args.get("max_budget").and_then(|v| v.as_u64()).filter(|&b| b > 0);
    let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or_else(|| {
        if samples[0].as_arr().is_some() {
            "ascd"
        } else {
            "cite"
        }
    });

    if mode == "ascd" {
        let first = samples[0]
            .as_arr()
            .ok_or("ASC-D mode expects probability arrays in 'samples'")?;
        let k = first.len();
        let mut stopper = AscdStopper::new(k, delta, max_budget).map_err(|e| e.to_string())?;
        let mut dec = stopper.decision();
        let mut buf = vec![0.0; k];
        for row in samples {
            let arr = row.as_arr().ok_or("Each ASC-D sample must be an array")?;
            if arr.len() != k {
                return Err(format!("Expected probability vector of length {}", k));
            }
            for (i, v) in arr.iter().enumerate() {
                buf[i] = v.as_f64().ok_or("Probability values must be numbers")?;
            }
            dec = stopper.update(&buf);
            if dec.certified {
                break;
            }
        }
        Ok(json_obj(vec![
            ("mode", Json::Str("ascd".to_string())),
            ("certified", Json::Bool(dec.certified)),
            ("deterministic_lock", Json::Bool(dec.deterministic_lock)),
            (
                "winner",
                dec.winner.map(|w| Json::Num(w as f64)).unwrap_or(Json::Null),
            ),
            (
                "leader",
                dec.leader.map(|l| Json::Num(l as f64)).unwrap_or(Json::Null),
            ),
            ("n_samples", Json::Num(dec.n_samples as f64)),
            ("samples_saved", Json::Num((samples.len() as u64).saturating_sub(dec.n_samples) as f64)),
            ("log_evidence", Json::Num(dec.log_evidence)),
            ("log_threshold", Json::Num(dec.log_threshold)),
            ("margin_estimate", Json::Num(dec.margin_estimate)),
            ("scope", Json::Str(scope_str(dec.scope).to_string())),
        ]))
    } else {
        let mut stopper = CiteStopper::new(delta, max_budget).map_err(|e| e.to_string())?;
        let mut dec = stopper.decision();
        for item in samples {
            dec = if let Some(s) = item.as_str() {
                stopper.update_str(s)
            } else if let Some(u) = item.as_u64() {
                stopper.update(u)
            } else {
                stopper.update_str(&item.to_json_string())
            };
            if dec.certified {
                break;
            }
        }
        let winner = stopper
            .winner_str()
            .map(|s| Json::Str(s.to_string()))
            .or_else(|| stopper.winner_label().map(|l| Json::Num(l as f64)))
            .unwrap_or(Json::Null);
        let leader = stopper
            .leader_str()
            .map(|s| Json::Str(s.to_string()))
            .or_else(|| stopper.leader_label().map(|l| Json::Num(l as f64)))
            .unwrap_or(Json::Null);
        Ok(json_obj(vec![
            ("mode", Json::Str("cite".to_string())),
            ("certified", Json::Bool(dec.certified)),
            ("deterministic_lock", Json::Bool(dec.deterministic_lock)),
            ("winner", winner),
            ("leader", leader),
            ("n_samples", Json::Num(dec.n_samples as f64)),
            ("samples_saved", Json::Num((samples.len() as u64).saturating_sub(dec.n_samples) as f64)),
            ("log_evidence", Json::Num(dec.log_evidence)),
            ("log_threshold", Json::Num(dec.log_threshold)),
            ("margin_estimate", Json::Num(dec.margin_estimate)),
            ("scope", Json::Str(scope_str(dec.scope).to_string())),
        ]))
    }
}

fn call_audit_panel(args: &Json) -> Result<Json, String> {
    let votes_rows = args
        .get("votes")
        .and_then(|v| v.as_arr())
        .ok_or("Missing required 2D array 'votes'")?;
    let gold_arr = args
        .get("gold")
        .and_then(|v| v.as_arr())
        .ok_or("Missing required array 'gold'")?;
    let n_items = gold_arr.len();
    if n_items == 0 || votes_rows.len() != n_items {
        return Err("votes and gold must be non-empty and have equal length".to_string());
    }
    let k_judges = votes_rows[0]
        .as_arr()
        .ok_or("Each row of 'votes' must be an array")?
        .len();
    if k_judges == 0 {
        return Err("k_judges must be >= 1".to_string());
    }
    let mut votes_flat = Vec::with_capacity(n_items * k_judges);
    for row in votes_rows {
        let r = row.as_arr().ok_or("Each row of 'votes' must be an array")?;
        if r.len() != k_judges {
            return Err("All vote rows must have equal length".to_string());
        }
        for v in r {
            let u = v.as_u64().ok_or("Votes must be non-negative integers")?;
            if u > u32::MAX as u64 {
                return Err("Vote exceeds u32::MAX".to_string());
            }
            votes_flat.push(u as u32);
        }
    }
    let mut gold = Vec::with_capacity(n_items);
    for g in gold_arr {
        let u = g.as_u64().ok_or("Gold labels must be non-negative integers")?;
        if u > u32::MAX as u64 {
            return Err("Gold label exceeds u32::MAX".to_string());
        }
        gold.push(u as u32);
    }
    let strata_vec = if let Some(st_arr) = args.get("strata").and_then(|v| v.as_arr()) {
        if st_arr.len() != n_items {
            return Err("strata length must match gold length".to_string());
        }
        let mut s = Vec::with_capacity(n_items);
        for v in st_arr {
            s.push(v.as_u64().ok_or("Strata must be non-negative integers")? as usize);
        }
        Some(s)
    } else {
        None
    };

    let n_perms = match args.get("n_perms") {
        None => 2000usize,
        Some(v) => {
            let p = v.as_u64().ok_or("n_perms must be a non-negative integer")?;
            if p > 1_000_000 {
                return Err("n_perms must be <= 1_000_000".to_string());
            }
            p as usize
        }
    };
    let seed = match args.get("seed") {
        None => 42u64,
        Some(v) => v.as_u64().ok_or("seed must be a non-negative integer")?,
    };
    let audit = PanelAudit::from_votes(&votes_flat, &gold, n_items, k_judges, strata_vec.as_deref())
        .map_err(|e| e.to_string())?;
    let cfg = PanelAuditConfig {
        n_condorcet_sims: n_perms.min(5000),
        n_permutations: n_perms,
        n_bootstrap: 1000,
        subset_target_ratio: 0.90,
        seed,
    };
    let rep = audit.run_with_config(&cfg);
    Ok(panel_report_to_json(&rep))
}

pub fn panel_report_to_json(rep: &cloture_core::panel::PanelReport) -> Json {
    let loo = rep
        .loo_analysis
        .iter()
        .map(|l| {
            json_obj(vec![
                ("judge_idx", Json::Num(l.judge_idx as f64)),
                ("delta_accuracy", Json::Num(l.delta_accuracy)),
                ("delta_n_eff", Json::Num(l.delta_n_eff)),
                ("redundant", Json::Bool(l.redundant)),
            ])
        })
        .collect();

    json_obj(vec![
        ("k_judges", Json::Num(rep.k_judges as f64)),
        ("n_items", Json::Num(rep.n_items as f64)),
        ("panel_accuracy", Json::Num(rep.panel_accuracy)),
        ("best_single_accuracy", Json::Num(rep.best_single_accuracy)),
        ("best_single_index", Json::Num(rep.best_single_index as f64)),
        ("panel_lift", Json::Num(rep.panel_lift)),
        ("mean_phi", Json::Num(rep.mean_phi)),
        ("std_phi", Json::Num(rep.std_phi)),
        ("n_eff_kish", Json::Num(rep.n_eff_kish)),
        ("n_eff_eigen", Json::Num(rep.n_eff_eigen)),
        ("lambda_max", Json::Num(rep.lambda_max)),
        ("effective_rank", Json::Num(rep.effective_rank)),
        ("independence_ratio", Json::Num(rep.independence_ratio)),
        (
            "asymptotic_ceiling",
            if rep.asymptotic_ceiling.is_finite() {
                Json::Num(rep.asymptotic_ceiling)
            } else {
                Json::Null
            },
        ),
        ("krippendorff_alpha", Json::Num(rep.krippendorff_alpha)),
        ("unanimous_rate", Json::Num(rep.unanimous_rate)),
        ("unanimous_accuracy", Json::Num(rep.unanimous_accuracy)),
        ("all_wrong_rate", Json::Num(rep.all_wrong_rate)),
        (
            "condorcet_predicted_accuracy",
            Json::Num(rep.condorcet_predicted_accuracy),
        ),
        ("condorcet_gap", Json::Num(rep.condorcet_gap)),
        ("permutation_p_value", Json::Num(rep.permutation_p_value)),
        ("n_eff_ci_low", Json::Num(rep.n_eff_ci_low)),
        ("n_eff_ci_high", Json::Num(rep.n_eff_ci_high)),
        (
            "judge_accuracies",
            Json::Arr(rep.judge_accuracies.iter().map(|&x| Json::Num(x)).collect()),
        ),
        ("loo_analysis", Json::Arr(loo)),
        (
            "optimal_subset",
            Json::Arr(rep.optimal_subset.iter().map(|&i| Json::Num(i as f64)).collect()),
        ),
    ])
}

fn call_population_cert(args: &Json) -> Result<Json, String> {
    let units = args
        .get("units")
        .and_then(|v| v.as_arr())
        .ok_or("Missing required 2D array 'units'")?;
    let delta = args.get("delta").and_then(|v| v.as_f64()).unwrap_or(0.05);
    let setup_cost = args.get("setup_cost").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let rollout_cost = args.get("rollout_cost").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let mut pop = PopulationCert::new(delta).map_err(|e| e.to_string())?;
    let mut dec = pop.decision();
    for u in units {
        let arr = u.as_arr().ok_or("Each unit must be an array of scores")?;
        let mut vals = Vec::with_capacity(arr.len());
        for v in arr {
            vals.push(v.as_f64().ok_or("Scores must be numbers")?);
        }
        dec = pop.add_unit_rollouts(&vals);
    }
    let opt_k = pop.optimal_rollouts_per_unit(setup_cost, rollout_cost);
    Ok(json_obj(vec![
        ("certified", Json::Bool(dec.certified)),
        ("n_units", Json::Num(dec.n_units as f64)),
        ("total_rollouts", Json::Num(dec.total_rollouts as f64)),
        ("theta_hat", Json::Num(dec.mean_unit_score)),
        ("sigma_b2", Json::Num(dec.sigma_b2)),
        ("sigma_w2", Json::Num(dec.sigma_w2)),
        ("rho_hat", Json::Num(dec.rho_hat)),
        ("n_eff_within", Json::Num(dec.n_eff_within)),
        (
            "within_unit_ceiling",
            if dec.within_unit_ceiling.is_finite() {
                Json::Num(dec.within_unit_ceiling)
            } else {
                Json::Null
            },
        ),
        ("log_evidence", Json::Num(dec.log_evidence)),
        ("log_threshold", Json::Num(dec.log_threshold)),
        ("optimal_rollouts_per_unit", Json::Num(opt_k as f64)),
    ]))
}

fn call_active_ppi(args: &Json) -> Result<Json, String> {
    let obs = args
        .get("observations")
        .and_then(|v| v.as_arr())
        .ok_or("Missing required array 'observations'")?;
    let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.05);
    let target_rate = args
        .get("target_audit_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.10);
    let min_prob = args
        .get("min_audit_prob")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.05);

    let mut ppi = ActivePPI::new(alpha, target_rate, min_prob).map_err(|e| e.to_string())?;
    for item in obs {
        let pred = item
            .get("pred")
            .and_then(|v| v.as_f64())
            .ok_or("Each observation requires numeric 'pred'")?;
        let audited = item
            .get("audited")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let gold = item.get("gold").and_then(|v| v.as_f64());
        let prob = item
            .get("prob")
            .and_then(|v| v.as_f64())
            .unwrap_or(target_rate);
        ppi.observe(pred, audited, gold, prob);
    }
    let st = ppi.status();
    Ok(json_obj(vec![
        ("n_total", Json::Num(st.n_total as f64)),
        ("n_audited", Json::Num(st.n_audited as f64)),
        ("audit_rate", Json::Num(st.audit_rate)),
        ("mean_estimate", Json::Num(st.mean_estimate)),
        ("lower", Json::Num(st.lower)),
        ("upper", Json::Num(st.upper)),
        ("width", Json::Num(st.width)),
        ("lambda_hat", Json::Num(st.lambda_hat)),
        ("variance_reduction_factor", Json::Num(st.variance_reduction_factor)),
        ("effective_sample_multiplier", Json::Num(st.effective_sample_multiplier)),
    ]))
}

fn call_agent_race(args: &Json) -> Result<Json, String> {
    let delta = args.get("delta").and_then(|v| v.as_f64()).unwrap_or(0.05);
    if let Some(evals_arr) = args.get("e_values").and_then(|v| v.as_arr()) {
        let mut es = Vec::with_capacity(evals_arr.len());
        for v in evals_arr {
            es.push(v.as_f64().ok_or("e_values must be numbers")?);
        }
        let disc = e_bh_discoveries(&es, delta);
        return Ok(json_obj(vec![
            ("alpha", Json::Num(delta)),
            (
                "discoveries",
                Json::Arr(disc.into_iter().map(|i| Json::Num(i as f64)).collect()),
            ),
        ]));
    }

    let rounds = args
        .get("rounds")
        .and_then(|v| v.as_arr())
        .ok_or("Provide either 'rounds' or 'e_values'")?;
    if rounds.is_empty() {
        return Err("'rounds' must be non-empty".to_string());
    }
    let n_arms = rounds[0]
        .as_arr()
        .ok_or("Each round must be an array of arm scores")?
        .len();
    let paired = args.get("paired").and_then(|v| v.as_bool()).unwrap_or(true);
    let mut race = AgentRace::new(n_arms, delta, paired).map_err(|e| e.to_string())?;
    let mut buf = vec![0.0; n_arms];
    let mut rounds_used = 0usize;
    for row in rounds {
        let arr = row.as_arr().ok_or("Each round must be an array")?;
        if arr.len() != n_arms {
            return Err("All rounds must have equal arm count".to_string());
        }
        for (i, v) in arr.iter().enumerate() {
            buf[i] = v.as_f64().unwrap_or(f64::NAN);
        }
        rounds_used += 1;
        let st = race.observe_paired(&buf);
        if st.finished {
            break;
        }
    }
    let st = race.status();
    Ok(json_obj(vec![
        ("finished", Json::Bool(st.finished)),
        (
            "winner",
            st.winner.map(|w| Json::Num(w as f64)).unwrap_or(Json::Null),
        ),
        ("rounds_used", Json::Num(rounds_used as f64)),
        ("total_evals", Json::Num(st.total_evals as f64)),
        (
            "active_arms",
            Json::Arr(st.active_arms.iter().map(|&a| Json::Num(a as f64)).collect()),
        ),
        (
            "eliminated_arms",
            Json::Arr(st.eliminated_arms.iter().map(|&a| Json::Num(a as f64)).collect()),
        ),
    ]))
}

fn call_safe_fuse(args: &Json) -> Result<Json, String> {
    let losses = args
        .get("losses")
        .and_then(|v| v.as_arr())
        .ok_or("Missing required 2D array 'losses'")?;
    if losses.is_empty() {
        return Err("'losses' must be non-empty".to_string());
    }
    let n_agents = losses[0]
        .as_arr()
        .ok_or("Each loss row must be an array")?
        .len();
    let q_alpha = args
        .get("quarantine_alpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01);
    let fb_alpha = args
        .get("fallback_alpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.05);
    let base_arr = args.get("baseline_losses").and_then(|v| v.as_arr());
    let preds_arr = args.get("preds").and_then(|v| v.as_arr());

    let mut fuse = SafeFuse::new(n_agents, q_alpha, fb_alpha).map_err(|e| e.to_string())?;
    let mut fused_preds = Vec::new();
    let mut row_buf = vec![0.0; n_agents];
    for (t, row) in losses.iter().enumerate() {
        if let Some(p_rows) = preds_arr {
            if let Some(pr) = p_rows.get(t).and_then(|r| r.as_arr()) {
                if pr.len() == n_agents {
                    for (i, v) in pr.iter().enumerate() {
                        row_buf[i] = v.as_f64().unwrap_or(f64::NAN);
                    }
                    fused_preds.push(Json::Num(fuse.predict(&row_buf).fused));
                }
            }
        }
        let lr = row.as_arr().ok_or("Each loss row must be an array")?;
        if lr.len() != n_agents {
            return Err("All loss rows must have equal length".to_string());
        }
        for (i, v) in lr.iter().enumerate() {
            row_buf[i] = v.as_f64().unwrap_or(f64::NAN);
        }
        let base = base_arr.and_then(|b| b.get(t)).and_then(|v| v.as_f64());
        fuse.update(&row_buf, base);
    }

    let quarantined: Vec<Json> = fuse
        .quarantined()
        .iter()
        .enumerate()
        .filter_map(|(i, &q)| if q { Some(Json::Num(i as f64)) } else { None })
        .collect();
    Ok(json_obj(vec![
        ("steps", Json::Num(fuse.steps() as f64)),
        ("fallback_active", Json::Bool(fuse.fallback_active())),
        ("quarantined_agents", Json::Arr(quarantined)),
        (
            "weights",
            Json::Arr(fuse.weights().iter().map(|&w| Json::Num(w)).collect()),
        ),
        ("fused_predictions", Json::Arr(fused_preds)),
    ]))
}
