//! `cloture` — Zero-dependency single-binary CLI & MCP server for anytime-valid
//! multi-agent consensus stopping, correlated-panel auditing, sequential racing,
//! and active prediction-powered evaluation.

mod json;
mod mcp;

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::io::{self, BufRead};
use std::process;
use std::time::Instant;

use cloture_core::betting::{BettingCS, EmpiricalBernsteinCS, PairedEProcess};
use cloture_core::cascade::OnlineConformal;
use cloture_core::fuse::SafeFuse;
use cloture_core::math::Rng;
use cloture_core::mode::{AscdStopper, CertScope, CiteStopper};
use cloture_core::panel::{PanelAudit, PanelAuditConfig};
use cloture_core::ppi::ActivePPI;
use cloture_core::race::AgentRace;
use cloture_core::sketch::EvidenceSketch;

use crate::json::{parse_json, Json};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_help() {
    println!(
        "cloture {VERSION} — End multi-agent debate the millisecond the math proves who won.

USAGE:
    cloture <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    stop     Stream answers or probability vectors from stdin; halt immediately on certification
    audit    Audit a correlated multi-agent judge panel (n_eff, Condorcet gap, permutation p, LOO)
    race     Stream multi-arm scores from stdin; certify winning policy/prompt via LUCB / paired e-process
    ppi      Stream LLM-judge + sparse gold audits from stdin; output anytime-valid Active-PPI++ bounds
    bench    Run built-in sub-microsecond microbenchmark suite on this CPU
    mcp      Start JSON-RPC 2.0 Model Context Protocol (MCP) server over stdio

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version

EXAMPLES:
    printf '42\\n42\\n42\\n42\\n42\\n42\\n42\\n42\\n42\\n42\\n' | cloture stop --mode cite --delta 0.05
    printf '[0.9,0.1]\\n[0.85,0.15]\\n[0.92,0.08]\\n' | cloture stop --mode ascd --k 2 --delta 0.05
    cloture audit --file panel.json
    cloture audit --csv judges.csv --gold-col gold --strata-col difficulty
    cloture bench
    cloture mcp"
    );
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return;
    }

    let code = match args[0].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            0
        }
        "-V" | "--version" => {
            println!("cloture {}", VERSION);
            0
        }
        "stop" => cmd_stop(&args[1..]),
        "audit" => cmd_audit(&args[1..]),
        "race" => cmd_race(&args[1..]),
        "ppi" => cmd_ppi(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "mcp" => match mcp::run_mcp_server() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("cloture mcp error: {}", e);
                1
            }
        },
        other => {
            eprintln!("Unknown subcommand '{}'. Run `cloture --help`.", other);
            2
        }
    };

    if code != 0 {
        process::exit(code);
    }
}

fn scope_name(s: CertScope) -> &'static str {
    match s {
        CertScope::RealizedUnit => "realized_unit",
        CertScope::Population => "population",
        CertScope::DeterministicBudget => "deterministic_budget",
    }
}

fn json_obj(entries: Vec<(&str, Json)>) -> Json {
    let mut map = BTreeMap::new();
    for (k, v) in entries {
        map.insert(k.to_string(), v);
    }
    Json::Obj(map)
}

fn cmd_stop(args: &[String]) -> i32 {
    let mut mode = String::from("cite");
    let mut delta: f64 = 0.05;
    let mut k_opt: Option<usize> = None;
    let mut max_budget: Option<u64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" if i + 1 < args.len() => {
                mode = args[i + 1].clone();
                i += 2;
            }
            "--delta" if i + 1 < args.len() => {
                delta = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --delta value");
                        return 2;
                    }
                };
                i += 2;
            }
            "--k" if i + 1 < args.len() => {
                let parsed: usize = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --k value");
                        return 2;
                    }
                };
                k_opt = Some(parsed);
                i += 2;
            }
            "--max-budget" if i + 1 < args.len() => {
                let b: u64 = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --max-budget value");
                        return 2;
                    }
                };
                max_budget = if b == 0 { None } else { Some(b) };
                i += 2;
            }
            "-h" | "--help" => {
                println!(
                    "USAGE: cloture stop [--mode cite|ascd|weighted] [--delta 0.05] [--k N] [--max-budget N]"
                );
                return 0;
            }
            other => {
                eprintln!("Unknown option for stop: {}", other);
                return 2;
            }
        }
    }

    if !matches!(mode.as_str(), "cite" | "ascd" | "weighted") {
        eprintln!(
            "Invalid --mode '{}': expected one of cite, ascd, weighted",
            mode
        );
        return 2;
    }

    let stdin = io::stdin();
    if mode == "ascd" {
        let mut stopper: Option<AscdStopper> = match k_opt {
            Some(k) => match AscdStopper::new(k, delta, max_budget) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    return 2;
                }
            },
            None => None,
        };
        let mut last_dec = None;

        for line_res in stdin.lock().lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("stdin read error: {}", e);
                    return 1;
                }
            };
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let probs: Vec<f64> = if t.starts_with('[') || t.starts_with('{') {
                match parse_json(t) {
                    Ok(Json::Arr(arr)) => arr.iter().filter_map(|v| v.as_f64()).collect(),
                    Ok(obj) => obj
                        .get("probs")
                        .and_then(|v| v.as_arr())
                        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                        .unwrap_or_default(),
                    Err(e) => {
                        eprintln!("Invalid JSON row: {}", e);
                        return 2;
                    }
                }
            } else {
                t.split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect()
            };
            if probs.len() < 2 {
                eprintln!("ASC-D expects >= 2 probabilities per line");
                return 2;
            }
            if stopper.is_none() {
                stopper = match AscdStopper::new(probs.len(), delta, max_budget) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        return 2;
                    }
                };
            }
            let st = stopper.as_mut().unwrap();
            let dec = st.update(&probs);
            let done = dec.certified;
            last_dec = Some(dec);
            if done {
                break;
            }
        }

        let dec = match last_dec {
            Some(d) => d,
            None => {
                eprintln!("No input rows received");
                return 1;
            }
        };
        let out = json_obj(vec![
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
            ("log_evidence", Json::Num(dec.log_evidence)),
            ("log_threshold", Json::Num(dec.log_threshold)),
            ("margin_estimate", Json::Num(dec.margin_estimate)),
            ("scope", Json::Str(scope_name(dec.scope).to_string())),
        ]);
        println!("{}", out.to_json_string());
        0
    } else {
        let weighted = mode == "weighted";
        let mut stopper = match CiteStopper::new(delta, max_budget) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 2;
            }
        };
        let mut last_dec = stopper.decision();

        for line_res in stdin.lock().lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("stdin read error: {}", e);
                    return 1;
                }
            };
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if t.starts_with('{') {
                if let Ok(obj) = parse_json(t) {
                    let ans = obj
                        .get("answer")
                        .or_else(|| obj.get("label"))
                        .map(|v| match v {
                            Json::Str(s) => s.clone(),
                            other => other.to_json_string(),
                        })
                        .unwrap_or_else(|| t.to_string());
                    if let Some(w) = obj.get("weight").and_then(|v| v.as_f64()) {
                        last_dec = stopper.update_str_weighted(&ans, w);
                    } else {
                        last_dec = stopper.update_str(&ans);
                    }
                } else {
                    last_dec = stopper.update_str(t);
                }
            } else if weighted {
                let mut parts = t.splitn(2, '\t');
                let ans = parts.next().unwrap_or("").trim();
                let w = parts
                    .next()
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .unwrap_or(1.0);
                last_dec = stopper.update_str_weighted(ans, w);
            } else {
                last_dec = stopper.update_str(t);
            }
            if last_dec.certified {
                break;
            }
        }

        let out = json_obj(vec![
            ("mode", Json::Str(mode)),
            ("certified", Json::Bool(last_dec.certified)),
            ("deterministic_lock", Json::Bool(last_dec.deterministic_lock)),
            (
                "winner",
                stopper
                    .winner_str()
                    .map(|s| Json::Str(s.to_string()))
                    .unwrap_or(Json::Null),
            ),
            (
                "leader",
                stopper
                    .leader_str()
                    .map(|s| Json::Str(s.to_string()))
                    .unwrap_or(Json::Null),
            ),
            ("n_samples", Json::Num(last_dec.n_samples as f64)),
            ("log_evidence", Json::Num(last_dec.log_evidence)),
            ("log_threshold", Json::Num(last_dec.log_threshold)),
            ("margin_estimate", Json::Num(last_dec.margin_estimate)),
            ("scope", Json::Str(scope_name(last_dec.scope).to_string())),
        ]);
        println!("{}", out.to_json_string());
        0
    }
}

struct ParsedPanelInput {
    votes_flat: Vec<u32>,
    gold: Vec<u32>,
    n_items: usize,
    k_judges: usize,
    strata: Option<Vec<usize>>,
    judge_names: Option<Vec<String>>,
}

/// Zero-dependency RFC-4180 CSV parser supporting optional quotes, escaped quotes (`""`),
/// embedded commas/newlines inside quoted fields, CRLF/LF line endings, and leading UTF-8 BOM.
fn parse_csv_records(raw: &str) -> Result<Vec<Vec<String>>, String> {
    let input = raw.strip_prefix('\u{FEFF}').unwrap_or(raw);
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut line_num = 1usize;
    let mut records: Vec<Vec<String>> = Vec::new();

    while pos < len {
        // Skip blank lines (CRLF, LF, or whitespace-only lines) between records.
        let mut scan = pos;
        while scan < len && (bytes[scan] == b' ' || bytes[scan] == b'\t') {
            scan += 1;
        }
        if scan == len {
            break;
        }
        if bytes[scan] == b'\r' {
            pos = scan + 1;
            if pos < len && bytes[pos] == b'\n' {
                pos += 1;
            }
            line_num += 1;
            continue;
        }
        if bytes[scan] == b'\n' {
            pos = scan + 1;
            line_num += 1;
            continue;
        }

        let row_start_line = line_num;
        let mut row: Vec<String> = Vec::new();

        loop {
            if pos < len && bytes[pos] == b'"' {
                // RFC-4180 quoted field.
                pos += 1;
                let mut field = String::new();
                let mut seg_start = pos;
                let mut closed = false;
                while pos < len {
                    if bytes[pos] == b'"' {
                        field.push_str(&input[seg_start..pos]);
                        if pos + 1 < len && bytes[pos + 1] == b'"' {
                            field.push('"');
                            pos += 2;
                            seg_start = pos;
                        } else {
                            pos += 1;
                            closed = true;
                            break;
                        }
                    } else {
                        if bytes[pos] == b'\n' {
                            line_num += 1;
                        }
                        pos += 1;
                    }
                }
                if !closed {
                    return Err(format!(
                        "CSV parse error: unterminated quoted field starting on line {}",
                        row_start_line
                    ));
                }
                if pos < len && !matches!(bytes[pos], b',' | b'\r' | b'\n') {
                    return Err(format!(
                        "CSV parse error: unexpected character after closing quote on line {}",
                        line_num
                    ));
                }
                row.push(field);
            } else {
                // Unquoted field: trim surrounding ASCII spaces/tabs while forbidding stray quotes.
                let start = pos;
                while pos < len {
                    match bytes[pos] {
                        b',' | b'\r' | b'\n' => break,
                        b'"' => {
                            return Err(format!(
                                "CSV parse error: unexpected quote in unquoted field on line {}",
                                line_num
                            ));
                        }
                        _ => pos += 1,
                    }
                }
                row.push(input[start..pos].trim().to_string());
            }

            if pos < len && bytes[pos] == b',' {
                pos += 1;
                continue;
            }
            if pos < len && bytes[pos] == b'\r' {
                pos += 1;
                if pos < len && bytes[pos] == b'\n' {
                    pos += 1;
                }
                line_num += 1;
            } else if pos < len && bytes[pos] == b'\n' {
                pos += 1;
                line_num += 1;
            }
            break;
        }

        records.push(row);
    }

    Ok(records)
}

fn intern_label(map: &mut HashMap<String, u32>, label: &str) -> Result<u32, String> {
    if let Some(&id) = map.get(label) {
        return Ok(id);
    }
    let next_id = u32::try_from(map.len()).map_err(|_| "Too many distinct labels (> u32::MAX)")?;
    map.insert(label.to_string(), next_id);
    Ok(next_id)
}

fn parse_csv_panel(
    raw: &str,
    gold_col: &str,
    strata_col: Option<&str>,
) -> Result<ParsedPanelInput, String> {
    if gold_col.is_empty() {
        return Err("--gold-col cannot be empty".to_string());
    }
    if let Some(sc) = strata_col {
        if sc.is_empty() {
            return Err("--strata-col cannot be empty".to_string());
        }
        if sc == gold_col {
            return Err(format!(
                "--gold-col and --strata-col must be distinct (both '{}')",
                gold_col
            ));
        }
    }

    let records = parse_csv_records(raw)?;
    if records.is_empty() {
        return Err("CSV input is empty".to_string());
    }

    let header = &records[0];
    let n_cols = header.len();
    if n_cols < 2 {
        return Err("CSV header must contain at least 2 columns (gold + >=1 judge)".to_string());
    }

    let mut gold_idx: Option<usize> = None;
    for (idx, col) in header.iter().enumerate() {
        if col == gold_col {
            if gold_idx.is_some() {
                return Err(format!("Duplicate gold column '{}' in CSV header", gold_col));
            }
            gold_idx = Some(idx);
        }
    }
    let gold_idx = gold_idx.ok_or_else(|| {
        format!(
            "Gold column '{}' not found in CSV header (available: {})",
            gold_col,
            header.join(", ")
        )
    })?;

    let strata_idx: Option<usize> = if let Some(sc) = strata_col {
        let mut found: Option<usize> = None;
        for (idx, col) in header.iter().enumerate() {
            if col == sc {
                if found.is_some() {
                    return Err(format!("Duplicate strata column '{}' in CSV header", sc));
                }
                found = Some(idx);
            }
        }
        let idx = found.ok_or_else(|| {
            format!(
                "Strata column '{}' not found in CSV header (available: {})",
                sc,
                header.join(", ")
            )
        })?;
        Some(idx)
    } else {
        None
    };

    let mut judge_col_indices: Vec<usize> = Vec::with_capacity(n_cols);
    let mut judge_names: Vec<String> = Vec::with_capacity(n_cols);
    for (idx, col) in header.iter().enumerate() {
        if idx == gold_idx || Some(idx) == strata_idx {
            continue;
        }
        if col.is_empty() {
            return Err(format!(
                "CSV header contains an empty judge column name at column {}",
                idx + 1
            ));
        }
        judge_col_indices.push(idx);
        judge_names.push(col.clone());
    }
    if judge_col_indices.is_empty() {
        return Err(
            "CSV must contain at least one judge column besides gold/strata".to_string(),
        );
    }

    if records.len() < 2 {
        return Err("CSV input has header row only and no data rows".to_string());
    }

    let n_items = records.len() - 1;
    let k_judges = judge_col_indices.len();
    let total_votes = n_items
        .checked_mul(k_judges)
        .ok_or_else(|| "CSV dimensions overflow usize".to_string())?;

    let mut label_map: HashMap<String, u32> = HashMap::new();
    let mut strata_map: HashMap<String, usize> = HashMap::new();
    let mut votes_flat = Vec::with_capacity(total_votes);
    let mut gold = Vec::with_capacity(n_items);
    let mut strata = strata_idx.map(|_| Vec::with_capacity(n_items));

    for (row_offset, row) in records[1..].iter().enumerate() {
        if row.len() != n_cols {
            return Err(format!(
                "CSV data row {} has {} column(s), expected {} (matching header)",
                row_offset + 2,
                row.len(),
                n_cols
            ));
        }
        gold.push(intern_label(&mut label_map, &row[gold_idx])?);
        for &j_col in &judge_col_indices {
            votes_flat.push(intern_label(&mut label_map, &row[j_col])?);
        }
        if let (Some(s_col), Some(ref mut s_vec)) = (strata_idx, &mut strata) {
            let s_val = &row[s_col];
            let sid = if let Some(&existing) = strata_map.get(s_val) {
                existing
            } else {
                let next_id = strata_map.len();
                strata_map.insert(s_val.clone(), next_id);
                next_id
            };
            s_vec.push(sid);
        }
    }

    Ok(ParsedPanelInput {
        votes_flat,
        gold,
        n_items,
        k_judges,
        strata,
        judge_names: Some(judge_names),
    })
}

fn parse_json_panel(raw: &str) -> Result<ParsedPanelInput, String> {
    let doc = parse_json(raw).map_err(|e| format!("Invalid JSON input: {}", e))?;
    let votes_rows = match doc.get("votes").and_then(|v| v.as_arr()) {
        Some(v) if !v.is_empty() => v,
        _ => return Err("Input JSON must contain non-empty 'votes' 2D array".to_string()),
    };
    let gold_arr = match doc.get("gold").and_then(|v| v.as_arr()) {
        Some(v) if v.len() == votes_rows.len() => v,
        _ => {
            return Err(
                "Input JSON must contain 'gold' array matching 'votes' rows".to_string(),
            );
        }
    };
    let n_items = gold_arr.len();
    let k_judges = match votes_rows[0].as_arr() {
        Some(r) if !r.is_empty() => r.len(),
        _ => return Err("Each row of 'votes' must be a non-empty array".to_string()),
    };
    let mut votes_flat = Vec::with_capacity(n_items * k_judges);
    for row in votes_rows {
        let r = match row.as_arr() {
            Some(a) if a.len() == k_judges => a,
            _ => return Err("Inconsistent row length in 'votes'".to_string()),
        };
        for v in r {
            let u = match v.as_u64() {
                Some(u) if u <= u32::MAX as u64 => u as u32,
                _ => return Err("All 'votes' entries must be integers in 0..=u32::MAX".to_string()),
            };
            votes_flat.push(u);
        }
    }
    let mut gold = Vec::with_capacity(n_items);
    for v in gold_arr {
        let u = match v.as_u64() {
            Some(u) if u <= u32::MAX as u64 => u as u32,
            _ => return Err("All 'gold' entries must be integers in 0..=u32::MAX".to_string()),
        };
        gold.push(u);
    }
    let strata: Option<Vec<usize>> = if let Some(st_val) = doc.get("strata") {
        let a = match st_val.as_arr() {
            Some(a) if a.len() == n_items => a,
            _ => return Err("'strata' must be an array matching 'gold' length".to_string()),
        };
        let mut s = Vec::with_capacity(n_items);
        for v in a {
            let u = match v.as_u64() {
                Some(u) => u as usize,
                None => {
                    return Err("All 'strata' entries must be non-negative integers".to_string());
                }
            };
            s.push(u);
        }
        Some(s)
    } else {
        None
    };

    Ok(ParsedPanelInput {
        votes_flat,
        gold,
        n_items,
        k_judges,
        strata,
        judge_names: None,
    })
}

fn cmd_audit(args: &[String]) -> i32 {
    let mut file: Option<String> = None;
    let mut csv: Option<String> = None;
    let mut gold_col = String::from("gold");
    let mut gold_col_explicit = false;
    let mut strata_col: Option<String> = None;
    let mut n_perms: usize = 2000;
    let mut n_condorcet: usize = 2000;
    let mut n_boot: usize = 1000;
    let mut seed: u64 = 42;
    let mut json_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--file" | "-f" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for {}", args[i]);
                    return 2;
                }
                file = Some(args[i + 1].clone());
                i += 2;
            }
            "--csv" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --csv");
                    return 2;
                }
                csv = Some(args[i + 1].clone());
                i += 2;
            }
            "--gold-col" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --gold-col");
                    return 2;
                }
                gold_col = args[i + 1].clone();
                gold_col_explicit = true;
                i += 2;
            }
            "--strata-col" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --strata-col");
                    return 2;
                }
                strata_col = Some(args[i + 1].clone());
                i += 2;
            }
            "--perms" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --perms");
                    return 2;
                }
                n_perms = match args[i + 1].parse() {
                    Ok(v) if v <= 1_000_000 => v,
                    _ => {
                        eprintln!("Invalid --perms value (must be integer in 0..=1_000_000)");
                        return 2;
                    }
                };
                i += 2;
            }
            "--condorcet" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --condorcet");
                    return 2;
                }
                n_condorcet = match args[i + 1].parse() {
                    Ok(v) if v <= 1_000_000 => v,
                    _ => {
                        eprintln!("Invalid --condorcet value (must be integer in 0..=1_000_000)");
                        return 2;
                    }
                };
                i += 2;
            }
            "--boot" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --boot");
                    return 2;
                }
                n_boot = match args[i + 1].parse() {
                    Ok(v) if v <= 1_000_000 => v,
                    _ => {
                        eprintln!("Invalid --boot value (must be integer in 0..=1_000_000)");
                        return 2;
                    }
                };
                i += 2;
            }
            "--seed" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --seed");
                    return 2;
                }
                seed = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --seed value");
                        return 2;
                    }
                };
                i += 2;
            }
            "--json" => {
                json_out = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!(
                    "USAGE:\n    cloture audit [--file panel.json] [--perms 2000] [--condorcet 2000] [--boot 1000] [--seed 42] [--json]\n    cloture audit --csv <path|-> [--gold-col gold] [--strata-col <col>] [--perms 2000] [--condorcet 2000] [--boot 1000] [--seed 42] [--json]"
                );
                return 0;
            }
            other => {
                eprintln!("Unknown option for audit: {}", other);
                return 2;
            }
        }
    }

    if file.is_some() && csv.is_some() {
        eprintln!("Cannot specify both --file and --csv");
        return 2;
    }
    if (gold_col_explicit || strata_col.is_some()) && csv.is_none() {
        eprintln!("--gold-col and --strata-col require --csv");
        return 2;
    }

    let parsed = if let Some(csv_path) = csv {
        let raw = if csv_path == "-" {
            let mut s = String::new();
            if let Err(e) = io::Read::read_to_string(&mut io::stdin(), &mut s) {
                eprintln!("Failed to read CSV from stdin: {}", e);
                return 2;
            }
            s
        } else {
            match fs::read_to_string(&csv_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read CSV '{}': {}", csv_path, e);
                    return 2;
                }
            }
        };
        match parse_csv_panel(&raw, &gold_col, strata_col.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}", e);
                return 2;
            }
        }
    } else {
        let raw = match file {
            Some(path) => match fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read {}: {}", path, e);
                    return 2;
                }
            },
            None => {
                let mut s = String::new();
                if let Err(e) = io::Read::read_to_string(&mut io::stdin(), &mut s) {
                    eprintln!("Failed to read stdin: {}", e);
                    return 2;
                }
                s
            }
        };
        match parse_json_panel(&raw) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}", e);
                return 2;
            }
        }
    };

    let audit = match PanelAudit::from_votes(
        &parsed.votes_flat,
        &parsed.gold,
        parsed.n_items,
        parsed.k_judges,
        parsed.strata.as_deref(),
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("PanelAudit error: {}", e);
            return 2;
        }
    };
    let cfg = PanelAuditConfig {
        n_condorcet_sims: n_condorcet,
        n_permutations: n_perms,
        n_bootstrap: n_boot,
        subset_target_ratio: 0.90,
        seed,
    };
    let rep = audit.run_with_config(&cfg);
    if json_out {
        let mut report_json = mcp::panel_report_to_json(&rep);
        if let (Some(names), Json::Obj(ref mut map)) = (&parsed.judge_names, &mut report_json) {
            map.insert(
                "judge_names".to_string(),
                Json::Arr(names.iter().map(|s| Json::Str(s.clone())).collect()),
            );
        }
        println!("{}", report_json.to_json_string());
    } else {
        println!(
            "cloture panel audit  (K={} judges, N={} items)",
            rep.k_judges, rep.n_items
        );
        println!("──────────────────────────────────────────────────────────────");
        println!(
            "  Effective Votes (Kish)     : {:.2} / {}  (95% CI [{:.2}, {:.2}])",
            rep.n_eff_kish, rep.k_judges, rep.n_eff_ci_low, rep.n_eff_ci_high
        );
        println!(
            "  Effective Votes (Eigen)    : {:.2} / {}  (lambda_max={:.2}, eff_rank={:.2})",
            rep.n_eff_eigen, rep.k_judges, rep.lambda_max, rep.effective_rank
        );
        println!(
            "  Mean Pairwise Error phi    : {:.3} +/- {:.3}  (asymptotic ceiling: {:.2})",
            rep.mean_phi, rep.std_phi, rep.asymptotic_ceiling
        );
        let best_label = match &parsed.judge_names {
            Some(names) => names[rep.best_single_index].clone(),
            None => format!("J{}", rep.best_single_index),
        };
        println!(
            "  Panel vs Best Single Acc   : {:.2}% vs {:.2}% (lift {:+.2} pp, best={})",
            rep.panel_accuracy * 100.0,
            rep.best_single_accuracy * 100.0,
            rep.panel_lift * 100.0,
            best_label
        );
        println!(
            "  Condorcet Independence Gap : {:+.2} pp  (predicted {:.2}% under independence)",
            rep.condorcet_gap * 100.0,
            rep.condorcet_predicted_accuracy * 100.0
        );
        println!(
            "  Stratified Permutation p   : {:.6}   (Krippendorff alpha={:.3})",
            rep.permutation_p_value, rep.krippendorff_alpha
        );
        println!(
            "  Unanimous Rate / All-Wrong : {:.1}% unanimous ({:.1}% acc), {:.1}% all-wrong",
            rep.unanimous_rate * 100.0,
            rep.unanimous_accuracy * 100.0,
            rep.all_wrong_rate * 100.0
        );
        if let Some(names) = &parsed.judge_names {
            let opt_names: Vec<&str> = rep
                .optimal_subset
                .iter()
                .map(|&idx| names[idx].as_str())
                .collect();
            let prunable_names: Vec<&str> = rep
                .loo_analysis
                .iter()
                .filter(|l| l.redundant)
                .map(|l| names[l.judge_idx].as_str())
                .collect();
            println!("  Optimal Diverse Subset     : {}", opt_names.join(", "));
            if prunable_names.is_empty() {
                println!("  Prunable Judges (LOO)      : (none)");
            } else {
                println!(
                    "  Prunable Judges (LOO)      : {}",
                    prunable_names.join(", ")
                );
            }
            println!("  Per-Judge LOO Diagnostics  :");
            let name_w = names.iter().map(|s| s.len()).max().unwrap_or(8).clamp(8, 32);
            for l in &rep.loo_analysis {
                let tag = if l.redundant { "[PRUNABLE]" } else { "[KEEP]    " };
                println!(
                    "    {:<width$}  acc={:5.1}%  LOO dAcc={:+6.2} pp  dNeff={:+6.2}  {}",
                    names[l.judge_idx],
                    rep.judge_accuracies[l.judge_idx] * 100.0,
                    l.delta_accuracy * 100.0,
                    l.delta_n_eff,
                    tag,
                    width = name_w
                );
            }
        } else {
            let redundant: Vec<usize> = rep
                .loo_analysis
                .iter()
                .filter(|l| l.redundant)
                .map(|l| l.judge_idx)
                .collect();
            println!("  Optimal Diverse Subset     : {:?}", rep.optimal_subset);
            println!("  Redundant Judges (LOO)     : {:?}", redundant);
        }
    }
    0
}

fn cmd_race(args: &[String]) -> i32 {
    let mut delta: f64 = 0.05;
    let mut paired = true;
    let mut n_arms_opt: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--delta" if i + 1 < args.len() => {
                delta = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --delta value");
                        return 2;
                    }
                };
                i += 2;
            }
            "--arms" if i + 1 < args.len() => {
                let parsed: usize = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --arms value");
                        return 2;
                    }
                };
                n_arms_opt = Some(parsed);
                i += 2;
            }
            "--unpaired" => {
                paired = false;
                i += 1;
            }
            "--paired" => {
                paired = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!("USAGE: cloture race [--arms M] [--delta 0.05] [--paired | --unpaired]");
                return 0;
            }
            other => {
                eprintln!("Unknown option for race: {}", other);
                return 2;
            }
        }
    }

    let mut race: Option<AgentRace> = match n_arms_opt {
        Some(n_arms) => match AgentRace::new(n_arms, delta, paired) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("Error: {}", e);
                return 2;
            }
        },
        None => None,
    };

    let stdin = io::stdin();
    let mut rounds = 0usize;
    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin error: {}", e);
                return 1;
            }
        };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let val = match parse_json(t) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid JSON line: {}", e);
                return 2;
            }
        };
        if let Some(arr) = val.as_arr() {
            if race.is_none() {
                race = match AgentRace::new(arr.len(), delta, paired) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        return 2;
                    }
                };
            }
            let scores: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
            rounds += 1;
            let st = race.as_mut().unwrap().observe_paired(&scores);
            if st.finished {
                break;
            }
        } else if let (Some(arm), Some(score)) = (
            val.get("arm").and_then(|v| v.as_u64()),
            val.get("score").and_then(|v| v.as_f64()),
        ) {
            let r = match race.as_mut() {
                Some(r) => r,
                None => {
                    eprintln!("Provide --arms M for unpaired object stream");
                    return 2;
                }
            };
            rounds += 1;
            let st = r.observe_arm(arm as usize, score);
            if st.finished {
                break;
            }
        }
    }

    let r = match race {
        Some(r) => r,
        None => {
            eprintln!("No rounds observed");
            return 1;
        }
    };
    let st = r.status();
    let out = json_obj(vec![
        ("finished", Json::Bool(st.finished)),
        (
            "winner",
            st.winner.map(|w| Json::Num(w as f64)).unwrap_or(Json::Null),
        ),
        ("rounds_used", Json::Num(rounds as f64)),
        ("total_evals", Json::Num(st.total_evals as f64)),
        (
            "active_arms",
            Json::Arr(st.active_arms.iter().map(|&a| Json::Num(a as f64)).collect()),
        ),
    ]);
    println!("{}", out.to_json_string());
    0
}

fn cmd_ppi(args: &[String]) -> i32 {
    let mut alpha: f64 = 0.05;
    let mut target_rate: f64 = 0.10;
    let mut min_prob: f64 = 0.05;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--alpha" if i + 1 < args.len() => {
                alpha = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --alpha value");
                        return 2;
                    }
                };
                i += 2;
            }
            "--target-rate" if i + 1 < args.len() => {
                target_rate = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --target-rate value");
                        return 2;
                    }
                };
                i += 2;
            }
            "--min-prob" if i + 1 < args.len() => {
                min_prob = match args[i + 1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("Invalid --min-prob value");
                        return 2;
                    }
                };
                i += 2;
            }
            "-h" | "--help" => {
                println!(
                    "USAGE: cloture ppi [--alpha 0.05] [--target-rate 0.10] [--min-prob 0.05]"
                );
                return 0;
            }
            other => {
                eprintln!("Unknown option for ppi: {}", other);
                return 2;
            }
        }
    }

    let mut ppi = match ActivePPI::new(alpha, target_rate, min_prob) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            return 2;
        }
    };
    let stdin = io::stdin();
    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin error: {}", e);
                return 1;
            }
        };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let obj = match parse_json(t) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid JSON line: {}", e);
                return 2;
            }
        };
        let pred = obj.get("pred").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let audited = obj.get("audited").and_then(|v| v.as_bool()).unwrap_or(false);
        let gold = obj.get("gold").and_then(|v| v.as_f64());
        let prob = obj.get("prob").and_then(|v| v.as_f64()).unwrap_or(target_rate);
        ppi.observe(pred, audited, gold, prob);
    }

    let st = ppi.status();
    let out = json_obj(vec![
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
    ]);
    println!("{}", out.to_json_string());
    0
}

fn cmd_bench(args: &[String]) -> i32 {
    let json_out = args.iter().any(|a| a == "--json");
    let mut rng = Rng::from_seed(0xC107_0001);
    let mut rows: Vec<(&'static str, f64, f64)> = Vec::new();
    let mut sink: f64 = 0.0;

    // 1. EmpiricalBernsteinCS update
    {
        let n = 1_000_000usize;
        let xs: Vec<f64> = (0..1024).map(|_| rng.next_f64()).collect();
        let mut cs = EmpiricalBernsteinCS::new(0.05).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            let b = cs.update(xs[i & 1023]);
            sink += b.lower;
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("EmpiricalBernsteinCS::update", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 2. PairedEProcess update
    {
        let n = 1_000_000usize;
        let a: Vec<f64> = (0..1024).map(|_| rng.next_f64()).collect();
        let b: Vec<f64> = (0..1024).map(|_| rng.next_f64()).collect();
        let mut pe = PairedEProcess::two_sided(0.05).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            let st = pe.update(a[i & 1023], b[i & 1023]);
            sink += st.anytime_p_value;
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("PairedEProcess::update", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 3. AscdStopper (K=4) update
    {
        let n = 500_000usize;
        let probs = [0.60, 0.25, 0.10, 0.05];
        let mut st = AscdStopper::new(4, 0.05, None).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            if i % 64 == 0 {
                st.reset();
            }
            let d = st.update(&probs);
            sink += d.log_evidence;
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("AscdStopper(K=4)::update", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 4. CiteStopper update
    {
        let n = 500_000usize;
        let labels: Vec<u64> = (0..1024).map(|i| (i % 6) as u64).collect();
        let mut st = CiteStopper::new(0.05, None).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            if i % 64 == 0 {
                st.reset();
            }
            let d = st.update(labels[i & 1023]);
            sink += d.log_evidence;
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("CiteStopper::update", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 5. ActivePPI observe
    {
        let n = 1_000_000usize;
        let fs: Vec<f64> = (0..1024).map(|_| rng.next_f64()).collect();
        let mut ppi = ActivePPI::new(0.05, 0.10, 0.05).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            let aud = (i % 10) == 0;
            let f = fs[i & 1023];
            let y = if aud { Some((f + 0.05).min(1.0)) } else { None };
            let st = ppi.observe(f, aud, y, 0.10);
            sink += st.lower;
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("ActivePPI::observe", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 6. BettingCS (grid=201) update
    {
        let n = 100_000usize;
        let xs: Vec<f64> = (0..1024).map(|_| rng.next_f64()).collect();
        let mut bcs = BettingCS::with_grid(0.05, 201).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            let b = bcs.update(xs[i & 1023]);
            sink += b.width();
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("BettingCS(G=201)::update", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 7. SafeFuse (K=5) update
    {
        let n = 200_000usize;
        let mut fuse = SafeFuse::new(5, 0.01, 0.05).unwrap();
        let losses = [0.12, 0.15, 0.14, 0.18, 0.13];
        let t0 = Instant::now();
        for _ in 0..n {
            let u = fuse.update(&losses, Some(0.20));
            sink += u.fused_loss;
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("SafeFuse(K=5)::update", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 8. OnlineConformal update
    {
        let n = 1_000_000usize;
        let ps: Vec<f64> = (0..1024).map(|_| rng.next_f64()).collect();
        let mut oc = OnlineConformal::new(0.10).unwrap();
        let t0 = Instant::now();
        for i in 0..n {
            sink += oc.update_pvalue(ps[i & 1023]);
        }
        let dt = t0.elapsed().as_secs_f64();
        rows.push(("OnlineConformal::update_pvalue", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 9. EvidenceSketch (712 B) merge
    {
        let n = 500_000usize;
        let mut a = EvidenceSketch::new();
        let mut b = EvidenceSketch::new();
        for i in 0..64 {
            a.observe(i as f64 * 0.01, Some(0.1));
            b.observe(i as f64 * 0.015, Some(-0.05));
        }
        let t0 = Instant::now();
        for _ in 0..n {
            a.merge(&b);
        }
        let dt = t0.elapsed().as_secs_f64();
        sink += a.mean();
        rows.push(("EvidenceSketch(712B)::merge", dt * 1e9 / (n as f64), (n as f64) / dt));
    }

    // 10. PanelAudit (K=9, N=500, 2000 perms + 2000 Condorcet + 1000 boot)
    {
        let k_j = 9usize;
        let n_it = 500usize;
        let mut votes = vec![0u32; k_j * n_it];
        let mut gold = vec![0u32; n_it];
        for i in 0..n_it {
            gold[i] = (i % 4) as u32;
            let hard = rng.next_f64() < 0.25;
            for j in 0..k_j {
                let err = if hard { rng.next_f64() < 0.65 } else { rng.next_f64() < 0.08 };
                votes[i * k_j + j] = if err { (gold[i] + 1) % 4 } else { gold[i] };
            }
        }
        let audit = PanelAudit::from_votes(&votes, &gold, n_it, k_j, None).unwrap();
        let cfg = PanelAuditConfig {
            n_condorcet_sims: 2000,
            n_permutations: 2000,
            n_bootstrap: 1000,
            subset_target_ratio: 0.90,
            seed: 42,
        };
        let t0 = Instant::now();
        let rep = audit.run_with_config(&cfg);
        let dt = t0.elapsed().as_secs_f64();
        sink += rep.n_eff_kish;
        rows.push((
            "PanelAudit(9x500, 2k perms+2k MC+1k boot)",
            dt * 1e9,
            1.0 / dt,
        ));
    }

    if json_out {
        let arr: Vec<Json> = rows
            .iter()
            .map(|(name, ns, ops)| {
                json_obj(vec![
                    ("benchmark", Json::Str((*name).to_string())),
                    ("ns_per_op", Json::Num(*ns)),
                    ("ops_per_sec", Json::Num(*ops)),
                ])
            })
            .collect();
        let out = json_obj(vec![
            ("version", Json::Str(VERSION.to_string())),
            ("benchmarks", Json::Arr(arr)),
            ("sink", Json::Num(sink)),
        ]);
        println!("{}", out.to_json_string());
    } else {
        println!("cloture {} — Zero-Dependency Rust Core Benchmarks", VERSION);
        println!("──────────────────────────────────────────────────────────────────────");
        println!("{:<44} {:>12} {:>14}", "Operation", "Latency", "Throughput");
        println!("──────────────────────────────────────────────────────────────────────");
        for (name, ns, ops) in &rows {
            if *ns >= 1_000_000.0 {
                println!("{:<44} {:>9.2} ms {:>11.1} /s", name, ns / 1e6, ops);
            } else if *ns >= 1_000.0 {
                println!("{:<44} {:>9.2} us {:>11.0} /s", name, ns / 1e3, ops);
            } else {
                println!("{:<44} {:>9.1} ns {:>10.2} M/s", name, ns, ops / 1e6);
            }
        }
        println!("──────────────────────────────────────────────────────────────────────");
        if sink.is_nan() {
            eprintln!("warn: nan sink");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_rfc4180_quotes_escapes_crlf_bom_and_newlines() {
        let raw = "\u{FEFF}gold,\"gpt-4o\",\"claude,sonnet\",\"llama \"\"3\"\"\"\r\n\
                   \"option, A\",\"option, A\",\"option \"\"B\"\"\",\"option\nC\"\r\n\
                   \"option \"\"B\"\"\",\"option \"\"B\"\"\",\"option \"\"B\"\"\",\"option, A\"\r\n\r\n";
        let recs = parse_csv_records(raw).unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0], vec!["gold", "gpt-4o", "claude,sonnet", "llama \"3\""]);
        assert_eq!(recs[1], vec!["option, A", "option, A", "option \"B\"", "option\nC"]);
        assert_eq!(recs[2], vec!["option \"B\"", "option \"B\"", "option \"B\"", "option, A"]);

        let parsed = parse_csv_panel(raw, "gold", None).unwrap();
        assert_eq!(parsed.n_items, 2);
        assert_eq!(parsed.k_judges, 3);
        assert_eq!(
            parsed.judge_names.as_deref(),
            Some(&[
                "gpt-4o".to_string(),
                "claude,sonnet".to_string(),
                "llama \"3\"".to_string()
            ][..])
        );
        // Consistent label-to-u32 mapping: gpt-4o matches gold on both rows.
        assert_eq!(parsed.votes_flat[0], parsed.gold[0]);
        assert_eq!(parsed.votes_flat[3], parsed.gold[1]);
        // claude,sonnet misses row 0 and matches row 1.
        assert_ne!(parsed.votes_flat[1], parsed.gold[0]);
        assert_eq!(parsed.votes_flat[4], parsed.gold[1]);
    }

    #[test]
    fn csv_panel_string_labels_and_strata_mapping() {
        let csv_str = "item_stratum,target,j_alpha,j_beta,j_clone\n\
                       easy,yes,yes,yes,yes\n\
                       easy,no,no,yes,no\n\
                       hard,yes,no,yes,no\n\
                       hard,no,yes,no,yes\n";
        let parsed = parse_csv_panel(csv_str, "target", Some("item_stratum")).unwrap();
        assert_eq!(parsed.n_items, 4);
        assert_eq!(parsed.k_judges, 3);
        assert_eq!(
            parsed.judge_names,
            Some(vec![
                "j_alpha".to_string(),
                "j_beta".to_string(),
                "j_clone".to_string()
            ])
        );
        assert_eq!(parsed.strata, Some(vec![0, 0, 1, 1]));

        let rep = PanelAudit::from_votes(
            &parsed.votes_flat,
            &parsed.gold,
            parsed.n_items,
            parsed.k_judges,
            parsed.strata.as_deref(),
        )
        .unwrap()
        .run_with_config(&PanelAuditConfig {
            n_condorcet_sims: 200,
            n_permutations: 200,
            n_bootstrap: 100,
            seed: 42,
            ..Default::default()
        });
        // j_alpha and j_clone are exact clones (phi == 1.0).
        assert!((rep.phi_matrix[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn csv_panel_rejects_malformed_inputs_and_bad_flags_with_code_2() {
        assert!(parse_csv_panel("", "gold", None).is_err());
        assert!(parse_csv_panel("   \r\n\n", "gold", None).is_err());
        assert!(parse_csv_panel("gold,j1,j2\n", "gold", None).is_err());
        assert!(parse_csv_panel("label,j1,j2\nA,A,B\n", "gold", None).is_err());
        assert!(parse_csv_panel("gold,j1\nA,A\n", "gold", Some("missing")).is_err());
        assert!(parse_csv_panel("gold,strata\nA,easy\n", "gold", Some("strata")).is_err());
        assert!(parse_csv_panel("gold,j1\nA,A\n", "gold", Some("gold")).is_err());
        assert!(parse_csv_panel("gold,gold,j1\nA,A,A\n", "gold", None).is_err());
        assert!(parse_csv_panel("gold,j1,j2\nA,A\n", "gold", None).is_err());
        assert!(parse_csv_panel("gold,j1\n\"unterminated,A\n", "gold", None).is_err());
        assert!(parse_csv_panel("gold,j1\nA\"bad,B\n", "gold", None).is_err());
        assert!(parse_csv_panel("gold,j1\n\"A\"trailing,B\n", "gold", None).is_err());

        // CLI flag validation returns exit code 2 without panicking.
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(cmd_audit(&s(&["--csv"])), 2);
        assert_eq!(cmd_audit(&s(&["--gold-col"])), 2);
        assert_eq!(cmd_audit(&s(&["--strata-col"])), 2);
        assert_eq!(cmd_audit(&s(&["--csv", "/nonexistent/does_not_exist.csv"])), 2);
        assert_eq!(
            cmd_audit(&s(&["--file", "a.json", "--csv", "b.csv"])),
            2
        );
        assert_eq!(cmd_audit(&s(&["--gold-col", "label"])), 2);
        assert_eq!(cmd_audit(&s(&["--strata-col", "diff"])), 2);
        assert_eq!(cmd_audit(&s(&["--unknown-flag"])), 2);
    }
}

