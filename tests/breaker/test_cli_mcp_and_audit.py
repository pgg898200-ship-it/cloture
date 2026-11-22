"""Adversarial CLI/MCP fuzzing, zero-dependency verification, and anonymization audit."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_BIN = ROOT / "target" / "release" / ("cloture.exe" if sys.platform == "win32" else "cloture")


def _run_cli(args: list[str], stdin_bytes: bytes = b"", timeout: float = 10.0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(CLI_BIN), *args],
        input=stdin_bytes,
        capture_output=True,
        timeout=timeout,
    )


def _assert_no_panic_or_signal(proc: subprocess.CompletedProcess[bytes], ctx: str) -> None:
    stderr_txt = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode >= 0 and proc.returncode < 128, (
        f"PROCESS CRASHED [{ctx}]: exit_code={proc.returncode}, stderr={stderr_txt!r}"
    )
    assert "panicked at" not in stderr_txt and "SIGABRT" not in stderr_txt and "SIGSEGV" not in stderr_txt, (
        f"RUST PANIC DETECTED [{ctx}]: exit_code={proc.returncode}, stderr={stderr_txt!r}"
    )


# ============================================================================
# 1. CLI fuzzing: malformed JSON, 500-deep nesting, binary garbage, empty stdin,
#    giant numbers (1e400), invalid flags, NaN-producing inputs
# ============================================================================


DEEP_ARR_500 = ("[" * 500 + "1" + "]" * 500).encode("ascii")
DEEP_OBJ_500 = ('{"a":' * 500 + "1" + "}" * 500).encode("ascii")
BINARY_GARBAGE = bytes([0xFF, 0xFE, 0x00, 0x80, 0xC0, 0xAF, 0x0A, 0xDE, 0xAD, 0xBE, 0xEF])

FUZZ_PAYLOADS = [
    ("empty", b""),
    ("whitespace_only", b"   \n\t\r\n  "),
    ("malformed_json_brace", b"{not valid json\n"),
    ("malformed_json_bracket", b"[1, 2, \n"),
    ("unterminated_string", b'{"a": "unterminated\n'),
    ("bad_escape", b'{"a": "\\q"}\n'),
    ("truncated_unicode_escape", b'{"a": "\\u12"}\n'),
    ("deep_array_500", DEEP_ARR_500 + b"\n"),
    ("deep_object_500", DEEP_OBJ_500 + b"\n"),
    ("binary_garbage", BINARY_GARBAGE),
    ("giant_number_1e400", b"[1e400, -1e400]\n"),
    ("giant_number_obj", b'{"pred": 1e400, "audited": true, "gold": 1e400, "prob": 1e400}\n'),
    ("nan_string_input", b"NaN, Inf, -Inf\n"),
    ("json_nulls", b"null\nnull\n"),
    ("json_scalar", b"12345\n"),
]


@pytest.mark.parametrize(
    "subcmd",
    [
        ["stop", "--mode", "cite"],
        ["stop", "--mode", "ascd", "--k", "2"],
        ["stop", "--mode", "weighted"],
        ["audit", "--json"],
        ["race", "--paired"],
        ["race", "--unpaired", "--arms", "3"],
        ["ppi"],
        ["mcp"],
    ],
)
@pytest.mark.parametrize("payload_name,payload", FUZZ_PAYLOADS)
def test_cli_subcommands_never_panic_on_fuzz_stdin(
    subcmd: list[str], payload_name: str, payload: bytes
):
    """All CLI subcommands & MCP server must NEVER panic or crash on malformed/hostile stdin."""
    proc = _run_cli(subcmd, stdin_bytes=payload)
    _assert_no_panic_or_signal(proc, f"{' '.join(subcmd)} < {payload_name}")


@pytest.mark.parametrize(
    "bad_args",
    [
        ["nonexistent_subcommand"],
        ["stop", "--unknown-flag"],
        ["stop", "--delta"],
        ["stop", "--delta", "not_a_float"],
        ["stop", "--delta", "0.0"],
        ["stop", "--delta", "1.0"],
        ["stop", "--delta", "NaN"],
        ["stop", "--delta", "-0.5"],
        ["stop", "--k", "not_an_int"],
        ["stop", "--max-budget", "-5"],
        ["stop", "--mode", "ascd", "--k", "0"],
        ["audit", "--file", "/nonexistent/path/to/nowhere.json"],
        ["audit", "--unknown"],
        ["race", "--unknown"],
        ["race", "--delta", "NaN", "--arms", "3"],
        ["ppi", "--unknown"],
        ["ppi", "--alpha", "0.0"],
        ["ppi", "--target-rate", "0.05", "--min-prob", "0.5"],
    ],
)
def test_cli_invalid_flags_never_panic(bad_args: list[str]):
    """Invalid CLI flags and out-of-range parameters must exit cleanly, never panic."""
    proc = _run_cli(bad_args, stdin_bytes=b"")
    _assert_no_panic_or_signal(proc, f"args={bad_args}")


@pytest.mark.parametrize(
    "label,args,stdin_bytes",
    [
        (
            "audit --boot usize::MAX",
            ["audit", "--boot", "18446744073709551615", "--perms", "10", "--condorcet", "10", "--json"],
            b'{"votes": [[0, 1], [1, 0]], "gold": [0, 1]}',
        ),
        (
            "race --arms usize::MAX",
            ["race", "--arms", "18446744073709551615"],
            b"",
        ),
        (
            "stop --mode ascd --k 100000000",
            ["stop", "--mode", "ascd", "--k", "100000000"],
            b"",
        ),
    ],
)
def test_cli_huge_numeric_flags_must_not_sigabrt(label: str, args: list[str], stdin_bytes: bytes):
    """Huge integer flags in CLI must return clean validation error, NEVER SIGABRT / capacity overflow!"""
    proc = _run_cli(args, stdin_bytes=stdin_bytes)
    _assert_no_panic_or_signal(proc, label)


# ============================================================================
# 2. MCP JSON-RPC 2.0 protocol fuzzing & semantic correctness attacks
# ============================================================================


def test_mcp_protocol_error_handling_and_edge_cases():
    """Feed unknown methods, missing fields, bad argument types, 1e400, NaN-inducing inputs to MCP."""
    reqs = [
        # Malformed JSON -> -32700
        "{broken json",
        # Deeply nested JSON -> -32700 (must hit MAX_JSON_DEPTH cleanly)
        "[" * 500 + "0" + "]" * 500,
        # Missing method with id -> -32600
        json.dumps({"jsonrpc": "2.0", "id": 10}),
        # Unknown method -> -32601
        json.dumps({"jsonrpc": "2.0", "id": 11, "method": "does/not/exist"}),
        # tools/call missing params -> -32602
        json.dumps({"jsonrpc": "2.0", "id": 12, "method": "tools/call"}),
        # tools/call missing tool name -> -32602
        json.dumps({"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {}}),
        # tools/call unknown tool -> -32601
        json.dumps({"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {"name": "evil_tool"}}),
        # tools/call missing required fields for every tool -> isError=true
        json.dumps({"jsonrpc": "2.0", "id": 15, "method": "tools/call", "params": {"name": "cloture_mode_stop", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 16, "method": "tools/call", "params": {"name": "cloture_audit_panel", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 17, "method": "tools/call", "params": {"name": "cloture_population_cert", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 18, "method": "tools/call", "params": {"name": "cloture_active_ppi", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 19, "method": "tools/call", "params": {"name": "cloture_agent_race", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 20, "method": "tools/call", "params": {"name": "cloture_safe_fuse", "arguments": {}}}),
        # Giant number 1e400 in delta & samples
        '{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"cloture_mode_stop","arguments":{"mode":"ascd","delta":1e400,"samples":[[0.5,0.5]]}}}',
        '{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"cloture_mode_stop","arguments":{"mode":"ascd","delta":0.05,"samples":[[1e400,-1e400],[0.0,0.0]]}}}',
        # Wrong argument types (string instead of object, empty inner arrays, mismatched rows)
        json.dumps({"jsonrpc": "2.0", "id": 23, "method": "tools/call", "params": {"name": "cloture_mode_stop", "arguments": "not_an_obj"}}),
        json.dumps({"jsonrpc": "2.0", "id": 24, "method": "tools/call", "params": {"name": "cloture_agent_race", "arguments": {"rounds": [[]]}}}),
        json.dumps({"jsonrpc": "2.0", "id": 25, "method": "tools/call", "params": {"name": "cloture_audit_panel", "arguments": {"votes": [[0, 1], [0]], "gold": [0, 1]}}}),
        json.dumps({"jsonrpc": "2.0", "id": 26, "method": "tools/call", "params": {"name": "cloture_safe_fuse", "arguments": {"losses": [[]]}}}),
        # Notification (no id) -> must produce NO response
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        # Ping -> ok
        json.dumps({"jsonrpc": "2.0", "id": 27, "method": "ping"}),
    ]
    proc = _run_cli(["mcp"], stdin_bytes=("\n".join(reqs) + "\n").encode("utf-8"))
    _assert_no_panic_or_signal(proc, "mcp protocol fuzz batch")
    assert proc.returncode == 0

    lines = proc.stdout.decode("utf-8").strip().splitlines()
    # Every request except the 1 notification should produce a valid JSON-RPC 2.0 response
    assert len(lines) == len(reqs) - 1, f"Expected {len(reqs) - 1} responses, got {len(lines)}"
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["error"]["code"] == -32700
    assert parsed[1]["error"]["code"] == -32700
    assert parsed[2]["error"]["code"] == -32600
    assert parsed[3]["error"]["code"] == -32601
    assert parsed[4]["error"]["code"] == -32602
    assert parsed[5]["error"]["code"] == -32602
    assert parsed[6]["error"]["code"] == -32601
    for resp in parsed[7:13]:
        assert resp["result"]["isError"] is True
    assert parsed[-1]["id"] == 27 and "result" in parsed[-1]


def test_mcp_cite_mode_must_not_truncate_distinct_floats_to_zero():
    """Json::as_u64() truncates arbitrary non-negative floats (0.1, 0.2, ..., 0.99) to 0u64,
    causing MCP cloture_mode_stop (cite) to FALSELY CERTIFY winner=0 on 25 all-distinct floats!
    """
    distinct_floats = [round(0.01 * i, 4) for i in range(1, 31)]  # 30 completely distinct values in (0, 0.31)
    req = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {
            "name": "cloture_mode_stop",
            "arguments": {"mode": "cite", "samples": distinct_floats, "delta": 0.05},
        },
    }
    proc = _run_cli(["mcp"], stdin_bytes=(json.dumps(req) + "\n").encode("utf-8"))
    _assert_no_panic_or_signal(proc, "mcp cite distinct floats")
    resp = json.loads(proc.stdout.decode("utf-8").strip())
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert not payload["certified"], (
        f"MCP CITE FALSELY CERTIFIED 30 COMPLETELY DISTINCT FLOATS {distinct_floats[:5]}... "
        f"as winner={payload['winner']!r} in n_samples={payload['n_samples']} "
        f"(root cause: Json::as_u64 truncates float -> u64)!"
    )


# ============================================================================
# 3. Zero-dependency & anonymization audit
# ============================================================================


def test_zero_external_crate_dependencies():
    """Verify Cargo.toml and Cargo.lock have ZERO external crate dependencies."""
    core_toml = (ROOT / "crates" / "cloture-core" / "Cargo.toml").read_text(encoding="utf-8")
    cli_toml = (ROOT / "crates" / "cloture-cli" / "Cargo.toml").read_text(encoding="utf-8")
    lock_text = (ROOT / "Cargo.lock").read_text(encoding="utf-8")

    # Parse [dependencies] blocks
    def deps_in_toml(text: str) -> list[str]:
        in_deps = False
        deps = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                in_deps = line in ("[dependencies]", "[dev-dependencies]", "[build-dependencies]")
                continue
            if in_deps and "=" in line:
                deps.append(line.split("=", 1)[0].strip())
        return deps

    assert deps_in_toml(core_toml) == [], f"cloture-core has external deps: {deps_in_toml(core_toml)}"
    assert deps_in_toml(cli_toml) == ["cloture-core"], f"cloture-cli has external deps: {deps_in_toml(cli_toml)}"

    lock_pkgs = re.findall(r'^name\s*=\s*"([^"]+)"', lock_text, flags=re.MULTILINE)
    assert sorted(lock_pkgs) == ["cloture-cli", "cloture-core"], (
        f"Cargo.lock contains external packages: {lock_pkgs}"
    )


def test_anonymization_no_leaked_private_identifiers():
    """Scan entire repo (excluding build/venv/cache and this audit regex) for private identifiers."""
    skip_dirs = {".venv", ".wheel-venv", "dist", "build", "telemetry", "target", ".pytest_cache", "__pycache__", ".git"}
    this_file = Path(__file__).resolve()
    banned = re.compile(
        r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|C:\\\\Users\\\\",
        re.IGNORECASE,
    )
    email_pat = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    hits: list[str] = []
    for p in ROOT.rglob("*"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if not p.is_file() or p.resolve() == this_file:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if banned.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{lineno}: {line.strip()}")
            if email_pat.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{lineno} [email]: {line.strip()}")

    assert not hits, "Leaked private workspace identifiers or emails found:\n" + "\n".join(hits)
