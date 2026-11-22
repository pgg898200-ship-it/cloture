#!/usr/bin/env python3
"""Zero-dependency PEP 427 wheel builder for `cloture`.

Because `cloture` loads `libcloture_core` via Python's stdlib `ctypes`, a single
`py3-none-<platform>.whl` wheel supports Python 3.9 through 3.14+ and PyPy on
that OS/architecture without CPython C-API headers or ABI lock-in.

Usage:
    python3 scripts/build_wheel.py \
        --lib target/release/libcloture_core.so \
        --plat-tag manylinux_2_28_x86_64 \
        --out-dir dist
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "cloture"
VERSION = "0.1.0"


def _default_lib_path() -> Path:
    sys_name = platform.system()
    if sys_name == "Darwin":
        candidates = ["libcloture_core.dylib"]
    elif sys_name == "Windows":
        candidates = ["cloture_core.dll", "libcloture_core.dll"]
    else:
        candidates = ["libcloture_core.so"]
    for name in candidates:
        p = ROOT / "target" / "release" / name
        if p.exists():
            return p
    return ROOT / "target" / "release" / candidates[0]


def _default_plat_tag() -> str:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if sys_name == "linux":
        arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
        return f"linux_{arch}"
    if sys_name == "darwin":
        if machine in ("arm64", "aarch64"):
            return "macosx_11_0_arm64"
        return "macosx_10_12_x86_64"
    if sys_name == "windows":
        return "win_amd64"
    return f"{sys_name}_{machine}"


def _sha256_record(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _build_metadata(readme_text: str) -> bytes:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {PACKAGE_NAME}",
        f"Version: {VERSION}",
        "Summary: Sub-microsecond anytime-valid consensus stopping, correlated-panel auditing, and active prediction-powered racing for multi-agent AI systems.",
        "Home-page: https://pgg898200-ship-it.github.io/cloture",
        "Author: Cloture Contributors",
        "License: MIT",
        "Project-URL: Homepage, https://pgg898200-ship-it.github.io/cloture",
        "Project-URL: Repository, https://github.com/pgg898200-ship-it/cloture",
        "Project-URL: Documentation, https://github.com/pgg898200-ship-it/cloture#readme",
        "Project-URL: Changelog, https://github.com/pgg898200-ship-it/cloture/blob/main/CHANGELOG.md",
        "Project-URL: Issues, https://github.com/pgg898200-ship-it/cloture/issues",
        "Keywords: ai-agents,multi-agent,self-consistency,anytime-valid,e-values,prediction-powered-inference,llm-evaluation,conformal-prediction,sequential-testing,rust",
        "Classifier: Development Status :: 4 - Beta",
        "Classifier: Intended Audience :: Developers",
        "Classifier: Intended Audience :: Science/Research",
        "Classifier: License :: OSI Approved :: MIT License",
        "Classifier: Programming Language :: Python :: 3",
        "Classifier: Programming Language :: Rust",
        "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Classifier: Topic :: Scientific/Engineering :: Mathematics",
        "Requires-Python: >=3.9",
        "Provides-Extra: numpy",
        'Requires-Dist: numpy>=1.24; extra == "numpy"',
        "Provides-Extra: dev",
        'Requires-Dist: pytest>=8.0; extra == "dev"',
        'Requires-Dist: numpy>=1.24; extra == "dev"',
        'Requires-Dist: scipy>=1.11; extra == "dev"',
        "Description-Content-Type: text/markdown",
        "",
        readme_text,
    ]
    return "\n".join(lines).encode("utf-8")


def _build_wheel_meta(plat_tag: str) -> bytes:
    lines = [
        "Wheel-Version: 1.0",
        "Generator: cloture-build-wheel 0.1.0",
        "Root-Is-Purelib: false",
        f"Tag: py3-none-{plat_tag}",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_wheel(lib_path: Path, plat_tag: str, out_dir: Path) -> Path:
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Native library not found at {lib_path}. "
            "Run `cargo build --release -p cloture-core` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_filename = f"{PACKAGE_NAME}-{VERSION}-py3-none-{plat_tag}.whl"
    wheel_path = out_dir / wheel_filename
    dist_info = f"{PACKAGE_NAME}-{VERSION}.dist-info"

    pkg_dir = ROOT / "python" / "cloture"
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    license_bytes = (ROOT / "LICENSE").read_bytes()
    notice_bytes = (ROOT / "NOTICE").read_bytes()

    entries: List[Tuple[str, bytes, int]] = []

    # Package Python modules and typing marker
    for src in sorted(pkg_dir.iterdir()):
        if src.name == "__pycache__" or src.suffix in (".pyc", ".pyo", ".so", ".dylib", ".dll"):
            continue
        if src.is_file():
            entries.append((f"{PACKAGE_NAME}/{src.name}", src.read_bytes(), 0o644))

    # Bundled native shared library
    entries.append((f"{PACKAGE_NAME}/{lib_path.name}", lib_path.read_bytes(), 0o755))

    # Dist-info metadata files
    entries.append((f"{dist_info}/METADATA", _build_metadata(readme_text), 0o644))
    entries.append((f"{dist_info}/WHEEL", _build_wheel_meta(plat_tag), 0o644))
    entries.append((f"{dist_info}/LICENSE", license_bytes, 0o644))
    entries.append((f"{dist_info}/NOTICE", notice_bytes, 0o644))
    entries.append((f"{dist_info}/top_level.txt", f"{PACKAGE_NAME}\n".encode("utf-8"), 0o644))

    record_lines: List[str] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arcname, data, mode in entries:
            zi = zipfile.ZipInfo(arcname, date_time=(2026, 11, 22, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o100000 | mode) << 16
            zf.writestr(zi, data)
            record_lines.append(f"{arcname},{_sha256_record(data)},{len(data)}")

        record_arcname = f"{dist_info}/RECORD"
        record_lines.append(f"{record_arcname},,")
        record_bytes = ("\n".join(record_lines) + "\n").encode("utf-8")
        zi = zipfile.ZipInfo(record_arcname, date_time=(2026, 11, 22, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = (0o100000 | 0o644) << 16
        zf.writestr(zi, record_bytes)

    return wheel_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build universal py3-none-<plat> wheel for cloture.")
    parser.add_argument("--lib", type=Path, default=_default_lib_path(), help="Path to libcloture_core.{so,dylib,dll}")
    parser.add_argument("--plat-tag", type=str, default=_default_plat_tag(), help="PEP 425 platform tag (e.g. manylinux_2_28_x86_64)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist", help="Output directory for wheel")
    parser.add_argument("--cargo-build", action="store_true", help="Run `cargo build --release -p cloture-core` before packaging")
    args = parser.parse_args()

    if args.cargo_build:
        subprocess.run(["cargo", "build", "--release", "-p", "cloture-core"], cwd=ROOT, check=True)

    wheel_path = build_wheel(args.lib, args.plat_tag, args.out_dir)
    size_mib = wheel_path.stat().st_size / (1024 * 1024)
    digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    print(f"Built {wheel_path} ({size_mib:.2f} MiB) sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
