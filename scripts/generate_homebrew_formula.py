#!/usr/bin/env python3
"""Generate Formula/cloture.rb from a release tag and SHA256SUMS.txt."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
REPO = "pgg898200-ship-it/cloture"


def parse_sums(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            out[parts[-1]] = parts[0]
    return out


def render_formula(version: str, sums: Dict[str, str]) -> str:
    tag = version if version.startswith("v") else f"v{version}"
    ver = tag.lstrip("v")
    base = f"https://github.com/{REPO}/releases/download/{tag}"

    def need(asset: str) -> str:
        if asset not in sums:
            raise KeyError(f"Missing {asset} in SHA256SUMS.txt")
        return sums[asset]

    mac_arm = f"cloture-{tag}-darwin-aarch64.tar.gz"
    mac_x64 = f"cloture-{tag}-darwin-x86_64.tar.gz"
    lin_arm = f"cloture-{tag}-linux-aarch64.tar.gz"
    lin_x64 = f"cloture-{tag}-linux-x86_64.tar.gz"

    return f'''class Cloture < Formula
  desc "Zero-dependency anytime-valid multi-agent consensus, panel auditing, racing & Active-PPI++"
  homepage "https://pgg898200-ship-it.github.io/cloture/"
  version "{ver}"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "{base}/{mac_arm}"
      sha256 "{need(mac_arm)}"
    else
      url "{base}/{mac_x64}"
      sha256 "{need(mac_x64)}"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "{base}/{lin_arm}"
      sha256 "{need(lin_arm)}"
    else
      url "{base}/{lin_x64}"
      sha256 "{need(lin_x64)}"
    end
  end

  def install
    bin.install "cloture"
    include.install "cloture.h" if File.exist?("cloture.h")
  end

  test do
    assert_match "cloture", shell_output("#{{bin}}/cloture --version")
  end
end
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="Release tag (e.g. v0.1.0)")
    ap.add_argument("--sums", type=Path, required=True, help="Path to SHA256SUMS.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "Formula" / "cloture.rb")
    args = ap.parse_args()
    sums = parse_sums(args.sums.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_formula(args.version, sums), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
