#!/bin/sh
# One-line verified binary installer for `cloture`
# Usage: curl -fsSL https://raw.githubusercontent.com/pgg898200-ship-it/cloture/main/scripts/install.sh | sh
# Optional env vars:
#   CLOTURE_VERSION=v0.1.0        (default: latest GitHub release)
#   CLOTURE_INSTALL_DIR=~/.local/bin

set -eu

REPO="pgg898200-ship-it/cloture"
INSTALL_DIR="${CLOTURE_INSTALL_DIR:-$HOME/.local/bin}"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)  OS_TAG="linux" ;;
  Darwin) OS_TAG="darwin" ;;
  *)
    echo "Unsupported OS: $OS (download binaries from https://github.com/$REPO/releases or build with cargo)" >&2
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64)  ARCH_TAG="x86_64" ;;
  aarch64|arm64) ARCH_TAG="aarch64" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

if [ -n "${CLOTURE_VERSION:-}" ]; then
  VERSION="$CLOTURE_VERSION"
else
  VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
  if [ -z "$VERSION" ]; then
    echo "Could not determine latest release tag from GitHub API; set CLOTURE_VERSION=v0.1.0." >&2
    exit 1
  fi
fi

ASSET="cloture-${VERSION}-${OS_TAG}-${ARCH_TAG}.tar.gz"
BASE_URL="https://github.com/$REPO/releases/download/$VERSION"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading cloture $VERSION ($OS_TAG-$ARCH_TAG)..."
curl -fsSL "$BASE_URL/$ASSET" -o "$TMPDIR/$ASSET"
curl -fsSL "$BASE_URL/SHA256SUMS.txt" -o "$TMPDIR/SHA256SUMS.txt"

EXPECTED="$(grep "  $ASSET\$" "$TMPDIR/SHA256SUMS.txt" | awk '{print $1}' | head -n 1)"
if [ -z "$EXPECTED" ]; then
  echo "Checksum entry for $ASSET not found in SHA256SUMS.txt; aborting." >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$TMPDIR/$ASSET" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$TMPDIR/$ASSET" | awk '{print $1}')"
else
  echo "Neither sha256sum nor shasum found; cannot verify download integrity." >&2
  exit 1
fi

if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "SHA-256 verification FAILED for $ASSET!" >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "$TMPDIR/$ASSET" -C "$TMPDIR"
install -m 755 "$TMPDIR/cloture" "$INSTALL_DIR/cloture"

echo "Installed cloture to $INSTALL_DIR/cloture (SHA-256 verified)."
"$INSTALL_DIR/cloture" --version || true
echo ""
echo "Next steps:"
echo "  $INSTALL_DIR/cloture bench          # run local sub-microsecond benchmark"
echo "  $INSTALL_DIR/cloture mcp            # start stdio JSON-RPC 2.0 MCP server"
case ":${PATH}:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    echo ""
    echo "Add $INSTALL_DIR to your PATH:"
    echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
    ;;
esac
