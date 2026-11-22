class Cloture < Formula
  desc "Zero-dependency anytime-valid multi-agent consensus, panel auditing, racing & Active-PPI++"
  homepage "https://pgg898200-ship-it.github.io/cloture/"
  version "0.2.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.2.0/cloture-v0.2.0-darwin-aarch64.tar.gz"
      sha256 "1fb13b8211d6b3693b7ad7204550c8afd4455189a57902f029b5b6376fd7819e"
    else
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.2.0/cloture-v0.2.0-darwin-x86_64.tar.gz"
      sha256 "e4f298ba5d3cb03411fbe726c85cca37158fd65bb3a4b56895929fa6c9ec5677"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.2.0/cloture-v0.2.0-linux-aarch64.tar.gz"
      sha256 "27f887912882966e2ebf2056e81f17304e63dd9c24c9b624b844bca088f14b25"
    else
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.2.0/cloture-v0.2.0-linux-x86_64.tar.gz"
      sha256 "be23cf40922cb2a51ebf35ba644e235f8a03635ed5d5ad2d2e1391c9405f08c1"
    end
  end

  def install
    bin.install "cloture"
    include.install "cloture.h" if File.exist?("cloture.h")
  end

  test do
    assert_match "cloture", shell_output("#{bin}/cloture --version")
  end
end
