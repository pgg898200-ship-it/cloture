class Cloture < Formula
  desc "Zero-dependency anytime-valid multi-agent consensus, panel auditing, racing & Active-PPI++"
  homepage "https://pgg898200-ship-it.github.io/cloture/"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.1.0/cloture-v0.1.0-darwin-aarch64.tar.gz"
      sha256 "9cb6d659b92bf13455ee26377c512d1c43e086af25bcf6894855ba478f090eea"
    else
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.1.0/cloture-v0.1.0-darwin-x86_64.tar.gz"
      sha256 "f701e992787ac535d55e9ed6f06900aad86f6edfbd653b15e70c538dd4c74d51"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.1.0/cloture-v0.1.0-linux-aarch64.tar.gz"
      sha256 "3db1079c7ab42b7c321905581bef76e6012aad3286caeefb2279568e87fde895"
    else
      url "https://github.com/pgg898200-ship-it/cloture/releases/download/v0.1.0/cloture-v0.1.0-linux-x86_64.tar.gz"
      sha256 "e1e10988e081a3c58b3b6f2190be85a0a0a3404ff36defea655014ac05e0ec06"
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
