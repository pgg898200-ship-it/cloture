import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { loadCloture } from "./cloture.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.resolve(__dirname, "..", "site", "wasm", "cloture_core.wasm");
const wasmBytes = fs.readFileSync(wasmPath);

test("WASM module instantiates with zero imports and runs all core engines", async () => {
  const mod = await WebAssembly.compile(wasmBytes);
  assert.deepEqual(WebAssembly.Module.imports(mod), [], "WASM must have zero host imports");

  const cloture = await loadCloture(wasmBytes);

  // 1. ASC-D grey-box consensus stopping
  const ascd = new cloture.AscdStopper(4, 0.05, 64);
  let dec;
  for (let i = 0; i < 20; i++) {
    dec = ascd.update([0.92, 0.04, 0.02, 0.02]);
    if (dec.certified) break;
  }
  assert.equal(dec.certified, true);
  assert.equal(dec.winner, 0);
  assert.ok(dec.nSamples <= 12);
  ascd.free();

  // 2. CITE unbounded consensus stopping + async cancellation
  const res = await cloture.consensus(
    async (i, signal) => {
      await new Promise((r) => setTimeout(r, 2));
      if (signal.aborted) return "aborted";
      return "SELECT 42;";
    },
    { delta: 0.05, maxCalls: 32, concurrency: 6 }
  );
  assert.equal(res.certified, true);
  assert.equal(res.winner, "SELECT 42;");
  assert.ok(res.nSamples <= 12);
  assert.ok(res.nCancelled >= 1);

  // 3. Correlated judge-panel audit (3 cloned judges -> n_eff = 1.0)
  const votes = [];
  const gold = [];
  for (let i = 0; i < 40; i++) {
    const v = i < 30 ? 1 : 0;
    votes.push([v, v, v]);
    gold.push(1);
  }
  const rep = cloture.auditPanel(votes, gold, { nCondorcetSims: 300, nPerms: 300, nBoot: 200, seed: 7 });
  assert.equal(rep.kJudges, 3);
  assert.equal(rep.nItems, 40);
  assert.ok(Math.abs(rep.nEffKish - 1.0) < 1e-6, `Expected n_eff=1.0 for clones, got ${rep.nEffKish}`);
  assert.equal(rep.looDeltaAcc.length, 3);

  // 4. PopulationCert
  const pop = new cloture.PopulationCert(0.05);
  let pdec;
  for (let u = 0; u < 30; u++) pdec = pop.addUnit([0.5, 0.6, 0.4, 0.5]);
  assert.equal(pdec.certified, true);
  assert.ok(pdec.thetaHat > 0.4);
  pop.free();

  // 5. ActivePPI + EBCS + BettingCS + OnlineConformal
  const ppi = new cloture.ActivePPI(0.05, 0.2, 0.05);
  let pres;
  let s = 123456789;
  const rnd = () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = 0; i < 400; i++) {
    const y = rnd() < 0.8 ? 1.0 : 0.0;
    const f = Math.min(1.0, Math.max(0.0, y * 0.85 + 0.1));
    const audited = rnd() < 0.2;
    pres = ppi.observe(f, audited, audited ? y : 0.0, 0.2);
  }
  assert.ok(pres.lower <= 0.8 && pres.upper >= 0.8, `Expected [lower,upper] to cover 0.8, got [${pres.lower}, ${pres.upper}]`);
  ppi.free();

  const eb = new cloture.EmpiricalBernsteinCS(0.05, 0.0, 1.0);
  const b = eb.update(0.75);
  assert.equal(b.n, 1);
  eb.free();

  const oc = new cloture.OnlineConformal(0.10);
  const a1 = oc.updatePvalue(0.02);
  assert.ok(Number.isFinite(a1));
  oc.free();

  // 6. 712-byte EvidenceSketch wire monoid roundtrip + merge + subtract
  const s1 = new cloture.EvidenceSketch();
  const s2 = new cloture.EvidenceSketch();
  for (let i = 0; i < 10; i++) {
    s1.observe(0.2 + 0.01 * i, 0.1);
    s2.observe(0.6 + 0.01 * i, 0.1);
  }
  const bytes = s1.toBytes();
  assert.equal(bytes.length, 712);
  const s1Clone = cloture.EvidenceSketch.fromBytes(bytes);
  s1Clone.merge(s2);
  s1Clone.subtract(s2);
  assert.ok(Math.abs(s1Clone.mean() - s1.mean()) < 1e-12);
  s1.free();
  s2.free();
  s1Clone.free();
});
