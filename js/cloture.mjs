/**
 * Zero-dependency WebAssembly JS/TS SDK for `cloture-core`.
 *
 * Works in browsers, Node.js 18+, Deno, Bun, Cloudflare Workers, and Vercel Edge.
 * Zero external dependencies and zero WASI/host imports.
 */

const SCOPE_NAMES = ["realized_unit", "population", "deterministic_lock"];

function readModeResult(view, ptr) {
  const certified = view.getInt32(ptr + 0, true) !== 0;
  const deterministicLock = view.getInt32(ptr + 4, true) !== 0;
  const hasWinner = view.getInt32(ptr + 8, true) !== 0;
  const hasLeader = view.getInt32(ptr + 12, true) !== 0;
  const winnerRaw = view.getBigUint64(ptr + 16, true);
  const leaderRaw = view.getBigUint64(ptr + 24, true);
  const nSamples = Number(view.getBigUint64(ptr + 32, true));
  const logEvidence = view.getFloat64(ptr + 40, true);
  const logThreshold = view.getFloat64(ptr + 48, true);
  const marginEstimate = view.getFloat64(ptr + 56, true);
  const scopeCode = view.getInt32(ptr + 64, true);
  return {
    certified,
    deterministicLock,
    winner: hasWinner ? Number(winnerRaw) : null,
    leader: hasLeader ? Number(leaderRaw) : null,
    nSamples,
    logEvidence,
    logThreshold,
    marginEstimate,
    scope: SCOPE_NAMES[scopeCode] ?? "realized_unit",
  };
}

function readPopResult(view, ptr) {
  return {
    nUnits: Number(view.getBigUint64(ptr + 0, true)),
    totalRollouts: Number(view.getBigUint64(ptr + 8, true)),
    certified: view.getInt32(ptr + 16, true) !== 0,
    thetaHat: view.getFloat64(ptr + 24, true),
    sigmaB2: view.getFloat64(ptr + 32, true),
    sigmaW2: view.getFloat64(ptr + 40, true),
    rhoHat: view.getFloat64(ptr + 48, true),
    nEffWithin: view.getFloat64(ptr + 56, true),
    saturationCeiling: view.getFloat64(ptr + 64, true),
    logEvidence: view.getFloat64(ptr + 72, true),
    logThreshold: view.getFloat64(ptr + 80, true),
  };
}

function readCsResult(view, ptr) {
  return {
    n: Number(view.getBigUint64(ptr + 0, true)),
    mean: view.getFloat64(ptr + 8, true),
    lower: view.getFloat64(ptr + 16, true),
    upper: view.getFloat64(ptr + 24, true),
    width: view.getFloat64(ptr + 32, true),
  };
}

function readPpiResult(view, ptr) {
  return {
    nTotal: Number(view.getBigUint64(ptr + 0, true)),
    nAudited: Number(view.getBigUint64(ptr + 8, true)),
    meanEstimate: view.getFloat64(ptr + 16, true),
    lower: view.getFloat64(ptr + 24, true),
    upper: view.getFloat64(ptr + 32, true),
    width: view.getFloat64(ptr + 40, true),
    lambdaHat: view.getFloat64(ptr + 48, true),
    varianceReduction: view.getFloat64(ptr + 56, true),
  };
}

function readPanelSummary(view, ptr) {
  return {
    kJudges: view.getUint32(ptr + 0, true),
    nItems: view.getUint32(ptr + 4, true),
    panelAccuracy: view.getFloat64(ptr + 8, true),
    bestSingleAccuracy: view.getFloat64(ptr + 16, true),
    bestSingleIndex: view.getUint32(ptr + 24, true),
    panelLift: view.getFloat64(ptr + 32, true),
    meanPhi: view.getFloat64(ptr + 40, true),
    stdPhi: view.getFloat64(ptr + 48, true),
    nEffKish: view.getFloat64(ptr + 56, true),
    nEffEigen: view.getFloat64(ptr + 64, true),
    lambdaMax: view.getFloat64(ptr + 72, true),
    effectiveRank: view.getFloat64(ptr + 80, true),
    independenceRatio: view.getFloat64(ptr + 88, true),
    asymptoticCeiling: view.getFloat64(ptr + 96, true),
    krippendorffAlpha: view.getFloat64(ptr + 104, true),
    unanimousRate: view.getFloat64(ptr + 112, true),
    unanimousAccuracy: view.getFloat64(ptr + 120, true),
    allWrongRate: view.getFloat64(ptr + 128, true),
    condorcetPredictedAccuracy: view.getFloat64(ptr + 136, true),
    condorcetGap: view.getFloat64(ptr + 144, true),
    permutationPValue: view.getFloat64(ptr + 152, true),
    nEffCiLow: view.getFloat64(ptr + 160, true),
    nEffCiHigh: view.getFloat64(ptr + 168, true),
  };
}

/**
 * Instantiate `cloture-core` WebAssembly from bytes, WebAssembly.Module, Response, or URL.
 * @param {BufferSource | WebAssembly.Module | Response | string | URL} source
 */
export async function loadCloture(source) {
  let instance;
  if (source instanceof WebAssembly.Module) {
    instance = await WebAssembly.instantiate(source, {});
  } else if (typeof source === "string" || source instanceof URL) {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to fetch cloture WASM (${resp.status}): ${source}`);
    const bytes = await resp.arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, {}));
  } else if (typeof Response !== "undefined" && source instanceof Response) {
    const bytes = await source.arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, {}));
  } else {
    ({ instance } = await WebAssembly.instantiate(source, {}));
  }

  const ex = instance.exports;
  const mem = () => ex.memory.buffer;
  const dv = () => new DataView(mem());

  function alloc(bytes) {
    const ptr = ex.cloture_alloc(bytes);
    if (!ptr) throw new Error(`cloture_alloc(${bytes}) failed`);
    return ptr;
  }

  function freeBuf(ptr, bytes) {
    if (ptr) ex.cloture_dealloc(ptr, bytes);
  }

  function withF64(arr, fn) {
    const n = arr.length;
    const bytes = Math.max(8, n * 8);
    const ptr = alloc(bytes);
    try {
      new Float64Array(mem(), ptr, n).set(arr);
      return fn(ptr, n);
    } finally {
      freeBuf(ptr, bytes);
    }
  }

  class AscdStopper {
    constructor(k, delta = 0.05, maxBudget = 0) {
      this._k = k;
      this._ptr = ex.cloture_ascd_new(k, delta, BigInt(maxBudget || 0));
      if (!this._ptr) throw new Error(`Invalid AscdStopper(k=${k}, delta=${delta})`);
      this._out = alloc(72);
    }
    update(probs) {
      if (probs.length !== this._k) {
        throw new Error(`Expected ${this._k} probabilities, got ${probs.length}`);
      }
      return withF64(probs, (pPtr, n) => {
        ex.cloture_ascd_update_into(this._ptr, pPtr, n, this._out);
        return readModeResult(dv(), this._out);
      });
    }
    reset() {
      ex.cloture_ascd_reset(this._ptr);
    }
    free() {
      if (this._ptr) {
        ex.cloture_ascd_free(this._ptr);
        this._ptr = 0;
      }
      if (this._out) {
        freeBuf(this._out, 72);
        this._out = 0;
      }
    }
  }

  class CiteStopper {
    constructor(delta = 0.05, maxBudget = 0) {
      this._ptr = ex.cloture_cite_new(delta, BigInt(maxBudget || 0));
      if (!this._ptr) throw new Error(`Invalid CiteStopper(delta=${delta})`);
      this._out = alloc(72);
    }
    update(labelId) {
      ex.cloture_cite_update_into(this._ptr, BigInt(labelId), this._out);
      return readModeResult(dv(), this._out);
    }
    reset() {
      ex.cloture_cite_reset(this._ptr);
    }
    free() {
      if (this._ptr) {
        ex.cloture_cite_free(this._ptr);
        this._ptr = 0;
      }
      if (this._out) {
        freeBuf(this._out, 72);
        this._out = 0;
      }
    }
  }

  class PopulationCert {
    constructor(delta = 0.05) {
      this._ptr = ex.cloture_pop_new(delta);
      if (!this._ptr) throw new Error(`Invalid PopulationCert(delta=${delta})`);
      this._out = alloc(88);
    }
    addUnit(scores) {
      return withF64(scores, (pPtr, n) => {
        ex.cloture_pop_add_unit_into(this._ptr, pPtr, n, this._out);
        return readPopResult(dv(), this._out);
      });
    }
    optimalRollouts(unitSetupCost = 1.0, perRolloutCost = 0.25) {
      return Number(ex.cloture_pop_optimal_rollouts(this._ptr, unitSetupCost, perRolloutCost));
    }
    free() {
      if (this._ptr) {
        ex.cloture_pop_free(this._ptr);
        this._ptr = 0;
      }
      if (this._out) {
        freeBuf(this._out, 88);
        this._out = 0;
      }
    }
  }

  class EmpiricalBernsteinCS {
    constructor(alpha = 0.05, minVal = 0.0, maxVal = 1.0) {
      this._ptr = ex.cloture_ebcs_new(alpha, minVal, maxVal);
      if (!this._ptr) throw new Error("Invalid EmpiricalBernsteinCS parameters");
      this._out = alloc(40);
    }
    update(x) {
      ex.cloture_ebcs_update_into(this._ptr, x, this._out);
      return readCsResult(dv(), this._out);
    }
    free() {
      if (this._ptr) { ex.cloture_ebcs_free(this._ptr); this._ptr = 0; }
      if (this._out) { freeBuf(this._out, 40); this._out = 0; }
    }
  }

  class BettingCS {
    constructor(alpha = 0.05, minVal = 0.0, maxVal = 1.0, gridPoints = 201) {
      this._ptr = ex.cloture_betcs_new(alpha, minVal, maxVal, gridPoints);
      if (!this._ptr) throw new Error("Invalid BettingCS parameters");
      this._out = alloc(40);
    }
    update(x) {
      ex.cloture_betcs_update_into(this._ptr, x, this._out);
      return readCsResult(dv(), this._out);
    }
    free() {
      if (this._ptr) { ex.cloture_betcs_free(this._ptr); this._ptr = 0; }
      if (this._out) { freeBuf(this._out, 40); this._out = 0; }
    }
  }

  class ActivePPI {
    constructor(alpha = 0.05, targetAuditRate = 0.10, minAuditProb = 0.05) {
      this._ptr = ex.cloture_ppi_new(alpha, targetAuditRate, minAuditProb);
      if (!this._ptr) throw new Error("Invalid ActivePPI parameters");
      this._out = alloc(64);
    }
    suggestProb(judgePred, uncertainty = 0.5) {
      return ex.cloture_ppi_suggest_prob(this._ptr, judgePred, uncertainty);
    }
    observe(judgePred, audited, goldLabel, auditProb) {
      ex.cloture_ppi_observe_into(this._ptr, judgePred, audited ? 1 : 0, goldLabel, auditProb, this._out);
      return readPpiResult(dv(), this._out);
    }
    free() {
      if (this._ptr) { ex.cloture_ppi_free(this._ptr); this._ptr = 0; }
      if (this._out) { freeBuf(this._out, 64); this._out = 0; }
    }
  }

  class OnlineConformal {
    constructor(targetAlpha = 0.10) {
      this._ptr = ex.cloture_conformal_new(targetAlpha);
      if (!this._ptr) throw new Error("Invalid OnlineConformal targetAlpha");
    }
    updatePvalue(p) {
      return ex.cloture_conformal_update_pvalue(this._ptr, p);
    }
    free() {
      if (this._ptr) { ex.cloture_conformal_free(this._ptr); this._ptr = 0; }
    }
  }

  class EvidenceSketch {
    constructor(ptr = 0) {
      this._ptr = ptr || ex.cloture_sketch_new();
      if (!this._ptr) throw new Error("Failed to create EvidenceSketch");
    }
    observe(score, betIncrement = 0.0) {
      ex.cloture_sketch_observe(this._ptr, score, betIncrement);
    }
    merge(other) {
      if (ex.cloture_sketch_merge(this._ptr, other._ptr) !== 0) throw new Error("Sketch merge failed");
    }
    subtract(other) {
      if (ex.cloture_sketch_subtract(this._ptr, other._ptr) !== 0) throw new Error("Sketch subtract failed");
    }
    mean() { return ex.cloture_sketch_mean(this._ptr); }
    quantile(q) { return ex.cloture_sketch_quantile(this._ptr, q); }
    eValue() { return ex.cloture_sketch_e_value(this._ptr); }
    toBytes() {
      const len = Number(ex.cloture_sketch_serialized_len(this._ptr));
      const ptr = alloc(len);
      try {
        const written = Number(ex.cloture_sketch_to_bytes(this._ptr, ptr, len));
        return new Uint8Array(mem().slice(ptr, ptr + written));
      } finally {
        freeBuf(ptr, len);
      }
    }
    static fromBytes(bytes) {
      const ptr = alloc(bytes.length);
      try {
        new Uint8Array(mem(), ptr, bytes.length).set(bytes);
        const sPtr = ex.cloture_sketch_from_bytes(ptr, bytes.length);
        if (!sPtr) throw new Error("Corrupt or invalid EvidenceSketch wire bytes");
        return new EvidenceSketch(sPtr);
      } finally {
        freeBuf(ptr, bytes.length);
      }
    }
    free() {
      if (this._ptr) { ex.cloture_sketch_free(this._ptr); this._ptr = 0; }
    }
  }

  function auditPanel(votes, gold, opts = {}) {
    const nItems = gold.length;
    if (nItems === 0 || votes.length !== nItems) {
      throw new Error(`votes length (${votes.length}) must match non-empty gold length (${nItems})`);
    }
    const kJudges = votes[0].length;
    if (kJudges === 0) throw new Error("Panel must have at least 1 judge");
    const flat = new Uint32Array(nItems * kJudges);
    for (let i = 0; i < nItems; i++) {
      if (votes[i].length !== kJudges) throw new Error(`Row ${i} has ${votes[i].length} judges, expected ${kJudges}`);
      for (let j = 0; j < kJudges; j++) flat[i * kJudges + j] = votes[i][j] >>> 0;
    }
    const goldArr = new Uint32Array(nItems);
    for (let i = 0; i < nItems; i++) goldArr[i] = gold[i] >>> 0;

    const vBytes = flat.byteLength;
    const gBytes = goldArr.byteLength;
    const sumBytes = 176;
    const phiBytes = kJudges * kJudges * 8;
    const kBytes = kJudges * 8;

    const vPtr = alloc(vBytes);
    const gPtr = alloc(gBytes);
    const sPtr = alloc(sumBytes);
    const phiPtr = alloc(phiBytes);
    const accPtr = alloc(kBytes);
    const looAccPtr = alloc(kBytes);
    const looNeffPtr = alloc(kBytes);
    try {
      new Uint32Array(mem(), vPtr, flat.length).set(flat);
      new Uint32Array(mem(), gPtr, goldArr.length).set(goldArr);
      const rc = ex.cloture_panel_audit(
        vPtr,
        gPtr,
        0,
        nItems,
        kJudges,
        opts.nCondorcetSims ?? 1000,
        opts.nPerms ?? 1000,
        opts.nBoot ?? 500,
        BigInt(opts.seed ?? 42),
        sPtr,
        phiPtr,
        accPtr,
        looAccPtr,
        looNeffPtr
      );
      if (rc !== 0) throw new Error(`cloture_panel_audit failed with code ${rc}`);
      const summary = readPanelSummary(dv(), sPtr);
      const phiFlat = Array.from(new Float64Array(mem(), phiPtr, kJudges * kJudges));
      const phiMatrix = [];
      for (let r = 0; r < kJudges; r++) phiMatrix.push(phiFlat.slice(r * kJudges, (r + 1) * kJudges));
      return {
        ...summary,
        phiMatrix,
        judgeAccuracies: Array.from(new Float64Array(mem(), accPtr, kJudges)),
        looDeltaAcc: Array.from(new Float64Array(mem(), looAccPtr, kJudges)),
        looDeltaNeff: Array.from(new Float64Array(mem(), looNeffPtr, kJudges)),
      };
    } finally {
      freeBuf(vPtr, vBytes);
      freeBuf(gPtr, gBytes);
      freeBuf(sPtr, sumBytes);
      freeBuf(phiPtr, phiBytes);
      freeBuf(accPtr, kBytes);
      freeBuf(looAccPtr, kBytes);
      freeBuf(looNeffPtr, kBytes);
    }
  }

  /**
   * Drop-in async multi-agent consensus with AbortController cancellation.
   * @param {(callIndex: number, signal: AbortSignal) => Promise<any>} spawnFn
   */
  async function consensus(spawnFn, opts = {}) {
    const delta = opts.delta ?? 0.05;
    const maxCalls = opts.maxCalls ?? 32;
    const concurrency = opts.concurrency ?? 4;
    const extractAnswer = opts.extractAnswer ?? ((x) => x);
    const stopper = new CiteStopper(delta, opts.deterministicLock === false ? 0 : maxCalls);
    const keyToId = new Map();
    const idToVal = new Map();
    let nextId = 0;
    let launched = 0;
    let cancelled = 0;
    let lastDec = null;
    const rawOutputs = [];
    const active = new Map();

    try {
      const launch = () => {
        while (launched < maxCalls && active.size < concurrency) {
          const idx = launched++;
          const ac = new AbortController();
          const p = Promise.resolve()
            .then(() => spawnFn(idx, ac.signal))
            .then((val) => ({ idx, val, ok: true }))
            .catch((err) => ({ idx, err, ok: false }));
          active.set(idx, { ac, p });
        }
      };
      launch();
      while (active.size > 0) {
        const res = await Promise.race(Array.from(active.values(), (v) => v.p));
        active.delete(res.idx);
        if (!res.ok) throw res.err;
        rawOutputs.push(res.val);
        const ans = extractAnswer(res.val);
        const key = typeof ans === "string" || typeof ans === "number" ? ans : JSON.stringify(ans);
        if (!keyToId.has(key)) {
          keyToId.set(key, nextId);
          idToVal.set(nextId, ans);
          nextId++;
        }
        lastDec = stopper.update(keyToId.get(key));
        if (lastDec.certified) break;
        launch();
      }
      for (const { ac } of active.values()) {
        ac.abort();
        cancelled++;
      }
      return {
        certified: lastDec?.certified ?? false,
        winner: lastDec?.winner != null ? idToVal.get(lastDec.winner) : null,
        leader: lastDec?.leader != null ? idToVal.get(lastDec.leader) : null,
        nSamples: lastDec?.nSamples ?? 0,
        nCancelled: cancelled,
        logEvidence: lastDec?.logEvidence ?? 0,
        logThreshold: lastDec?.logThreshold ?? 0,
        rawOutputs,
      };
    } finally {
      stopper.free();
    }
  }

  return {
    AscdStopper,
    CiteStopper,
    PopulationCert,
    EmpiricalBernsteinCS,
    BettingCS,
    ActivePPI,
    OnlineConformal,
    EvidenceSketch,
    auditPanel,
    consensus,
  };
}
