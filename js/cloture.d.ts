export interface ModeDecision {
  certified: boolean;
  deterministicLock: boolean;
  winner: number | null;
  leader: number | null;
  nSamples: number;
  logEvidence: number;
  logThreshold: number;
  marginEstimate: number;
  scope: "realized_unit" | "population" | "deterministic_lock";
}

export interface PopDecision {
  nUnits: number;
  totalRollouts: number;
  certified: boolean;
  thetaHat: number;
  sigmaB2: number;
  sigmaW2: number;
  rhoHat: number;
  nEffWithin: number;
  saturationCeiling: number;
  logEvidence: number;
  logThreshold: number;
}

export interface CsResult {
  n: number;
  mean: number;
  lower: number;
  upper: number;
  width: number;
}

export interface PpiResult {
  nTotal: number;
  nAudited: number;
  meanEstimate: number;
  lower: number;
  upper: number;
  width: number;
  lambdaHat: number;
  varianceReduction: number;
}

export interface PanelReport {
  kJudges: number;
  nItems: number;
  panelAccuracy: number;
  bestSingleAccuracy: number;
  bestSingleIndex: number;
  panelLift: number;
  meanPhi: number;
  stdPhi: number;
  nEffKish: number;
  nEffEigen: number;
  lambdaMax: number;
  effectiveRank: number;
  independenceRatio: number;
  asymptoticCeiling: number;
  krippendorffAlpha: number;
  unanimousRate: number;
  unanimousAccuracy: number;
  allWrongRate: number;
  condorcetPredictedAccuracy: number;
  condorcetGap: number;
  permutationPValue: number;
  nEffCiLow: number;
  nEffCiHigh: number;
  phiMatrix: number[][];
  judgeAccuracies: number[];
  looDeltaAcc: number[];
  looDeltaNeff: number[];
}

export interface ClotureModule {
  AscdStopper: new (k: number, delta?: number, maxBudget?: number) => {
    update(probs: number[]): ModeDecision;
    reset(): void;
    free(): void;
  };
  CiteStopper: new (delta?: number, maxBudget?: number) => {
    update(labelId: number | bigint): ModeDecision;
    reset(): void;
    free(): void;
  };
  PopulationCert: new (delta?: number) => {
    addUnit(scores: number[]): PopDecision;
    optimalRollouts(unitSetupCost?: number, perRolloutCost?: number): number;
    free(): void;
  };
  EmpiricalBernsteinCS: new (alpha?: number, minVal?: number, maxVal?: number) => {
    update(x: number): CsResult;
    free(): void;
  };
  BettingCS: new (alpha?: number, minVal?: number, maxVal?: number, gridPoints?: number) => {
    update(x: number): CsResult;
    free(): void;
  };
  ActivePPI: new (alpha?: number, targetAuditRate?: number, minAuditProb?: number) => {
    suggestProb(judgePred: number, uncertainty?: number): number;
    observe(judgePred: number, audited: boolean, goldLabel: number, auditProb: number): PpiResult;
    free(): void;
  };
  OnlineConformal: new (targetAlpha?: number) => {
    updatePvalue(p: number): number;
    free(): void;
  };
  EvidenceSketch: {
    new (): {
      observe(score: number, betIncrement?: number): void;
      merge(other: unknown): void;
      subtract(other: unknown): void;
      mean(): number;
      quantile(q: number): number;
      eValue(): number;
      toBytes(): Uint8Array;
      free(): void;
    };
    fromBytes(bytes: Uint8Array): unknown;
  };
  auditPanel(
    votes: number[][],
    gold: number[],
    opts?: { nCondorcetSims?: number; nPerms?: number; nBoot?: number; seed?: number }
  ): PanelReport;
  consensus<T>(
    spawnFn: (callIndex: number, signal: AbortSignal) => Promise<T>,
    opts?: {
      delta?: number;
      maxCalls?: number;
      concurrency?: number;
      deterministicLock?: boolean;
      extractAnswer?: (val: T) => unknown;
    }
  ): Promise<{
    certified: boolean;
    winner: unknown;
    leader: unknown;
    nSamples: number;
    nCancelled: number;
    logEvidence: number;
    logThreshold: number;
    rawOutputs: T[];
  }>;
}

export function loadCloture(
  source: BufferSource | WebAssembly.Module | Response | string | URL
): Promise<ClotureModule>;
