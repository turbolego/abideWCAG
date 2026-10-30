import type { CheckPhase, Rule, Thresholds, Usage, Verdict } from "@coldtea/abide-schema";
import type { FileDiff } from "./git.js";
import { checkWithModel, isModelRule, type ModelRule } from "./jev.js";
import { checkWithLaya, isLayaRule, type LayaModelRule } from "./layaWcag.js";
import { ruleAppliesTo } from "./scope.js";

export type CheckRequest = {
  phase: CheckPhase;
  /** The change, one entry per touched file, paths repo-relative. */
  fileDiffs: readonly FileDiff[];
  task?: string;
  rules: readonly Rule[];
  thresholds: Thresholds;
  timeoutMs: number;
  /** Transient gateway failures to retry. Hooks leave it at zero. */
  retries?: number;
};

export type CheckOutcome = {
  verdicts: Verdict[];
  modelRules: ModelRule[];
  /** How many model calls the check took: one per distinct set of in-scope files. */
  calls: number;
  usage: Usage;
  modelLatencyMs: number;
};

/** Only model rules run. A lint-shaped rule is the linter's to enforce, and abide only reports it. */
const runsInPhase = (rule: Rule, phase: CheckPhase): boolean =>
  rule.status === "active" && rule.check.type === "model" && rule.when === phase;

/** Rules that are active, belong to this phase, and apply to at least one touched file. */
export const selectRules = (
  rules: readonly Rule[],
  phase: CheckPhase,
  files: readonly string[],
): Rule[] =>
  rules.filter((rule) => runsInPhase(rule, phase) && files.some((f) => ruleAppliesTo(rule, f)));

const renderFiles = (fileDiffs: readonly FileDiff[]): string =>
  fileDiffs.map((f) => `--- a/${f.file}\n+++ b/${f.file}\n${f.text}`).join("\n");

/** Rules grouped by the files each one applies to, so no rule ever sees a file outside its scope. */
export const groupByScope = (
  rules: readonly (ModelRule | LayaModelRule)[],
  fileDiffs: readonly FileDiff[],
): { rules: (ModelRule | LayaModelRule)[]; fileDiffs: FileDiff[] }[] => {
  const groups = new Map<string, { rules: (ModelRule | LayaModelRule)[]; fileDiffs: FileDiff[] }>();
  for (const rule of rules) {
    const inScope = fileDiffs.filter((f) => ruleAppliesTo(rule as Rule, f.file));
    if (inScope.length === 0) continue;
    const key = inScope.map((f) => f.file).join("\n");
    const group = groups.get(key) ?? { rules: [], fileDiffs: inScope };
    group.rules.push(rule);
    groups.set(key, group);
  }
  return [...groups.values()];
};

const addUsage = (a: Usage, b: Usage): Usage => ({
  inputTokens: (a.inputTokens ?? 0) + (b.inputTokens ?? 0),
  outputTokens: (a.outputTokens ?? 0) + (b.outputTokens ?? 0),
  costUsd: (a.costUsd ?? 0) + (b.costUsd ?? 0),
});

/** One verdict per rule: the loudest. */
export const loudestVerdicts = (verdicts: readonly Verdict[]): Verdict[] => {
  const best = new Map<string, Verdict>();
  for (const v of verdicts) {
    const have = best.get(v.ruleId);
    if (have === undefined || v.probability > have.probability) best.set(v.ruleId, v);
  }
  return [...best.values()];
};

export const mergeOutcomes = (outcomes: readonly CheckOutcome[]): CheckOutcome => {
  const modelRules: ModelRule[] = [];
  for (const o of outcomes) {
    for (const rule of o.modelRules) if (!modelRules.includes(rule)) modelRules.push(rule);
  }
  return {
    verdicts: loudestVerdicts(outcomes.flatMap((o) => o.verdicts)),
    modelRules,
    calls: outcomes.reduce((sum, o) => sum + o.calls, 0),
    usage: outcomes.reduce<Usage>((sum, o) => addUsage(sum, o.usage), {}),
    modelLatencyMs: Math.max(0, ...outcomes.map((o) => o.modelLatencyMs)),
  };
};

export const runCheck = async (request: CheckRequest): Promise<CheckOutcome> => {
  const files = request.fileDiffs.map((f) => f.file);
  const allRules = selectRules(request.rules, request.phase, files);
  const jevRules = allRules.filter(isModelRule) as ModelRule[];
  const layaRules = allRules.filter(isLayaRule) as LayaModelRule[];
  const modelRules: ModelRule[] = [...jevRules, ...(layaRules as unknown as ModelRule[])];

  const jevGroups = groupByScope(jevRules, request.fileDiffs);
  const layaGroups = groupByScope(layaRules as (ModelRule | LayaModelRule)[], request.fileDiffs);

  if (jevGroups.length === 0 && layaGroups.length === 0) {
    return { verdicts: [], modelRules, calls: 0, usage: {}, modelLatencyMs: 0 };
  }

  const started = performance.now();

  const [jevResults, layaResults] = await Promise.all([
    // Jev (cloud) backend
    Promise.all(
      jevGroups.map(async (group) => {
        const only = group.fileDiffs.length === 1 ? group.fileDiffs[0] : undefined;
        const result = await checkWithModel(
          group.rules as readonly ModelRule[],
          {
            ...(request.task === undefined ? {} : { task: request.task }),
            ...(request.phase === "edit" && only !== undefined
              ? { file: only.file, diff: only.text }
              : { files: group.fileDiffs.map((f) => f.file), diff: renderFiles(group.fileDiffs) }),
          },
          request.thresholds,
          request.timeoutMs,
          request.retries ?? 0,
        );
        return only === undefined
          ? result
          : { ...result, verdicts: result.verdicts.map((v) => ({ ...v, file: only.file })) };
      }),
    ),
    // LayaWCAG (local ONNX) backend
    layaGroups.length === 0
      ? Promise.resolve([])
      : (async () => {
          const out: { verdicts: Verdict[]; latencyMs: number }[] = [];
          for (const group of layaGroups) {
            const only = group.fileDiffs.length === 1 ? group.fileDiffs[0] : undefined;
            const state = {
              ...(request.task === undefined ? {} : { task: request.task }),
              ...(request.phase === "edit" && only !== undefined
                ? { file: only.file, diff: only.text }
                : { files: group.fileDiffs.map((f) => f.file), diff: renderFiles(group.fileDiffs) }),
            };
            const result = await checkWithLaya(group.rules as readonly LayaModelRule[], state, request.thresholds);
            out.push({
              verdicts:
                only === undefined
                  ? result.verdicts
                  : result.verdicts.map((v) => ({ ...v, file: only.file })),
              latencyMs: result.latencyMs,
            });
          }
          return out;
        })(),
  ]);

  const jevVerdicts = jevResults.flatMap((r) => r.verdicts);
  const jevUsage = jevResults.reduce<Usage>((sum, r) => addUsage(sum, r.usage), {});
  const layaVerdicts = layaResults.flatMap((r) => r.verdicts);

  return {
    verdicts: [...jevVerdicts, ...layaVerdicts],
    modelRules,
    calls: jevGroups.length + layaGroups.length,
    usage: jevUsage,
    modelLatencyMs: Math.round(performance.now() - started),
  };
};