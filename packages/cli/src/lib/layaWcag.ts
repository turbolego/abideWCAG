import {
  AbideError,
  type Rule,
  type Thresholds,
  type Usage,
  type Verdict,
} from "@coldtea/abide-schema";
import type { InferenceSession, Tensor } from "onnxruntime-common";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { bandFor } from "./band.js";
import type { CheckState } from "./jev.js";
import {
  LAYA_WCAG_MODEL_DIR_ENV,
  LAYA_WCAG_MODEL_DIR_DEFAULT,
  LAYA_WCAG_MAX_SEQ_LEN,
  LAYA_WCAG_LOAD_TIMEOUT_MS,
} from "./constants.js";

/**
 * LayaWCAG backend — runs the local ONNX model to score WCAG 2.2 violations.
 *
 * The model is a frozen ModernBERT-large encoder with a 2-layer head producing
 * 63 logits, one per WCAG 2.2 success criterion. Each logit is the signed
 * evidence for that criterion; sigmoid gives P(violation).
 *
 * Model bundle (in a directory):
 *   laya-wcag.onnx         — ONNX graph (307 KB)
 *   laya-wcag.onnx.data    — externalized int8 weights (~395 MB, assembled from chunks)
 *   tokenizer.json         — HuggingFace BPE tokenizer
 *   wcag_criteria.json     — 63 criteria in output order
 */

/** Canonical 63 WCAG 2.2 criteria, in model-output order. */
export const LAYA_WCAG_CRITERIA_COUNT = 63;

let session: InferenceSession | null = null;
let tokenizer: any = null;
let loadPromise: Promise<void> | null = null;
let criteriaLabels: string[] = [];

/** Resolve the model bundle directory from env or default. */
export const layaModelDir = (): string =>
  process.env[LAYA_WCAG_MODEL_DIR_ENV] ?? LAYA_WCAG_MODEL_DIR_DEFAULT;

export async function initLayaWcag(): Promise<void> {
  if (session && tokenizer) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    // Must import onnxruntime-node first to register the native backend.
    // Use it directly for InferenceSession — vitest's ESM interop of the CJS
    // module can drop registerBackend's side effect on onnxruntime-common.
    const ort = await import("onnxruntime-node");
    const { AutoTokenizer } = await import("@huggingface/transformers");

    const modelDir = layaModelDir();
    const modelPath = resolve(modelDir, "laya-wcag.onnx");
    const criteriaPath = resolve(modelDir, "wcag_criteria.json");

    // Load criteria labels
    const criteriaRaw = JSON.parse(readFileSync(criteriaPath, "utf8")) as { criterion: string }[];
    criteriaLabels = criteriaRaw.map((c) => c.criterion);
    if (criteriaLabels.length !== LAYA_WCAG_CRITERIA_COUNT) {
      throw new AbideError(
        "RUBRIC_INVALID",
        `LayaWCAG: criteria file has ${criteriaLabels.length} entries, expected ${LAYA_WCAG_CRITERIA_COUNT}`,
      );
    }

    // Load ONNX model with timeout
    session = (await Promise.race([
      ort.InferenceSession.create(modelPath),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new AbideError("CHECK_TIMEOUT", "LayaWCAG: ONNX model load timed out")),
          LAYA_WCAG_LOAD_TIMEOUT_MS,
        ),
      ),
    ])) as unknown as InferenceSession;

    // Load tokenizer
    tokenizer = await AutoTokenizer.from_pretrained(modelDir);
  })();

  try {
    await loadPromise;
  } catch (error) {
    loadPromise = null;
    session = null;
    tokenizer = null;
    throw error;
  }
}

export function isLayaInitialized(): boolean {
  return session !== null && tokenizer !== null;
}

function assertReady(): void {
  if (!session || !tokenizer) {
    throw new AbideError(
      "CHECK_FAILED",
      "LayaWCAG: not initialized — call initLayaWcag() first",
    );
  }
}

/** Run the model on a text blob → 63 P(violation) values in criteria order. */
export async function runLayaWcag(text: string): Promise<Float32Array> {
  assertReady();
  const enc = await tokenizer(text, {
    max_length: LAYA_WCAG_MAX_SEQ_LEN,
    truncation: true,
    padding: "max_length",
  });
  const inputIds: BigInt64Array = enc.input_ids.data;
  const attentionMask: BigInt64Array = enc.attention_mask.data;
  const ort = await import("onnxruntime-node");
  const { Tensor } = ort;
  const results = await session!.run({
    input_ids: new Tensor("int64", inputIds, [1, LAYA_WCAG_MAX_SEQ_LEN]),
    attention_mask: new Tensor("int64", attentionMask, [1, LAYA_WCAG_MAX_SEQ_LEN]),
  });
  const rawLogs = results.criterion_scores?.data as Float32Array | undefined;
  if (!rawLogs || rawLogs.length !== LAYA_WCAG_CRITERIA_COUNT) {
    throw new AbideError(
      "CHECK_FAILED",
      `LayaWCAG: output shape mismatch (got ${rawLogs?.length ?? 0}, expected ${LAYA_WCAG_CRITERIA_COUNT})`,
    );
  }
  const logits = rawLogs!;
  const n = logits.length;
  const probs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = logits[i] as number;
    probs[i] = 1 / (1 + Math.exp(-v));
  }
  return probs;
}

/** Look up the model output index for a WCAG criterion like "1.4.3". */
export const criterionIndex = (criterion: string): number => criteriaLabels.indexOf(criterion);

/** A model rule backed by the local LayaWCAG ONNX model. */
export type LayaModelRule = {
  id: string;
  check: {
    type: "model";
    backend: "laya";
    criterion: string;
    question: import("@coldtea/abide-schema").Question;
  };
  when?: import("@coldtea/abide-schema").RuleWhen;
};

/** True when the rule targets the local LayaWCAG backend. */
export const isLayaRule = (rule: Rule): boolean =>
  rule.check.type === "model" && (rule.check as { backend?: string }).backend === "laya";

/** Clamp to [0,1]. */
const clamp01 = (n: number): number => Math.min(1, Math.max(0, n));

/**
 * LayaWCAG check — the local-model counterpart of checkWithModel().
 * Runs the ONNX model once on the diff text, then answers every rule with the
 * probability of its own criterion.
 */
export const checkWithLaya = async (
  rules: readonly LayaModelRule[],
  state: CheckState,
  thresholds: Thresholds,
): Promise<{ verdicts: Verdict[]; usage: Usage; latencyMs: number }> => {
  if (rules.length === 0) return { verdicts: [], usage: {}, latencyMs: 0 };
  await initLayaWcag();

  const started = performance.now();
  const probs = await runLayaWcag(state.diff);
  const latencyMs = Math.round(performance.now() - started);

  const verdicts: Verdict[] = [];
  for (const rule of rules) {
    const idx = criterionIndex(rule.check.criterion);
    if (idx === -1) continue; // criterion not in the model's 63
    // The Laya model outputs sigmoid P(violation) directly per criterion —
    // it is the violation probability, not an answer to a semantic question.
    const violationProb = clamp01(probs[idx] as number);
    verdicts.push({
      ruleId: rule.id,
      probability: violationProb,
      band: bandFor(violationProb, thresholds),
      ...(state.file === undefined ? {} : { file: state.file }),
    });
  }

  // Local inference — no token usage cost.
  return { verdicts, usage: {}, latencyMs };
};