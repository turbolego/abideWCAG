// WCAG 2.2 violation detector runtime for the browser.
// Port of the Python runtime in notebooks/laya_wcag_training.ipynb: ModernBERT-large encoder
// (frozen) + a Linear(1024, 63) multi-label head trained with BCEWithLogitsLoss, exported to ONNX
// as input_ids/attention_mask -> criterion_scores [batch, 63] logits.
// The browser applies the sigmoid that Python's F.binary_cross_entropy_with_logits would apply
// at inference time (torch.sigmoid in the notebook's evaluation cell).

const CRITERIA_URL = "./wcag_criteria.json";

// Encode a single HTML snippet: [CLS] tokens [SEP], padded to max_len.
export function buildWcagSequence(tok, sp, text, maxLen = 512) {
  const enc = Array.from(tok.encode(text, { add_special_tokens: false }).ids);
  const ids = [sp.cls, ...enc.slice(0, maxLen - 2), sp.sep].slice(0, maxLen);
  return ids;
}

export function collateOne(seq, padId) {
  const L = seq.length;
  const ids = new BigInt64Array(L);
  const att = new BigInt64Array(L);
  for (let j = 0; j < L; j++) { ids[j] = BigInt(seq[j]); att[j] = 1n; }
  return { ids, att, L };
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const r4 = (x) => Math.round(x * 1e4) / 1e4;

export class WcagLaya {
  constructor(ort, session, tokenizer, criteria) {
    this.ort = ort;
    this.session = session;
    this.tok = tokenizer;
    this.criteria = criteria;
    this.sp = specialIdsWcag(tokenizer);
  }

  // Evaluate one HTML snippet against all 63 WCAG 2.2 criteria.
  // Returns { criteria: [{criterion, level, description, violation, probability}], violations: [...], latency_ms }
  async systemOne(state, options = {}) {
    const maxLen = options.maxLen ?? 512;
    const threshold = options.threshold ?? 0.5;
    const text = typeof state === "string" ? state : JSON.stringify(state);
    const seq = buildWcagSequence(this.tok, this.sp, text, maxLen);
    const b = collateOne(seq, this.sp.pad);
    const feeds = {
      input_ids: new this.ort.Tensor("int64", b.ids, [1, b.L]),
      attention_mask: new this.ort.Tensor("int64", b.att, [1, b.L]),
    };
    const t0 = performance.now();
    const out = await this.session.run(feeds);
    const ms = performance.now() - t0;
    const logits = out.criterion_scores.data;
    const results = [];
    const violations = [];
    this.criteria.forEach((c, i) => {
      const p = sigmoid(logits[i]);
      const violation = p >= threshold;
      const entry = {
        criterion: c.criterion,
        principle: c.principle,
        level: c.level,
        description: c.description,
        violation,
        probability: r4(p),
      };
      results.push(entry);
      if (violation) violations.push(entry);
    });
    results.sort((a, b) => b.probability - a.probability);
    violations.sort((a, b) => b.probability - a.probability);
    return { criteria: results, violations, latency_ms: ms };
  }

  // Evaluate many snippets. The graph supports batch > 1; pad to the longest sequence.
  async systemBatch(states, options = {}) {
    const maxLen = options.maxLen ?? 512;
    const threshold = options.threshold ?? 0.5;
    const seqs = states.map((s) => buildWcagSequence(this.tok, this.sp, typeof s === "string" ? s : JSON.stringify(s), maxLen));
    const n = seqs.length;
    const L = Math.max(...seqs.map((s) => s.length));
    const ids = new BigInt64Array(n * L).fill(BigInt(this.sp.pad));
    const att = new BigInt64Array(n * L);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < seqs[i].length; j++) {
        ids[i * L + j] = BigInt(seqs[i][j]);
        att[i * L + j] = 1n;
      }
    }
    const feeds = {
      input_ids: new this.ort.Tensor("int64", ids, [n, L]),
      attention_mask: new this.ort.Tensor("int64", att, [n, L]),
    };
    const t0 = performance.now();
    const out = await this.session.run(feeds);
    const ms = performance.now() - t0;
    const logits = out.criterion_scores.data;
    return states.map((s, r) => {
      const results = [];
      const violations = [];
      this.criteria.forEach((c, i) => {
        const p = sigmoid(logits[r * this.criteria.length + i]);
        const violation = p >= threshold;
        const entry = { criterion: c.criterion, principle: c.principle, level: c.level, description: c.description, violation, probability: r4(p) };
        results.push(entry);
        if (violation) violations.push(entry);
      });
      results.sort((a, b) => b.probability - a.probability);
      violations.sort((a, b) => b.probability - a.probability);
      return { state: s, criteria: results, violations, latency_ms: ms };
    });
  }
}

function specialIdsWcag(tok) {
  const g = (t) => tok.token_to_id(t);
  return { cls: g("[CLS]"), sep: g("[SEP]"), pad: g("[PAD]") };
}

export async function loadCriteria() {
  const r = await fetch(CRITERIA_URL);
  if (!r.ok) throw new Error(CRITERIA_URL + ": HTTP " + r.status);
  return r.json();
}
