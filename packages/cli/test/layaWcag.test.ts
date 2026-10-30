import { describe, expect, it } from "vitest";
import type { Rule } from "@coldtea/abide-schema";
import { checkWithLaya, isLayaRule, LAYA_WCAG_CRITERIA_COUNT } from "../src/lib/layaWcag.js";

const layaRule = (id: string, criterion: string): Rule =>
  ({
    id,
    text: id,
    source: { path: "WCAG.md" },
    status: "active",
    when: "edit",
    check: {
      type: "model",
      backend: "laya",
      criterion,
      question: { type: "boolean", instructions: "Does this diff violate the criterion?" },
    },
  }) as unknown as Rule;

describe("LayaWCAG backend", () => {
  it("isLayaRule selects only laya-backed model rules", () => {
    expect(isLayaRule(layaRule("a", "1.4.3"))).toBe(true);
    const jevRule = {
      id: "b",
      text: "b",
      source: { path: "AGENTS.md" },
      status: "active" as const,
      when: "edit" as const,
      check: {
        type: "model" as const,
        backend: "jev" as const,
        question: { type: "boolean" as const, instructions: "?" },
      },
    };
    expect(isLayaRule(jevRule as Rule)).toBe(false);
  });

  it("runLayaWcag scores a low-contrast + missing-alt snippet with real model output", async () => {
    const { initLayaWcag, runLayaWcag, criterionIndex } = await import("../src/lib/layaWcag.js");
    await initLayaWcag();
    const html =
      '<div style="color: #ccc; background: #fff;">Low contrast text here</div><img src="x.jpg">';
    const probs = await runLayaWcag(html);
    expect(probs.length).toBe(LAYA_WCAG_CRITERIA_COUNT);
    // The trained model should rank contrast (1.4.3) and non-text content (1.1.1) in its top outputs.
    const top = Array.from(probs)
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 10)
      .map((x) => x.i);
    expect(top).toContain(criterionIndex("1.4.3"));
    expect(top).toContain(criterionIndex("1.1.1"));
  }, 120_000);

  it("checkWithLaya returns one verdict per rule with the criterion's probability", async () => {
    const result = await checkWithLaya(
      [
        layaRule("contrast", "1.4.3"),
        layaRule("alt", "1.1.1"),
      ] as never,
      { diff: '<div style="color: #ccc; background: #fff;">Low contrast text here</div><img src="y.png">' },
      { act: 0.7, flag: 0.4 },
    );
    expect(result.verdicts.length).toBe(2);
    expect(result.verdicts[0]!.ruleId).toBe("contrast");
    // Model ranks 1.4.3 in its top outputs for this input.
    expect(result.verdicts[0]!.probability).toBeGreaterThan(0.1);
    expect(result.usage).toEqual({});
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
  }, 120_000);
});
