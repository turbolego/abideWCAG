import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import type { Rule } from "@coldtea/abide-schema";
import { leakWarning } from "../src/commands/login.js";
import { Callout } from "../src/ui/components/Callout.js";
import { Picker } from "../src/ui/components/Picker.js";
import { Verdicts } from "../src/ui/components/Verdicts.js";
import { RuleTable } from "../src/ui/components/RuleTable.js";
import { ReportView } from "../src/ui/views/ReportView.js";
import { CheckView } from "../src/ui/views/CheckView.js";
import { CompileDoneView } from "../src/ui/views/CompileDoneView.js";
import { scopeLabel, truncate, meter } from "../src/ui/theme.js";

const rule = (over: Partial<Rule> & Pick<Rule, "id" | "check">): Rule => ({
  text: "t",
  source: { path: "AGENTS.md" },
  status: "active",
  ...over,
});

const rules: Rule[] = [
  rule({
    id: "no-interface",
    check: { type: "lint", how: "@typescript-eslint/consistent-type-definitions" },
    scope: ["**/*.ts", "**/*.tsx"],
  }),
  rule({
    id: "raw-error-to-user",
    when: "edit",
    check: { type: "model", backend: "jev", question: { type: "boolean", instructions: "?" } },
    calibration: {
      at: "x",
      hunks: 20,
      median: 0.04,
      min: 0,
      max: 0.9,
      fired: 1,
      verdict: "decisive",
    },
  }),
  rule({
    id: "imports-ordered-by-length",
    when: "edit",
    status: "weak",
    check: { type: "model", backend: "jev", question: { type: "boolean", instructions: "?" } },
    calibration: {
      at: "x",
      hunks: 15,
      median: 0.32,
      min: 0.1,
      max: 0.4,
      fired: 0,
      verdict: "weak",
    },
  }),
  rule({ id: "ask-first", check: { type: "unenforceable", reason: "about the conversation" } }),
];

describe("theme helpers", () => {
  it("reads globs as extensions and truncates with an ellipsis", () => {
    expect(scopeLabel(["**/*.ts", "**/*.tsx"])).toBe("ts, tsx");
    expect(scopeLabel(["apps/web/**"])).toBe("apps/web/**");
    expect(scopeLabel(undefined)).toBe("everywhere");
    expect(truncate("abcdefgh", 5)).toBe("abcd…");
    expect(truncate("abcdefgh", 5, "tail")).toBe("…efgh");
    expect(meter(0.5, 10)).toBe("█████░░░░░");
  });
});

describe("views", () => {
  it("draws every runnable rule with its status and hides the rest", () => {
    const { lastFrame } = render(<RuleTable rules={rules} runnableOnly />);
    const frame = lastFrame() ?? "";
    expect(frame).toContain("raw-error-to-user");
    expect(frame).toContain("weak");
    expect(frame).not.toContain("no-interface");
    expect(frame).not.toContain("ask-first");
  });

  it("folds clear verdicts into one line and shows the loud ones first", () => {
    const { lastFrame } = render(
      <Verdicts
        verdicts={[
          { ruleId: "quiet-a", probability: 0.05, band: "clear" },
          { ruleId: "loud", probability: 0.94, band: "act", answer: "loose-functions" },
          { ruleId: "quiet-b", probability: 0.1, band: "clear" },
          { ruleId: "maybe", probability: 0.62, band: "flag" },
        ]}
      />,
    );
    const frame = lastFrame() ?? "";
    expect(frame.indexOf("loud")).toBeLessThan(frame.indexOf("maybe"));
    expect(frame).toContain("loose-functions");
    expect(frame).toContain("2 rules clear");
    expect(frame).not.toContain("quiet-a");
  });

  it("report names the dead rules out loud", () => {
    const { lastFrame } = render(
      <ReportView
        data={{ root: "/r", rules, events: [], stats: new Map(), dead: [], problems: [] }}
      />,
    );
    const frame = lastFrame() ?? "";
    expect(frame).toContain("1 rule is switched off");
    expect(frame).toContain("imports-ordered-by-length");
    expect(frame).toContain("none recorded yet");
    expect(frame).toContain("For your linter");
    expect(frame).toContain("consistent-type");
    expect(frame).toContain("not configured");
    expect(frame).toContain("Not checked");
    expect(frame).toContain("ask-first");
  });

  it("check ends with the repair count", () => {
    const { lastFrame } = render(
      <CheckView
        data={{
          root: "/r",
          all: false,
          spendUsd: 0.0001,
          sections: [
            {
              phase: "edit",
              files: ["a.ts"],
              modelRules: 3,
              calls: 1,
              latencyMs: 400,
              verdicts: [{ ruleId: "x", probability: 0.9, band: "act" }],
            },
          ],
        }}
      />,
    );
    expect(lastFrame()).toContain("1 to repair");
  });
});

describe("Picker", () => {
  const until = async (ready: () => boolean): Promise<void> => {
    for (let i = 0; i < 50 && !ready(); i += 1) await new Promise((r) => setTimeout(r, 5));
  };
  const items = [
    { value: "a", label: "TypeSafe API key", hint: "from typesafe.ai" },
    { value: "b", label: "Vercel AI Gateway key" },
  ];

  it("moves with the arrows and keeps only the answer on screen after enter", async () => {
    const chosen: (string | undefined)[] = [];
    const { lastFrame, stdin } = render(
      <Picker title="Which key?" items={items} onDone={(item) => chosen.push(item?.value)} />,
    );
    expect(lastFrame()).toContain("→ TypeSafe API key");
    await new Promise((r) => setTimeout(r, 0));
    stdin.write("\u001B[B");
    await until(() => (lastFrame() ?? "").includes("→ Vercel AI Gateway key"));
    expect(lastFrame()).toContain("→ Vercel AI Gateway key");
    stdin.write("\r");
    await until(() => chosen.length > 0);
    expect(chosen).toEqual(["b"]);
    expect(lastFrame()).not.toContain("TypeSafe");
    expect(lastFrame()).toContain("Vercel AI Gateway key");
  });

  it("hands back nothing on escape", async () => {
    const chosen: (string | undefined)[] = [];
    const { stdin } = render(
      <Picker title="Which key?" items={items} onDone={(item) => chosen.push(item?.value)} />,
    );
    await new Promise((r) => setTimeout(r, 0));
    stdin.write("\u001B");
    await until(() => chosen.length > 0);
    expect(chosen).toEqual([undefined]);
  });
});

describe("leakWarning", () => {
  it("renders a warning for a repo that would commit .env.local, and nothing otherwise", () => {
    const root = mkdtempSync(path.join(tmpdir(), "abide-leak-"));
    spawnSync("git", ["init", "-q"], { cwd: root });
    const warning = leakWarning({ kind: "project", root });
    expect(warning).not.toBeNull();
    if (warning === null) return;
    const { lastFrame } = render(warning);
    expect(lastFrame()).toContain(".env.local is not ignored by git");
    expect(lastFrame()).toContain("Add .env.local to .gitignore");
    expect(leakWarning({ kind: "user" })).toBeNull();
  });
});

describe("Callout", () => {
  it("renders a bare string child without throwing", () => {
    const { lastFrame } = render(
      <Callout tone="warn" title="The turn finished but .abide/rubric.json was not written">
        {"Start Claude Code in this repo and ask it to compile the rubric."}
      </Callout>,
    );
    expect(lastFrame()).toContain("ask it to compile the rubric");
  });

  it("renders a multi-line string child on separate lines", () => {
    const { lastFrame } = render(
      <Callout tone="bad" title="Fix the rubric first">
        {"first issue\nsecond issue"}
      </Callout>,
    );
    expect(lastFrame()).toMatch(/first issue[^\n]*\n[^\n]*second issue/);
  });
});

describe("CompileDoneView", () => {
  const model = (id: string, status: Rule["status"]): Rule =>
    rule({
      id,
      status,
      when: "edit",
      check: { type: "model", backend: "jev", question: { type: "boolean", instructions: "?" } },
    });

  it("names the buckets and the follow-up commands when nothing is off", () => {
    const { lastFrame } = render(
      <CompileDoneView
        data={{
          which: "project",
          file: ".abide/rubric.json",
          rules: [model("a", "active"), model("b", "active")],
        }}
      />,
    );
    const frame = lastFrame() ?? "";
    expect(frame).toContain("Next: read the rubric");
    expect(frame).not.toContain("switched off");
    expect(frame).toContain("2 rules at .abide/rubric.json");
    expect(frame).toContain("teammates get the same rules");
    expect(frame).toContain("abide audit src/");
    expect(frame).not.toContain("abide tune");
  });

  it("lists weak, noisy and hand-disabled rules and points a global rubric at tune --global", () => {
    const { lastFrame } = render(
      <CompileDoneView
        data={{
          which: "global",
          file: "~/.abide/global.json",
          rules: [
            model("a", "weak"),
            model("b", "noisy"),
            model("c", "disabled"),
            model("d", "active"),
          ],
        }}
      />,
    );
    const frame = lastFrame() ?? "";
    expect(frame).toContain("3 rules are switched off");
    expect(frame).toContain("Weak: a");
    expect(frame).toContain("Noisy: b");
    expect(frame).toContain("Off by hand: c");
    expect(frame).toContain("abide tune --global");
    expect(frame).not.toContain("teammates get the same rules");
    expect(frame).not.toContain("abide audit");
  });
});
