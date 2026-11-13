#!/usr/bin/env node
/**
 * Judge generated Mermaid diagrams with Mermaid parser.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { JSDOM } from "jsdom";

// Mermaid expects to run in a browser and reaches for DOM globals while it is
// being imported. jsdom provides a minimal, real DOM so the parser can be used
// from Node without launching a headless browser.
const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  pretendToBeVisual: true,
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
for (const name of [
  "navigator",
  "getComputedStyle",
  "DOMParser",
  "XMLSerializer",
  "Node",
  "HTMLElement",
  "SVGElement",
]) {
  // Some of these are read-only on newer Node versions; a failure to override
  // one is harmless because jsdom's copy is only needed for the parser.
  try { globalThis[name] = dom.window[name]; } catch {}
}

// Import mermaid after the DOM globals exist.
const { default: mermaid } = await import("mermaid");
mermaid.initialize({ startOnLoad: false, logLevel: "fatal", securityLevel: "loose" });
mermaid.parseError = () => {};

/**
 * Error thrower helper
 */
function reasonOf(error) {
  const message = String(error?.message ?? error).trim();
  const firstLine = message.split(/\r?\n/, 1)[0].trim();
  return firstLine || "Unknown parse error";
}

/**
 * Extract mermaid charts diagram from responses that does not return solely the mermaid charts.
 */
function extractDiagram(text) {
  const match = text.match(/```[ \t]*[A-Za-z0-9_-]*[ \t]*\r?\n([\s\S]*?)```/);
  if (match) {
    return match[1].trim();
  }
  return text.trim();
}

/** Return true if the text is valid Mermaid, otherwise the failure reason. */
async function check(text) {
  const diagram = extractDiagram(text);
  try {
    const result = await mermaid.parse(diagram);
    // parse() throws on invalid input, but returns false when errors are
    // suppressed; treat both as failure so the result cannot silently pass.
    if (result === false) {
      return { ok: false, reason: "Parser rejected diagram" };
    }
    return { ok: true };
  } catch (error) {
    return { ok: false, reason: reasonOf(error) };
  }
}

/** Read a JSONL file into an array of row objects. */
function readRows(path) {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line));
}

/** Path where a given input's score file lives*/
function scorePathFor(input) {
  return input.replace(/\.jsonl$/i, "") + ".score.json";
}

async function judge(input) {
  const rows = readRows(input);
  const reasons = new Map();
  let passed = 0;

  for (const row of rows) {
    const text = typeof row.generated === "string" ? row.generated : "";
    const { ok, reason } = await check(text);
    if (ok) {
      passed += 1;
    } else {
      reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
    }
  }

  const total = rows.length;
  const failed = total - passed;

  // Stable sort by descending count. Insertion order is preserved for ties.
  const failure_reasons = Object.fromEntries(
    [...reasons.entries()].sort((a, b) => b[1] - a[1]),
  );

  const score = {
    source: relative(process.cwd(), resolve(input)),
    total,
    passed,
    failed,
    pass_rate_percent: total === 0 ? 0 : Math.round((passed / total) * 1000) / 10,
    failure_reasons,
  };

  writeFileSync(scorePathFor(input), JSON.stringify(score, null, 2) + "\n", "utf8");

  const rate = total === 0 ? "0" : ((passed / total) * 100).toFixed(1);
  console.log(`${input}: ${passed}/${total} valid (${rate}%) -> ${scorePathFor(input)}`);
  return score;
}

const inputs = process.argv.slice(2);
if (inputs.length === 0) {
  console.error("Usage: node judge.mjs <generations.jsonl> [more.jsonl ...]");
  process.exit(2);
}

for (const input of inputs) {
  await judge(input);
}
