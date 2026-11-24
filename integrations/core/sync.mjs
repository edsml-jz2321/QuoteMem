#!/usr/bin/env node
// Copy the adapter core into every adapter that ships it, or check the copies.
//
//   node integrations/core/sync.mjs           # write the copies
//   node integrations/core/sync.mjs --check   # exit 1 when a copy differs
//
// An installed plugin (an npm package, a Claude Code or Codex plugin cache)
// holds only its own folder, so each adapter carries a byte-identical copy of
// the core in its own `core/` folder. Edit integrations/core/, then sync.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const integrations = path.dirname(here);

/** Adapter folders that ship a copy, relative to integrations/. */
export const TARGETS = ['claude-code/core', 'codex/core', 'hooks/core', 'gemini-cli/core', 'openclaw/core', 'opencode/core'];

/** The core files that are shipped: every .mjs module except this script. */
export function coreFiles() {
  return fs.readdirSync(here).filter((name) => name.endsWith('.mjs') && name !== 'sync.mjs').sort();
}

/** The differences between the core and its copies: ["dsh/core/http.mjs differs", …]. */
export function check() {
  const problems = [];
  const files = coreFiles();
  for (const target of TARGETS) {
    const dir = path.join(integrations, target);
    let present = [];
    try {
      present = fs.readdirSync(dir).filter((name) => name.endsWith('.mjs')).sort();
    } catch {
      problems.push(`${target} is missing`);
      continue;
    }
    for (const name of files) {
      const copy = path.join(dir, name);
      if (!fs.existsSync(copy)) problems.push(`${target}/${name} is missing`);
      else if (!fs.readFileSync(copy).equals(fs.readFileSync(path.join(here, name)))) problems.push(`${target}/${name} differs from integrations/core/${name}`);
    }
    for (const name of present) if (!files.includes(name)) problems.push(`${target}/${name} is not in integrations/core`);
  }
  return problems;
}

function write() {
  const files = coreFiles();
  for (const target of TARGETS) {
    const dir = path.join(integrations, target);
    fs.mkdirSync(dir, { recursive: true });
    for (const name of fs.readdirSync(dir)) {
      if (name.endsWith('.mjs') && !files.includes(name)) fs.unlinkSync(path.join(dir, name));
    }
    for (const name of files) fs.copyFileSync(path.join(here, name), path.join(dir, name));
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.includes('--check')) {
    const problems = check();
    for (const problem of problems) console.error(`FAIL ${problem}; run node integrations/core/sync.mjs`);
    if (problems.length > 0) process.exit(1);
    console.log(`ok: ${TARGETS.length === 1 ? 'the core copy matches' : `${TARGETS.length} core copies match`} integrations/core`);
  } else {
    write();
    console.log(`synced ${coreFiles().length} core files to ${TARGETS.join(', ')}`);
  }
}
