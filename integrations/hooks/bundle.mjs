#!/usr/bin/env node
// Copy the hook runner and the MCP server into the self-contained bundles, or
// check the copies.
//
//   node integrations/hooks/bundle.mjs           # write the copies
//   node integrations/hooks/bundle.mjs --check   # exit 1 when a copy differs
//
// A Gemini CLI extension or an OpenClaw plugin is installed by copying its
// own folder, so it cannot reach integrations/hooks or integrations/dsh. Edit
// the sources below, then run this script and `node integrations/core/sync.mjs`
// (which copies the core into each bundle's core/ folder).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const integrations = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Bundle folder (relative to integrations/) -> {copy name: source relative to integrations/}. */
export const BUNDLES = {
  'gemini-cli': {
    'quotemem-hook.mjs': 'hooks/quotemem-hook.mjs',
    'hosts.mjs': 'hooks/hosts.mjs',
    'env.mjs': 'hooks/env.mjs',
    'mcp.mjs': 'hooks/mcp.mjs',
    'server.mjs': 'dsh/server.mjs',
  },
  openclaw: {
    'env.mjs': 'hooks/env.mjs',
    'mcp.mjs': 'hooks/mcp.mjs',
    'server.mjs': 'dsh/server.mjs',
  },
};

export function check() {
  const problems = [];
  for (const [bundle, files] of Object.entries(BUNDLES)) {
    for (const [name, source] of Object.entries(files)) {
      const copy = path.join(integrations, bundle, name);
      if (!fs.existsSync(copy)) problems.push(`${bundle}/${name} is missing`);
      else if (!fs.readFileSync(copy).equals(fs.readFileSync(path.join(integrations, source)))) problems.push(`${bundle}/${name} differs from integrations/${source}`);
    }
  }
  return problems;
}

function write() {
  for (const [bundle, files] of Object.entries(BUNDLES)) {
    for (const [name, source] of Object.entries(files)) {
      const copy = path.join(integrations, bundle, name);
      fs.mkdirSync(path.dirname(copy), { recursive: true });
      fs.copyFileSync(path.join(integrations, source), copy);
      if (name === 'quotemem-hook.mjs' || name === 'mcp.mjs' || name === 'server.mjs') fs.chmodSync(copy, 0o755);
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.includes('--check')) {
    const problems = check();
    for (const problem of problems) console.error(`FAIL ${problem}; run node integrations/hooks/bundle.mjs`);
    if (problems.length > 0) process.exit(1);
    console.log(`ok: ${Object.keys(BUNDLES).length} hook bundles match their sources`);
  } else {
    write();
    console.log(`copied the hook runner and MCP server into ${Object.keys(BUNDLES).join(', ')}`);
  }
}
