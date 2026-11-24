// The plugins and `quotemem setup` must speak the service's language: the
// QUOTEMEM_* values setup writes and the HTTP routes the adapters call are
// checked against the Python service in the same checkout (src/quotemem).
// A checkout or package without the service source skips these tests.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const integrations = path.resolve(here, '..', '..');
const root = path.resolve(integrations, '..');
const SERVICE = path.join(root, 'src', 'quotemem');
const skip = fs.existsSync(path.join(SERVICE, 'agent', 'settings.py')) ? false : 'no service source (src/quotemem) in this checkout';

function pythonSources(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) pythonSources(file, out);
    else if (entry.name.endsWith('.py')) out.push(fs.readFileSync(file, 'utf8'));
  }
  return out;
}

/** The string members of `<name> = Literal[...]` (or `type <name> = Literal[...]`) in the service source. */
function literal(source, name) {
  const match = new RegExp(`(?:^|\\n)(?:type\\s+)?${name}\\s*=\\s*Literal\\[([^\\]]*)\\]`).exec(source);
  assert.ok(match, `the service defines ${name} = Literal[...]`);
  return [...match[1].matchAll(/"([^"]*)"|'([^']*)'/g)].map((m) => m[1] ?? m[2]);
}

const service = skip ? '' : pythonSources(SERVICE).join('\n');
const quoted = (value) => service.includes(`"${value}"`) || service.includes(`'${value}'`);

test('setup writes only thinking values the service accepts', { skip }, async () => {
  const { CHAT_REASONING } = await import(pathToFileURL(path.join(integrations, 'cli', 'service.mjs')).href);
  const accepted = literal(service, 'ReasoningParameters');
  for (const value of Object.values(CHAT_REASONING)) {
    assert.ok(accepted.includes(value), `QUOTEMEM_CHAT_REASONING=${value} is one of the service's ${accepted.join('|')}`);
  }
});

test('setup offers exactly the service\'s reviews', { skip }, async () => {
  const { REVIEWS } = await import(pathToFileURL(path.join(integrations, 'cli', 'service.mjs')).href);
  assert.deepEqual(Object.keys(REVIEWS).sort(), literal(service, 'ReviewMode').sort());
});

test('every variable setup writes is read by the service', { skip }, async () => {
  const { MANAGED } = await import(pathToFileURL(path.join(integrations, 'cli', 'service.mjs')).href);
  for (const name of MANAGED) assert.ok(quoted(name), `the service reads ${name}`);
});

test('every route the adapters and the MCP server call is served', { skip }, () => {
  const callers = [
    ...fs.readdirSync(path.join(integrations, 'core')).filter((f) => f.endsWith('.mjs')).map((f) => path.join(integrations, 'core', f)),
    path.join(integrations, 'dsh', 'server.mjs'),
  ];
  const routes = new Set();
  for (const file of callers) {
    for (const m of fs.readFileSync(file, 'utf8').matchAll(/['"`](\/(?:memory|agent)\/[a-z/-]+|\/health)['"`]/g)) routes.add(m[1]);
  }
  assert.ok(routes.has('/memory/add') && routes.has('/memory/search'), [...routes].join(', '));
  const missing = [...routes].filter((route) => !quoted(route));
  assert.deepEqual(missing, [], `routes the plugins call that the service does not name: ${missing.join(', ')}`);
});
