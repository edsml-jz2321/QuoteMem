// Checks server.json, the entry for the official MCP Registry, against
// package.json and the MCP server. check_package.mjs runs it.
//
// The registry accepts a version only when the npm package of that version
// names the server in its `mcpName`, so the two names must be equal, and
// server.json must name the version of package.json.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

export const SCHEMA = 'https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json';
export const SERVER_NAME = 'io.github.edsml-jz2321/quotemem';

/** The sources of the MCP server that `npx quotemem mcp` runs: its environment is read here. */
export function serverSources() {
  const files = [path.join(here, 'dsh', 'server.mjs'), path.join(here, 'core', 'token.mjs')];
  return files.map((file) => fs.readFileSync(file, 'utf8')).join('\n');
}

/**
 * The problems of server.json, as messages.
 * @param {object} server - server.json
 * @param {object} pkg - package.json
 * @param {string} source - the MCP server's source code (serverSources())
 */
export function checkServerJson(server, pkg, source) {
  const problems = [];
  const ok = (value, message) => { if (!value) problems.push(message); };
  const version = pkg?.version;
  ok(server?.$schema === SCHEMA, `server.json $schema must be ${SCHEMA}`);
  ok(pkg?.mcpName === SERVER_NAME, `package.json mcpName must be "${SERVER_NAME}"`);
  ok(server?.name === pkg?.mcpName, `server.json name ${JSON.stringify(server?.name)} differs from package.json mcpName ${JSON.stringify(pkg?.mcpName)}`);
  ok(typeof server?.description === 'string' && server.description.length >= 1 && server.description.length <= 100, 'server.json description must have 1 to 100 characters');
  ok(server?.version === version, `server.json version ${server?.version} differs from package.json ${version}`);
  const packages = Array.isArray(server?.packages) ? server.packages : [];
  ok(packages.length === 1, 'server.json must list exactly one package, the npm package');
  for (const entry of packages) {
    ok(entry?.registryType === 'npm', 'server.json package registryType must be "npm"');
    ok(entry?.registryBaseUrl === undefined || entry.registryBaseUrl === 'https://registry.npmjs.org', 'server.json package registryBaseUrl must be https://registry.npmjs.org');
    ok(entry?.identifier === pkg?.name, `server.json package identifier ${JSON.stringify(entry?.identifier)} differs from package.json name ${JSON.stringify(pkg?.name)}`);
    ok(entry?.version === version, `server.json package version ${entry?.version} differs from package.json ${version}`);
    ok(entry?.transport?.type === 'stdio', 'server.json package transport must be stdio');
    const args = (entry?.packageArguments ?? []).map((arg) => arg?.value);
    ok(args.length === 1 && args[0] === 'mcp', 'server.json packageArguments must be the one positional argument "mcp" (npx quotemem mcp)');
    ok(pkg?.bin?.[pkg?.name] !== undefined, `package.json has no bin "${pkg?.name}" for npx ${pkg?.name} mcp`);
    for (const variable of entry?.environmentVariables ?? []) {
      const name = variable?.name;
      ok(typeof name === 'string' && new RegExp(`\\benv\\.${name}\\b`).test(source), `server.json names ${name}, which the MCP server does not read`);
      ok(variable?.isRequired !== true, `server.json marks ${name} as required; the server needs no variable`);
      if (variable?.default !== undefined) {
        ok(source.includes(`env.${name} || '${variable.default}'`), `server.json default of ${name} (${variable.default}) differs from the MCP server's`);
      }
    }
  }
  return problems;
}
