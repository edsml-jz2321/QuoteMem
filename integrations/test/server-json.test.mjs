#!/usr/bin/env node
// server.json, the MCP Registry entry, against package.json and the MCP server.
//
//   node --test integrations/test/server-json.test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { SCHEMA, SERVER_NAME, checkServerJson, serverSources } from '../check_server_json.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const readJson = (name) => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
const SERVER = readJson('server.json');
const PKG = readJson('package.json');
const SOURCE = serverSources();
const copy = (value) => JSON.parse(JSON.stringify(value));
const problemsWith = (edit, pkgEdit = () => {}) => {
  const server = copy(SERVER);
  const pkg = copy(PKG);
  edit(server);
  pkgEdit(pkg);
  return checkServerJson(server, pkg, SOURCE);
};

test('the repository server.json passes', () => {
  assert.deepEqual(checkServerJson(SERVER, PKG, SOURCE), []);
  assert.equal(SERVER.$schema, SCHEMA);
  assert.equal(SERVER.name, SERVER_NAME);
  assert.equal(PKG.mcpName, SERVER_NAME);
});

test('server.json runs `npx -y quotemem mcp` over stdio and requires no variable', () => {
  const [entry] = SERVER.packages;
  assert.equal(entry.runtimeHint, 'npx');
  assert.deepEqual(entry.runtimeArguments.map((arg) => arg.name), ['-y']);
  assert.deepEqual(entry.packageArguments.map((arg) => arg.value), ['mcp']);
  assert.deepEqual(entry.environmentVariables.map((variable) => variable.name), ['QUOTEMEM_URL', 'QUOTEMEM_TOKEN_FILE', 'QUOTEMEM_USER_ID']);
  assert.ok(entry.environmentVariables.every((variable) => variable.isRequired === false && variable.isSecret === false));
});

test('a server.json version other than package.json fails', () => {
  const problems = problemsWith((server) => { server.version = '9.9.9'; });
  assert.equal(problems.length, 1);
  assert.match(problems[0], /server\.json version 9\.9\.9 differs from package\.json/);
});

test('a package version other than package.json fails', () => {
  const problems = problemsWith((server) => { server.packages[0].version = '9.9.9'; });
  assert.equal(problems.length, 1);
  assert.match(problems[0], /server\.json package version 9\.9\.9 differs/);
});

test('a version bump in package.json alone fails twice', () => {
  const problems = problemsWith(() => {}, (pkg) => { pkg.version = '9.9.9'; });
  assert.equal(problems.length, 2);
});

test('mcpName must equal the server.json name', () => {
  const renamed = problemsWith((server) => { server.name = 'io.github.someone/quotemem'; });
  assert.equal(renamed.length, 1);
  assert.match(renamed[0], /differs from package\.json mcpName/);
  const missing = problemsWith(() => {}, (pkg) => { delete pkg.mcpName; });
  assert.ok(missing.some((problem) => /package\.json mcpName must be/.test(problem)));
  assert.ok(missing.some((problem) => /differs from package\.json mcpName undefined/.test(problem)));
});

test('the schema, the package identifier, the registry and the transport are checked', () => {
  assert.equal(problemsWith((server) => { server.$schema = SCHEMA.replace('2025-12-11', '2025-09-29'); }).length, 1);
  assert.equal(problemsWith((server) => { server.packages[0].identifier = 'quotemem-mcp'; }).length, 1);
  assert.equal(problemsWith((server) => { server.packages[0].registryType = 'pypi'; }).length, 1);
  assert.equal(problemsWith((server) => { server.packages[0].registryBaseUrl = 'https://registry.example.org'; }).length, 1);
  assert.equal(problemsWith((server) => { server.packages[0].transport = { type: 'streamable-http' }; }).length, 1);
  assert.equal(problemsWith((server) => { server.packages[0].packageArguments = []; }).length, 1);
  assert.equal(problemsWith((server) => { server.packages.push(copy(server.packages[0])); }).length, 1);
  assert.equal(problemsWith((server) => { server.description = 'x'.repeat(101); }).length, 1);
});

test('every variable is one the server reads, none is required, and defaults match', () => {
  const unknown = problemsWith((server) => { server.packages[0].environmentVariables.push({ name: 'QUOTEMEM_TOKEN_PATH', isRequired: false }); });
  assert.deepEqual(unknown, ['server.json names QUOTEMEM_TOKEN_PATH, which the MCP server does not read']);
  const required = problemsWith((server) => { server.packages[0].environmentVariables[2].isRequired = true; });
  assert.deepEqual(required, ['server.json marks QUOTEMEM_USER_ID as required; the server needs no variable']);
  const port = problemsWith((server) => { server.packages[0].environmentVariables[0].default = 'http://127.0.0.1:8766'; });
  assert.equal(port.length, 1);
  assert.match(port[0], /default of QUOTEMEM_URL/);
});
