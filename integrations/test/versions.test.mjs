#!/usr/bin/env node
// The versions the documentation names in its commands (check_versions.mjs).
//
//   node --test integrations/test/versions.test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { docFiles, docVersionProblems, versionMentions } from '../check_versions.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const VERSION = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version;

test('every release command in the documentation names the package version', () => {
  assert.deepEqual(docVersionProblems(root, VERSION), []);
  const dshReadme = path.join('integrations', 'dsh', 'README.md');
  assert.ok(docFiles(root).some((file) => file.endsWith(dshReadme)));
  const dsh = fs.readFileSync(path.join(root, 'integrations', 'dsh', 'README.md'), 'utf8');
  assert.ok(versionMentions(dsh).length >= 3, 'the DSH README commands are found');
});

test('a command left at an earlier version is reported with its file and line', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qm-versions-'));
  fs.mkdirSync(path.join(dir, 'integrations', 'dsh'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'docs'));
  fs.writeFileSync(path.join(dir, 'README.md'), 'uvx --from quotemem==9.9.9 quotemem serve\n');
  fs.writeFileSync(path.join(dir, 'integrations', 'dsh', 'README.md'), [
    'dsh plugin --profile quotemem add file:/path/to/quotemem-9.9.8.tgz',
    'dsh plugin --profile quotemem add quotemem@9.9.9 and uvx quotemem[bench]==9.9.7',
  ].join('\n'));
  fs.writeFileSync(path.join(dir, 'docs', 'CHANGELOG.md'), 'quotemem==0.0.1 was the first\n');
  fs.writeFileSync(path.join(dir, 'docs', 'cli.md'), 'pip install quotemem==9.9.9\n');
  assert.deepEqual(docVersionProblems(dir, '9.9.9'), [
    'integrations/dsh/README.md:1 names quotemem 9.9.8, not 9.9.9',
    'integrations/dsh/README.md:2 names quotemem 9.9.7, not 9.9.9',
  ]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('the check is public text: it names no document that is not shipped', () => {
  const source = fs.readFileSync(path.join(root, 'integrations', 'check_versions.mjs'), 'utf8');
  assert.doesNotMatch(source, /REVISION|private/i);
});
