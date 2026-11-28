#!/usr/bin/env node
// Which host really runs a hook, whether its own QuoteMem hooks are
// installed, and whether a Cursor prompt hook may add context.
//
//   node --test integrations/core/test/*.test.mjs   (the shell expands the glob)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { CURSOR_CLI_CONTEXT_VERSION, CURSOR_CONTEXT_VERSION, cursorCliVersion, cursorPromptContext, foreignHost, markNative, nativeAdapterFor, nativeInstalled, nativeMarkers, resolveSettings } from '../index.mjs';
import { tempDir } from './helpers.mjs';

const root = tempDir('qm-native-');
let seq = 0;
function home() {
  seq += 1;
  const dir = path.join(root, `home-${seq}`);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

test('foreignHost tells Claude Code, Cursor and Devin apart from the payload', () => {
  const devinEnv = { DEVIN_PROJECT_DIR: '/work/project' };
  // Claude Code always sends transcript_path, even where DEVIN_PROJECT_DIR was inherited.
  assert.equal(foreignHost({ session_id: 's', prompt_id: 'p', transcript_path: '/t.jsonl', hook_event_name: 'Stop' }, devinEnv), null);
  assert.equal(foreignHost({ session_id: 's', transcript_path: '/t.jsonl', hook_event_name: 'SessionStart' }), null);
  assert.equal(foreignHost({ session_id: 's', cursor_version: '3.24.9', transcript_path: null, prompt_id: 'p' }), 'cursor');
  // Devin: prompt_id, no transcript_path, and DEVIN_PROJECT_DIR set for its hooks.
  assert.equal(foreignHost({ session_id: 's', prompt_id: 'p', hook_event_name: 'UserPromptSubmit', prompt: 'hi' }, devinEnv), 'devin');
  assert.equal(foreignHost({ session_id: 's', prompt_id: 'p', cwd: '/x' }), null, 'another host with prompt_id is not Devin');
  assert.equal(foreignHost({ session_id: 's', prompt_id: 'p' }, { DEVIN_PROJECT_DIR: '' }), null);
  assert.equal(foreignHost({ session_id: 's', prompt_id: '', hook_event_name: 'Stop' }, devinEnv), null);
  // Devin's SessionStart (no prompt_id yet) counts as Claude Code: it only tidies up.
  assert.equal(foreignHost({ session_id: 's', hook_event_name: 'SessionStart' }, devinEnv), null);
  assert.equal(foreignHost(null), null);
  assert.equal(foreignHost('x'), null);
});

test('nativeAdapterFor: a recent marker, or the host\'s own hook files that run quotemem-hook', () => {
  const ours = JSON.stringify({ Stop: [{ hooks: [{ type: 'command', command: 'quotemem-hook devin Stop' }] }] });
  const other = JSON.stringify({ Stop: [{ hooks: [{ type: 'command', command: 'my-hook.sh' }] }] });

  const none = home();
  assert.equal(nativeAdapterFor('devin', { HOME: none, USERPROFILE: none, DEVIN_PROJECT_DIR: none }), false);
  assert.equal(nativeAdapterFor(null, { HOME: none, USERPROFILE: none }), false);

  // Cursor's hooks are per user: its marker counts. Devin's are per project:
  // its marker does not, or one project's hooks would silence every other.
  const marked = home();
  markNative({ HOME: marked, USERPROFILE: marked }, 'cursor');
  markNative({ HOME: marked, USERPROFILE: marked }, 'devin');
  assert.equal(nativeAdapterFor('cursor', { HOME: marked, USERPROFILE: marked }), true);
  assert.equal(nativeAdapterFor('devin', { HOME: marked, USERPROFILE: marked, DEVIN_PROJECT_DIR: none }), false, 'the Devin marker is not used');
  const cursorOnly = home();
  markNative({ HOME: cursorOnly, USERPROFILE: cursorOnly }, 'cursor');
  assert.equal(nativeAdapterFor('devin', { HOME: cursorOnly, USERPROFILE: cursorOnly, DEVIN_PROJECT_DIR: none }), false, 'a marker is per host');

  const project = home();
  fs.mkdirSync(path.join(project, '.devin'));
  fs.writeFileSync(path.join(project, '.devin', 'hooks.v1.json'), ours);
  assert.equal(nativeAdapterFor('devin', { HOME: none, USERPROFILE: none, DEVIN_PROJECT_DIR: project }), true, 'the project hooks file');
  fs.writeFileSync(path.join(project, '.devin', 'hooks.v1.json'), other);
  assert.equal(nativeAdapterFor('devin', { HOME: none, USERPROFILE: none, DEVIN_PROJECT_DIR: project }), false, 'another hook is not ours');

  // Devin reads hook files in the folders above too, up to the repository root.
  const repo = home();
  fs.mkdirSync(path.join(repo, '.git'));
  fs.mkdirSync(path.join(repo, '.devin'));
  fs.writeFileSync(path.join(repo, '.devin', 'hooks.v1.json'), ours);
  const sub = path.join(repo, 'packages', 'app');
  fs.mkdirSync(sub, { recursive: true });
  assert.equal(nativeAdapterFor('devin', { HOME: none, USERPROFILE: none, DEVIN_PROJECT_DIR: sub }), true, 'a hooks file in a folder above');
  // Not past the repository root.
  const outer = home();
  fs.mkdirSync(path.join(outer, '.devin'));
  fs.writeFileSync(path.join(outer, '.devin', 'hooks.v1.json'), ours);
  const inner = path.join(outer, 'repo');
  fs.mkdirSync(path.join(inner, '.git'), { recursive: true });
  assert.equal(nativeAdapterFor('devin', { HOME: none, USERPROFILE: none, DEVIN_PROJECT_DIR: inner }), false, 'not above the repository root');

  const user = home();
  fs.mkdirSync(path.join(user, '.config', 'devin'), { recursive: true });
  fs.writeFileSync(path.join(user, '.config', 'devin', 'config.json'), JSON.stringify({ hooks: JSON.parse(ours) }));
  assert.equal(nativeAdapterFor('devin', { HOME: user, USERPROFILE: user, DEVIN_PROJECT_DIR: none }), true, 'the user config');

  const cursorHome = home();
  const workspace = home();
  fs.mkdirSync(path.join(workspace, '.cursor'));
  fs.writeFileSync(path.join(workspace, '.cursor', 'hooks.json'), JSON.stringify({ version: 1, hooks: { stop: [{ command: 'quotemem-hook cursor stop' }] } }));
  assert.equal(nativeAdapterFor('cursor', { HOME: cursorHome, USERPROFILE: cursorHome }, { workspace_roots: [workspace] }), true);
  assert.equal(nativeAdapterFor('cursor', { HOME: cursorHome, USERPROFILE: cursorHome }, { workspace_roots: ['relative/path'] }), false);
});

test('markers: in QUOTEMEM_STATE_DIR when it is set, nothing under the home; a marker already in the home still counts', () => {
  const profile = home();
  const state = path.join(home(), 'state');
  const env = { HOME: profile, USERPROFILE: profile, QUOTEMEM_STATE_DIR: state };
  markNative(env, 'devin');
  assert.ok(fs.existsSync(path.join(state, 'native', 'devin.json')));
  assert.equal(fs.existsSync(path.join(profile, '.quotemem')), false, 'nothing under the home');
  assert.ok(nativeInstalled(env, 'devin'));
  assert.equal(nativeInstalled({ HOME: profile, USERPROFILE: profile }, 'devin'), false, 'without the state folder it is not seen');
  // A marker written before the state folder was set.
  markNative({ HOME: profile, USERPROFILE: profile }, 'cursor');
  assert.ok(nativeInstalled(env, 'cursor'));
  assert.deepEqual(nativeMarkers(env), ['cursor', 'devin']);
  // `~` in the state folder is the home.
  const tilde = { HOME: profile, USERPROFILE: profile, QUOTEMEM_STATE_DIR: '~/qm-state' };
  markNative(tilde, 'gemini-cli');
  assert.ok(fs.existsSync(path.join(profile, 'qm-state', 'native', 'gemini-cli.json')));
});

test('cursorPromptContext: auto from the version, on, off', () => {
  assert.equal(CURSOR_CONTEXT_VERSION, '3.20.17');
  const auto = (version) => cursorPromptContext('auto', version);
  for (const version of ['3.20.17', '3.20.18', '3.21.0', '3.24.9', '4.0', 'v3.25.1', '10.0.0']) assert.equal(auto(version), true, version);
  for (const version of ['3.20.16', '3.19.0', '2.6.3', '3.7.36', '3', '', 'nightly', null, undefined, 42]) assert.equal(auto(version), false, String(version));
  // The CLI numbers its versions by date: its interactive mode reads the
  // field from 2026.11.19 on (the version as it sends it); earlier dates were
  // not seen to.
  assert.equal(CURSOR_CLI_CONTEXT_VERSION, '2026.11.19');
  for (const version of ['2026.11.19-f005e9d', '2026.11.19', '2026.11.20', '2026.12.1-abc', '2027.1.2']) assert.equal(auto(version), true, version);
  for (const version of ['2026.11.18-0a1b2c3', '2026.11.17', '2026.9.30', '2025.12.31']) assert.equal(auto(version), false, version);
  for (const version of ['2026.11.19-f005e9d', '2026.11.17', '2027.1.2']) assert.equal(cursorCliVersion(version), true, version);
  for (const version of ['3.20.17', '3.24.9', '10.0.0', '', 'nightly', null, undefined]) assert.equal(cursorCliVersion(version), false, String(version));
  assert.equal(cursorPromptContext('on', '2.1.0'), true);
  assert.equal(cursorPromptContext('on', undefined), true);
  assert.equal(cursorPromptContext('off', '3.24.9'), false);
  // The setting.
  const env = { QUOTEMEM_TOKEN: 't', HOME: root, USERPROFILE: root };
  assert.equal(resolveSettings({}, env).cursorPromptContext, 'auto');
  assert.equal(resolveSettings({}, { ...env, QUOTEMEM_CURSOR_PROMPT_CONTEXT: 'off' }).cursorPromptContext, 'off');
  assert.equal(resolveSettings({}, { ...env, QUOTEMEM_CURSOR_PROMPT_CONTEXT: 'ON' }).cursorPromptContext, 'on');
  assert.equal(resolveSettings({}, { ...env, QUOTEMEM_CURSOR_PROMPT_CONTEXT: 'maybe' }).cursorPromptContext, 'auto');
});
