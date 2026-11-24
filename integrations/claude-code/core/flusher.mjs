#!/usr/bin/env node
// The background flusher: a detached process that drains one state folder's
// disk queue and exits when it is idle. Started by DiskQueue.startFlusher,
// which passes the settings (without the token) in QUOTEMEM_FLUSHER_SETTINGS
// and the token in QUOTEMEM_TOKEN. It writes nothing to stdout or stderr;
// warnings go to <stateDir>/quotemem.log. A hook host's flusher
// (settings.idleWait) then stays while chats wait for their idle time, and
// saves their last turns (runIdleWaiter).
import path from 'node:path';

import { capLog } from './autostart.mjs';
import { runFlusher } from './diskqueue.mjs';
import { runIdleWaiter } from './hooks.mjs';
import { fileSink, makeLog } from './log.mjs';

let settings;
try {
  settings = { ...JSON.parse(process.env.QUOTEMEM_FLUSHER_SETTINGS || 'null'), token: process.env.QUOTEMEM_TOKEN || '' };
} catch {
  settings = null;
}
// Without a token only a service on this computer can be used: it was set
// up to start on demand, and its token is read once it exists.
if (!settings || typeof settings.stateDir !== 'string' || typeof settings.url !== 'string' || !(settings.token || settings.localService)) process.exit(0);

const log = makeLog(settings, fileSink(path.join(settings.stateDir, 'quotemem.log')));
Object.defineProperty(settings, 'autostartLog', { value: log, enumerable: false });
// A service started on demand appends to its log for as long as it runs.
if (settings.localService?.dir) capLog(path.join(settings.localService.dir, 'service.log'));
const stop = new AbortController();
for (const name of ['SIGTERM', 'SIGINT', 'SIGHUP']) {
  try { process.on(name, () => stop.abort()); } catch { /* not on this platform */ }
}
try {
  const outcome = settings.idleWait === true
    ? await runIdleWaiter(settings, log, { signal: stop.signal, env: process.env })
    : await runFlusher(settings, log, { signal: stop.signal });
  log.debug(`flusher exit: ${outcome}`);
} catch (error) {
  log.warnOnce('flusher', `background flusher error (${error?.code ?? error?.name ?? 'error'})`);
}
process.exit(0);
