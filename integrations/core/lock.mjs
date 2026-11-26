// Cross-process file locks and atomic file writes.
//
// A lock is a file created with O_EXCL that records its owner (host, boot,
// pid) and a random nonce. A lock whose owner is gone, or that was not
// refreshed within `staleMs`, is broken by renaming it aside; a release
// removes the file only while it still carries the owner's nonce.
import { randomBytes } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { ownerGone, ownerRecord } from './proc.mjs';

// A lock file still empty or half written this long after its creation was
// left by a process that died between create and write.
const PARTIAL_STALE_MS = 5000;

/** Write a file atomically (temporary file, then rename), mode 0600, creating its folder 0700. */
export function writeFileAtomic(file, text) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const tmp = `${file}.${randomBytes(4).toString('hex')}.tmp`;
  try {
    fs.writeFileSync(tmp, text, { mode: 0o600 });
    fs.renameSync(tmp, file);
  } catch (error) {
    try { fs.unlinkSync(tmp); } catch { /* none */ }
    throw error;
  }
}

const stateKeys = new Map();

/**
 * A random 32-byte key of a state folder (`state.key`, 0600), made once by
 * whichever process needs it first: hashes of a secret kept in the folder
 * (a token) are HMACs under it, so a copy of the folder gives no value to
 * test guesses against, and none that matches another folder's.
 */
export function stateKey(stateDir) {
  const known = stateKeys.get(stateDir);
  if (known) return known;
  const file = path.join(stateDir, 'state.key');
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const key = fs.readFileSync(file);
      if (key.length === 32) {
        stateKeys.set(stateDir, key);
        return key;
      }
      // another process is writing it
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      fs.mkdirSync(stateDir, { recursive: true, mode: 0o700 });
      const key = randomBytes(32);
      try {
        fs.writeFileSync(file, key, { flag: 'wx', mode: 0o600 });
        stateKeys.set(stateDir, key);
        return key;
      } catch (made) {
        if (made.code !== 'EEXIST') throw made;
      }
    }
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
  }
  throw Object.assign(new Error('the state folder key could not be read'), { code: 'ESTATEKEY' });
}

/** Read and parse a JSON file; null when missing or unreadable. */
export function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

function lockState(file, staleMs, ignoreHost = false) {
  let stat;
  try {
    stat = fs.statSync(file);
  } catch {
    return { exists: false };
  }
  const data = readJson(file);
  const age = Date.now() - stat.mtimeMs;
  if (!data || !Number.isInteger(data.pid)) return { exists: true, data, stale: age > PARTIAL_STALE_MS };
  const stale = ownerGone(data, data.pid, stat.mtimeMs, { ignoreHost }) || (staleMs > 0 && age > staleMs);
  return { exists: true, data, stale };
}

/** Whether a live owner holds the lock. */
export function lockHeld(file, { staleMs = 0 } = {}) {
  const state = lockState(file, staleMs);
  return state.exists && !state.stale;
}

function breakStale(file, seen) {
  const aside = `${file}.stale-${randomBytes(4).toString('hex')}`;
  try {
    fs.renameSync(file, aside);
  } catch {
    return; // someone else broke or released it
  }
  const taken = readJson(aside);
  if (seen && taken && taken.nonce !== seen.nonce) {
    // A new owner took the lock between our check and the rename: put it back.
    try { fs.linkSync(aside, file); } catch { /* a third owner exists */ }
  }
  try { fs.unlinkSync(aside); } catch { /* gone */ }
}

/**
 * Try to take a lock once (and break a stale one).
 * @param {{staleMs?: number, ignoreHost?: boolean}} options - `ignoreHost`:
 *   the owner's host name is not compared (see ownerGone).
 * @returns {{file: string, nonce: string} | null} the handle, or null when a live owner holds it.
 */
export function tryLock(file, { staleMs = 0, ignoreHost = false } = {}) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  for (let round = 0; round < 2; round += 1) {
    const nonce = randomBytes(8).toString('hex');
    let fd;
    try {
      fd = fs.openSync(file, 'wx', 0o600);
    } catch (error) {
      if (error?.code !== 'EEXIST') throw error;
      const state = lockState(file, staleMs, ignoreHost);
      if (!state.exists) continue;
      if (!state.stale) return null;
      breakStale(file, state.data);
      continue;
    }
    try {
      fs.writeSync(fd, JSON.stringify(ownerRecord({ nonce, at: Date.now() })));
    } finally {
      fs.closeSync(fd);
    }
    return { file, nonce };
  }
  return null;
}

/** Whether the lock file still carries this handle's nonce. */
export function lockOwned(handle) {
  return readJson(handle.file)?.nonce === handle.nonce;
}

/** Refresh a held lock's time, so it is not taken as stale. False when it is no longer ours. */
export function touchLock(handle) {
  const data = readJson(handle.file);
  if (!data || data.nonce !== handle.nonce) return false;
  try {
    const now = new Date();
    fs.utimesSync(handle.file, now, now);
    return true;
  } catch {
    return false;
  }
}

/**
 * Hand a held lock to another process (`pid`, one this process started): the
 * lock then lives as long as that process does. Its holder takes it over
 * with adoptLock. False when it is no longer ours.
 */
export function handLock(handle, pid) {
  const data = readJson(handle.file);
  if (!data || data.nonce !== handle.nonce || !Number.isInteger(pid)) return false;
  try {
    writeFileAtomic(handle.file, JSON.stringify({ ...data, pid }));
    return true;
  } catch {
    return false;
  }
}

/**
 * Take over a lock another process took for this one (its `nonce`, passed
 * at the start). Null when it is gone or now someone else's.
 * @returns {{file: string, nonce: string} | null}
 */
export function adoptLock(file, nonce) {
  if (typeof nonce !== 'string' || !nonce) return null;
  const data = readJson(file);
  if (!data || data.nonce !== nonce) return null;
  try {
    writeFileAtomic(file, JSON.stringify(ownerRecord({ nonce, at: Date.now() })));
  } catch {
    return null;
  }
  return lockOwned({ file, nonce }) ? { file, nonce } : null;
}

/** Release a lock if it is still ours. */
export function releaseLock(handle) {
  if (!handle) return;
  const data = readJson(handle.file);
  if (data && data.nonce === handle.nonce) {
    try { fs.unlinkSync(handle.file); } catch { /* gone */ }
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Run `fn` while holding the lock, waiting for it up to `timeoutMs`.
 * @throws {Error} code `ELOCKED` when the lock could not be taken in time.
 */
export async function withLock(file, fn, { timeoutMs = 2000, staleMs = 30_000 } = {}) {
  const end = Date.now() + timeoutMs;
  let handle = tryLock(file, { staleMs });
  let pause = 5;
  while (!handle) {
    if (Date.now() >= end) throw Object.assign(new Error('lock busy'), { code: 'ELOCKED' });
    await sleep(pause);
    pause = Math.min(50, pause * 2);
    handle = tryLock(file, { staleMs });
  }
  try {
    return await fn();
  } finally {
    releaseLock(handle);
  }
}
