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

// On Windows a file that another process has open, or that was just removed
// while another process still reads it (it is "pending delete" until then),
// cannot be renamed over, created again or read: the call fails with EPERM,
// EACCES or EBUSY for a few milliseconds. Such a call is tried again there,
// for at most about 0.3 s; on other systems the first failure is final.
const WINDOWS_BUSY = new Set(['EPERM', 'EACCES', 'EBUSY']);
const BUSY_TRIES = 25;

/**
 * Run `op`, trying it again while Windows reports the file busy.
 * @param {{platform?: string, wait?: Function}} options - for tests.
 */
function whileBusy(op, { platform = process.platform, wait = sleepMs } = {}) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return op();
    } catch (error) {
      if (platform !== 'win32' || attempt >= BUSY_TRIES || !WINDOWS_BUSY.has(error?.code)) throw error;
      wait(1 + attempt);
    }
  }
}

/**
 * fs.renameSync, tried again while Windows reports the target busy.
 * @param {{platform?: string, rename?: Function, wait?: Function}} options - for tests.
 */
export function renameOver(from, to, { platform = process.platform, rename = fs.renameSync, wait = sleepMs } = {}) {
  whileBusy(() => rename(from, to), { platform, wait });
}

/**
 * Create a file exclusively (fs.openSync with 'wx'), tried again while
 * Windows reports it busy: a lock file another process just released is
 * still there, pending delete, while a third one reads it.
 * @param {{platform?: string, open?: Function, wait?: Function}} options - for tests.
 * @returns {number} the file descriptor.
 */
export function openExclusive(file, mode, { platform = process.platform, open = fs.openSync, wait = sleepMs } = {}) {
  return whileBusy(() => open(file, 'wx', mode), { platform, wait });
}

function sleepMs(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/** Write a file atomically (temporary file, then rename), mode 0600, creating its folder 0700. */
export function writeFileAtomic(file, text) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const tmp = `${file}.${randomBytes(4).toString('hex')}.tmp`;
  try {
    fs.writeFileSync(tmp, text, { mode: 0o600 });
    renameOver(tmp, file);
  } catch (error) {
    try { fs.unlinkSync(tmp); } catch { /* none */ }
    throw error;
  }
}

const stateKeys = new Map();
// A state.key of the wrong length this old is a write that was cut short.
const STATE_KEY_STALE_MS = 5000;

/**
 * A random 32-byte key of a state folder (`state.key`, 0600), made once by
 * whichever process needs it first: hashes of a secret kept in the folder
 * (a token) are HMACs under it, so a copy of the folder gives no value to
 * test guesses against, and none that matches another folder's. The key is
 * written to a temporary file and linked into place whole, so no reader
 * sees a part of it; a key file of the wrong length older than a few
 * seconds (a write cut short, a full disk) is replaced.
 */
export function stateKey(stateDir) {
  const known = stateKeys.get(stateDir);
  if (known) return known;
  const file = path.join(stateDir, 'state.key');
  for (let attempt = 0; attempt < 50; attempt += 1) {
    let key = null;
    try {
      key = fs.readFileSync(file);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (key && key.length === 32) {
      stateKeys.set(stateDir, key);
      return key;
    }
    let stale = false;
    try {
      stale = key !== null && Date.now() - fs.statSync(file).mtimeMs > STATE_KEY_STALE_MS;
    } catch {
      stale = false;
    }
    if (stale) {
      // A key cut short is replaced by one process at a time (the guard
      // file, made exclusively), which reads it again under the guard: a
      // whole key another process put there meanwhile is kept, so every
      // process ends up with the same key.
      const fresh = replaceStateKey(file);
      if (fresh) {
        stateKeys.set(stateDir, fresh);
        return fresh;
      }
    } else if (key === null) {
      fs.mkdirSync(stateDir, { recursive: true, mode: 0o700 });
      const fresh = randomBytes(32);
      const tmp = `${file}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
      try {
        fs.writeFileSync(tmp, fresh, { flag: 'wx', mode: 0o600 });
        try {
          fs.linkSync(tmp, file);
        } catch (error) {
          // A file system without hard links: created in place, as before.
          if (!['EPERM', 'ENOTSUP', 'EOPNOTSUPP', 'ENOSYS'].includes(error.code)) throw error;
          fs.writeFileSync(file, fresh, { flag: 'wx', mode: 0o600 });
        }
        stateKeys.set(stateDir, fresh);
        return fresh;
      } catch (made) {
        if (made.code !== 'EEXIST') throw made;
      } finally {
        try { fs.unlinkSync(tmp); } catch { /* linked, or never made */ }
      }
    }
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
  }
  throw Object.assign(new Error('the state folder key could not be read'), { code: 'ESTATEKEY' });
}

/**
 * Replace a key file of the wrong length, holding `<file>.repair` (made
 * exclusively; one left by a process that died is removed after
 * STATE_KEY_STALE_MS). The new key, or null when another process holds the
 * guard or the file is whole again (the caller reads it again).
 */
function replaceStateKey(file) {
  const guard = `${file}.repair`;
  let fd;
  try {
    fd = openExclusive(guard, 0o600);
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    try {
      if (Date.now() - fs.statSync(guard).mtimeMs > STATE_KEY_STALE_MS) fs.unlinkSync(guard);
    } catch {
      // gone meanwhile
    }
    return null;
  }
  const tmp = `${file}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
  try {
    let again = null;
    try { again = fs.readFileSync(file); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (again === null || again.length === 32) return null;
    const fresh = randomBytes(32);
    fs.writeFileSync(tmp, fresh, { flag: 'wx', mode: 0o600 });
    renameOver(tmp, file);
    return fresh;
  } finally {
    try { fs.unlinkSync(tmp); } catch { /* renamed, or never made */ }
    fs.closeSync(fd);
    try { fs.unlinkSync(guard); } catch { /* removed as stale */ }
  }
}

/**
 * Read and parse a JSON file; null when missing or unreadable. A read that
 * Windows refuses for a moment (see whileBusy) is tried again: a lock whose
 * owner read it as missing would never be released.
 */
export function readJson(file) {
  let text;
  try {
    text = whileBusy(() => fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
  try {
    return JSON.parse(text);
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

// Breaks a stale lock. One process at a time does it, holding `<file>.break`
// (made exclusively, with its owner's record and a nonce), and only while
// the lock still holds the record that was seen stale. Without the guard,
// two processes could break the same dead lock: the second then moved aside
// the live lock a third had taken meanwhile, and put it back after that
// owner had released it, so the lock stayed held by a process that no longer
// held it. Another process removes the guard only once its breaker is gone
// (dropGuard); a breaker removes its guard only while it holds its nonce.
function breakStale(file, seen, staleMs, ignoreHost) {
  const guard = `${file}.break`;
  const nonce = randomBytes(8).toString('hex');
  let fd;
  try {
    fd = openExclusive(guard, 0o600);
  } catch (error) {
    if (error?.code === 'EEXIST') dropGuard(guard, ignoreHost);
    return; // another process is breaking it, or its guard was just removed
  }
  try {
    try {
      fs.writeSync(fd, JSON.stringify(ownerRecord({ nonce, at: Date.now() })));
    } finally {
      fs.closeSync(fd);
    }
    const state = lockState(file, staleMs, ignoreHost);
    if (!state.exists || !state.stale || state.data?.nonce !== seen?.nonce) return;
    const aside = `${file}.stale-${randomBytes(4).toString('hex')}`;
    try {
      fs.renameSync(file, aside);
    } catch {
      return; // released meanwhile
    }
    const taken = readJson(aside);
    if (taken?.nonce !== seen?.nonce) {
      // A lock seen stale by its age, whose owner released it and a new owner
      // took it between the check and the rename: put it back.
      try { fs.linkSync(aside, file); } catch { /* a third owner exists */ }
    }
    try { fs.unlinkSync(aside); } catch { /* gone */ }
  } finally {
    releaseOwn(guard, nonce);
  }
}

/**
 * Remove a guard whose process is gone (ownerless), holding `<guard>.clear`
 * the same way: only one process at a time removes it, and no other process
 * removes a guard, so the guard it saw ownerless is the one it removes, never
 * a live breaker's made meanwhile. A `.clear` file left by a process that
 * died is removed like a guard.
 */
function dropGuard(guard, ignoreHost) {
  const clear = `${guard}.clear`;
  const nonce = randomBytes(8).toString('hex');
  let fd;
  try {
    fd = openExclusive(clear, 0o600);
  } catch (error) {
    if (error?.code === 'EEXIST' && ownerless(clear, ignoreHost)) {
      try { fs.unlinkSync(clear); } catch { /* gone */ }
    }
    return;
  }
  try {
    try {
      fs.writeSync(fd, JSON.stringify(ownerRecord({ nonce, at: Date.now() })));
    } finally {
      fs.closeSync(fd);
    }
    if (ownerless(guard, ignoreHost)) {
      try { fs.unlinkSync(guard); } catch { /* gone */ }
    }
  } finally {
    releaseOwn(clear, nonce);
  }
}

/**
 * Whether the process that made a guard file is gone: the one its owner
 * record names no longer runs, or, with no record yet (it died between the
 * create and the write), the file is still empty or unreadable
 * PARTIAL_STALE_MS after it was made. A live process's guard is never
 * ownerless, however old.
 */
function ownerless(file, ignoreHost) {
  let stat;
  try {
    stat = fs.statSync(file);
  } catch {
    return false; // gone meanwhile
  }
  const data = readJson(file);
  if (data && Number.isInteger(data.pid)) return ownerGone(data, data.pid, stat.mtimeMs, { ignoreHost });
  return Date.now() - stat.mtimeMs > PARTIAL_STALE_MS;
}

/** Remove a guard file this process made, while it still holds its nonce. */
function releaseOwn(file, nonce) {
  if (readJson(file)?.nonce === nonce) {
    try { fs.unlinkSync(file); } catch { /* gone */ }
  }
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
      fd = openExclusive(file, 0o600);
    } catch (error) {
      if (error?.code !== 'EEXIST') throw error;
      const state = lockState(file, staleMs, ignoreHost);
      if (!state.exists) continue;
      if (!state.stale) return null;
      breakStale(file, state.data, staleMs, ignoreHost);
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
