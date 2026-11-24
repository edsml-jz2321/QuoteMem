// Warnings and debug lines. They carry only counts, HTTP status, error codes
// and request ids, never the token, a query or message text. A host that
// must not write to stdout or stderr (a hook whose output is parsed, a TUI)
// logs to a file instead.
import fs from 'node:fs';
import path from 'node:path';

const MAX_LOG_BYTES = 256 * 1024;

/**
 * A logger. Logging never throws.
 * @param {object} settings - `debug` turns debug lines on.
 * @param {(line: string) => void} [write] - the sink; default console.warn.
 */
export function makeLog(settings, write = (line) => console.warn(line)) {
  const seen = new Set();
  const emit = (line) => {
    try {
      write(line);
    } catch {
      // logging must never fail a turn
    }
  };
  return {
    warn(message) { emit(`[quotemem] ${message}`); },
    warnOnce(key, message) {
      if (seen.has(key)) return;
      if (seen.size > 500) seen.clear();
      seen.add(key);
      emit(`[quotemem] ${message}`);
    },
    debug(message) { if (settings.debug) emit(`[quotemem] debug: ${message}`); },
  };
}

/**
 * A sink that appends timestamped lines to a file (mode 0600), keeping one
 * rotated copy (`<file>.1`) once it passes 256 KiB.
 */
export function fileSink(file) {
  return (line) => {
    try {
      fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
      try {
        if (fs.statSync(file).size > MAX_LOG_BYTES) fs.renameSync(file, `${file}.1`);
      } catch {
        // no file yet
      }
      fs.appendFileSync(file, `${new Date().toISOString()} pid=${process.pid} ${line}\n`, { mode: 0o600 });
    } catch {
      // logging must never fail a turn
    }
  };
}
