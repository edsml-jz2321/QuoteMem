// Client-side secret redaction: best-effort and pattern based. The service
// refuses to make a memory of a secret as a second line of defence.

// Text beyond the per-message (or query) cap is cut before redaction, so a
// huge paste costs no regex time; this margin keeps a secret that straddles
// the cut detectable.
export const REDACT_MARGIN = 512;

function luhn(digits) {
  let sum = 0;
  for (let i = 0; i < digits.length; i += 1) {
    let d = digits.charCodeAt(digits.length - 1 - i) - 48;
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
  }
  return sum % 10 === 0;
}

function prcIdValid(id) {
  const weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
  const checks = '10X98765432';
  let sum = 0;
  for (let i = 0; i < 17; i += 1) sum += (id.charCodeAt(i) - 48) * weights[i];
  return checks[sum % 11] === id[17].toUpperCase();
}

function entropy(text) {
  const counts = new Map();
  for (const ch of text) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let h = 0;
  for (const n of counts.values()) {
    const p = n / text.length;
    h -= p * Math.log2(p);
  }
  return h;
}

const SECRET_RULES = [
  ['private_key', /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z0-9 ]*PRIVATE KEY-----|$)/g],
  ['aws_key', /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g],
  ['github_token', /\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b/g],
  ['api_key', /\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}/g],
  ['stripe_key', /\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,}\b/g],
  ['huggingface_token', /\bhf_[A-Za-z0-9]{30,}\b/g],
  ['npm_token', /\bnpm_[A-Za-z0-9]{36,}\b/g],
  ['slack_token', /\bxox[abposr]-[A-Za-z0-9-]{10,}/g],
  ['google_key', /\bAIza[0-9A-Za-z_-]{35}\b/g],
  ['jwt', /\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}/g],
  ['bearer', /\b(?:Bearer|Token|Basic)\s+[A-Za-z0-9._~+/=-]{16,}/g],
];

const ASSIGNMENT = /((?:password|passwd|passphrase|pwd|secret|api[_ -]?key|access[_ -]?key|access[_ -]?token|auth[_ -]?token|token|private[_ -]?key|client[_ -]?secret|密码|口令|密钥|令牌)\s*(?:[:=：]|is\s+|是|为)\s*)(["']?)([^\s"'，。,;；]{4,})\2/giu;
const URL_CREDENTIALS = /\b([a-z][a-z0-9+.-]*:\/\/)([^\s/:@]+):([^\s/@]+)@/gi;
const CARD = /(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])/g;
// A US social security number in its usual 3-2-4 form; never area 000, 666
// or 9xx, group 00 or serial 0000.
const SSN = /(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])/g;
const PRC_ID = /(?<![0-9A-Za-z])\d{17}[\dXx](?![0-9A-Za-z])/g;
const LONG_TOKEN = /[A-Za-z0-9+/_=-]{40,}/g;

/**
 * Replace secrets and credentials with `[REDACTED:<kind>]` markers.
 * @param {string} input - message or query text.
 * @returns {{text: string, secretOnly: boolean, kinds: string[]}} the redacted text;
 *   `secretOnly` means too little non-secret text is left to be worth storing.
 */
export function redactText(input) {
  let text = String(input ?? '');
  const kinds = new Set();
  const mark = (kind) => {
    kinds.add(kind);
    return `[REDACTED:${kind}]`;
  };
  for (const [kind, pattern] of SECRET_RULES) text = text.replace(pattern, () => mark(kind));
  text = text.replace(URL_CREDENTIALS, (_m, scheme) => `${scheme}${mark('url_credentials')}@`);
  text = text.replace(ASSIGNMENT, (match, prefix, quote, value) => (value.startsWith('[REDACTED:') ? match : `${prefix}${quote}${mark('credential')}${quote}`));
  text = text.replace(SSN, () => mark('id_number'));
  text = text.replace(CARD, (match) => {
    const digits = match.replace(/\D/g, '');
    return digits.length >= 13 && digits.length <= 19 && luhn(digits) ? mark('card_number') : match;
  });
  text = text.replace(PRC_ID, (match) => (prcIdValid(match) ? mark('id_number') : match));
  text = text.replace(LONG_TOKEN, (match) => {
    if (/^[0-9a-fA-F]+$/.test(match)) return match; // hashes such as git SHAs
    const classes = [/[a-z]/, /[A-Z]/, /\d/].filter((re) => re.test(match)).length;
    return classes >= 3 && entropy(match) > 4.5 ? mark('token') : match;
  });
  const rest = text.replace(/\[REDACTED:[a-z_]+\]/g, '').replace(/\s+/g, '');
  const secretOnly = kinds.has('private_key') || (kinds.size > 0 && [...rest].length < 8);
  return { text, secretOnly, kinds: [...kinds] };
}
