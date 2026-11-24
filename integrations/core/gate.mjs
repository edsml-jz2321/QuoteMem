// Cheap local tests on a user message: trivial, slash command, memory related.

// Pure acknowledgements and greetings. Short answers such as "yes", "no",
// "对", "不是" or "没有" are kept: as replies to a question they carry facts.
const TRIVIAL = /^(ok(ay)?|k|thanks?( you)?|thx|ty|cool|nice|great|got it|hi|hello|hey|bye|好的?|好滴|嗯+|哦+|噢+|谢谢(你)?|多谢|收到|明白了?|知道了|行|继续|在吗|你好)[\s.!?,。！？，~～]*$/i;

/** True for acknowledgements and other messages with nothing to remember. */
export function isTrivial(text) {
  const t = String(text ?? '').trim();
  return t.length === 0 || TRIVIAL.test(t);
}

/** True when the text ends with a question mark. */
export function endsWithQuestion(text) {
  return /[?？]\s*$/.test(String(text ?? ''));
}

// Words whose period never ends a sentence: titles before a name, and
// abbreviations that end a sentence only when a capital follows.
const TITLES = new Set(['mr', 'mrs', 'ms', 'mx', 'dr', 'prof', 'sr', 'jr', 'st', 'mt', 'rev', 'hon', 'fr', 'gen', 'col', 'capt', 'lt', 'sgt', 'gov', 'sen', 'rep']);
const ABBREVIATIONS = new Set(['etc', 'vs', 'e.g', 'i.e', 'eg', 'ie', 'a.m', 'p.m', 'u.s', 'u.k', 'u.n', 'no', 'nos', 'inc', 'ltd', 'co', 'corp', 'approx', 'dept', 'est', 'fig', 'vol', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec']);
const BOUNDARY = /[.!?;]+(?=\s|$)|[。！？；]+|\n+/gu;

/**
 * Split a message into sentences, each with its end marks; the pieces join
 * back to the whole text. A Latin mark ends a sentence only before a space
 * or the end, so "v2.3", "a.b@example.com" and "3.5" stay whole; a period
 * after a title ("Dr. Sato") or a single initial ("J. Smith") never ends one,
 * and after an abbreviation ("etc.", "U.S.") only when a capital follows.
 */
export function sentences(text) {
  const t = String(text ?? '');
  const out = [];
  let start = 0;
  for (const match of t.matchAll(BOUNDARY)) {
    const end = match.index + match[0].length;
    if (match[0] === '.') {
      const word = /(\S+)$/.exec(t.slice(start, match.index))?.[1] ?? '';
      const bare = word.replace(/^[("'“‘]+/, '').toLowerCase();
      const next = /^\s*(\S)/u.exec(t.slice(end))?.[1] ?? '';
      if (TITLES.has(bare) || /^\p{Lu}$/u.test(word)) continue;
      if (ABBREVIATIONS.has(bare) && next && !/\p{Lu}/u.test(next)) continue;
    }
    if (end > start) out.push(t.slice(start, end));
    start = end;
  }
  if (start < t.length) out.push(t.slice(start));
  return out.filter((piece) => piece.trim().length > 0 || out.length === 1);
}

/** True for a slash command (`/name …`) or a shell escape (`!…`). */
export function isCommand(text) {
  const t = String(text ?? '').trimStart();
  return /^\/[\w:-]+(\s|$)/.test(t) || t.startsWith('!');
}

// The gate must stay quiet on ordinary coding requests ("fix the memory
// leak", "run the tests before deploying", "explain recall and precision")
// and fire on questions about the user or earlier conversations. Bare time
// words and bare "memory"/"recall" are therefore not enough on their own:
// they must come with a first or second person or a question form.
// Words about code that follow "my" in an ordinary coding question ("what is
// my bug here?"): not a fact about the user.
const CODE_NOUN = '(?:bug|bugs|error|errors|code|config|configuration|build|test|tests|function|file|files|repo|repository|branch|script|program|query|output|logs?|stack|stack trace|issue|problem|commit|pr|class|method|variable|schema|server|app|project|setup|environment|env|dependency|dependencies|package|module|api|endpoint|database|db|table|component|page|site|website|container|cluster|pipeline|job|workflow|command|terminal|shell)';
const STRONG_EN = [
  /\b(do|can|could|did|will|would)\s+you\s+(still\s+)?(remember|recall)\b/i,
  /\b(remember|recall|remind me)\s+(what|when|where|who|how|which)\s+(i|my|me|we|our|us)\b/i,
  /\b(recall|remind me)\s+(that\s+)?(i|my|me|we|our|us)\b/i,
  /\b(your|my|our)\s+(memory|memories)\b/i,
  /\b(last|previous|earlier|prior|other)\s+(time|session|conversation|chat|discussion)\b/i,
  /\b(last|previous)\s+(day|week|month|year|meeting)\b[^.\n]{0,40}\b(i|we|you|my|our)\b/i,
  /\b(as usual|like before|as before|as always|the usual|same as last|same as before)\b/i,
  /\b(you|i|we)\s+(said|told|mentioned|discussed|talked|asked|promised|shared)\b/i,
  /\b(as|like|what|whatever)\s+(you|i|we)\s+(had\s+|have\s+)?(decided|agreed|chose|picked|planned|settled)\b/i,
  /\b(did|have|had|do|does)\s+(i|we)\s+(ever\s+)?(say|tell|mention|ask|share|decide|agree|choose|pick|use|prefer|like|own|have|buy|book|visit|live|work)\b/i,
  new RegExp(String.raw`\b(what|where|when|who|which|how)('s|\s+(is|are|was|were|did|do|does))\s+(my|our)\s+(?!${CODE_NOUN}\b)`, 'i'),
  /\b(what|where|when|who|which)\s+(\w+\s+)?(do|did|does|am|was|have|had)\s+(i|we)\s+(usually\s+|normally\s+|still\s+)?(use|prefer|like|love|hate|live|work|do\s+for|own|have|eat|drink|drive|play|study|decide|choose|chose|pick|call|say|tell|mention|agree|settle|want|plan|go|buy)\b/i,
  // "What sport do I do these days?": the verb "do" needs a time of the
  // user's life, so "what do I do with this error" stays out.
  /\b(what|which)\s+(\w+\s+)?(do|did)\s+(i|we)\s+(usually\s+|normally\s+|still\s+)?do\s+(these days|nowadays|lately|for fun|for exercise|for a living|on (the )?weekends?|at weekends?|in (my|our) (free|spare) time|after work|in the evenings?)\b/i,
  /\bhow\s+(do|did|does)\s+(i|we)\s+(usually|normally|always|often|typically)\b/i,
  /\bhow\s+(do|did)\s+(i|we)\s+(get|go|commute|travel)\s+(to\s+)?(work|school|the office|office|campus|my job|home)\b/i,
  /\bhow\s+old\s+am\s+i\b/i,
  /\b(yesterday|last night|the other day|a while ago)\b[^.\n]{0,40}\b(i|we|you|my|our)\b/i,
  /\b(i|we|you)\b[^.\n]{0,40}\b(yesterday|last night|the other day|a while ago)\b/i,
  /\b(continue|resume|pick up|carry on)\b[^.\n]{0,30}\b(where\s+(we|you|i)\s+(left|stopped)|from\s+(the\s+)?(last|previous|earlier)\s+(time|session|conversation|chat))\b/i,
];
// Telling the assistant to remember, a decision stated with its content, or
// "I forgot my …": statements, so they count only in a question.
const SOFT_EN = [
  /\b(remember|forgot|forget)\s+(that\s+)?(i|my|me|we|our|us)\b/i,
  /\bremember\s+(this|that)\b/i,
  /\b(you|i|we)\s+(decided|agreed|chose|picked|planned|settled)\b/i,
];
const ME = '(?:我|我们|咱们)';
const STRONG_ZH = [
  /(还?记得|回忆一下|想起来)/,
  new RegExp(`(上次|上回|那次|那天|前几天|上周|上个?星期|上个?月|去年)[^。！？\\n]*(${ME}|你|说|讲|提|聊|讨论)|(${ME}|你)[^。！？\\n]*(上次|上回|那次|那天|前几天|上周|上个?星期|上个?月|去年)`),
  new RegExp(`(之前|以前|此前|先前|昨天|前天)[^。！？\\n]{0,12}(${ME}|你)(说|讲|提|聊|讨论|定|决定|选|用)`),
  new RegExp(`(${ME}|你)[^。！？\\n]{0,8}(之前|以前|此前|先前|昨天|前天)[^。！？\\n]{0,8}(说|讲|提|聊|讨论|定|决定|选|用|告诉)`),
  /(我说过|我讲过|我提过|我提到过|我告诉过?你|跟你说过|和你说过|你说过|你告诉过?我|你提过)/,
  new RegExp(`${ME}(聊|谈|讨论|说|商量|约定|决定|定)(过|了|的)`),
  /(关于我|你知道我|我是谁|我叫什么|我多大)/,
  new RegExp(`${ME}[^。！？\\n]{0,10}(什么|哪|几|多少|谁)[^。！？\\n]{0,6}[？?吗]?$`),
  /(老样子|照旧|还是那样|跟以前一样|和以前一样|继续上次|接着上次)/,
];

// Patterns about the user's own facts. A plain statement ("I prefer Rust",
// "我喜欢咖啡", "my wife is called Ana") matches them too, so in the
// `questions` gate they count only when the message asks something.
const SELF_EN = [
  /\b(about me|know (about )?me|who am i|my name)\b/i,
  /\bmy\s+(birthday|age|wife|husband|partner|girlfriend|boyfriend|kids?|children|son|daughter|mom|mother|dad|father|parents|family|sister|brother|siblings?|grand(?:ma|pa|mother|father|parents)|aunt|uncle|cousin|friends?|dog|cat|pets?|job|boss|address|email|phone|hometown|preferences?|favou?rites?|usual|habits?|schedule|goals?|allerg\w*|diet|hobb\w+)\b/i,
  /\bi\s+(prefer|usually|always|never|live|moved|am allergic)\b/i,
];
const SELF_ZH = [
  /我的?(名字|生日|年龄|老婆|妻子|老公|丈夫|对象|女朋友|男朋友|孩子|儿子|女儿|父母|爸|妈|姐姐|妹妹|哥哥|弟弟|姐妹|兄弟|爷爷|奶奶|外公|外婆|家人|朋友|宠物|狗|猫|工作|老板|地址|邮箱|电话|家乡|喜好|偏好|习惯|爱好|计划|目标|日程|安排)/,
  /我家(的|里)?[^\s，。？！]{0,4}(叫|是|在|有|几)/,
  new RegExp(`${ME}(喜欢|偏好|习惯|讨厌|不喜欢|一般|通常|总是|从不|平时|住在|住哪|对什么)`),
];
const SOFT_ZH = [/(记住|记一下|帮我记|忘了|忘记|别忘)/];
const STRONG = [...STRONG_ZH, ...STRONG_EN];
// Like SELF: these match statements too, so they count in a question only.
const SELF = [...SELF_ZH, ...SELF_EN, ...SOFT_ZH, ...SOFT_EN];
const QUESTION_EN = /^\s*(?:(?:hey|so|and|also|ok(?:ay)?|well)[\s,]+)?(what|where|when|who|whom|whose|which|how|why|do|does|did|is|are|am|was|were|can|could|will|would|should|have|has|had)\b/i;
const QUESTION_ZH = /(吗|呢|什么|啥|哪|几|谁|多少|怎么|是不是|有没有)/;

/** True when the message asks something: a question mark, a question word or form. */
export function isQuestion(text) {
  const t = String(text ?? '');
  return /[?？]/.test(t) || QUESTION_EN.test(t) || QUESTION_ZH.test(t);
}

/**
 * Cheap local check: does this user message refer to the past or to the
 * user's own facts or preferences? English and Chinese keyword heuristics.
 * @param {string} text - the user message.
 * @param {{statements?: boolean}} options - `statements: false` (the
 *   `questions` gate) ignores plain first-person statements about the user
 *   unless the message asks something; the default keeps the broad gate.
 */
export function isMemoryRelated(text, { statements = true } = {}) {
  const t = String(text ?? '');
  // The whole message, then each sentence: some patterns are anchored at a
  // sentence end ("我姐姐叫什么名字？她住在哪里？").
  const parts = [t, ...sentences(t)];
  const strong = parts.some((part) => STRONG.some((re) => re.test(part)));
  if (strong) return true;
  const self = parts.some((part) => SELF.some((re) => re.test(part)));
  return self && (statements || isQuestion(t));
}
