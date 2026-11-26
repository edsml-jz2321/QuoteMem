// Silent forget: find a user's request to forget something, and carry it out
// through QuoteMem's forget by statement (prepare, then confirm the plan).
//
// Detection follows the service's own in-chat rule (src/quotemem/ingestion/
// forget_requests.py) closely, so a sentence the service would read as a
// request is read as one here, and a dismissal ("forget it, let's move on"),
// a reminder ("don't forget that I …") or a question about forgetting never
// is. Chinese also accepts the 把…忘掉 form and 不要记…, and a request that
// refers back ("forget what I just said", "forget that I said that",
// "别记住我刚才说的") forgets the earlier sentences of the message, else the
// previous user message.
//
// Beyond the service's rule, a retraction of something the user asked for
// ("please forget that I asked you for a shell command to count the lines in
// data.csv", "忘掉我问过你怎么统计行数") is a request too: the service may have
// stored the ask, or the conversation lines that state it. A bare "forget
// that I asked" (never mind the question) stays a dismissal, and so does a
// change of instructions ("forget that I asked you to write it in Python, do
// it in Rust instead").
//
// The plugin also reads these everyday forms beyond the service's rule:
// another person's or a pet's name or detail without "my" ("forget Ben's
// name", "forget the dog's name", "忘掉小李的名字", "忘掉阿杰"), a fact about
// "we" or "our" ("forget our wedding anniversary date"), and a request after
// "from now on" or 以后 ("以后不要记得我恐高了"). A request followed by a clear
// replacement in its own sentence ("forget my old address, I live at 27 Birch
// Lane now", "忘掉我的旧地址，我现在住在银杏路6号", "忘掉我的猫的名字，它现在叫咪咪")
// forgets only what comes before the replacement, which stays in the turn and
// is stored as usual (replaces()). Any other clause after it ("forget that I'm
// pregnant, I'm currently in my second trimester") is forgotten with it.
//
// A bare "don't remember …" or "never remember …" with a plain noun phrase
// and nothing else that asks is not read: without "I" it is as often the
// user's own "(I) don't remember the dog's name, can you remind me". Nor is
// "(I/you) forgot …" in Chinese ("忘记我的密码了", "你忘了我的生日吧",
// "忘记了我的生日"), nor 记错/记混 ("别记错我的生日"). The plugin also reads "I
// don't want you to remember my …", "could you please not remember …",
// "don't ever remember …", "你能不能别记住…" and "忘了…吧". Each form is read
// only where it cannot be a complaint, a question or an instruction about
// files: a false request deletes the user's data without a word, a missed
// one is said again more plainly.
//
// Each request also says whether the service's own in-chat check will see it
// (`inChat`): when the request's own turn is stored, the service forgets what
// it asks from every batch stored before, so a batch the plugin's forget did
// not wait for is still covered. A request only the plugin reads (a
// retraction, a reference back to an earlier message, the 把…忘掉 form, the
// forms above), or one it reads with fewer words than the service does (a
// new fact cut off), has no such second check: its forget must wait until the
// earlier batches are stored.
//
// Identity: every forget gets a deterministic request id from the session,
// the turn and the statement, so a repeated hook, a second install of the
// same hooks or the background flusher's retry replays the same operation.
// The service's own in-chat forget (run after a /memory/add that holds the request)
// finds what this forget removed already gone (`forgotten_earlier`), so the
// two never forget twice; the plugin always runs its forget first, because
// the request's own turn is captured after it.
import path from 'node:path';

import { MAX_ID_CHARS } from './config.mjs';
import { SESSION_KEEP_MS, ownerKey, sha16 } from './capture.mjs';
import { hitMemory } from './format.mjs';
import { isCommand, sentences } from './gate.mjs';
import { codeOf, errKind, postJson, retryAfterMs, retryable } from './http.mjs';
import { readJson, withLock, writeFileAtomic } from './lock.mjs';
import { homeDir } from './proc.mjs';
import { redactText } from './redact.mjs';

export const MAX_FORGET_STATEMENT_CHARS = 1000;
const MAX_FORGETS = 4;

// --------------------------------------------------------------- detection

const W = String.raw`[^\p{L}\p{N}_]`;
const OPENER = String.raw`(?:(?:ok(?:ay)?|also|and|so|oh|hey|hi|actually|now|well|lastly|finally|one more thing|by the way|btw)\b[\s,:\-—–]*)*`;
const POLITE = String.raw`(?<polite>(?:please|kindly|(?:can|could|would) you(?: please)?|i(?:'d| would) like (?:you )?to|i want you to|i need you to|i(?:'m| am) asking you to)[\s,]+)?(?:please[\s,]+)?`;
const MEMORY = String.raw`(?:(?:your |the )?(?:memory|memories)|your (?:records?|notes|history))`;
const FORGET = new RegExp(String.raw`^${W}*${OPENER}${POLITE}(?<verb>forget(?: all)?(?: about)?)\b[\s,]*(?<object>.*)$`, 'isu');
const REMEMBER = new RegExp(String.raw`^${W}*${OPENER}${POLITE}(?<verb>(?:do not|don't|dont|never)(?: ever)? remember|stop remembering|no longer remember|not remember)\b\s*(?<object>.*)$`, 'isu');
// "I don't want you to remember my address": only "remember", only one
// clause, about something of the user's ("my …", "that I …"), and not a
// complaint or a correction ("… my name wrong", "… my birthday as the 5th",
// "… me as rude", "… everything about me") nor the work (WANT_THING).
const WANT_NOT = new RegExp(String.raw`^${W}*${OPENER}(?<polite>i (?:don't|do not|dont) want you to|i(?:'d| would) (?:rather|prefer) (?:that )?you (?:didn't|did not|don't|do not|not))\s+(?<verb>remember)\b\s*(?<object>.*)$`, 'isu');
const WANT_OBJECT = /^(?:my|our)\s|^that\s+(?:i|i'm|i've|i'd|i'll|my|we|we're|we've|our)\b/i;
// Any "remember" form (REMEMBER, WANT_NOT) whose object is a noun phrase:
// a correction or a complaint ("please don't remember my name wrong", "… my
// birthday as the 5th", "… everything I say") is never a request. A
// that-clause states the fact itself ("… that I work as a nurse"), and a
// condition asks ("… my address until I'm settled").
const WANT_REFUSED = /\b(?:wrong|wrongly|incorrectly|right|correctly|as|again|like (?:this|that)|instead|remember)\b/i;
const REMEMBER_ALL = /^(?:everything|anything|all|whatever|me)\b/i;
const WANT_THING = /^(?:my|our)\s+(?:\S+\s+)?(?:files?|documents?|docs?|work|progress|sessions?|settings|config\w*|data|database|forms?|cookies?|cache|logs?|history|edits?|output|results?|screens?|screenshots?|uploads?|downloads?|changes?|code|tabs?)\b/i;
// A bare "don't remember …" asks when something marks it as a request: a
// that-clause in the first person ("Don't remember that I live in Lisbon"),
// "do not" written out or emphasis ("Do NOT", "DON'T"), neither when the
// whole sentence is in capitals, or a closing please or thanks; and nothing
// in it asks to be reminded ("…, can you remind me", "tbh", "help"). A
// that-clause that only refers back ("that I said that") is no such mark. A
// bare "never remember …" asks only with please ("never remember my keys!"
// is a complaint).
const ASKS_THAT = /^that\s+(?:i|i'm|i've|i'd|i'll|my|we|we're|we've|our)\b/i;
const CLOSING = /\b(?:please|pls|plz|thanks|thank you|thx)[^\p{L}\p{N}]*$/iu;
const PLEASE = /\b(?:please|pls|plz)[^\p{L}\p{N}]*$/iu;
const EMPHASIS = /\b(?:NOT|DON'?T|NEVER)\b/;
const RECALL = /\b(?:remind me|tell me|can you|could you|would you|do you|help|how|huh|list (?:it|them)|look (?:it|that) up|what(?:'s| is| was| are| were)|tbh|lol|sorry|idk|either)\b/i;
const RECORD = new RegExp(String.raw`^${W}*${OPENER}${POLITE}(?<verb>(?:(?:do not|don't|dont|never) (?:keep|store|save|record|retain))|stop (?:keeping|storing|saving|recording)|remove|delete|erase|wipe|clear|drop|purge)\s+(?<object>.+?)\s+(?:in|into|to|from|out of) ${MEMORY}\b${W}*(?<rest>.*)$`, 'isu');
const SPEAKER = /\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b/i;
const NAMES_MEMORY = new RegExp(String.raw`\b(?:from|in|into|out of) ${MEMORY}\b`, 'i');
const TRAILING_MEMORY = new RegExp(String.raw`\s*,?\s*(?:from|out of|in) ${MEMORY}${W}*$`, 'iu');
// What ends a request's object besides its words: punctuation and spaces, but never a +, #
// or % ("C++", "C#", "50%"): "forget C++" is not "forget C". As the service reads it, a
// detached one stays too, with any courtesy word before it.
const TAIL = String.raw`[^\p{L}\p{N}_+#%]`;
const TRAILING = new RegExp(String.raw`(?:[\s,]+(?:please|pls|plz|for me|thanks|thank you|thx))*${TAIL}*$`, 'iu');
const DISMISSAL = new RegExp(String.raw`^(?:about )?(?:it|this|that|them|everything)\b(?:${W}*$|\s*[,—–-].*$)`, 'isu');
const RETRACTION = /^(?:that |what )?i (?:just |ever )?asked\b/i;
// What a retraction names, after "I asked (you)": at least two content words.
const ASKED = /^i (?:just |ever |once |also |earlier )?asked(?: you)?\b\s*(?<rest>.*)$/i;
const ASKED_MIN_TERMS = 2;
// "I asked you to <do something>" is an instruction to the assistant: taking
// it back ("forget that I asked you to use tabs") changes the work, it does
// not ask for a memory to go. Only a polite or memory-naming retraction of an
// instruction about the user ("please forget that I asked you to look up my
// test results") is a request.
const ASKED_TO = /^to\s+\S/i;
// A clause after the retraction that gives another instruction ("…, do it in
// Rust instead", "…; use spaces") makes it a change of instructions.
const INSTEAD = /\b(?:instead|rather)\b|^(?:just |now |and |but |so )?(?:do|use|write|make|skip|try|give|switch|go|keep|let's|lets|show|answer|tell|explain|help|run|add|change|rename|leave|stick|focus|continue)\b/i;
// The same after any other request: "forget my branch name, use main
// instead" (or "… and use main instead") names what to use in its place, so
// it changes the instructions too. "Forget that I use vim instead of emacs"
// says a fact: its "instead" is in the first clause, after no instruction.
// A new name to use is a change of instructions too: "forget the project's
// name, call it Atlas", "Forget Ben's name, call him Benny".
const RENAME = String.raw`(?:call|name|rename) (?:it|him|her|them|this|that)\b`;
const REPLACED = new RegExp(String.raw`\binstead\b|\brather than\b|^(?:(?:and|then|just|so|now|please)\s+)*${RENAME}`, 'i');
const USE_INSTEAD = new RegExp(String.raw`\s(?:and|then|but|so|just)\s+(?:(?:use|switch to|go with|stick with|try)\b.*\binstead\b|${RENAME})`, 'i');
// The Chinese side: "忘掉我的分支名，改用main", "…，用main代替", "…，叫他小王", "…，改叫咪咪".
const ZH_INSTEAD = /^\s*(?:就|请|那就|直接|还是|以后|之后)?(?:改用|换用|改成|换成|改为|改叫|改名|叫(?:他|她|它)们?|用.+?(?:来)?(?:代替|替代|取代|替换))/u;
// And in the next sentence: "Forget my branch name. Use main instead.",
// "forget my editor; I want you to use vim instead", "Call him Benny.", "…。改用main".
const NEXT_INSTEAD = new RegExp(String.raw`^\W*(?:(?:just|now|and|but|so|then|please|i want you to|i'd like you to)\s+)*(?:(?:use|switch to|go with|stick with)\b.*\binstead\b|${RENAME})`, 'iu');
// The service's narrower reference back (src/quotemem/ingestion/forget_requests.py):
// a reference only the plugin reads is a literal statement to the service.
const SERVICE_REFERENCE = /^(?:(?:it|this|that|those|these)(?: information| detail| details)?|(?:what|everything|all|anything|whatever) (?:that )?i (?:just |have |'ve |had )?(?:said|told you|mentioned|shared|wrote|typed)(?: (?:earlier|before|just now))?)\W*$/i;
// "Forget that I said that": a reference back, not a fact.
const SAID_BACK = /^i (?:just |ever |also )?(?:said|told you|mentioned|wrote|typed)(?:\s+(?:that|it|this|those|these|so|earlier|before|just now|anything))*$/i;
const SAID = String.raw`i (?:just |have |'ve |had )?(?:said|told you|mentioned|shared|wrote|typed|sent)(?: (?:earlier|before|just now))?`;
const REFERENCE = new RegExp(String.raw`^(?:(?:it|this|that|those|these)(?: information| detail| details)?|(?:what|everything|all|anything|whatever) (?:that )?${SAID}|(?:the |my )?(?:last|previous) (?:thing|things|sentence|message)s? (?:that )?${SAID}|my (?:last|previous) (?:message|sentence))${W}*$`, 'iu');
// Instructions, plans and code are not facts about the user: "forget my
// previous instructions" or "forget the approach I suggested" asks for
// another way of working, not for a memory to be deleted.
const NOT_A_FACT = /^(?:(?:my|our|your|the|all|any|these|those|this|that)\s+)?(?:(?:previous|prior|earlier|last|above|old|original|initial|first|current|other|whole|entire)\s+)*(?:instructions?|approach(?:es)?|suggestions?|plans?|prompts?|ideas?|requests?|directions?|rules?|commands?|steps?|tasks?|attempts?|drafts?|versions?|changes?|edits?|code|solutions?|guidelines?|context|implementations?|designs?|branch(?:es)?|functions?|query|queries|scripts?|schemas?|prs?|pull requests?|commits?)\b/i;
// The same in Chinese ("忘掉我的代码，重新开始").
const ZH_NOT_A_FACT = /^我们?的(?:代码|方案|分支|实现|函数|脚本|提交|查询|架构|改动|修改|草稿|接口|PR)/u;
// What was said, as a whole ("my questions", "that stuff", "our chats"), is
// no fact; with what it was about ("the things I told you about my ex", "my
// messages to my ex") it is one.
// "Forget the stuff we discussed", "forget the things from our last session":
// starting the work over, not a fact (SAID_HEAD with "we", "our" or "you"
// and no "I", "me" or "my" after it). "Forget the conversation we had about
// my divorce" is one.
const SAID_HEAD = /^(?:(?:the|that|this|those|these|all|our|your|my)\s+){0,2}(?:(?:previous|prior|earlier|last|old|whole|entire|little|other)\s+)*(?:stuff|things?|conversations?|chats?|discussions?|sessions?|talks?|everything)\b/i;
const CONVERSING = /\b(?:we|we've|we'd|us|our|you|your)\b/i;
// A clause or the next sentence that starts over makes a forget about no fact
// of the user's a dismissal: "…, let's start fresh", "Forget the stuff about
// the movie. Let's talk about work.". After a fact of the user's (I, me, my;
// 我) it only closes the request, and is no part of what is forgotten:
// "Please forget that I'm pregnant, let's move on.", "忘掉我怀孕了，说点别的吧".
// "Never mind" and 算了 withdraw the request itself (NEVERMIND, ZH_NEVERMIND).
// The closing clause is cut off only when it is the last clause and goes on
// with nothing (RESET_GOES_ON, ZH_RESET_GOES_ON): "Forget that I have HIV,
// let's move on, my partner doesn't know." keeps the whole object.
const RESET = /^\W*(?:(?:ok(?:ay)?|so|and|now|then|just|alright)[\s,]+)*(?:let'?s|let us|start (?:over|fresh|again|anew)|starting (?:over|fresh)|moving on|move on|never ?mind|new topic|different topic|change (?:of )?(?:the )?(?:topic|subject))\b/iu;
const RESET_GOES_ON = /[,;:—–、]|\s-\s|\b(?:because|since|as|so|but|cause|cuz|though|although)\b/iu;
const ZH_RESET_GOES_ON = /[，,；;、]|因为|由于|所以|我|还|但/u;
const NEVERMIND = /^\W*(?:(?:ok(?:ay)?|so|and|oh|just|actually)[\s,]+)*never ?mind\b/iu;
const ZH_NEVERMIND = /^\P{L}*(?:那|好|好吧|那就|行)?算了/u;
const ZH_RESET = /^\P{L}*(?:那|好|好吧|那就|行)?(?:我们|咱们)?(?:重新开始|从头开始|从头来|换个话题|换一个话题|说点别的|聊点别的|聊别的|说别的|算了)/u;
// "忘掉我们聊过的东西", "忘掉我们之前说的": as SAID_HEAD.
const ZH_SAID_RESET = /^(?:我们|咱们)(?:之前|刚才|上次|昨天|今天|刚刚)?(?:聊|说|讨论|谈)(?:过|了)?的(?:东西|内容|事情?|话|那些|这些)?$/u;
const SAID_WHOLE = /^(?:(?:my|our|your|the|all|any|these|those|this|that)\s+){0,2}(?:(?:previous|prior|earlier|last|above|old|other|whole|entire|little|stupid|silly|dumb)\s+)*(?:questions?|messages?|answers?|responses?|replies|stuff|things|mistakes?|chats?|conversations?|discussions?)$/i;
const TOLD = /^(?:the|that|this|any|all|those|these)\s+(?:detail|details|fact|facts|information|info|preference|preferences|memory|memories|thing|things|part|stuff|bit)\s+(?:about|that|of|regarding|concerning|on)\b/i;
const CLAUSE = /\s*[,;:—–]\s*|\s+-\s+/;
const KNOWN = /^(?:that\s+)?you(?:'ve| have)?\s+(?:remember(?:ed)?|know|knew|noted|learned|stored|saved)\s+(?:that\s+|about\s+)?/i;
const SUBJECT_THAT = /^that\s+(?=(?:i|i'm|i've|i'd|my|me|we|our|it's|it|there)\b)/i;
// Forms only the plugin reads (inChat false). A fact about "we" or "our"
// that names at least two content terms ("our wedding anniversary date", not
// "our differences"), and not a reference such as "what we discussed".
const PLURAL = /\b(?:we|we're|we've|we'd|we'll|us|our|ours|ourselves)\b/i;
const PLURAL_MIN_TERMS = 2;
const VAGUE = /^(?:what|everything|all|anything|whatever|how|where|when|why)\b/i;
// "Forget our fight last night", "forget our little argument": letting a
// quarrel go, not a fact to delete ("forget my fight with cancer" is one).
const DISPUTE = /^(?:(?:our|the|that|this|those|these|all)\s+)?(?:(?:little|small|stupid|silly|dumb|big|huge|last|earlier|recent|whole|petty|awkward)\s+)*(?:fights?|arguments?|disagreements?|quarrels?|spats?|tiffs?|falling[- ]outs?|misunderstandings?|differences|beef|drama)\b/i;
// A person's (or a pet's) name or personal detail without "my": "Ben's
// name", "Amy's phone number", "the dog's name", "the kids' names". After
// "the" (or "this", "that"), the owner is a person or an animal (ROLES_EN,
// PEOPLE_EN) or a capitalised name ("the Smiths' address"), never a thing
// ("the function's name", "the server's address"). Without it, any word but
// a pronoun or an acronym ("API's address").
const DETAIL = String.raw`(?:(?:full|first|last|real|middle|maiden|nick|pet)\s*)?(?:names?|nicknames?|surnames?)|birthdays?|birth ?dates?|(?:(?:phone|cell|mobile)\s+)?numbers?|phone|address(?:es)?|e-?mails?(?:\s+address(?:es)?)?|ages?`;
const NAMED = new RegExp(String.raw`^(?:(?<det>the|this|that|these|those)\s+)?(?<owner>\p{L}[\p{L}\p{M}.-]*?)(?:'s|s')\s+(?:${DETAIL})$`, 'iu');
const NOT_OWNER = new Set('it that this what who there here let he she one everyone everybody someone somebody anyone anybody'.split(' '));
// Software, services and machines are not people: "forget Docker's address",
// "forget main's address", 忘掉Redis的地址.
const TECH = new Set(('docker kubernetes k8s python java javascript typescript node nodejs deno bun npm yarn pnpm pip redis postgres postgresql '
  + 'mysql mariadb mongo mongodb sqlite nginx apache caddy linux ubuntu debian fedora centos windows macos mac ios android git github gitlab '
  + 'bitbucket aws azure gcp google chrome firefox safari edge slack discord teams zoom jira confluence notion figma react vue angular svelte '
  + 'django flask fastapi rails ruby rust go golang php swift kotlin scala bash zsh helm terraform ansible jenkins elasticsearch kafka rabbitmq '
  + 'vercel netlify heroku cloudflare openai claude chatgpt gpt gemini llama main master dev develop prod production staging server servers '
  + 'api app apps service website site localhost database db cluster vm container repo branch').split(' '));
const PEOPLE_EN = new Set(('boy girl guy man woman lady person people kids children baby babies client customer patient student classmate '
  + 'tutor coach mentor nanny babysitter vet bride groom stepson stepdaughter stepmother stepfather kitty pony turtle lizard snake').split(' '));
// "From now on, please don't remember that I smoke", "以后不要记得我恐高了":
// the request after its time frame.
const LATER = /^\s*(?:(?:from now on|going forward|in (?:the )?future|henceforth|from here on(?: out)?)\b[\s,]*|(?:以后|今后|之后|往后|从今以后|从今往后|从现在起|从现在开始)[\s，,]*)(?=\S)/iu;
// A clear replacement after the request, in the last clause of its own
// sentence, is not forgotten with it (replaces()): the clause restates the
// same thing (an address, a name, a number, an email: VALUE_NOUN) with a new
// value, and says nothing else. The value is one or more words that each
// start with a digit, hold an @ or are in title case (VALUES), say no "not"
// (VALUE_NEGATED), and are followed
// at most by "now":
// - "my new <the same thing> is …" or "my <the same thing> is now …"
//   (NEW_SAME, SAME_NOW: it shares a content term or a kind (KINDS) with
//   what is forgotten; "my new one" after "my old <thing>" (OLD_NP)):
//   "forget my number, my new number is 555";
// - "it's … now" after "my old <thing>" (IT_NOW): "forget my old address,
//   it's 27 Birch Lane now";
// - a new place to live after an address or a past place (ADDRESS_NP,
//   LIVED; MOVED_TO): "forget my old address, I live at 27 Birch Lane now",
//   "forget that I lived in Leeds, I live in York now";
// - what a name is now (NAME_WORD; RENAMED_NOW, RENAMED_NEW): "forget my
//   cat's name, it's called Max now", "…, its new name is Max".
// Anything else is forgotten with it: a state ("my number is now unlisted",
// "it's 3 weeks old now", "I'm a patient now"), a clause that only
// describes the thing or the person ("I'm currently on parole"), one that
// goes on after the value ("…, the old one was 444-1234", "… now and I'm
// scared", "…, I used to be called Sarah"), and one that points back ("I
// moved there in 2019"). "Old", "prior" or "ex" elsewhere than before a noun
// of the user's ("a prior conviction", "my ex hit me") is no old value.
const VALUE = String.raw`(?:\d[\p{L}\p{N}./#-]*|[\p{L}\p{N}._%+-]*@[\p{L}\p{N}.-]+|\p{Lu}\p{Ll}[\p{Ll}\p{M}'.-]*)`;
// A value says no "not" ("it's NOT 12 Oak Street now").
const VALUE_NEGATED = /\b(?:not|no|never|except|instead)\b|n't\b/i;
// Case-sensitive: a capital starts a value. The patterns below capture what
// follows their own words as `value`, which VALUES must match whole.
const VALUES = new RegExp(String.raw`^${VALUE}(?:\s+${VALUE}){0,5}(?:\s+now)?[^\p{L}\p{N}]*$`, 'u');
const OLD_NP = /^(?:my|our)\s+(?:old|previous|former|prior|ex-?)\s*[\p{L}'-]+(?:\s+[\p{L}'-]+){0,2}$/iu;
const NEW_SAME = new RegExp(String.raw`^(?:now[\s,]+)?(?:my|our|the) new (?<noun>[\p{L}'-]+(?:\s+[\p{L}'-]+)?) (?:is|are|will be)\s+(?<value>\S.*)$`, 'iu');
const SAME_NOW = new RegExp(String.raw`^(?:my|our) (?<noun>[\p{L}'-]+(?:\s+[\p{L}'-]+)?) (?:is|are) now\s+(?<value>\S.*)$`, 'iu');
const IT_NOW = new RegExp(String.raw`^(?=.*\bnow\b)(?:now[\s,]+)?(?:it's|it is)\s+(?:now\s+)?(?<value>\S.*)$`, 'iu');
// Only a value is restated so: an address, a name, a number, an email.
const VALUE_NOUN = /\b(?:address|home|place|flat|apartment|house|email|e-mail|mail|name|nickname|surname|number|phone|cell|mobile|username|handle)\b/i;
const KINDS = [['address', 'home', 'place', 'flat', 'apartment', 'house'], ['email', 'e-mail', 'mail'], ['name', 'nickname']];
const ADDRESS_NP = /^(?:my|our)\s+(?:(?:old|previous|former|prior|home|current|street|mailing|postal)\s+)*(?:address|home|place|flat|apartment|house)$/i;
const LIVED = /^(?:i|we) (?:lived|used to live|was living|were living)\b/i;
const MOVED_TO = new RegExp(String.raw`^(?:now[\s,]+)?(?:i|we)(?:'m|'re| am| are|'ve| have)?(?:\s+(?:now|just|recently|currently))?\s+(?:live|living|moved|relocated)\s+(?:(?:now|just|recently|currently)\s+)?(?:at|in|on|to|into|near)\s+(?:the\s+)?(?<value>\S.*)$`, 'iu');
const NAME_WORD = /\b(?:nick|sur|dead|maiden)?names?\b|\bcalled\b/i;
const RENAMED_NOW = new RegExp(String.raw`^(?=.*\bnow\b)(?:now[\s,]+)?(?:it|he|she|they|i|we)(?:'s| is|'re| are|'m| am)\s+(?:now\s+)?(?:called|named|known as)\s+(?<value>\S.*)$`, 'iu');
const RENAMED_NEW = new RegExp(String.raw`^(?:now[\s,]+)?(?:its|his|her|their|my|our) (?:new (?:nick)?names? (?:is|are)|(?:nick)?names? (?:is|are) now)\s+(?<value>\S.*)$`, 'iu');
// A clause after the replacement, or a word that goes on, keeps it in.
const BOUNDARY = /[,;:—–、]|\s-\s/u;
const SUBORDINATE = /^(?:that\s+)?(?:when|if|because|since|although|though|while|after|before|as|until|once|whenever)\b/i;
const CLAUSES = new RegExp(CLAUSE.source, 'g');

const ZH = new RegExp(String.raw`^${W}*(?:请|麻烦|能不能|可以)?(?:你)?(?:帮我)?(?:(?<verb>忘记|忘掉|(?:不要|别|不用)(?:再)?(?:记住|记得|记录|保存|记(?![错混反岔乱])))(?<object>.*)|(?:删除|删掉|清除)(?<deleted>.*?)(?:的)?(?:记忆|记录))${W}*$`, 'su');
const ZH_BA = new RegExp(String.raw`^${W}*(?:请|麻烦)?(?:你)?(?:帮我)?把(?<object>.+?)(?:给)?(?<verb>忘掉|忘记|忘了|删掉|删除|清除)(?:吧|啊|哈|呢|掉)?(?:[，,]\s*(?<rest>[^，,].*?))?${W}*$`, 'su');
const ZH_EDGES = ' \t，,：:。！？.!?；;、';
const ZH_REFERENCE = /^我?(?:刚才|刚刚|之前|前面|上面|方才)?(?:说|讲|提|写|告诉你)(?:过)?的(?:话|内容|那些|这些|事|东西|那句话?|这句话?)?$/;
// "忘掉我刚才问的" without what was asked: never mind the question.
const ZH_ASKED_BARE = /^我?(?:刚才|刚刚|之前|前面|上面|方才)?(?:问|提问)(?:过)?(?:你)?的(?:话|问题|那个问题|这个问题|事|东西|那些|这些)?$/;
// Another person's or a pet's name or detail ("小李的名字", "Emma的名字",
// "他的生日", "老师的名字"), or a person called by 阿 and one character ("阿杰"),
// or by 小 or 老 and a common surname ("小李", "老王"). The owner is a person
// or an animal (ROLES_ZH, ZH_PEOPLE, a pronoun), such a name, or a
// capitalised Latin name; never a thing ("公司的地址", "这个函数的名字"), a word
// such as 阿姨 or 阿里, or 小三, 老歌. Read only by the plugin (inChat false).
const ZH_NAMED = /^(?<owner>[^\s，,。；;、：:！？!?我]{1,10})的(?:名字|姓名|全名|名|昵称|外号|小名|绰号|英文名|中文名|电话|电话号码|手机号|手机号码|生日|地址|住址|年龄|邮箱)$/u;
const ZH_SURNAMES = '王李张刘陈杨黄赵吴周徐孙朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘蒋蔡杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范廖贾夏韦邹孟熊秦邱尹薛段雷侯陶贺顾郝龚邵严覃武戴莫孔汤';
const ZH_NAME = new RegExp(String.raw`^(?:阿\p{Script=Han}|[小老][${ZH_SURNAMES}])$`, 'u');
const ZH_A_WORDS = new Set('阿姨 阿婆 阿公 阿爸 阿妈 阿哥 阿嫂 阿婶 阿伯 阿叔 阿爷 阿嬷 阿妹 阿弟 阿姐 阿里 阿胶 阿片 阿拉 阿门'.split(' '));
const ZH_PEOPLE = ['男友', '女友', '前任', '闺蜜', '兄弟', '姐妹', '同学', '客户', '领导', '经理', '教练', '导师', '宝宝', '小孩', '狗狗', '猫咪'];
const ZH_PRONOUN = /^(?:他|她|它|TA|ta)们?$/u;
const LATIN_NAME = /^\p{Lu}\p{Ll}+(?:[ -]\p{Lu}\p{Ll}+)?$/u;
const ZH_PARTICLES = /(?:吧|啊|呀|哈|呢)+$/u;
// "(I/you) forgot …", not a request: 忘记 + "…的<a noun>了" or "我家…了"
// (then 吗, 吧, 呢 or 怎么办) without 请 or 帮我 ("忘记我的密码了", "忘记我家地址了吗");
// 忘记了 or 忘掉了 ("忘记了我的生日"); and an accusation of "you": 你(…)忘了…,
// 你 + 也/又/已经/是不是… + 忘, or 你忘记/忘掉… + 了吗/了没, or + 了吧 after a
// noun phrase ("你忘了我的生日吧", "你忘掉我的生日了吗", "你忘记我的生日了吧").
// "你忘掉我的地址吧" and "你忘掉我离婚了吧" ask; so does "忘掉我离婚了": a
// sentence-final 了 is ordinary Chinese.
const ZH_POLITE = /^\P{L}*(?:请|麻烦|能不能|可以|帮我)/u;
const ZH_FORGOT = /^忘记$/u;
const ZH_FORGOT_NP = /^(?:[^，,。；;！？!?]*的\p{Script=Han}{1,6}|我们?家\p{Script=Han}{0,6})了(?:吗|吧|呢|怎么办)?$/u;
const ZH_YOU_FORGOT = /^\P{L}*你(?:(?:也|又|还|都|已经|是不是|是否|是|真的|竟然)(?:忘记|忘掉|忘了)|忘了)/u;
const ZH_YOU_ASKED = /^\P{L}*你(?:忘记|忘掉)/u;
const ZH_ASKED_END = /了(?:吗|没有?)\P{L}*$/u;
const ZH_GUESS_END = /了吧\P{L}*$/u;
// "忘了我的地址吧": a request (忘了带钥匙 is not one); "你能不能别记住…".
const ZH_LE = new RegExp(String.raw`^${W}*(?:请|麻烦)?(?:帮我)?忘了(?<object>.+?)吧${W}*$`, 'su');
const ZH_ASK = /^\s*你(?:能不能|可不可以|能否)(?=别|不要|不用|忘|把|删)/u;
// "你可以忘掉我的生日吗" asks; "你可以忘掉我的生日" may only allow it.
const ZH_ASK_KEYI = /^\s*你可以(?=别|不要|不用|忘掉|忘记|把|删)/u;
const ZH_MA_END = /吗(?=[^\p{L}\p{N}]*$)/u;
// 删掉, 删除 or 清除 asks to forget only what is named as a memory ("删除关于我
// 生日的记忆", "把你记住的我的地址删掉"), never work ("把我的分支删掉",
// "删除我的提交记录").
const ZH_MEMORY_HEAD = /^你?(?:记住|记得|记下|保存)的/u;
const ZH_MEMORY_TAIL = /(?:的)?(?:记忆|记录)$/u;
const ZH_WORK = /文件|分支|代码|目录|提交|日志|缓存|配置|测试|仓库|项目|数据库|表格|数据表|表单|镜像|聊天|浏览|通话|搜索|下载|订单|表$/u;
// "忘掉我们昨晚吵架的事吧": letting a quarrel go (DISPUTE).
const ZH_DISPUTE = /^(?:我们|咱们|我和你|你和我)(?:之间)?(?:昨天|昨晚|今天|上次|那次|刚才|之前)?(?:的)?(?:吵架|争吵|吵的架|矛盾|不愉快|误会|过节|别扭|争执)(?:的事情?|这件事|那件事)?$/u;
// A clear replacement after the request (replaces(), as in English), in the
// last clause and with no word that goes on (ZH_CONTINUES: "…，以前叫小红",
// "…，我现在还住在杭州"): the clause gives 新<the same thing> (ZH_NEW_SAME: a
// character pair of terms() or a kind (ZH_KINDS) it shares: "忘掉我原来的手机号，
// 新号码是123", "…，它的新名字是咪咪"); or a new place to live after an address
// or a past place (ZH_ADDRESS_NP, ZH_LIVED; ZH_MOVED_TO: "…，我现在住在银杏路6号");
// or the target is a name and the clause says what it is called now
// (ZH_RENAMED: "…，它现在叫咪咪", "忘掉我的名字，我现在叫小明").
const ZH_NEW_SAME = /^(?:(?:我|它|他|她)们?的)?新(?:的)?(?<noun>\p{Script=Han}{1,4}?)(?:是|为|叫|：|:)/u;
const ZH_VALUE = /地址|住址|住处|名字|称呼|号码|手机|电话|邮箱|账号/u;
const ZH_KINDS = [['手机', '电话', '号码'], ['地址', '住址', '住处'], ['名字', '称呼', '叫']];
const ZH_ADDRESS_NP = /^我们?的?(?:旧|以前的|原来的|之前的|原先的|从前的)?(?:家庭?|家的)?(?:地址|住址|住处|家)$/u;
const ZH_LIVED = /^我们?(?:以前|之前|原来|原先|从前|曾经)(?:住在|住过)/u;
const ZH_MOVED_TO = /^(?:现在|目前|如今)?(?:我们?)?(?:现在|目前|如今|已经|刚刚?)?(?:住在|搬到|搬去|搬进|搬来|住到)(?![那这])\S/u;
const ZH_NAME_WORD = /名|叫/u;
const ZH_RENAMED = /^(?:(?:我|它|他|她)们?(?:现在|如今)(?:改)?叫|(?:现在|如今)(?:它|他|她)?们?(?:改)?叫|(?:(?:我|它|他|她)们?)?现在的名字(?:是|叫))\S/u;
const ZH_BOUNDARY = /[，,；;、]/u;
const ZH_CONTINUES = /以前|旧|原来|原先|之前|从前|曾经|但|因为|由于|所以|而且|不过|可是|然后|并且|和|跟|一起|为了|还|也|不是/u;
const ZH_SUBORDINATE = /^(?:当|如果|要是|因为|虽然|自从)/u;
const ZH_CLAUSES = /\s*[，,；;]\s*/gu;

const fold = (text) => text.replace(/[’ʼ]/g, "'");

// A list marker before a request ("③ Please forget …", "1) …", "1）…",
// "1．…", "Q1) …", "a)Please …", "a. …", "iv) …", "• …", "（一）请忘掉…",
// "一、…", "一）…", "第一，…"): enclosed or circled numbers; digits (after at
// most two letters) with "." or ")"; a letter or a roman numeral with ")",
// or with "." and a space; any of these in brackets; a bullet; a Chinese
// ordinal; then spaces and punctuation. Brackets and stops may be full width.
// A number that is part of what is said ("2 cats …", "1.5 …") has no "." or
// ")" after it and stays, and so does a word made of roman letters that is
// not a numeral from i to xxxix in one case ("Mix) …").
const ENCLOSED = '\u2460-\u24ff\u2776-\u2793\u3251-\u325f\u32b1-\u32bf';
const ZH_NUMBER = '一二三四五六七八九十百零〇两';
const ROMAN = '(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})|(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})';
const ENUMERATOR = new RegExp(String.raw`^\s*(?:[${ENCLOSED}]`
  + String.raw`|[(（](?:\d{1,3}|[A-Za-z]|${ROMAN}|[${ZH_NUMBER}]{1,4})[)）]`
  + String.raw`|(?:[A-Za-z]|${ROMAN})(?:[)）]|[.．](?=\s))`
  + String.raw`|[A-Za-z]{0,2}\d{1,3}[.)、．）](?!\d)`
  + String.raw`|[${ZH_NUMBER}]{1,4}[、.．)）]`
  + String.raw`|第[${ZH_NUMBER}\d]{1,4}(?:[条点项个步])?`
  + String.raw`|[•·◦▪‣●○■□\-*–—])[\s\p{P}]*`, 'u');

/** A sentence without a list marker at its start (ENUMERATOR). */
export function withoutEnumerator(sentence) {
  const text = String(sentence ?? '');
  const match = ENUMERATOR.exec(text);
  return match && match[0].length < text.length ? text.slice(match[0].length) : text;
}

function strip(text, chars) {
  let start = 0;
  let end = text.length;
  while (start < end && chars.includes(text[start])) start += 1;
  while (end > start && chars.includes(text[end - 1])) end -= 1;
  return text.slice(start, end);
}

function bounded(text) {
  const t = String(text).split(/\s+/).filter(Boolean).join(' ');
  if (t.length <= MAX_FORGET_STATEMENT_CHARS) return t;
  const cut = t.slice(0, MAX_FORGET_STATEMENT_CHARS);
  return cut.includes(' ') ? cut.slice(0, cut.lastIndexOf(' ')) : cut;
}

// The kind of a text (KINDS, ZH_KINDS): the index of the first group it
// names, else -1.
function kindOf(text, zh) {
  if (zh) return ZH_KINDS.findIndex((words) => words.some((word) => text.includes(word)));
  const have = terms(text);
  return KINDS.findIndex((words) => words.some((word) => have.has(word)));
}

// Whether a clause after a request clearly replaces what it forgets (OLD_NP
// … RENAMED, ZH_NEW_SAME … ZH_RENAMED above), in English or (zh) Chinese.
function replaces(kept, rest, zh) {
  if (zh) {
    if (ZH_BOUNDARY.test(rest) || ZH_CONTINUES.test(rest)) return false;
    const noun = ZH_NEW_SAME.exec(rest)?.groups.noun;
    const same = noun !== undefined && ZH_VALUE.test(noun) && (covered(terms(noun), terms(kept)) > 0 || (kindOf(noun, true) >= 0 && kindOf(noun, true) === kindOf(kept, true)));
    return same
      || ((ZH_ADDRESS_NP.test(kept) || ZH_LIVED.test(kept)) && ZH_MOVED_TO.test(rest))
      || (ZH_NAME_WORD.test(kept) && ZH_RENAMED.test(rest));
  }
  if (BOUNDARY.test(rest)) return false;
  const valued = (pattern) => {
    const match = pattern.exec(rest);
    return match !== null && VALUES.test(match.groups.value) && !VALUE_NEGATED.test(match.groups.value) ? match : null;
  };
  const noun = (valued(NEW_SAME) ?? valued(SAME_NOW))?.groups.noun;
  const one = /^one$/i.test(noun ?? '');
  const value = one ? OLD_NP.test(kept) && VALUE_NOUN.test(kept) : VALUE_NOUN.test(noun ?? '');
  const same = noun !== undefined && value && (one || covered(terms(noun), terms(kept)) > 0
    || (kindOf(noun, false) >= 0 && kindOf(noun, false) === kindOf(kept, false)));
  return same
    || (OLD_NP.test(kept) && VALUE_NOUN.test(kept) && valued(IT_NOW) !== null)
    || ((ADDRESS_NP.test(kept) || LIVED.test(kept)) && valued(MOVED_TO) !== null)
    || (NAME_WORD.test(kept) && (valued(RENAMED_NOW) ?? valued(RENAMED_NEW)) !== null);
}

// What a request asks to forget, without a clear replacement in a later
// clause of its own (replaces()): the words before the first such clause,
// else all of them.
function beforeNewFact(target, zh) {
  if ((zh ? ZH_SUBORDINATE : SUBORDINATE).test(target)) return target;
  // A sentence in capitals tells no value from a word: never cut.
  if (/\p{Lu}/u.test(target) && !/\p{Ll}/u.test(target)) return target;
  for (const boundary of target.matchAll(zh ? ZH_CLAUSES : CLAUSES)) {
    const kept = target.slice(0, boundary.index).trim();
    if (kept && replaces(kept, target.slice(boundary.index + boundary[0].length), zh)) return kept;
  }
  return target;
}

// Whether the owner of a name or detail without "my" (NAMED) is a person or
// an animal.
function namedOwner(clause) {
  const match = NAMED.exec(clause.trim());
  if (!match) return false;
  const { det, owner } = match.groups;
  const lower = owner.toLowerCase();
  if (NOT_OWNER.has(lower) || TECH.has(lower) || /^\p{Lu}{2,}$/u.test(owner)) return false;
  if (ROLES_EN.has(lower) || PEOPLE_EN.has(lower)) return true;
  return !det || LATIN_NAME.test(owner);
}

// Whether a Chinese owner (ZH_NAMED) or a bare word is a person or an animal.
function zhName(word) {
  return ZH_NAME.test(word) && !ZH_A_WORDS.has(word);
}

// Whether what a request forgets is about the user or someone of theirs:
// I, me, my (SPEAKER) or 我; a name or detail without "my" (namedOwner,
// zhOwner, zhName); "we" or "our" with a fact (PLURAL).
function personal(text) {
  const value = String(text).trim();
  const named = ZH_NAMED.exec(value.replace(ZH_PARTICLES, ''))?.groups.owner;
  return SPEAKER.test(value) || value.includes('我') || namedOwner(value) || zhName(value.replace(ZH_PARTICLES, ''))
    || (named !== undefined && zhOwner(named))
    || (PLURAL.test(value) && !VAGUE.test(value) && terms(value).size >= PLURAL_MIN_TERMS);
}

function zhOwner(owner) {
  return ZH_PRONOUN.test(owner) || zhName(owner) || (LATIN_NAME.test(owner) && !TECH.has(owner.toLowerCase()))
    || [...ROLES_ZH, ...ZH_PEOPLE].some((role) => owner.endsWith(role));
}

function objectOf(raw) {
  let value = raw.trim().replace(TRAILING, '');
  value = value.replace(TRAILING_MEMORY, '').replace(TRAILING, '').trim().replace(KNOWN, '');
  return value.replace(SUBJECT_THAT, '').trim();
}

// The object of an English request sentence, whether it refers back and
// whether the service's in-chat check reads it the same way, else null.
function english(sentence) {
  const text = fold(sentence).trim();
  const namesMemory = NAMES_MEMORY.test(text);
  let verb;
  let match = FORGET.exec(text);
  if (match) verb = 'forget';
  else if ((match = REMEMBER.exec(text))) verb = 'remember';
  else if ((match = RECORD.exec(text))) verb = 'record';
  else if ((match = WANT_NOT.exec(text))) verb = 'want';
  else return null;
  const politeText = (match.groups.polite ?? '').trim().toLowerCase();
  if (text.endsWith('?') && !/^(can|could|would)/.test(politeText)) return null;
  const polite = politeText.length > 0;
  if (verb === 'forget' && match.groups.verb.toLowerCase().endsWith('about') && !(polite || namesMemory)) return null;
  const verbText = match.groups.verb;
  // Forms the service's in-chat check does not read: "don't ever remember",
  // "could you please not remember", "I don't want you to remember …".
  const serviceForm = verb !== 'want' && !/\bever\b|^not\b/i.test(verbText);
  // "Can you not remember my name? I told you twice." is a complaint: only
  // "… please not remember …" asks.
  if (verb === 'remember' && /^not\b/i.test(verbText) && !/\bplease\b/.test(politeText)) return null;
  const object = match.groups.object ?? '';
  let target = objectOf(object);
  if ((verb === 'remember' || verb === 'want') && !ASKS_THAT.test(object.trim()) && (WANT_REFUSED.test(object) || REMEMBER_ALL.test(object.trim()))) return null;
  if (verb === 'want' && !(WANT_OBJECT.test(object.trim()) && !WANT_THING.test(target) && target.split(CLAUSE).length === 1)) return null;
  // A bare "don't remember" or "never remember" may be the user's own "(I)
  // don't remember …": it asks only when polite, naming the memory, or
  // marked as a request (ASKS_THAT … RECALL).
  if (verb === 'remember' && /^(?:do not|don't|dont|never)\b/i.test(verbText) && !(polite || namesMemory)) {
    const back = SAID_BACK.test(target) || REFERENCE.test(target);
    const shouted = !/\p{Ll}/u.test(text);
    const marked = /^never\b/i.test(verbText)
      ? PLEASE.test(text)
      : (!back && ASKS_THAT.test(object.trim())) || (!shouted && (/^do not\b/i.test(verbText) || EMPHASIS.test(verbText))) || CLOSING.test(text);
    if (!marked || RECALL.test(object)) return null;
  }
  // A clause that starts over (RESET): see above.
  let closed = false;
  for (const boundary of target.matchAll(CLAUSES)) {
    const rest = target.slice(boundary.index + boundary[0].length);
    if (!RESET.test(rest)) continue;
    const before = target.slice(0, boundary.index).trim();
    if (NEVERMIND.test(rest) || !personal(before) || SAID_BACK.test(before) || REFERENCE.test(before) || RETRACTION.test(before)) return null;
    if (!RESET_GOES_ON.test(rest.slice(RESET.exec(rest)[0].length))) {
      target = before;
      closed = true;
    }
    break;
  }
  const head = SAID_HEAD.exec(target);
  if (head && !SPEAKER.test(target.slice(head[0].length)) && CONVERSING.test(target)) return null;
  if (RETRACTION.test(target)) {
    // "Forget that I asked you for X" names X; "forget (what) I asked",
    // "forget that I asked, just answer" and a change of instructions
    // ("forget that I asked you to use tabs; use spaces") do not.
    const [asked, ...after] = target.split(CLAUSE).map((part) => part.trim());
    const rest = ASKED.exec(asked)?.groups.rest ?? '';
    if (/^what\b/i.test(asked) || terms(rest).size < ASKED_MIN_TERMS) return null;
    if (after.some((part) => INSTEAD.test(part))) return null;
    if (ASKED_TO.test(rest) && !((polite || namesMemory) && SPEAKER.test(rest))) return null;
    return { target: asked, back: false, inChat: false };
  }
  if (SAID_BACK.test(target)) return { target: '', back: true, inChat: false };
  if (verb === 'forget' && !namesMemory && DISMISSAL.test(target)) return null;
  if (REFERENCE.test(target)) return { target: '', back: true, inChat: serviceForm && SERVICE_REFERENCE.test(target) };
  if (NOT_A_FACT.test(target) || SAID_WHOLE.test(target) || DISPUTE.test(target)) return null;
  const [clause, ...after] = target.split(CLAUSE);
  if (after.some((part) => REPLACED.test(part)) || USE_INSTEAD.test(target)) return null;
  if (target.split(/\s+/).filter(Boolean).length < 2) return null;
  const own = namesMemory || SPEAKER.test(clause) || TOLD.test(clause);
  const plural = PLURAL.test(clause) && !VAGUE.test(clause) && terms(clause).size >= PLURAL_MIN_TERMS;
  if (!(own || plural || namedOwner(clause))) return null;
  const kept = beforeNewFact(target, false);
  return { target: kept, back: false, inChat: serviceForm && own && kept === target && !closed };
}

// The object of a Chinese request sentence, else null (as english()). The
// service reads the plain forms, not 把…忘掉, a bare 不要记 or a reference back.
function chinese(sentence) {
  const text = sentence.trim();
  const plain = ZH.exec(text);
  if (ZH_YOU_FORGOT.test(text)) return null;
  const match = plain ?? ZH_BA.exec(text) ?? ZH_LE.exec(text);
  if (!match) return null;
  let object = match.groups.object ?? match.groups.deleted ?? '';
  // 删掉, 删除, 清除: only a memory ("把你记住的我的地址删掉"), never work.
  const deletes = match.groups.deleted !== undefined || (match !== plain && /^[删清]/u.test(match.groups.verb ?? ''));
  if (deletes && match.groups.deleted === undefined) {
    object = strip(object, ZH_EDGES);
    if (!(ZH_MEMORY_HEAD.test(object) || ZH_MEMORY_TAIL.test(object))) return null;
    object = object.replace(ZH_MEMORY_HEAD, '').replace(ZH_MEMORY_TAIL, '');
  }
  if (deletes && ZH_WORK.test(object)) return null;
  // "忘记了我的生日": "(I) forgot my birthday".
  if (plain && /^(?:忘记|忘掉)$/u.test(plain.groups.verb ?? '') && /^了/u.test(object)) return null;
  let target = strip(object, ZH_EDGES);
  target = strip(target.replace(/^(?:关于|有关)/, ''), ZH_EDGES);
  // 把…忘掉 followed by a clause: only a clear replacement ("把我的旧地址忘掉，现在我住在杭州").
  if (match.groups.rest !== undefined && !replaces(target, match.groups.rest, true)) return null;
  // "忘记我的密码了": "(I) forgot my password"; "你忘记我的生日了吗", "你忘记我的生日了吧".
  if (plain && ZH_FORGOT.test(plain.groups.verb ?? '') && ZH_FORGOT_NP.test(target) && !ZH_POLITE.test(text)) return null;
  if (ZH_YOU_ASKED.test(text) && (ZH_ASKED_END.test(text) || (ZH_GUESS_END.test(text) && ZH_FORGOT_NP.test(target)))) return null;
  if (ZH_DISPUTE.test(target.replace(ZH_PARTICLES, ''))) return null;
  if (ZH_REFERENCE.test(target)) return { target: '', back: true, inChat: false };
  if (ZH_ASKED_BARE.test(target)) return null;
  let closed = false;
  for (const boundary of target.matchAll(ZH_CLAUSES)) {
    const rest = target.slice(boundary.index + boundary[0].length);
    if (!ZH_RESET.test(rest)) continue;
    const before = target.slice(0, boundary.index).trim();
    if (ZH_NEVERMIND.test(rest) || !personal(before) || ZH_REFERENCE.test(before)) return null;
    if (!ZH_RESET_GOES_ON.test(rest.slice(ZH_RESET.exec(rest)[0].length))) {
      target = before;
      closed = true;
    }
    break;
  }
  if (ZH_NOT_A_FACT.test(target)) return null;
  if (target.split(/[，,；;]/).slice(1).some((part) => ZH_INSTEAD.test(part))) return null;
  if (ZH_SAID_RESET.test(target.replace(ZH_PARTICLES, ''))) return null;
  if (target.includes('我') && [...target].length >= 3) {
    const kept = beforeNewFact(target, true);
    return { target: kept, back: false, inChat: Boolean(plain) && kept === target && !closed && !/(?:不要|别|不用)(?:再)?记$/.test(plain.groups.verb ?? '') };
  }
  // Another person, by name: "小李的名字", "阿杰" (a particle such as 吧 left out).
  const named = target.replace(ZH_PARTICLES, '');
  const owner = ZH_NAMED.exec(named)?.groups.owner;
  if ((owner !== undefined && zhOwner(owner)) || zhName(named)) return { target: named, back: false, inChat: false };
  return null;
}

/**
 * Find the forget requests of one user message.
 * @param {string} text - the user message.
 * @param {{previous?: string}} options - the previous user message, for a
 *   request that refers back to "what I said".
 * @returns {string[]} the statements to forget, in the user's words (at most
 *   four, each at most 1000 characters); empty when the message asks nothing.
 */
export function detectForget(text, options) {
  return detectForgetRequests(text, options).map((request) => request.statement);
}

/**
 * Find the forget requests of one user message, as detectForget, each with
 * whether the service's in-chat check reads it too (`inChat`): then the
 * service forgets it again from every batch stored before the request's own
 * turn, and a forget need not wait long for those batches. A request the
 * check does not read, after something its own message says
 * ("1. My locker code is HERON. 2) Please forget my locker code."), has
 * `ownTurn`: what the message says before it is stored with the message's
 * own batch, after the forget ran, so the forget runs again after that batch.
 * What the message says after the request is not forgotten when it says
 * nothing before.
 * @returns {{statement: string, inChat: boolean, ownTurn: boolean}[]}
 */
export function detectForgetRequests(text, { previous = '' } = {}) {
  const content = String(text ?? '');
  if (!content.trim() || isCommand(content)) return [];
  const found = [];
  let before = '';
  const pieces = sentences(content);
  for (const [index, piece] of pieces.entries()) {
    let hit = english(piece) ?? chinese(piece);
    // After a list marker ("③ Please forget …"), a time frame ("From now
    // on, …", "以后…") or 你能不能. The service's in-chat check does not read
    // such a request, so it is the plugin's alone.
    if (!hit) {
      const later = withoutEnumerator(piece).replace(LATER, '');
      const keyi = ZH_ASK_KEYI.exec(later);
      const body = keyi && ZH_MA_END.test(later) ? later.slice(keyi[0].length).replace(ZH_MA_END, '') : later.replace(ZH_ASK, '');
      const listed = body !== piece ? english(body) ?? chinese(body) : null;
      if (listed) hit = { ...listed, inChat: false };
    }
    const next = withoutEnumerator(pieces[index + 1] ?? '');
    if (hit && !hit.back && (NEXT_INSTEAD.test(next) || ZH_INSTEAD.test(next))) hit = null;
    if (hit && (RESET.test(next) || ZH_RESET.test(next))) {
      if (hit.back || !personal(hit.target) || NEVERMIND.test(next) || ZH_NEVERMIND.test(next)) hit = null;
    }
    let statement = null;
    // A reference back to the previous message names a turn the service's
    // check of this message does not hold.
    let inChat = Boolean(hit?.inChat);
    if (hit?.back) {
      statement = before.trim() || null;
      if (!statement) {
        statement = String(previous ?? '').trim() || null;
        inChat = false;
      }
    } else if (hit) {
      statement = hit.target;
    }
    if (statement) {
      const value = bounded(statement);
      const ownTurn = !inChat && before.trim().length > 0;
      const known = found.find((request) => request.statement === value);
      if (known) {
        known.inChat ||= inChat;
        known.ownTurn = !known.inChat && (known.ownTurn || ownTurn);
      } else if (value && found.length < MAX_FORGETS) found.push({ statement: value, inChat, ownTurn });
    }
    // What a later "forget what I just said" refers to: the sentences that
    // say something, not a request or a filler such as "Ignore that;".
    if (!hit && contentful(piece)) before += piece;
  }
  return found;
}

// Words that carry no fact of their own in a sentence before a request.
const FILLER = new Set(('ignore never mind nevermind wait sorry oops actually scratch cancel undo hmm um uh ok okay yes yeah yep nope '
  + 'right fine well hey oh so now then instead').split(' '));
const ZH_FILLER = /算了|不对|等等|等一下|没事|好吧|对不起|抱歉|不好意思|嗯|哦|噢|啊|呃/gu;

function contentful(piece) {
  for (const term of terms(String(piece).replace(ZH_FILLER, ''))) if (!FILLER.has(term)) return true;
  return false;
}

/** True when the message asks to forget something. */
export function isForgetRequest(text, options) {
  return detectForget(text, options).length > 0;
}

// --------------------------------------------------------------- the flow

/** The deterministic request id of one forget. */
// ------------------------------------------- forgets to apply again
// What the user asked to forget, statements only (redacted of secrets, but
// in the user's words), in ~/.quotemem/forgotten/<owner>.json (0600 in a
// 0700 folder, one file per user and service): a forget a hook ran, and one
// the model ran through the MCP tools, which run in another process. A
// batch queued later that holds the user's words from before a forget gets
// that forget again, right behind it (HookMemory.enqueue), so words saved
// after a forget never bring the fact back: a turn read late from a
// transcript, or a turn of this run that Cursor's CLI saves only when the run
// ends. Kept as long as a turn said before it can still wait unsaved in a
// session file: a week (SESSION_KEEP_MS); a session file older than that is
// closed and its turns queued.
export const FORGOTTEN_KEEP_MS = SESSION_KEEP_MS;
const MAX_FORGOTTEN = 200;

function homeOf(env) {
  return homeDir(env ?? {});
}

/**
 * The files of an owner's forgets to apply again: in QUOTEMEM_STATE_DIR when
 * the environment sets it, and in ~/.quotemem. A forget is kept in each, and
 * read from each: an MCP server given QUOTEMEM_STATE_DIR in its own config
 * and hooks without it (or the other way round) still share
 * ~/.quotemem/forgotten. Without QUOTEMEM_STATE_DIR there is one file.
 */
export function forgottenFiles(env, { url, userId }) {
  const name = `${ownerKey(url, userId)}.json`;
  const home = path.join(homeOf(env), '.quotemem', 'forgotten', name);
  const stateDir = String(env?.QUOTEMEM_STATE_DIR ?? '').trim();
  if (!stateDir) return [home];
  const own = path.join(path.resolve(homeOf(env), stateDir.replace(/^~(?=$|[\\/])/, homeOf(env))), 'forgotten', name);
  return own === home ? [home] : [own, home];
}

/** The first of forgottenFiles: QUOTEMEM_STATE_DIR's when set, else ~/.quotemem's. */
export function forgottenFile(env, owner) {
  return forgottenFiles(env, owner)[0];
}

function keptIn(file) {
  const items = readJson(file)?.items;
  const since = Date.now() - FORGOTTEN_KEEP_MS;
  return (Array.isArray(items) ? items : []).filter((item) => typeof item?.statement === 'string' && Number(item.at) > since);
}

/** An owner's forgets of the last FORGOTTEN_KEEP_MS, from every file, oldest first: [{statement, at}]. */
export function forgottenOf(env, owner) {
  const seen = new Set();
  const out = [];
  for (const item of forgottenFiles(env, owner).flatMap(keptIn)) {
    const key = JSON.stringify([item.statement, Number(item.at)]);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out.sort((a, b) => Number(a.at) - Number(b.at));
}

/**
 * Keep a forget the user asked for, in every file of forgottenFiles: `at`,
 * when they asked (a batch with the user's words from before it gets the
 * forget again).
 * @param {object} owner - {url, userId}: the settings' service and user.
 */
export async function noteForgotten(env, owner, statement, at = Date.now()) {
  const redacted = redactText(String(statement ?? ''));
  const text = bounded(redacted.text);
  if (redacted.secretOnly || !text || !Number.isFinite(at)) return;
  for (const file of forgottenFiles(env, owner)) {
    await withLock(`${file}.lock`, () => {
      const kept = keptIn(file).filter((item) => !(item.statement === text && item.at === at));
      writeFileAtomic(file, JSON.stringify({ version: 1, items: [...kept, { statement: text, at }].slice(-MAX_FORGOTTEN) }));
    }, { timeoutMs: 2000, staleMs: 15_000 });
  }
}

/**
 * A key of a forget request's words, which tells the outcomes of the two
 * forget paths for the same request apart from others: the plugin's forget
 * by statement keeps it with its result, and the service's in-chat outcome
 * of the request is given it (chatForgetsOf). A hash of the words folded,
 * lower-cased and without edge punctuation (a # or % right after a word stays:
 * "C#" is not "C"); no word is kept. Null for no words.
 */
export function requestKey(words) {
  const text = fold(String(words ?? '')).normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').replace(/^[\s\p{P}]+|(?:\s|(?![#%])\p{P}|(?<![\p{L}\p{N}_#%])[#%])+$/gu, '');
  return text ? sha16(['forget-request', text]) : null;
}

export function forgetRequestId(sessionId, turnKey, statement) {
  const suffix = `:forget:${sha16([String(turnKey ?? ''), statement])}`;
  return `${String(sessionId).slice(0, MAX_ID_CHARS - suffix.length)}${suffix}`;
}

/**
 * The prepare body of one forget, with the statement redacted.
 * @returns {object|null} null when the statement is only a secret.
 */
export function forgetBody(settings, sessionId, turnKey, statement) {
  const redacted = redactText(statement);
  if (redacted.secretOnly) return null;
  const text = bounded(redacted.text);
  if (!text) return null;
  const session = String(sessionId).slice(0, MAX_ID_CHARS);
  return {
    schema_version: 1,
    user_id: settings.userId,
    client_session_id: session,
    request_id: forgetRequestId(session, turnKey, statement),
    statement: text,
  };
}

// Codes after which a plan must be prepared again: it expired, the scope
// changed under it, or the erasure budget it was bound to is spent. The
// forget routes answer all of them as 422 `operation_conflict`; the internal
// names are accepted too.
const PREPARE_AGAIN = new Set(['operation_conflict', 'intent_expired', 'reception_stale', 'budget_exceeded']);

const EXCLUSIONS = new Set(['other_subject', 'negated']);

/**
 * A forget's counts, summed over its rounds. `earlier` counts memories and
 * `earlierSentences` sentences that were gone before this forget. `lost`:
 * the memories outside the plan's targets that the forget left unreadable or
 * reworded, as [memory id, effect] (see addLost).
 */
function tally() {
  return { forgotten: 0, sentences: 0, earlier: 0, earlierSentences: 0, changed: 0, bundled: [], excluded: [], lost: [] };
}

// The effects of `affected` a forget counts, and the count each one gives;
// any other effect is not counted.
export const LOST_EFFECTS = new Map([['forgotten', 'dependents'], ['unreadable', 'unreadable'], ['reworded', 'reworded']]);

/**
 * Count the effects a forget had on memories outside its targets, each
 * given when not 0: `dependents` (forgotten with it), `unreadable` (stored,
 * but cannot be read until the user says it again) and `reworded` (stored,
 * rewritten to leave out what was forgotten). Any other effect is not counted.
 */
export function effectCounts(effects) {
  const out = {};
  for (const [effect, field] of LOST_EFFECTS) {
    const n = effects.filter((value) => value === effect).length;
    if (n > 0) out[field] = n;
  }
  return out;
}

/** A key of a memory id: it names a memory a forget affected, without the id. */
export function memoryKey(id) {
  return sha16(['memory', String(id)]);
}

// The service's `affected` list as [memory id, effect], without the words a
// prepare gives with it: `forgotten`, a memory that relied only on what the
// forget removes, so it goes too; `unreadable`, one that stays stored but
// cannot be read until the user says it again; `reworded`, one that stays
// stored, rewritten to leave out what is forgotten. Any other effect is left out.
function affectedOf(list) {
  return (Array.isArray(list) ? list : [])
    .filter((entry) => typeof entry?.memory_id === 'string' && LOST_EFFECTS.has(entry.effect))
    .map((entry) => [entry.memory_id, entry.effect]);
}

// Add memories a forget left unreadable or reworded, once each; a later
// report of the same memory replaces an earlier one.
function addLost(sum, entries) {
  for (const [id, effect] of entries) {
    const at = sum.lost.findIndex(([known]) => known === id);
    if (at >= 0) sum.lost[at] = [id, effect];
    else sum.lost.push([id, effect]);
  }
}

// The entries of a plan's `affected` that are candidates the plan kept: read
// by the pick and not among its targets.
function keptAffected(plan) {
  const ids = (list) => new Set((Array.isArray(list) ? list : []).map((entry) => entry?.memory_id).filter((id) => typeof id === 'string'));
  const candidates = ids(plan.confirmation?.candidates);
  const targets = ids(plan.confirmation?.targets);
  return (plan.affected ?? []).filter(([id]) => candidates.has(id) && !targets.has(id));
}

function failure(response, extra = {}) {
  const code = codeOf(response);
  if (retryable(response)) return { status: 'retry', code, delay: retryAfterMs(response), ...extra };
  if (response.status === 401 || response.status === 403) return { status: 'auth', code };
  return { status: 'failed', code: `${response.status} ${code}` };
}

// Read a prepare answer's candidates: those the service vetoed (about another
// person or animal, or stating the opposite) are kept and reported, and the
// targets it forgets whole because they also hold other facts are named.
function readCandidates(prepared, sum) {
  const bundled = [];
  for (const candidate of Array.isArray(prepared?.candidates) ? prepared.candidates : []) {
    const statement = typeof candidate?.statement === 'string' ? candidate.statement.trim() : '';
    if (!statement) continue;
    if (!candidate.target && EXCLUSIONS.has(candidate.excluded)) {
      if (!sum.excluded.some((item) => item.statement === statement)) sum.excluded.push({ statement, why: candidate.excluded });
    } else if (candidate.target && candidate.bundled === true && typeof candidate.memory_id === 'string') {
      bundled.push([candidate.memory_id, statement]);
    }
  }
  return bundled;
}

// Add one confirmation's outcomes. Only `verified` counts as forgotten now:
// a memory outcome forgets that memory, and the outcome whose memory_id is
// null (a fact only the user's sentences stated) forgets `withheld_spans`
// sentences. `forgotten_earlier` was gone before; `changed` was kept.
// `resent`: this plan was confirmed once already and that answer was lost or
// was an error (the service may have forgotten some targets before it).
// Every target was current when the plan was prepared, so one gone now was
// forgotten for this same request: it counts as forgotten, and a bundled one
// is named.
function addOutcomes(sum, outcomes, plan, resent) {
  const named = new Map(plan.bundled ?? []);
  let memories = 0;
  let sentences = 0;
  for (const item of Array.isArray(outcomes) ? outcomes : []) {
    const now = item?.outcome === 'verified' || (resent && item?.outcome === 'forgotten_earlier');
    if (now) {
      if (item.memory_id !== null) {
        memories += 1;
        const statement = named.get(item.memory_id);
        if (statement && !sum.bundled.includes(statement)) sum.bundled.push(statement);
        // A bundled target whose words were not kept on disk (carryOnDisk): counted.
        else if (statement === '') sum.whole = (Number(sum.whole) || 0) + 1;
      } else {
        const n = Number(item.withheld_spans);
        const planned = Number(plan.spans);
        sentences += Number.isInteger(n) && n > 0 ? n : Number.isInteger(planned) && planned > 0 ? planned : 1;
      }
    } else if (item?.outcome === 'forgotten_earlier') {
      sum.earlier += 1;
    } else if (item?.outcome === 'changed') {
      sum.changed += 1;
    }
  }
  return { memories, sentences };
}

// Add what one confirmation forgot now. An agent-mode service also answers
// its own counts (`forgotten`: every memory it removed, the cascade's
// included; `withheld_spans`: every sentence it withheld, those withheld with
// the first forgotten memory included), which are taken when this is the
// plan's first confirmation. Otherwise (a repeated confirmation, whose own
// counts say only what it did this time, or a service without them) the
// outcomes are counted, and a forgotten memory withheld the plan's
// sentences with it unless the sentences had an outcome of their own.
//
// The memories outside the targets that the confirmation left unreadable
// come from its `affected` (an agent-mode service's `forgotten` never counts
// them). A repeated confirmation finds them gone already and lists none, and
// a service without its own counts lists none either: the plan's prepare
// named them beforehand, so that list is taken then. A repeated confirmation
// takes only the candidates it kept (keptAffected): the prepare's list also
// names targets and memories its cascade removes, which the outcomes and the
// service's record count already.
function addConfirmed(sum, data, plan, resent) {
  const own = addOutcomes(sum, data?.outcomes, plan, resent);
  const count = (value) => (Number.isInteger(value) && value >= 0 ? value : null);
  const spans = Number.isInteger(plan.spans) && plan.spans > 0 ? plan.spans : 0;
  const memories = count(data?.forgotten);
  if (resent) addLost(sum, keptAffected(plan));
  else if (memories === null) addLost(sum, plan.affected ?? []);
  addLost(sum, affectedOf(data?.affected));
  const withheld = count(data?.withheld_spans);
  sum.forgotten += !resent && memories !== null ? Math.max(memories, own.memories) : own.memories;
  if (!resent && withheld !== null) sum.sentences += Math.max(withheld, own.sentences);
  else sum.sentences += own.memories > 0 && own.sentences === 0 ? spans : own.sentences;
}

// Whether a service's `earlier` record says something was forgotten for the request.
function recordDone(record) {
  return Boolean(record && typeof record === 'object' && (record.in_chat === 'forgotten' || record.statement === 'forgotten'));
}

// Note, at this forget's first prepare, whether either forget path had
// already forgotten something for the request (`prior`). Later records
// also count this forget's own confirmations, so only the first one tells.
function notePrior(sum, data) {
  if (sum.prior === undefined) sum.prior = recordDone(data?.earlier);
}

// What the service's `earlier` record (FORGET-ONE: an agent-mode service
// answers it on prepare and confirm) adds to a forget's counts. The record
// sums what both forget paths removed for this request, this forget's own
// confirmations included. What the record holds beyond this forget's own
// counts was gone before only when another path had done something: the
// in-chat forget of an Add holding the same request, or a forget by
// statement before this one (`prior`). Otherwise all of it is this forget's
// own (a confirmation whose answer was lost, or the memories its cascade
// removed), and it counts as forgotten now. Memories and sentences are kept
// apart. A service without the record adds nothing.
function addEarlier(sum, data) {
  const record = data?.earlier;
  if (!recordDone(record)) return;
  const n = (value) => (Number.isInteger(value) && value > 0 ? value : 0);
  const memories = Math.max(0, n(record.forgotten) - sum.forgotten);
  const sentences = Math.max(0, n(record.withheld) - sum.sentences);
  if (record.in_chat === 'forgotten' || sum.prior === true) {
    sum.earlier = Math.max(sum.earlier, memories);
    sum.earlierSentences = Math.max(sum.earlierSentences, sentences);
  } else {
    sum.forgotten += memories;
    sum.sentences += sentences;
  }
}

// The result of a finished forget from its counts and the service's last status.
/**
 * A retry's carry as it is written to the queue file: ids, counts and the
 * plan (the service's confirmation names memory ids and hashes, no words),
 * without the statements of bundled or kept memories, which only a note
 * given at once quotes. A forgotten-whole memory is then only counted
 * (`whole`). The stored statement a match is forgotten by (`matched`, its
 * request `body`) stays while the job waits: the retry prepares it again.
 */
export function carryOnDisk(carry) {
  if (!carry || typeof carry !== 'object') return carry ?? null;
  const out = { ...carry };
  if (carry.sum) out.sum = { ...carry.sum, whole: (Number(carry.sum.whole) || 0) + (carry.sum.bundled?.length ?? 0), bundled: [], excluded: [] };
  if (carry.plan) out.plan = { ...carry.plan, bundled: (carry.plan.bundled ?? []).map(([id]) => [id, '']) };
  return out;
}

// `dependents`, `unreadable` and `reworded` count `lost` by effect
// (effectCounts); `affectedKeys` lists them as [memoryKey, effect], so the
// service's in-chat outcome of the same request does not count them again
// (DiskQueue.takeChatForgets). A reworded memory is kept, but it no longer
// shows what was forgotten, so the forget did something.
function finished(sum, status) {
  const { prior: _prior, lost = [], ...counts } = sum;
  const others = effectCounts(lost.map(([, effect]) => effect));
  Object.assign(counts, others);
  if (lost.length > 0) counts.affectedKeys = lost.map(([id, effect]) => [memoryKey(id), effect]);
  const done = counts.forgotten + counts.sentences + (others.dependents ?? 0) + (others.reworded ?? 0);
  if (status === 'incomplete' && counts.changed > 0) return { status: 'incomplete', ...counts };
  if (done > 0) return { status: 'forgotten', ...counts };
  if (counts.earlier + counts.earlierSentences > 0) return { status: 'already', ...counts };
  return { status: 'nothing', ...counts };
}

/**
 * One forget: prepare a plan, then confirm it exactly as prepared. The plan
 * is prepared once more when the service asks for that (it expired, or the
 * scope changed) or when a target changed meanwhile.
 * @param {{plan?: object, sum?: object}} carry - a plan whose confirmation
 *   failed retryably, with the counts so far: it is confirmed again first (a
 *   repeated confirmation resumes it).
 */
async function forgetOnce(settings, body, signal, carry = {}) {
  const sum = { ...tally(), ...(carry.sum ?? {}) };
  sum.bundled = [...(carry.sum?.bundled ?? [])];
  sum.excluded = [...(carry.sum?.excluded ?? [])];
  sum.lost = [...(carry.sum?.lost ?? [])];
  let plan = carry.plan && typeof carry.plan.confirmation === 'object' ? carry.plan : null;
  // A carried plan was sent before: its confirmation failed after it may
  // have reached the service.
  let resent = plan !== null;
  let again = false;
  for (;;) {
    if (!plan) {
      let prepared;
      try {
        prepared = await postJson(settings, '/memory/forget/prepare', body, signal);
      } catch (error) {
        return { status: 'retry', code: errKind(error), delay: null, transport: true, carry: { sum } };
      }
      if (prepared.status < 200 || prepared.status >= 300) {
        if (prepared.status === 422 && PREPARE_AGAIN.has(codeOf(prepared))) {
          if (!again) {
            again = true;
            continue;
          }
          // Still in conflict: tried again later, a bounded number of times.
          return { status: 'retry', code: codeOf(prepared), delay: null, conflict: true, carry: { sum } };
        }
        return failure(prepared, { carry: { sum } });
      }
      const bundled = readCandidates(prepared.data, sum);
      notePrior(sum, prepared.data);
      if (prepared.data?.status === 'nothing_matched') {
        // Nothing left to forget: the in-chat forget of the request may have
        // carried it out first, and then it is already gone.
        addEarlier(sum, prepared.data);
        return finished(sum, 'verified');
      }
      const confirmation = prepared.data?.confirmation;
      if (prepared.data?.status !== 'confirmation_required' || !confirmation || typeof confirmation !== 'object') {
        return { status: 'failed', code: 'unexpected_answer' };
      }
      const spans = (Array.isArray(prepared.data?.spans) ? prepared.data.spans : []).filter((span) => span?.target === true).length;
      plan = { confirmation, bundled, spans, affected: affectedOf(prepared.data?.affected) };
      resent = false;
    }
    let confirmed;
    try {
      confirmed = await postJson(settings, '/memory/forget/confirm', plan.confirmation, signal);
    } catch (error) {
      // The confirmation may have reached the service: a retry resumes this
      // plan, and must not forget a neighbour when it finds it already gone.
      return { status: 'retry', code: errKind(error), delay: null, transport: true, maybeDone: true, carry: { plan, sum } };
    }
    if (confirmed.status < 200 || confirmed.status >= 300) {
      if (confirmed.status === 422 && PREPARE_AGAIN.has(codeOf(confirmed))) {
        if (!again) {
          again = true;
          plan = null;
          continue;
        }
        return { status: 'retry', code: codeOf(confirmed), delay: null, conflict: true, maybeDone: true, carry: { sum } };
      }
      // A busy service (another forget, a rebuilt index) may have forgotten
      // some targets already; the same confirmation resumes the plan later.
      return failure(confirmed, { maybeDone: true, carry: { plan, sum } });
    }
    const changedBefore = sum.changed;
    addConfirmed(sum, confirmed.data, plan, resent);
    addEarlier(sum, confirmed.data);
    if (confirmed.data?.status === 'incomplete' && sum.changed > changedBefore && !again) {
      // A target changed after the plan: prepare once more for its new head.
      again = true;
      sum.changed = changedBefore;
      plan = null;
      continue;
    }
    return finished(sum, confirmed.data?.status === 'verified' ? 'verified' : 'incomplete');
  }
}

// As many as recall asks for: a short list is not the head of a longer one.
const FALLBACK_HITS = 10;
const CLOSEST = 3;
const ENTITIES = { '&lt;': '<', '&gt;': '>', '&amp;': '&' };
const STOP = new Set(('a an the and or but of to in on at for from with by as is am are was were be been being it its this that these those '
  + 'i me my mine myself we us our you your user user\'s users please forget remember about something thing things fact anything do does did '
  + 'not no never any all some just also really very').split(' '));
const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+/gu;
const CJK_FILLER = /我们|你们|咱们|这件事|那件事|这个|那个|用户|我|你|的|了|吧|呢|啊|请|把|给/gu;

/**
 * The statements of /memory/search hits, in rank order: evidence hits are skipped.
 * Both hit renderings are read (hitMemory): the agent service's compact hit,
 * whose statement is followed by a `Status:` line, and the full one with a
 * `Statement:` line.
 */
export function statementsOf(hits) {
  const out = [];
  for (const hit of Array.isArray(hits) ? hits : []) {
    const statement = hitMemory(hit?.content)?.statement.replace(/&(lt|gt|amp);/g, (m) => ENTITIES[m]).trim();
    if (statement && !out.includes(statement)) out.push(statement);
  }
  return out;
}

// The content terms of a text: Latin words without function words (a plural
// or third-person "s" dropped), and the character pairs of CJK runs.
function terms(text) {
  const t = fold(String(text ?? '')).toLowerCase();
  const out = new Set();
  for (const word of t.replace(CJK, ' ').match(/[\p{L}\p{N}][\p{L}\p{N}'_-]*/gu) ?? []) {
    if (STOP.has(word) || word.length < 2) continue;
    out.add(word.length > 3 && word.endsWith('s') && !word.endsWith('ss') ? word.slice(0, -1) : word);
  }
  for (const run of t.match(CJK) ?? []) {
    const chars = [...run.replace(CJK_FILLER, '')];
    if (chars.length === 1) out.add(chars[0]);
    for (let i = 0; i + 1 < chars.length; i += 1) out.add(chars[i] + chars[i + 1]);
  }
  return out;
}

const NEGATION = /\b(?:not|no|never|none|nor|without|cannot)\b|n't\b|[不没别未无]/iu;
// People and animals: "my sister" and "the user's brother" are other facts,
// however many other words they share.
const ROLES_EN = new Set(('mother mom mum mommy father dad daddy parent wife husband spouse partner girlfriend boyfriend fiance fiancee '
  + 'son daughter child children kid baby sister brother sibling twin grandmother grandma grandfather grandpa grandparent grandson '
  + 'granddaughter aunt uncle cousin niece nephew friend roommate flatmate neighbor neighbour boss manager colleague coworker teacher '
  + 'doctor dentist landlord landlady dog cat puppy kitten pet bird fish hamster rabbit horse parrot').split(' '));
const ROLES_ZH = ['女朋友', '男朋友', '妈妈', '母亲', '爸爸', '父亲', '父母', '老婆', '妻子', '老公', '丈夫', '对象', '儿子', '女儿',
  '孩子', '姐姐', '妹妹', '哥哥', '弟弟', '爷爷', '奶奶', '外公', '外婆', '姥姥', '姥爷', '叔叔', '阿姨', '舅舅', '姑姑', '表哥',
  '表姐', '表弟', '表妹', '堂哥', '堂姐', '朋友', '室友', '邻居', '老板', '同事', '老师', '医生', '房东', '宠物', '狗', '猫'];

function rolesOf(text) {
  const out = new Set();
  for (const term of terms(text)) if (ROLES_EN.has(term)) out.add(term);
  let rest = String(text ?? '');
  for (const role of ROLES_ZH) {
    if (rest.includes(role)) {
      out.add(role);
      rest = rest.split(role).join(' ');
    }
  }
  return [...out].sort().join(' ');
}

function covered(part, whole) {
  let shared = 0;
  for (const term of part) if (whole.has(term)) shared += 1;
  return shared;
}

/**
 * Whether a stored statement says what the user's words name, and nothing
 * else. Both ways: at least 60% of the words' content terms (and at least
 * two, when there are two) appear in the statement, and at least 60% of the
 * statement's content terms appear in the words. Both name the same people
 * and animals, and both are negated or neither is. Deliberately strict: a
 * statement that only shares a topic, or says more, is left for the user to
 * choose.
 */
export function sameFact(words, statement) {
  const wanted = terms(words);
  const have = terms(statement);
  if (wanted.size === 0 || have.size === 0) return false;
  if (NEGATION.test(fold(String(words))) !== NEGATION.test(fold(String(statement)))) return false;
  if (rolesOf(words) !== rolesOf(statement)) return false;
  const shared = covered(wanted, have);
  return shared >= Math.min(2, wanted.size) && shared / wanted.size >= 0.6 && covered(have, wanted) / have.size >= 0.6;
}

const QUOTE_SHARE = 0.75;

/**
 * Whether the user's own quoted message says what the user's words name:
 * at least 75% of the words' content terms (and at least two, when there
 * are two) appear in the quote, both name the same people and animals, and
 * both are negated or neither is. One way only: the quote is the whole
 * message the memory came from, so it may say more. A question is never a
 * match: it states nothing.
 */
export function saidIn(words, quote) {
  const text = String(quote ?? '').replace(/&(lt|gt|amp);/g, (m) => ENTITIES[m]);
  if (/[?？]\s*$/.test(text)) return false;
  const wanted = terms(words);
  const have = terms(text);
  if (wanted.size === 0 || have.size === 0) return false;
  if (NEGATION.test(fold(String(words))) !== NEGATION.test(fold(text))) return false;
  if (rolesOf(words) !== rolesOf(text)) return false;
  const shared = covered(wanted, have);
  return shared >= Math.min(2, wanted.size) && shared / wanted.size >= QUOTE_SHARE;
}

/**
 * The one memory among /memory/search hits whose quoted user message says what the
 * user's words name (saidIn), or null when none or more than one does. This
 * finds a memory stored in another language than the request: its statement
 * is in one language, its quote in the user's own.
 */
export function quotedMatch(words, hits) {
  const found = [];
  for (const hit of Array.isArray(hits) ? hits : []) {
    const memory = hitMemory(hit?.content);
    if (!memory) continue;
    const statement = memory.statement.replace(/&(lt|gt|amp);/g, (m) => ENTITIES[m]).trim();
    if (!statement || found.includes(statement)) continue;
    if (memory.quotes.some((quote) => /^user$/i.test(quote.who) && saidIn(words, quote.text))) found.push(statement);
  }
  return found.length === 1 ? found[0] : null;
}

// A result of the forget of a stored statement found by /memory/search: it says
// which statement was forgotten, and what a retry carries keeps that
// statement's own body, so a resumed or re-prepared plan is for the match,
// never for the user's original words.
function withMatched(result, matched, body) {
  if (['forgotten', 'incomplete', 'already'].includes(result.status)) return { ...result, matched };
  if (result.status === 'retry') return { ...result, carry: { ...(result.carry ?? {}), matched, body } };
  return result;
}

// The forget body of a stored statement found for the user's words.
function matchBodyOf(body, match) {
  return { ...body, request_id: `${body.request_id.slice(0, MAX_ID_CHARS - 3)}:m1`, statement: bounded(match) };
}

/**
 * Carry out one forget: prepare the plan, then confirm it exactly as
 * prepared. A plan that expired or changed before the confirmation is
 * prepared once more. The user's own request is the consent.
 *
 * When the service finds nothing for the user's words (they were cut, or
 * the memory is stored in another language), the words are looked up with
 * /memory/search. A stored statement that says the same thing (sameFact), or else
 * the one memory whose quoted user message says it (quotedMatch), is then
 * forgotten by its own text; otherwise the closest statements come back in
 * `closest`, so the note can show them and claims nothing. A statement the
 * service kept because it is about another person or animal, or states the
 * opposite (`excluded`), is never forgotten in its place nor offered.
 * @param {object} body - from forgetBody.
 * @param {{signal?: AbortSignal, substitute?: boolean, carry?: object}} options -
 *   `signal` aborts every call. `substitute: false` skips the lookup: an
 *   earlier attempt may have forgotten the statement already, so "nothing
 *   matched" then means done, and a neighbour must not be forgotten instead.
 *   `carry`: what a `retry` result carried, to resume its plan.
 * @returns {Promise<{status: 'forgotten'|'incomplete'|'already'|'nothing', forgotten: number,
 *     sentences: number, earlier: number, earlierSentences: number, changed: number, bundled: string[],
 *     dependents?: number, unreadable?: number, reworded?: number, affectedKeys?: [string, string][],
 *     excluded: {statement: string, why: string}[], matched?: string, closest?: string[]}
 *   | {status: 'retry', code: string, delay: number|null, maybeDone?: boolean, carry: object}
 *   | {status: 'auth'|'failed', code: string}>}
 *   `forgotten` counts the memories the service verified forgotten now,
 *   `sentences` the sentences of the saved conversation that stated it,
 *   `earlier` the memories and `earlierSentences` the sentences already gone
 *   (forgotten before, by this request's in-chat forget or an earlier forget
 *   of the same words), and `changed` those kept because they changed
 *   meanwhile. `bundled`: forgotten memories that also held other
 *   facts, so they went whole. `dependents`: memories outside the targets
 *   that relied only on what was forgotten, so they were forgotten too;
 *   `unreadable`: such memories that stay stored but cannot be read until the
 *   user says them again; `reworded`: such memories that stay stored,
 *   rewritten to leave out what was forgotten (each only when not 0).
 *   `affectedKeys`: all of these memories, as [memoryKey, effect].
 *   `already`: everything it named was gone before. `retry`: the service was
 *   busy or unreachable; try again later with `carry`.
 */
export async function runForget(settings, body, { signal, substitute = true, carry = null } = {}) {
  if (carry?.matched) {
    // A forget of a stored statement found by /memory/search resumes as one: its
    // plan, or else a new plan, is for that statement, never for the user's
    // original words. (A carry written before the match's body was kept is
    // given that body from the matched statement.)
    const own = carry.body && typeof carry.body.statement === 'string' && typeof carry.body.request_id === 'string'
      ? carry.body
      : matchBodyOf(body, carry.matched);
    const resumed = await forgetOnce(settings, own, signal, carry);
    return withMatched(resumed, carry.matched, own);
  }
  const first = await forgetOnce(settings, body, signal, carry ?? {});
  if (first.status !== 'nothing' || !substitute) return first;
  const kept = new Set(first.excluded.map((item) => item.statement));
  let closest = [];
  let hits = [];
  try {
    const found = await postJson(settings, '/memory/search', { user_id: body.user_id, query: body.statement, top_k: FALLBACK_HITS }, signal);
    if (found.status >= 200 && found.status < 300) {
      hits = Array.isArray(found.data?.data) ? found.data.data : [];
      closest = statementsOf(hits).filter((statement) => !kept.has(statement));
    }
  } catch {
    // the lookup is a best effort: the answer stays "nothing matched"
  }
  const quoted = quotedMatch(body.statement, hits);
  const match = closest.find((statement) => sameFact(body.statement, statement)) ?? (quoted && !kept.has(quoted) ? quoted : null);
  if (!match) return { ...first, closest: closest.slice(0, CLOSEST) };
  const matchBody = matchBodyOf(body, match);
  // The look-alikes the first prepare kept are carried into the match's
  // counts, so they are still reported after a retry.
  const again = await forgetOnce(settings, matchBody, signal, { sum: { excluded: first.excluded } });
  if (again.status === 'nothing') return { ...first, closest: closest.slice(0, CLOSEST) };
  return withMatched(again, match, matchBody);
}
