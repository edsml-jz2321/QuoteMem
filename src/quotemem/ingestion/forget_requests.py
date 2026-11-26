"""Explicit in-chat forget requests: the user's own sentences asking to forget a fact.

An Add whose frozen extraction policy names ``chat_forget`` carries out the forget
requests of its user messages once the Add has completed, through forget by statement
(``service/chat_forget.py``). This module only finds the requests and the
statement each one asks to forget; it reads nothing else and calls nothing.

A sentence of a user message is a request when, after an optional opener ("ok",
"also", "one more thing", ...) and an optional polite frame ("please", "kindly",
"can/could/would you", "I'd like you to", "I want/need you to"), it starts with:

- ``forget`` or ``forget about``, followed by what to forget; a bare "Forget about
  ..." with neither a polite frame nor the memory named is the idiom for "stop
  thinking about" ("Forget about my ex, she's not worth it"), never a request;
- a remembering verb that it negates or stops: "don't/do not/never remember",
  "stop remembering", "no longer remember";
- a verb that keeps or drops a record, with the memory it names: "don't keep, store,
  save, record or retain <it> in your memory", "remove, delete, erase, wipe, clear or
  drop <it> from your memory" (or records, notes, history).

What it asks to forget must be about the speaker or what the speaker told: its
first clause says I, me, my, mine or myself, or names something told ("the detail
about ...", "the fact that ...", "any information about ..."), or its sentence names
the memory; and it must be more than one word. A bare
"it", "this" or "that" after ``forget`` is a dismissal ("Forget it, let's move on"),
never a request; after a remembering or record verb, or as "what I said / told you /
mentioned / shared", it refers to what the user said just before: the earlier
sentences of the same message, else the previous user message of the Add. "Forget
that I asked" retracts a question and asks nothing.

Each form must begin its sentence, so a report ("She told me to forget about it"),
a statement about forgetting ("I keep forgetting my keys", "history books too often
forget"), a reminder ("Don't forget that I ...") and a question about forgetting
("Do you ever forget what I tell you?", "Did you forget my birthday?") never
qualify. A sentence ending in a question mark qualifies only through a polite
"can/could/would you" frame. A message may start with its speaker's label ("Mira:
Please forget that I ..."), as transcripts write it; the label is not part of the
sentence. Unless the sentence names the memory, the Add must be a conversation with
the memory: it holds an assistant message, or its user messages come from one speaker
(their ``speaker``, else their label). Between people, two speakers and no assistant
("Forget about my ex, she's not worth it"), the imperative is advice to the other
person, not an instruction to the memory.

Chinese requests follow the same rule with their own words: an optional ``请`` (and
``帮我``) before ``忘记``, ``忘掉`` or ``不要/别 记住/记得/记录``, or ``删除/删掉 ...
记忆/记录``, about the speaker (``我``). The ``agent`` Add mode (``complete``, the
``explicit-request-v2`` rule) also reads the forms its plugins' users write: ``把 ... 忘掉/
忘了/忘记/删掉`` ("把我在学日语这件事忘掉吧") and a bare ``不要/别/不用 记 ...`` ("不要记我在学
日语"); every other mode reads only the forms above.

The ``agent`` mode (``complete``) follows the plugins' own reading of a request
(``integrations/core/forget.mjs``), so both read one sentence alike, the narrower
one where they differ:

- it also reads a request after a time frame ("From now on, please don't remember ...",
  "以后不要记得我恐高了"), another person's name ("forget Ben's name") and, after 忘掉 only, a
  person called by name ("忘掉阿杰", "忘掉小李的名字", "忘掉Emma的名字": 阿 and one character, or
  小 or 老 and a common surname; a Latin name before 的名字);
- it reads no bare "don't remember ..." unless marked as a request (a that-clause in the
  first person, "do not" written out, emphasis, a closing please or thanks) and asking nothing
  back ("don't remember my password, can you remind me" is the user's own), no 忘记 and
  someone's password, number ... then 了 without 请 or 帮我 ("忘记我的密码了": what the user
  forgot), no reference back only the plugins read ("forget I said it", "不要记我刚才说的"), no
  slash command, no instruction or plan ("forget my previous instructions"), no quarrel let go
  ("forget our fight", "忘掉我们吵架的事"), no change of instructions ("..., use main instead",
  "Call him Benny."), no fresh start about no fact of the user's ("forget the stuff we
  discussed, let's start over", "忘掉我们聊过的东西") and no request withdrawn ("Forget my phone
  number. Never mind.") and no software's name ("forget Docker's name"); it splits a message into
  sentences as the plugins do ("Dr. Sato" is one, `recall_agent.request_sentences`);
- what it asks to forget ends before a closing fresh start ("Please forget that I'm pregnant,
  let's move on." asks to forget "I'm pregnant") and before a clear replacement in the same
  sentence (`_kept`):
  "Forget my old address, I live at 27 Birch Lane now." asks to forget "my old address", and
  `request_reach` says where that request ends in its sentence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from quotemem.core.forget_match import TIES
from quotemem.core.forget_pick import MAX_FORGET_STATEMENT_CHARS
from quotemem.core.recall_agent import CALENDAR, SENTENCE, request_sentences
from quotemem.core.run_inputs import InputMessageV1
from quotemem.core.sensitivity_rules import classify

MAX_CHAT_FORGETS: Final = 16
"""The requests one Add carries out at most, in message order; later ones are left."""
_OPENER = (r"(?:(?:ok(?:ay)?|also|and|so|oh|hey|hi|actually|now|well|lastly|finally|"
           r"one more thing|by the way|btw)\b[\s,:\-—–]*)*")
_POLITE = (r"(?P<polite>(?:please|kindly|(?:can|could|would) you(?: please)?|"
           r"i(?:'d| would) like (?:you )?to|i want you to|i need you to|"
           r"i(?:'m| am) asking you to)[\s,]+)?(?:please[\s,]+)?")
_MEMORY = r"(?:(?:your |the )?(?:memory|memories)|your (?:records?|notes|history))"
_FORGET = re.compile(
    rf"^\W*{_OPENER}{_POLITE}(?P<verb>forget(?: all)?(?: about)?)\b[\s,]*(?P<object>.*)$",
    re.IGNORECASE | re.DOTALL)
_REMEMBER = re.compile(
    rf"^\W*{_OPENER}{_POLITE}(?P<verb>(?:do not|don't|dont|never) remember|"
    rf"stop remembering|no longer remember)\b\s*(?P<object>.*)$",
    re.IGNORECASE | re.DOTALL)
_RECORD = re.compile(
    rf"^\W*{_OPENER}{_POLITE}(?P<verb>(?:(?:do not|don't|dont|never) "
    rf"(?:keep|store|save|record|retain))|stop (?:keeping|storing|saving|recording)|"
    rf"remove|delete|erase|wipe|clear|drop|purge)\s+(?P<object>.+?)\s+"
    rf"(?:in|into|to|from|out of) {_MEMORY}\b\W*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL)
_ZH = re.compile(
    r"^\W*(?:请|麻烦|能不能|可以)?(?:你)?(?:帮我)?(?:(?P<verb>忘记|忘掉|(?:不要|别|不用)(?:再)?"
    r"(?:记住|记得|记录|保存))(?P<object>.*)|(?:删除|删掉|清除)(?P<deleted>.*?)(?:的)?"
    r"(?:记忆|记录))\W*$", re.DOTALL)
_END: Final = r"[^\w+#%]*$"
"""The marks after a request's last word, which its statement leaves out: sentence
punctuation, never the symbols that end a word ("C++", "C#", "100%")."""
_ZH_COMPLETE = re.compile(
    r"^\W*(?:请|麻烦|能不能|可以)?(?:你)?(?:帮我)?(?:把(?P<object>.+?)(?:给)?"
    r"(?P<baverb>忘掉|忘了|忘记|删掉|删除|去掉)(?:吧|了|啊|呀)*|(?:不要|别|不用)(?:再)?"
    r"记(?![住得录错混反岔乱])(?P<bare>.+?))" + _END, re.DOTALL)
# The agent mode's forms (``complete``). They follow the plugins' own reading of a request
# (``integrations/core/forget.mjs``, ``detectForget``), so the statement a plugin sends and the
# one the in-chat forget reads of the same sentence agree; where the two differ,
# the narrower one is kept.
_LATER = re.compile(
    r"^\W*(?:(?:from now on|going forward|in (?:the )?future|henceforth|from here on(?: out)?)"
    r"\b[\s,]*|(?:以后|今后|之后|往后|从今以后|从今往后|从现在起|从现在开始)[\s，,]*)(?=\S)",
    re.IGNORECASE)
"""A time frame before a request: "From now on, please don't remember ...", "以后不要记得..."."""
_NOT_A_FACT = re.compile(
    r"^(?:(?:my|our|your|the|all|any|these|those|this|that)\s+)?(?:(?:previous|prior|earlier|last|"
    r"above|old|original|initial|first|current|other|whole|entire)\s+)*(?:instructions?|"
    r"approach(?:es)?|suggestions?|prompts?|ideas?|requests?|directions?|rules?|commands?|"
    r"steps?|tasks?|attempts?|drafts?|versions?|changes?|edits?|solutions?|guidelines?|"
    r"context)\b", re.IGNORECASE)
"""An instruction, never a fact: "forget my previous instructions"."""
_WORK_HEAD = re.compile(
    r"^(?:(?:my|our|your|the|all|any|these|those|this|that)\s+)?(?:(?:previous|prior|earlier|last|"
    r"above|old|original|initial|first|current|other|whole|entire|new)\s+)*(?:plans?|code|"
    r"implementations?|designs?|branch(?:es)?|functions?|query|queries|scripts?|schemas?|prs?|"
    r"pull requests?|commits?)\b", re.IGNORECASE)
_WORK_TAIL = re.compile(
    r"^(?:\W*$|\s*[,;:—–]|\s+(?:and|then|but|so|or)\b|\s+(?:for now|so far|above|here|again|too|"
    r"as well)\b|\s+(?:that\s+)?(?:i|you|we)\s+(?:just\s+)?(?:suggested|gave|wrote|made|proposed|"
    r"sent|had|did|tried|described|mentioned|shared|asked|said|used|pushed|opened|pasted|"
    r"copied|posted|uploaded|attached|showed|shown|ran|run|deployed|merged|reviewed|committed|"
    r"edited|fixed|changed|added|drafted|typed|generated|created|built|refactored)\b)",
    re.IGNORECASE)
_WORK_CONTEXT = re.compile(
    r"\b(?:api|apis|endpoints?|pages?|components?|modules?|services?|repo|repository|app|"
    r"functions?|class(?:es)?|methods?|tests?|build|bugs?|features?|database|db|server|ui|"
    r"frontend|backend|files?|scripts?|branch(?:es)?|query|schema|code|config|pipeline|"
    r"deploy(?:ment)?|login|buttons?|models?|migrations?|refactor|changes|edits|diff|patch|"
    r"commits?|prs?|implementation|docs|readme|project|codebase|variables?|loop|algorithm|sql|"
    r"css|html|js|python|java)\b", re.IGNORECASE)
_ZH_NOT_A_FACT = re.compile(
    r"^我们?的(?:代码|方案|分支|实现|函数|脚本|提交|查询|架构|改动|修改|草稿|接口|PR)+(?:吧|啊|了|呢)?$")
"""Work in a coding agent is no fact either ("forget my implementation", "forget my branch,
let's start over", "forget the code I pasted"), but these words also name the user's own life
("my plan to leave my husband", "my code name", "Forget my branch manager's name."), which
this module reads as it always did. So a work word refuses the request only as the whole
object (`_WORK_HEAD` and nothing more), or when what follows it goes on with the work
(`_WORK_TAIL`: "and ...", "I suggested", "I pasted", "for now") or names it (`_WORK_CONTEXT`:
"my design for the API", "my code changes"); in Chinese, only as the whole object ("忘掉我的
代码", "忘掉我的代码改动", not "我的方案是离开我老公")."""
_SAID_WHOLE = re.compile(
    r"^(?:(?:my|our|your|the|all|any|these|those|this|that)\s+){0,2}(?:(?:previous|prior|earlier|"
    r"last|above|old|other|whole|entire|little|stupid|silly|dumb)\s+)*(?:questions?|messages?|"
    r"answers?|responses?|replies|stuff|things|mistakes?|chats?|conversations?|discussions?)$",
    re.IGNORECASE)
"""What was said, as a whole ("my questions", "that stuff"), is no fact; with what it was
about ("the things I told you about my ex", "my messages to my ex") it is one."""
_CORRECTING = re.compile(
    r"\b(?:wrong|wrongly|incorrectly|right|correctly|as|again|like (?:this|that)|instead|"
    r"remember)\b", re.IGNORECASE)
_REMEMBER_ALL = re.compile(r"^(?:everything|anything|all|whatever|me)\b", re.IGNORECASE)
"""A "remember" form whose object is a noun phrase and that corrects or complains ("please
don't remember my name wrong", "... my birthday as the 5th", "... anything I say", "stop
remembering me as rude") is no request. A that-clause states the fact itself ("... that I
work as a nurse"), and a condition asks ("... my address until I'm settled")."""
_SAID_HEAD = re.compile(
    r"^(?:(?:the|that|this|those|these|all|our|your|my)\s+){0,2}(?:(?:previous|prior|earlier|"
    r"last|old|whole|entire|little|other)\s+)*(?:stuff|things?|conversations?|chats?|"
    r"discussions?|sessions?|talks?|everything)\b", re.IGNORECASE)
_CONVERSING = re.compile(r"\b(?:we|we've|we'd|us|our|you|your)\b", re.IGNORECASE)
"""Starting the work over, not a fact: "forget the stuff we discussed", "... the things from
our last session" (`_SAID_HEAD` with "we", "our" or "you" and no "I", "me" or "my" after it);
"forget the conversation we had about my divorce" is a request."""
_RESET = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|and|now|then|just|alright)[\s,]+)*(?:let'?s|let us|"
    r"start (?:over|fresh|again|anew)|starting (?:over|fresh)|moving on|move on|never ?mind|"
    r"new topic|different topic|change (?:of )?(?:the )?(?:topic|subject))\b", re.IGNORECASE)
_ZH_RESET = re.compile(
    r"^[\W\d_]*(?:那|好|好吧|那就|行)?(?:我们|咱们)?(?:重新开始|从头开始|从头来|换个话题|换一个话题|"
    r"说点别的|聊点别的|聊别的|说别的|算了)")
_NEVERMIND = re.compile(r"^\W*(?:(?:ok(?:ay)?|so|and|oh|just|actually)[\s,]+)*never ?mind\b",
                        re.IGNORECASE)
_ZH_NEVERMIND = re.compile(r"^[\W\d_]*(?:那|好|好吧|那就|行)?算了")
_RESET_GOES_ON = re.compile(
    r"[,;:—–、]|\s-\s|\b(?:because|since|as|so|but|cause|cuz|though|although)\b", re.IGNORECASE)
_ZH_RESET_GOES_ON = re.compile(r"[，,；;、]|因为|由于|所以|我|还|但")
"""A clause or the next sentence that starts over makes a forget about no fact of the user's
a dismissal: "..., let's start fresh", "Forget the stuff about the movie. Let's talk about
work.". After a fact of the user's (I, me, my; 我) it only closes the request and is no part
of what is forgotten: "Please forget that I'm pregnant, let's move on.", "忘掉我怀孕了，说点别的
吧". "Never mind" and 算了 withdraw the request itself (`_NEVERMIND`). The closing clause is cut off
only when it is the last clause and goes on with nothing (`_RESET_GOES_ON`): "Forget that I
have ``HIV``, let's move on, my partner doesn't know." keeps the whole object."""
_ZH_SAID_RESET = re.compile(
    r"^(?:我们|咱们)(?:之前|刚才|上次|昨天|今天|刚刚)?(?:聊|说|讨论|谈)(?:过|了)?的"
    r"(?:东西|内容|事情?|话|那些|这些)?$")
""""忘掉我们聊过的东西", "忘掉我们之前说的": as `_SAID_HEAD`."""
_DISPUTE = re.compile(
    r"^(?:(?:our|the|that|this|those|these|all)\s+)?(?:(?:little|small|stupid|silly|dumb|big|"
    r"huge|last|earlier|recent|whole|petty|awkward)\s+)*(?:fights?|arguments?|disagreements?|"
    r"quarrels?|spats?|tiffs?|falling[- ]outs?|misunderstandings?|differences|beef|drama)\b",
    re.IGNORECASE)
"""Letting a quarrel go ("forget our fight last night"), not a fact to delete."""
_ZH_DISPUTE = re.compile(
    r"^(?:我们|咱们|我和你|你和我)(?:之间)?(?:昨天|昨晚|今天|上次|那次|刚才|之前)?(?:的)?"
    r"(?:吵架|争吵|吵的架|矛盾|不愉快|误会|过节|别扭|争执)(?:的事情?|这件事|那件事)?$")
_RENAME = r"(?:call|name|rename) (?:it|him|her|them|this|that)\b"
_REPLACED = re.compile(
    rf"\binstead\b|\brather than\b|^(?:(?:and|then|just|so|now|please)\s+)*{_RENAME}",
    re.IGNORECASE)
_USE_INSTEAD = re.compile(
    rf"\s(?:and|then|but|so|just)\s+(?:(?:use|switch to|go with|stick with|try)\b.*\binstead\b|"
    rf"{_RENAME})", re.IGNORECASE)
_NEXT_INSTEAD = re.compile(
    rf"^\W*(?:(?:just|now|and|but|so|then|please|i want you to|i'd like you to)\s+)*"
    rf"(?:(?:use|switch to|go with|stick with)\b.*\binstead\b|{_RENAME})", re.IGNORECASE)
_ZH_INSTEAD = re.compile(
    r"^\s*(?:就|请|那就|直接|还是|以后|之后)?(?:改用|换用|改成|换成|改为|改叫|改名|叫(?:他|她|它)们?|"
    r"用.+?(?:来)?(?:代替|替代|取代|替换))")
"""A change of instructions after a request ("forget my branch name, use main instead",
"…，改用main", "Call him Benny."): no fact is asked to be forgotten."""
_EN_NAMED = re.compile(
    r"^(?P<owner>[A-Z][\w-]*(?: [A-Z][\w-]*)?)'s\s+(?:(?:first|last|full|nick)\s*)?name$")
"""An English request's object that is the name of a person it names: "Ben's name"."""
_NOT_OWNER: Final = frozenset(
    "it that this what who there here let he she one everyone everybody someone somebody "
    "anyone anybody".split())
_ZH_SURNAMES: Final = ("王李张刘陈杨黄赵吴周徐孙朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾"
                       "肖田董袁潘蒋蔡杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范廖贾夏韦邹孟"
                       "熊秦邱尹薛段雷侯陶贺顾郝龚邵严覃武戴莫孔汤")
_ZH_NAME = re.compile(rf"^(?:阿[\u4e00-\u9fff]|[小老][{_ZH_SURNAMES}])$")
"""A person called by 阿 and one character ("阿杰"), or by 小 or 老 and a common surname ("小李",
"老王"): never 小时 or 老歌."""
_ZH_A_WORDS: Final = frozenset(
    "阿姨 阿婆 阿公 阿爸 阿妈 阿哥 阿嫂 阿婶 阿伯 阿叔 阿爷 阿嬷 阿妹 阿弟 阿姐 "
    "阿里 阿胶 阿片 阿拉 阿门".split())
_LATIN_NAME = re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+)?$")
_ZH_NAMED = re.compile(
    r"^(?P<owner>[^\s，,。；;、：:！？!?我]{1,10})的(?:名字|姓名|全名|名|昵称|外号|小名|绰号|英文名|"
    r"中文名|电话|电话号码|手机号|手机号码|生日|地址|住址|年龄|邮箱)$")
_ZH_PARTICLES = re.compile(r"(?:吧|啊|呀|哈|呢)+$")
_ZH_POLITE = re.compile(r"^\W*(?:请|麻烦|能不能|可以|帮我)")
_ZH_FORGOT_NP = re.compile(
    r"^(?:[^，,。；;！？!?]*的[\u4e00-\u9fff]{1,6}|我们?家[\u4e00-\u9fff]{0,6})了(?:吗|吧|呢|怎么办)?$")
"""忘记 and something of someone's, then 了 (and 吗, 吧, 呢 or 怎么办), without 请 or 帮我,
says what the user forgot ("忘记我的密码了", "忘记我家地址了吗"); "忘掉我离婚了" is a request."""
_ZH_YOU_FORGOT = re.compile(
    r"^\W*你(?:(?:也|又|还|都|已经|是不是|是否|是|真的|竟然)(?:忘记|忘掉|忘了)|忘了)")
_ZH_YOU_ASKED = re.compile(r"^\W*你(?:忘记|忘掉)")
_ZH_ASKED_END = re.compile(r"了(?:吗|没有?)\W*$")
_ZH_GUESS_END = re.compile(r"了吧\W*$")
"""An accusation of "you" ("你忘了我的生日吧", "你也忘了...", "你忘掉我的生日了吗", "你忘记我的
生日了吧"): what the assistant forgot, no request; "你忘掉我的地址吧" asks."""
_ZH_MEMORY_HEAD = re.compile(r"^你?(?:记住|记得|记下|保存)的")
_ZH_MEMORY_TAIL = re.compile(r"(?:的)?(?:记忆|记录)$")
_ZH_WORK = re.compile(r"文件|分支|代码|目录|提交|日志|缓存|配置|测试|仓库|项目|数据库|表格|"
                      r"数据表|表单|镜像|聊天|浏览|通话|搜索|下载|订单|表$")
"""删掉, 删除 or 清除 asks to forget only what is named as a memory ("把你记住的我的地址删掉"),
never work ("把我的分支删掉", "删除我的提交记录")."""
_TRAILING_AGENT = re.compile(r"(?:[\s,]+(?:pls|plz|thx))+\W*$", re.IGNORECASE)
_SAID_BACK = re.compile(
    r"^i (?:just |ever |also )?(?:said|told you|mentioned|wrote|typed)(?:\s+(?:that|it|this|those|"
    r"these|so|earlier|before|just now|anything))*$", re.IGNORECASE)
_SAID = (r"i (?:just |have |'ve |had )?(?:said|told you|mentioned|shared|wrote|typed|sent)"
         r"(?: (?:earlier|before|just now))?")
_BACK = re.compile(
    rf"^(?:(?:it|this|that|those|these)(?: information| detail| details)?|(?:what|everything|all|"
    rf"anything|whatever) (?:that )?{_SAID}|(?:the |my )?(?:last|previous) (?:thing|things|"
    rf"sentence|message)s? (?:that )?{_SAID}|my (?:last|previous) (?:message|sentence))\W*$",
    re.IGNORECASE)
"""A reference back only the plugins read ("forget I said it", "forget my last message"): the
in-chat forget reads no such request."""
_ZH_BACK = re.compile(
    r"^我?(?:刚才|刚刚|之前|前面|上面|方才)?(?:(?:说|讲|提|写|告诉你)(?:过)?的(?:话|内容|那些|这些|事|东西|"
    r"那句话?|这句话?)?|(?:问|提问)(?:过)?(?:你)?的(?:话|问题|那个问题|这个问题|事|东西|那些|这些)?)$")
_ASKS_THAT = re.compile(r"^that\s+(?:i|i'm|i've|i'd|i'll|my|we|we're|we've|our)\b", re.IGNORECASE)
_CLOSING = re.compile(r"\b(?:please|pls|plz|thanks|thank you|thx)\W*$", re.IGNORECASE)
_PLEASE = re.compile(r"\b(?:please|pls|plz)\W*$", re.IGNORECASE)
_EMPHASIS = re.compile(r"\b(?:NOT|DON'?T|NEVER)\b")
_RECALL = re.compile(
    r"\b(?:remind me|tell me|can you|could you|would you|do you|help|how|huh|list (?:it|them)|"
    r"look (?:it|that) "
    r"up|what(?:'s| is| was| are| were)|tbh|lol|sorry|idk|either)\b", re.IGNORECASE)
"""What marks a bare "don't remember ..." as a request (the plugins' gate): a that-clause in
the first person, "do not" written out or emphasis (not when the sentence is all capitals), a
closing please or thanks, and nothing in it asking back ("..., can you remind me", "help"); a
bare "never remember ..." only with please ("never remember my keys!" is a complaint)."""
_TECH: Final = frozenset((
    "docker kubernetes k8s python java javascript typescript node nodejs deno bun npm yarn pnpm "
    "pip redis postgres postgresql mysql mariadb mongo mongodb sqlite nginx apache caddy linux "
    "ubuntu debian fedora centos windows macos mac ios android git github gitlab bitbucket aws "
    "azure gcp google chrome firefox safari edge slack discord teams zoom jira confluence notion "
    "figma react vue angular svelte django flask fastapi rails ruby rust go golang php swift "
    "kotlin scala bash zsh helm terraform ansible jenkins elasticsearch kafka rabbitmq vercel "
    "netlify heroku cloudflare openai claude chatgpt gpt gemini llama main master dev develop prod "
    "production staging server servers api app apps service website site localhost database db "
    "cluster vm container repo branch").split())
"""Software, services and machines are no people: "forget Docker's name", "忘掉Redis的地址"."""
_COMMAND = re.compile(r"^\s*(?:/[\w:-]+(?:\s|$)|!)")
"""A slash command or a shell escape (the plugins' ``isCommand``): no forget request."""
_FILLER: Final = frozenset((
    "ignore never mind nevermind wait sorry oops actually scratch cancel undo hmm um uh ok okay "
    "yes yeah yep nope right fine well hey oh so now then instead").split())
_ZH_FILLER = re.compile(r"算了|不对|等等|等一下|没事|好吧|对不起|抱歉|不好意思|嗯|哦|噢|啊|呃")


# A clear replacement after what a request forgets, in the last clause of its sentence, is no
# part of it (`_kept`, the plugins' ``replaces``): the clause restates the same thing (an
# address, a name, a number, an email) with a new value and says nothing else, the value being
# words that each start with a digit, hold an @ or start with a capital, then at most "now":
# "my new number is 555" or "my number is now 555" after "my number"; "it's 27 Birch Lane now"
# after "my old address"; a new place to live after an address or a past place ("forget my old
# address, I live at 27 Birch Lane now", "forget that I lived in Leeds, I live in York now");
# what a name is now ("forget my cat's name, it's called Max now"). Anything else is forgotten
# with it: a state ("my number is now unlisted", "I'm a patient now"), an elaboration ("I'm
# currently on parole", "我目前在吃药"), one that goes on after the value ("..., the old one was
# 444-1234", "... now and I'm scared", "…，以前叫小红") and one that points back ("我现在住在那里").
# The clause starts at a clause mark (the plugins' boundaries) and, without one, at "and",
# "but", "so", "since" or "because" (after "and", "remember that ..." asks to keep the rest:
# "Please forget my old address and remember that I live at 27 Birch Lane now."), and in
# Chinese where a new clause starts after a full word ("忘掉我的旧地址我现在住在银杏路6号"). When
# unsure, nothing is cut: forgetting too much is the safer error.
_CUT = re.compile(r"\s*[,;:—–]\s*|\s+-\s+")
_JOIN = re.compile(r"\s+(?:and|but|so|since|because)\s+", re.IGNORECASE)
_TELL = re.compile(r"^(?:(?:please|also|just)\s+)*(?:remember|note|know|keep in mind)"
                   r"(?:\s+that)?\s+", re.IGNORECASE)
_OLD_NP = re.compile(r"^(?:my|our)\s+(?:old|previous|former|prior|ex-?)\s*[^\W\d_]['\w-]*"
                     r"(?:\s+[^\W\d_]['\w-]*){0,2}$", re.IGNORECASE)
_VALUE = r"(?:\d[\w./#-]*|[\w.%+-]*@[\w.-]+|[A-Z][a-z][a-z'.-]*)"
_VALUES = re.compile(rf"^{_VALUE}(?:\s+{_VALUE}){{0,5}}(?:\s+now)?\W*$")
"""A new value: words that each start with a digit, hold an @ or are in title case ("27 Birch
Lane", "jon@example.com", "Max"), then at most "now"; case-sensitive. It says no "not"
(`_VALUE_NEGATED`: "it's NOT 12 Oak Street now")."""
_VALUE_NEGATED = re.compile(r"\b(?:not|no|never|except|instead)\b|n't\b", re.IGNORECASE)
_NEW_SAME = re.compile(r"^(?:now[\s,]+)?(?:my|our|the) new (?P<noun>[^\W\d_]['\w-]*"
                       r"(?:\s+[^\W\d_]['\w-]*)?) (?:is|are|will be)\s+(?P<value>\S.*)$",
                       re.IGNORECASE)
_SAME_NOW = re.compile(r"^(?:my|our) (?P<noun>[^\W\d_]['\w-]*(?:\s+[^\W\d_]['\w-]*)?) "
                       r"(?:is|are) now\s+(?P<value>\S.*)$", re.IGNORECASE)
_IT_NOW = re.compile(r"^(?=.*\bnow\b)(?:now[\s,]+)?(?:it's|it is)\s+(?:now\s+)?(?P<value>\S.*)$",
                     re.IGNORECASE)
_VALUE_NOUN = re.compile(r"\b(?:address|home|place|flat|apartment|house|email|e-mail|mail|name|"
                         r"nickname|surname|number|phone|cell|mobile|username|handle)\b",
                         re.IGNORECASE)
"""Only a value is restated so: an address, a name, a number, an email."""
_KINDS: Final = (("address", "home", "place", "flat", "apartment", "house"),
                 ("email", "e-mail", "mail"), ("name", "nickname"))
_ADDRESS_NP = re.compile(r"^(?:my|our)\s+(?:(?:old|previous|former|prior|home|current|street|"
                         r"mailing|postal)\s+)*(?:address|home|place|flat|apartment|house)$",
                         re.IGNORECASE)
_LIVED = re.compile(r"^(?:i|we) (?:lived|used to live|was living|were living)\b", re.IGNORECASE)
_MOVED_TO = re.compile(
    r"^(?:now[\s,]+)?(?:i|we)(?:'m|'re| am| are|'ve| have)?(?:\s+(?:now|just|recently|currently))?"
    r"\s+(?:live|living|moved|relocated)\s+(?:(?:now|just|recently|currently)\s+)?"
    r"(?:at|in|on|to|into|near)\s+(?:the\s+)?(?P<value>\S.*)$", re.IGNORECASE)
_NAME_WORD = re.compile(r"\b(?:nick|sur|dead|maiden)?names?\b|\bcalled\b", re.IGNORECASE)
_RENAMED_NOW = re.compile(
    r"^(?=.*\bnow\b)(?:now[\s,]+)?(?:it|he|she|they|i|we)(?:'s| is|'re| are|'m| am)\s+"
    r"(?:now\s+)?(?:called|named|known as)\s+(?P<value>\S.*)$", re.IGNORECASE)
_RENAMED_NEW = re.compile(
    r"^(?:now[\s,]+)?(?:its|his|her|their|my|our) (?:new (?:nick)?names? (?:is|are)|"
    r"(?:nick)?names? (?:is|are) now)\s+(?P<value>\S.*)$", re.IGNORECASE)
_GOES_ON = re.compile(r"[,;:—–、]|\s-\s")
"""A clause after the replacement keeps it in what is forgotten."""
_SUBORDINATE = re.compile(r"^(?:that\s+)?(?:when|if|because|since|although|though|while|after|"
                          r"before|as|until|once|whenever)\b", re.IGNORECASE)
_ZH_CUT = re.compile(r"\s*[，,；;]\s*")
_ZH_NEW_SAME = re.compile(r"^(?:(?:我|它|他|她)们?的)?新(?:的)?(?P<noun>[一-鿿]{1,4}?)"
                          r"(?:是|为|叫|：|:)")
_ZH_KINDS: Final = (("手机", "电话", "号码"), ("地址", "住址", "住处"), ("名字", "称呼", "叫"))
_ZH_VALUE = re.compile(r"地址|住址|住处|名字|称呼|号码|手机|电话|邮箱|账号")
_ZH_ADDRESS_NP = re.compile(
    r"^我们?的?(?:旧|以前的|原来的|之前的|原先的|从前的)?(?:家庭?|家的)?(?:地址|住址|住处|家)$")
_ZH_LIVED = re.compile(r"^我们?(?:以前|之前|原来|原先|从前|曾经)(?:住在|住过)")
_ZH_MOVED_TO = re.compile(r"^(?:现在|目前|如今)?(?:我们?)?(?:现在|目前|如今|已经|刚刚?)?"
                          r"(?:住在|搬到|搬去|搬进|搬来|住到)(?![那这])\S")
_ZH_RENAMED = re.compile(
    r"^(?:(?:我|它|他|她)们?(?:现在|如今)(?:改)?叫|(?:现在|如今)(?:它|他|她)?们?(?:改)?叫|"
    r"(?:(?:我|它|他|她)们?)?现在的名字(?:是|叫))\S")
_ZH_GOES_ON = re.compile(r"[，,；;、]|以前|旧|原来|原先|之前|从前|曾经|但|因为|由于|所以|而且|不过|"
                         r"可是|然后|并且|和|跟|一起|为了|还|也|不是")
"""A further clause or a word that goes on ("…，以前叫小红", "…，我现在还住在杭州") keeps the
replacement in what is forgotten."""
_ZH_CLAUSE_START = re.compile(r"(?:而且|但是?|不过|可是)?(?:现在|目前|如今|我们?|它|他|她|新)")
"""Where a Chinese clause may start without a mark ("我现在住在...", "现在叫..."): the rest is
read for a replacement only there, and only after a full word (not inside "我现在")."""
_ZH_SUBORDINATE = re.compile(r"^(?:当|如果|要是|因为|虽然|自从)")
# How far a request reaches beyond a clear replacement (`_separate`, never the plugins' words
# or the key): the rest is a new fact when the asked part names an old value the rest replaces
# and the rest is the user's or its own clause of the change ("Forget my old job, I'm now a
# teacher at Lincoln High.", "忘掉我以前那辆车，我现在开特斯拉"), or when the rest opens as a
# fact of its own ("..., and remember that I'm engaged now", "..., but remember my sister's
# number is 555-0101", "..., and by the way I moved to Boston", "…，另外…"). An elaboration
# ("forget that I'm pregnant, I'm currently in my second trimester") and a change with no old
# value named ("forget that I'm vegetarian, I eat meat again now", "Forget my 3pm meeting, it
# moved to 4pm.") stay in the request: forgetting too much is the safer error.
_OLD_MARK = re.compile(r"\b(?:old|previous|former|prior|earlier|ex)\b|\bex-", re.IGNORECASE)
_NOW_CLAUSE = re.compile(
    r"^(?:(?:and|but|so)\s+)?(?:i|i'm|i've|i'd|we|we're|we've|my|our|it|it's|its|they|they're)\b"
    r"(?=.*\b(?:now|again|no longer|anymore|any more|moved|changed|switched|instead|nowadays|"
    r"these days)\b)", re.IGNORECASE)
_SEPARATE = re.compile(
    r"^(?:(?:and|but)\s+)?(?:(?:please|also)\s+)*(?:remember(?:\s+that)?|by the way|btw|also|"
    r"note that|keep in mind)\b", re.IGNORECASE)
_CHANGE_KIND = re.compile(
    r"\b(?:job|work|position|role|title|employer|company|office|team|boss|manager|car|bike|"
    r"vehicle|phone|laptop|computer|school|class|course|major|gym|club|bank|doctor|dentist)\b",
    re.IGNORECASE)
_ZH_CHANGE_KIND = re.compile(r"工作|职位|岗位|公司|单位|老板|车|手机|电脑|学校|专业|班|"
                             r"健身房|俱乐部|银行|医生")
"""The kinds whose old value a change replaces (a job, a car), never a state of the user's
("my previous diagnosis, I'm now in therapy", "我以前的病，我现在在吃药": an elaboration,
forgotten with it)."""
_NEGATED = re.compile(r"\b(?:not|never)\b|n't\b|不是", re.IGNORECASE)
_ZH_OLD_MARK = re.compile(r"以前|之前|原来|原先|旧|从前|过去|曾经")
_ZH_NOW_CLAUSE = re.compile(r"^(?:而且|但是?|不过)?(?:我们?|它|那|现在|目前|如今)"
                            r"(?=.*(?:现在|改成|改到|改为|换成|搬到|已经|不再|又|如今|目前))")
_ZH_SEPARATE = re.compile(r"^(?:但是?|不过|而且)?请?(?:另外|顺便|记住|还有|记得)")
_TERM_STOP: Final = frozenset((
    "a an the and or but of to in on at for from with by as is am are was were be been being it "
    "its this that these those i me my mine myself we us our you your user user's users please "
    "forget remember about something thing things fact anything do does did not no never any all "
    "some just also really very").split())
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+")
_CJK_FILLER = re.compile(
    r"我们|你们|咱们|这件事|那件事|这个|那个|用户|我|你|的|了|吧|呢|啊|请|把|给")
_LABEL = re.compile(r"^\s*([A-Z][\w'\u2019.-]*(?: [A-Z][\w'\u2019.-]*){0,2}):[ \t]+(?=\S)")
_ZH_EDGES = " \t，,：:。！？.!?；;、"
_SPEAKER = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)
_NAMES_MEMORY = re.compile(rf"\b(?:from|in|into|out of) {_MEMORY}\b", re.IGNORECASE)
_TRAILING_MEMORY = re.compile(rf"\s*,?\s*(?:from|out of|in) {_MEMORY}\W*$", re.IGNORECASE)
_TRAILING = re.compile(r"(?:[\s,]+(?:please|for me|thanks|thank you))*" + _END,
                       re.IGNORECASE)
_DISMISSAL = re.compile(r"^(?:about )?(?:it|this|that|them|everything)\b(?:\W*$|\s*[,—–-].*$)",
                        re.IGNORECASE | re.DOTALL)
_RETRACTION = re.compile(r"^(?:that |what )?i (?:just |ever )?asked\b", re.IGNORECASE)
_REFERENCE = re.compile(
    r"^(?:(?:it|this|that|those|these)(?: information| detail| details)?|"
    r"(?:what|everything|all|anything|whatever) (?:that )?i (?:just |have |'ve |had )?"
    r"(?:said|told you|mentioned|shared|wrote|typed)(?: (?:earlier|before|just now))?)\W*$",
    re.IGNORECASE)
_TOLD = re.compile(
    r"^(?:the|that|this|any|all|those|these)\s+(?:detail|details|fact|facts|information|info|"
    r"preference|preferences|memory|memories|thing|things|part|stuff|bit)\s+"
    r"(?:about|that|of|regarding|concerning|on)\b", re.IGNORECASE)
_CLAUSE = re.compile(r"\s*[,;:—–]\s*|\s+-\s+")
_KNOWN = re.compile(r"^(?:that\s+)?you(?:'ve| have)?\s+(?:remember(?:ed)?|know|knew|noted|learned|"
                    r"stored|saved)\s+(?:that\s+|about\s+)?", re.IGNORECASE)
_SUBJECT_THAT = re.compile(r"^that\s+(?=(?:i|i'm|i've|i'd|my|me|we|our|it's|it|there)\b)",
                           re.IGNORECASE)


@dataclass(frozen=True)
class ForgetRequest:
    """One request found in an Add: where it is and the statement it asks to forget.

    Attributes:
        message (int): The index of its user message in the Add.
        statement (str): What to forget, in the user's words (at most
            ``MAX_FORGET_STATEMENT_CHARS`` characters). A named speaker's request is
            written as its transcript writes it ("Leah: I do swimming"), so the judge
            reads whose fact the first person is, as the memories name that speaker.
    """

    message: int
    statement: str


def _fold(text: str) -> str:
    """The text with curly apostrophes straight, character for character."""
    return text.replace("’", "'").replace("ʼ", "'")


def _bounded(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_FORGET_STATEMENT_CHARS:
        return text
    cut = text[:MAX_FORGET_STATEMENT_CHARS]
    return cut[:cut.rfind(" ")] if " " in cut else cut


def _object(raw: str) -> str:
    """What follows the verb, without the memory it names and trailing courtesy."""
    value = _TRAILING.sub("", raw.strip())
    value = _KNOWN.sub("", _TRAILING.sub("", _TRAILING_MEMORY.sub("", value)).strip())
    return _SUBJECT_THAT.sub("", value).strip()


def _speaker(message: InputMessageV1) -> str:
    """Who wrote a user message: its ``speaker``, else its leading label, else nobody named."""
    if message.speaker is not None:
        return message.speaker
    label = _LABEL.match(message.content)
    return "" if label is None else label.group(1)


@dataclass(frozen=True)
class _Found:
    """A request sentence's object (``target``), what the request asks to forget as the plugins
    read it (``kept``: all of it, or what comes before a clear replacement after a clause mark,
    `_kept`), how far the request reaches in its sentence (``reach``: before a clear
    replacement after a conjunction too, "since I moved to 27 Birch Lane"; empty for
    ``kept``); ``back`` when it refers back to what the user said before."""

    target: str
    kept: str
    back: bool = False
    reach: str = ""


def _terms(text: str) -> set[str]:
    """The plugins' content terms of a text: Latin words without function words (a plural "s"
    dropped), and the character pairs of CJK runs without fillers."""
    folded = _fold(text).lower()
    found: set[str] = set()
    for word in re.findall(r"[^\W_][\w'-]*", _CJK_RUN.sub(" ", folded)):
        if word in _TERM_STOP or len(word) < 2:
            continue
        found.add(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss")
                  else word)
    for run in _CJK_RUN.findall(folded):
        chars = _CJK_FILLER.sub("", run)
        if len(chars) == 1:
            found.add(chars)
        found.update(chars[index:index + 2] for index in range(len(chars) - 1))
    return found


def _kind(text: str, *, zh: bool) -> int:
    """The index of the first kind (`_KINDS`, `_ZH_KINDS`) a text names, else -1."""
    if zh:
        return next((index for index, words in enumerate(_ZH_KINDS)
                     if any(word in text for word in words)), -1)
    have = _terms(text)
    return next((index for index, words in enumerate(_KINDS) if have & set(words)), -1)


def _replaces(kept: str, rest: str, *, zh: bool) -> bool:
    """Whether ``rest``, the last clause after what a request forgets (``kept``), clearly
    replaces it with a new value and says nothing else (the comment above `_CUT`; the
    plugins' ``replaces``)."""
    if zh:
        if _ZH_GOES_ON.search(rest):
            return False
        found = _ZH_NEW_SAME.match(rest)
        same = found is not None and _ZH_VALUE.search(found["noun"]) is not None and (
            bool(_terms(found["noun"]) & _terms(kept)) or (
                _kind(found["noun"], zh=True) >= 0
                and _kind(found["noun"], zh=True) == _kind(kept, zh=True)))
        return bool(same or ((_ZH_ADDRESS_NP.match(kept) or _ZH_LIVED.match(kept))
                             and _ZH_MOVED_TO.match(rest))
                    or ("名" in kept or "叫" in kept) and _ZH_RENAMED.match(rest))
    rest = _TELL.sub("", rest)
    if _GOES_ON.search(rest):
        return False

    def valued(pattern: re.Pattern[str]) -> re.Match[str] | None:
        found = pattern.match(rest)
        return found if found is not None and _VALUES.match(found["value"]) and not (
            _VALUE_NEGATED.search(found["value"])) else None

    found = valued(_NEW_SAME) or valued(_SAME_NOW)
    noun = "" if found is None else found["noun"]
    one = noun.lower() == "one"
    value = (_OLD_NP.match(kept) is not None and _VALUE_NOUN.search(kept) is not None) if one \
        else _VALUE_NOUN.search(noun) is not None
    same = found is not None and value and (
        one or bool(_terms(noun) & _terms(kept))
        or (_kind(noun, zh=False) >= 0 and _kind(noun, zh=False) == _kind(kept, zh=False)))
    return bool(same
                or (_OLD_NP.match(kept) and _VALUE_NOUN.search(kept) and valued(_IT_NOW))
                or ((_ADDRESS_NP.match(kept) or _LIVED.match(kept)) and valued(_MOVED_TO))
                or (_NAME_WORD.search(kept) and (valued(_RENAMED_NOW) or valued(_RENAMED_NEW))))


def _kept(target: str, *, zh: bool, marks: bool = False) -> str:
    """What a request asks to forget of its object: the words before the first clear
    replacement after them (the comment above `_CUT`), else all of them. A target that opens
    with a subordinate clause ("when I was young, I lived in Paris") is never cut. With
    ``marks``, only a clause mark ends it, as the plugins cut (the request's own words, its
    request key); without, a conjunction or a Chinese clause without a mark does too (how
    far the request reaches, so what the user told after it is stored)."""
    if (_ZH_SUBORDINATE if zh else _SUBORDINATE).match(target) or (
            any(char.isupper() for char in target) and not any(char.islower() for char in target)):
        return target  # A sentence in capitals tells no value from a word: never cut.
    cuts = [(found.start(), found.end())
            for found in (_ZH_CUT if zh else _CUT).finditer(target)]
    if zh and not marks:
        cuts += [(index, index) for index in range(2, len(target))
                 if target[index - 1] not in "我们它他她你的新" and _ZH_CLAUSE_START.match(
                     target, index)]
    elif not marks:
        cuts += [(found.start(), found.end()) for found in _JOIN.finditer(target)]
    for start, end in sorted(cuts):
        kept = target[:start].strip(_ZH_EDGES if zh else None)
        if kept and (_replaces(kept, target[end:], zh=zh)
                     or not marks and _separate(kept, target[end:], zh=zh)):
            return kept
    return target


def _separate(kept: str, rest: str, *, zh: bool) -> bool:
    """Whether ``rest``, told after what a request asks to forget (``kept``), is a fact of its
    own the request does not reach (the comment above `_OLD_MARK`). A change of a value
    (address, number, email, name) is read by `_replaces` alone, and a change that is
    negated or sensitive ("I'm now in therapy", "我现在在吃药") is an elaboration."""
    if (_ZH_SEPARATE if zh else _SEPARATE).match(rest):
        return True
    if (_ZH_VALUE if zh else _VALUE_NOUN).search(kept) or _NEGATED.search(rest) or classify(
            [kept, rest]):
        return False
    if zh:
        return bool(_ZH_CHANGE_KIND.search(kept) and _ZH_OLD_MARK.search(kept)
                    and _ZH_NOW_CLAUSE.match(rest))
    return bool(_CHANGE_KIND.search(kept) and _OLD_MARK.search(kept)
                and _NOW_CLAUSE.match(rest))


def _english(sentence: str, *, assistant: bool, complete: bool = False) -> _Found | None:
    """The object of an English request sentence, else None (``complete``: the ``agent``
    mode's forms and limits too).

    ``assistant`` says whether the Add is a conversation with the memory.
    """
    text = _fold(sentence).strip()
    if complete:
        text = _LATER.sub("", text)
    names_memory = _NAMES_MEMORY.search(text) is not None
    match = _FORGET.match(text)
    verb: Literal["forget", "remember", "record"]
    if match is not None:
        verb = "forget"
    elif (match := _REMEMBER.match(text)) is not None:
        verb = "remember"
    elif (match := _RECORD.match(text)) is not None:
        verb = "record"
    else:
        return None
    if text.endswith("?") and not (match.group("polite") or "").strip().lower().startswith(
            ("can", "could", "would")):
        return None
    if not (assistant or names_memory):
        return None
    polite = bool((match.group("polite") or "").strip())
    if verb == "forget" and match.group("verb").lower().endswith("about") and not (
            polite or names_memory):
        return None
    target = _object(match.group("object"))
    if complete:
        target = _TRAILING_AGENT.sub("", target)
    if complete and verb == "remember" and not _ASKS_THAT.match(match.group("object").strip()) and (
                _CORRECTING.search(match.group("object"))
                or _REMEMBER_ALL.match(match.group("object").strip())):
        return None
    if complete and verb == "remember" and re.match(
            r"(?:do not|don't|dont|never)\b", match.group("verb"), re.IGNORECASE) and not (
                polite or names_memory):
        # A bare "don't remember ..." is as often the user's own "(I) don't remember my
        # password, can you remind me": only a marked one asks (`_RECALL`).
        back = _SAID_BACK.match(target) or _BACK.match(target)
        shouted = not any(char.islower() for char in text)
        marked = (_PLEASE.search(text) if re.match(r"never\b", match.group("verb"), re.I) else (
            (not back and _ASKS_THAT.match(match.group("object").strip()))
            or (not shouted and (re.match(r"do not\b", match.group("verb"), re.IGNORECASE)
                                 or _EMPHASIS.search(match.group("verb"))))
            or _CLOSING.search(text)))
        if not marked or _RECALL.search(match.group("object")):
            return None
    full = target
    if complete:
        for boundary in _CLAUSE.finditer(target):
            rest = target[boundary.end():]
            if (reset := _RESET.match(rest)) is None:
                continue
            before = target[:boundary.start()].strip()
            if _NEVERMIND.match(rest) or not _personal(before) or any(
                    rule.match(before) for rule in (_SAID_BACK, _BACK, _REFERENCE, _RETRACTION)):
                return None  # "Forget what I just said, let's start over.": a dismissal.
            if not _RESET_GOES_ON.search(rest[reset.end():]):
                target = before  # "Please forget that I'm pregnant, let's move on.": it closes.
            break
    if complete and (head := _SAID_HEAD.match(target)) is not None and not _SPEAKER.search(
            target[head.end():]) and _CONVERSING.search(target):
        return None
    if _RETRACTION.match(target) or (complete and _SAID_BACK.match(target)):
        return None
    if verb == "forget" and not names_memory and _DISMISSAL.match(target):
        return None
    if _REFERENCE.match(target):
        return _Found("", "", back=True)
    if complete and _BACK.match(target):
        return None
    clause, *after = _CLAUSE.split(target)
    work = _WORK_HEAD.match(target) if complete else None
    if work is not None and (_WORK_TAIL.match(target[work.end():])
                             or _WORK_CONTEXT.search(target[work.end():])):
        return None
    if complete and (_NOT_A_FACT.match(target) or _SAID_WHOLE.match(target)
                     or _DISPUTE.match(target)
                     or any(_REPLACED.search(part) for part in after)
                     or _USE_INSTEAD.search(target)):
        return None
    named = complete and _en_named(target)
    if len(target.split()) < 2 or not (
            names_memory or _SPEAKER.search(clause) or _TOLD.match(clause) or named):
        return None
    if not complete:
        return _Found(full, target)
    reach = _kept(target, zh=False)
    kept = _kept(target, zh=False, marks=True)
    return _Found(full, kept, reach="" if reach == kept else reach)


def _chinese(sentence: str, *, assistant: bool, complete: bool = False) -> _Found | None:
    """The object of a Chinese request sentence, else None (``complete``: the ``agent``
    mode's forms and limits too)."""
    text = sentence.strip()
    if complete:
        text = _LATER.sub("", text)
        if _ZH_YOU_FORGOT.match(text):
            return None
    match = _ZH.match(text)
    if match is not None and not (assistant or match.group("deleted") is not None):
        return None
    if match is not None:
        target = (match.group("object") or match.group("deleted") or "").strip(_ZH_EDGES)
        if complete and match.group("deleted") is not None and _ZH_WORK.search(target):
            return None  # "删除我的提交记录": work, not a memory.
    elif complete and assistant and (extended := _ZH_COMPLETE.match(text)) is not None:
        target = (extended.group("object") or extended.group("bare") or "").strip(_ZH_EDGES)
        if (extended.group("baverb") or "").startswith(("删", "去")):
            # 把…删掉 forgets only what it names as a memory ("把你记住的我的地址删掉").
            if not (_ZH_MEMORY_HEAD.match(target) or _ZH_MEMORY_TAIL.search(target)):
                return None
            target = _ZH_MEMORY_TAIL.sub("", _ZH_MEMORY_HEAD.sub("", target))
            if _ZH_WORK.search(target):
                return None
    else:
        return None
    verb = None if match is None else match.group("verb")
    if complete and match is not None and verb in ("忘记", "忘掉") and (
            match.group("object") or "").startswith("了"):
        return None  # "忘记了我的生日": what the user forgot.
    target = re.sub(r"^(?:关于|有关)", "", target).strip(_ZH_EDGES)
    if complete and verb == "忘记" and _ZH_FORGOT_NP.match(target) and not _ZH_POLITE.match(text):
        return None  # "忘记我的密码了": what the user forgot, not a fact ("忘掉我怀孕了").
    if complete and _ZH_YOU_ASKED.match(text) and (_ZH_ASKED_END.search(text) or (
            _ZH_GUESS_END.search(text) and _ZH_FORGOT_NP.match(target))):
        return None  # "你忘记我的生日了吗": a question to the assistant.
    if complete and _ZH_BACK.match(target):
        return None  # "不要记我刚才说的", "忘掉我刚才问的问题": the plugins' alone, or none.
    if complete and _ZH_DISPUTE.match(_ZH_PARTICLES.sub("", target)):
        return None
    full = target
    if complete:
        for boundary in _ZH_CUT.finditer(target):
            rest = target[boundary.end():]
            if (reset := _ZH_RESET.match(rest)) is None:
                continue
            before = target[:boundary.start()].strip()
            if _ZH_NEVERMIND.match(rest) or not _personal(before) or _ZH_BACK.match(before):
                return None
            if not _ZH_RESET_GOES_ON.search(rest[reset.end():]):
                target = before  # "忘掉我怀孕了，说点别的吧": the closing is no part of it.
            break
    if complete and (_ZH_NOT_A_FACT.match(target)
                     or _ZH_SAID_RESET.match(_ZH_PARTICLES.sub("", target))
                     or any(_ZH_INSTEAD.match(part) for part in re.split("[，,；;]", target)[1:])):
        return None  # "忘掉我的代码，重新开始": work; "忘掉我们聊过的东西": starting over.
    if "我" in target and len(target) >= 3:
        if not complete:
            return _Found(full, target)
        reach = _kept(target, zh=True)
        kept = _kept(target, zh=True, marks=True)
        return _Found(full, kept, reach="" if reach == kept else reach)
    if not complete or verb != "忘掉":
        return None
    # Another person by name, after 忘掉 only: "忘掉阿杰", "忘掉小李的名字", "忘掉Emma的名字".
    named = _ZH_PARTICLES.sub("", target)
    return _Found(named, named) if _zh_named(named) else None


def _zh_named(text: str) -> bool:
    """Whether a Chinese object is a person called by name or a detail of one: "阿杰", "小李的
    名字", "Emma的名字" (`_ZH_NAME`, `_ZH_NAMED`)."""
    found = _ZH_NAMED.match(text)
    owner = text if found is None else found["owner"]
    return bool((_ZH_NAME.match(owner) and owner not in _ZH_A_WORDS) or (
        found is not None and _LATIN_NAME.match(owner) and owner.lower() not in _TECH))


def _en_named(text: str) -> bool:
    """Whether an English object is another person's name: "Ben's name" (`_EN_NAMED`), never a
    software's, a day's or one written in capitals."""
    found = _EN_NAMED.match(text)
    return found is not None and not (
        found["owner"].lower() in _NOT_OWNER | _TECH | CALENDAR or found["owner"].isupper())


def _personal(text: str) -> bool:
    """Whether what a request forgets is about the user or someone of theirs: I, me, my or 我,
    or a person called by name (the plugins' ``personal``, as far as this module reads names)."""
    value = text.strip()
    return bool(_SPEAKER.search(value) or "我" in value or _en_named(value)
                or _zh_named(_ZH_PARTICLES.sub("", value)))


def _contentful(sentence: str) -> bool:
    """Whether a sentence says something: a content term that is no filler ("Sorry.")."""
    return any(term not in _FILLER for term in _terms(_ZH_FILLER.sub("", sentence)))


def _request(sentence: str, *, assistant: bool, complete: bool) -> _Found | None:
    return (_english(sentence, assistant=assistant, complete=complete)
            or _chinese(sentence, assistant=assistant, complete=complete))


def forget_requests(
    messages: Sequence[InputMessageV1], *, complete: bool = False,
) -> tuple[ForgetRequest, ...]:
    """Find the explicit forget requests of an Add's user messages (module docstring).

    Args:
        messages: The Add's messages in order.
        complete: Whether the ``agent`` mode's Chinese forms count too.

    Returns:
        tuple[ForgetRequest, ...]: At most ``MAX_CHAT_FORGETS`` requests in message
            order, each statement once.

    Examples:
        ```python
        from quotemem.core.run_inputs import InputMessageV1

        found = forget_requests([
            InputMessageV1(role="user", content="Please forget that I like jazz."),
            InputMessageV1(role="assistant", content="Done."),
        ])
        assert [request.statement for request in found] == ["I like jazz"]
        ```
    """
    users = [message for message in messages if message.role == "user"]
    assistant = any(message.role == "assistant" for message in messages) or len(
        {_speaker(message) for message in users}) <= 1
    found: list[ForgetRequest] = []
    seen: set[str] = set()
    previous = ""
    for index, message in enumerate(messages):
        if message.role != "user":
            continue
        before = ""
        speaker = _speaker(message)
        label = _LABEL.match(message.content)
        content = message.content if label is None else message.content[label.end():]
        if complete and _COMMAND.match(content):
            previous = content
            continue
        pieces = ([content[start:end] for start, end in request_sentences(content)] if complete
                  else [sentence.group() for sentence in SENTENCE.finditer(content)])
        for position, text in enumerate(pieces):
            hit = _request(text, assistant=assistant, complete=complete)
            following = pieces[position + 1] if position + 1 < len(pieces) else ""
            if complete and hit is not None and not hit.back and (
                    _NEXT_INSTEAD.match(following) or _ZH_INSTEAD.match(following.strip())):
                hit = None  # "Forget my branch name. Use main instead.": instructions.
            if complete and hit is not None and (
                    _RESET.match(following) or _ZH_RESET.match(following)):
                # "Forget my phone number. Never mind.", "Forget the stuff about the movie.
                # Let's talk about work.": a dismissal; after a fact of the user's it closes.
                if hit.back or not _personal(hit.kept) or _NEVERMIND.match(following) or (
                        _ZH_NEVERMIND.match(following)):
                    hit = None
            statement: str | None
            if hit is None:
                statement = None
            elif hit.back:
                statement = before.strip() or previous.strip() or None
            else:
                statement = hit.kept
            if statement:
                statement = _bounded(f"{speaker}: {statement}" if speaker else statement)
                if statement not in seen and len(found) < MAX_CHAT_FORGETS:
                    seen.add(statement)
                    found.append(ForgetRequest(index, statement))
            if not complete or (hit is None and _contentful(text)):
                # What a later "forget what I just said" refers to: a sentence saying
                # something, not a request or a filler ("Ignore that;").
                before += text
        previous = content
    return tuple(found)


_OLD_EN = re.compile(
    r"^(?P<owner>my|our)\s+(?:old|former|previous|earlier|prior|past)\s+"
    r"(?P<noun>[a-z][a-z'-]*(?: [a-z][a-z'-]*){0,2})$", re.IGNORECASE)
_OLD_ZH = re.compile(
    r"^我(?P<plural>们)?的?(?:以前|之前|原来|过去|旧|上一)的?(?:那|这)?"
    r"(?:个|辆|只|条|部|台|套|份|家|位|任|款|间)?的?(?P<noun>[\u4e00-\u9fff]{1,6})$")
_OLD_LIVED_EN = re.compile(
    r"^(?P<who>I|we) (?:lived|used to live|was living|were living) (?P<rest>(?:in|at|on|near) .+)$",
    re.IGNORECASE)
_OLD_LIVED_ZH = re.compile(r"^(?P<who>我们?)(?:以前|之前|原来|过去|曾经|曾)住在(?P<rest>.+)$")
_WHOLE_EN = re.compile(
    r"^(?:everything|all|anything)\s+(?:you know\s+)?(?:about|regarding)\s+(?P<who>.+)$",
    re.IGNORECASE)
_WHOLE_ZH = re.compile(
    r"^(?:关于|有关)?(?P<who>[^，,。；;\s]+?)的(?:一切|所有事情?|所有|全部)$")
_WHOLE_NAME_EN = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)?$")
_WHOLE_TIE_EN = re.compile(r"^(?:my|our)\s+(?P<tie>[a-z][a-z' -]*)$", re.IGNORECASE)
_WHOLE_TIE_ZH = re.compile(r"^我们?的?(?P<tie>[\u4e00-\u9fff]{1,6})$")


def _said_object(statement: str) -> str:
    """The object of a statement that is a whole request sentence, as a client may pass the
    user's own words ("Forget my old number, my new number is 555-9876." reads "my old number,
    my new number is 555-9876", "forget that I lived in Shanghai" "I lived in Shanghai"); the
    statement itself otherwise (the plugins' words, an in-chat request's object)."""
    text = statement.strip()
    found = _request(text, assistant=True, complete=True)
    return text if found is None or found.back or not found.target else found.target


def old_value(statement: str) -> str | None:
    """The plain words of a request for an old value of the user's: "my old address" gives
    "my address" and "我以前那辆车" "我的车" (`StatementForget`'s old-value rule takes the
    memories of that kind a later one replaced); None for any other request, for where the user
    lived (`lived_value`) and for a person ("my old friend" is a long-standing one). A whole
    request sentence is read by its object (`_said_object`)."""
    text = _said_object(statement)
    text = _kept(text, zh=not text.isascii())  # "my old address since I moved to ..."
    if (found := _OLD_EN.match(text)) is not None:
        if found["noun"].split()[-1].lower() in TIES:
            return None
        return f"{found['owner']} {found['noun']}"
    if lived_value(text) is not None:
        return None  # Where the user lived: `lived_value`, read for past values only.
    if (found := _OLD_ZH.match(text)) is not None:
        if any(found["noun"].endswith(word) for word in TIES if not word.isascii()):
            return None
        return f"我{found['plural'] or ''}的{found['noun']}"
    return None


def asked_part(statement: str) -> str:
    """The words of a request's object it asks to forget, before a clear replacement said
    after them with or without a clause mark: "my old address" of "my old address since I
    moved to 27 Birch Lane". What hides or tombstones a forgotten fact reads these, never the
    new value the user told with it."""
    text = statement.strip()
    return _kept(text, zh=not text.isascii())


def lived_value(statement: str) -> str | None:
    """Where a request says the user lived ("I lived in Shanghai as a kid", "我以前住在上海"), in
    the present tense the stored past value states it ("I live in Shanghai as a kid"), else None.
    Such a request takes only past values: a memory a later one replaced or ended, or one
    stated in the past tense; never where the user lives now (`StatementForget`). A whole
    request sentence ("忘掉我以前住在上海") is read by its object (`_said_object`)."""
    text = asked_part(_said_object(statement))
    if (found := _OLD_LIVED_EN.match(text)) is not None:
        return f"{found['who']} live {found['rest']}"
    if (found := _OLD_LIVED_ZH.match(text)) is not None:
        return f"{found['who']}住在{found['rest']}"
    return None


def tells_new(statement: str) -> bool:
    """Whether a request's words carry a clear replacement after what they ask to forget
    ("my old address since I moved to 27 Birch Lane", "我的旧地址，我现在住在银杏路6号")."""
    return asked_part(statement) != statement.strip()


def whole_referent(statement: str) -> tuple[Literal["name", "tie"], str] | None:
    """Whom a request to forget a person as a whole names: "everything about my ex" a tie
    ("my ex"), "everything about Jordan" and "忘掉阿杰" a name, "我前男友的一切" a tie; None
    for any other request (`StatementForget`'s whole-person rule takes every memory of exactly
    that one person, and nothing when the words name several or none). A whole request
    sentence is read by its object (`_said_object`)."""
    text = _ZH_PARTICLES.sub("", _said_object(statement))
    who = None
    if (found := _WHOLE_EN.match(text)) is not None:
        who = found["who"].strip()
        if _WHOLE_NAME_EN.match(who):
            return "name", who
        return ("tie", who) if _WHOLE_TIE_EN.match(who) else None
    if (found := _WHOLE_ZH.match(text)) is not None:
        who = found["who"]
    if who is None:
        who = text
        return ("name", who) if _ZH_NAME.match(who) and who not in _ZH_A_WORDS else None
    if _ZH_NAME.match(who) and who not in _ZH_A_WORDS:
        return "name", who
    return ("tie", who) if _WHOLE_TIE_ZH.match(who) else None


def is_request(text: str, *, complete: bool = False) -> bool:
    """Whether a text, read as one user message, makes a forget request: it records a request.

    An in-chat forget never forgets or marks the record of its own request: the memory
    that only quotes it ("The user asked to forget that ...") and the request sentence stay.
    ``complete`` reads the ``agent`` mode's forms too.
    """
    return bool(forget_requests([InputMessageV1(role="user", content=text)], complete=complete))


def request_reach(sentence: str, *, complete: bool = True) -> int | None:
    """Where the request of a sentence ends, read as one user message of a conversation with
    the memory: after the words it asks to forget (`_kept`), so a clear replacement after them
    ("Forget my old address, I live at 27 Birch Lane now.") is no part of the request; the
    whole sentence for any other request; None for a sentence that makes none.

    Args:
        sentence: One sentence of a user message, a speaker's label (``Mira:``) allowed.
        complete: Whether the ``agent`` mode's forms count too.

    Returns:
        The character offset in ``sentence`` where its request ends, or None.
    """
    label = _LABEL.match(sentence)
    offset = 0 if label is None else label.end()
    body = sentence[offset:]
    found = _request(body, assistant=True, complete=complete)
    if found is None:
        return None
    reached = found.reach or found.kept
    if found.back or reached == found.target:
        return len(sentence)
    at = _fold(body).find(found.target)
    return len(sentence) if at < 0 else offset + at + len(reached)


def requested(text: str, *, complete: bool = False) -> tuple[str, ...]:
    """The statements a text, read as one user message, asks to forget (`forget_requests`;
    ``complete`` reads the ``agent`` mode's forms too)."""
    return tuple(found.statement for found in forget_requests(
        [InputMessageV1(role="user", content=text)], complete=complete))
