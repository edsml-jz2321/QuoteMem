# Forgetting

A forget takes the memories it matches out of Search, History, Provenance
and the memory list, and withholds the words it removes. Its tombstone is
then checked against every later Add: a restatement with the same words is
refused, and a close one goes to review, which is shown the forgotten
statement. The `agent` Add mode makes one exception: the tombstone does not
refuse the user's own statement of the fact after the forget, and review
judges it as a new fact. That statement must meet every one of these
conditions:

- It is in a user message that the service received after the forget, by
  the service's clock. The message's own date must be after the forget too,
  so a late batch of older messages never counts.
- It states something: it is not a question or a forget request, and it does
  not name forgetting, erasing or the memory ("forget", "remember",
  "delete", "忘", "记住" and the like).
- It states the forgotten fact: it holds at least two of the fact's content
  words (all of them for a shorter fact) and at least half of them, or the
  message's embedding has a cosine of 0.6 or more to the forgotten statement.
- It shows no name a forget took, unless its words give that name (see the
  names below). After the forget of "I work at Ruijin Hospital", which also
  took "A hospital is named Ruijin Hospital.", "I still work at Ruijin
  Hospital." shows the hospital's name, so it is cut around the name or
  refused.

The exception never covers:

- the assistant's messages;
- the forget request and the words it quotes;
- a question;
- a message without a date, or one dated or received before the forget, such
  as a late batch from another tool;
- a later user sentence about something else, such as one that only names the
  place of a forgotten fact ("I visited Ruijin Hospital yesterday."): it is
  checked as a fact of its own.

Search does not drop such a statement, or a memory resting on it, for the
forgotten words it states again. The old messages stay withheld. The
`lean-local` and `full` Add modes make no exception.

In the `agent` Add mode, a forget also works inside sentences and on names:

- A forget of one detail of a sentence withholds the words of that detail
  and leaves readable the words that state the sentence's other facts, in
  Chinese and mixed-script sentences too, so their memories keep their
  quotes. After "请忘掉我的猫的名字" on "我的猫叫Bob，它三岁了", the memories
  that the user has a cat and that it is three years old can stay.
- When the detail cannot be cut out cleanly, for example when it is a word
  inside the user's phrase ("我的橘猫" for the cat's colour), the whole quote
  is withheld, and a memory that rested only on it can no longer be recalled
  (listed as `unreadable` in `affected`).
- A request to forget the name of the user's own person or pet, ending with
  the name phrase ("forget my brother's name", "forget my friend Alice's name",
  "忘掉我弟弟的名字", "忘掉我朋友小雪的名字") takes the memories that give that
  name, when they are about just one person or pet. Other memories about that
  person or pet that show the name are reworded to leave it out, or taken when
  they cannot be reworded; the prepare lists each in `affected` with its effect
  (`reworded`, `unreadable` or `forgotten`). A request for a nickname ("my
  cat's nickname", "小名") is handled like any other forget.
- A memory or a saved sentence that names a name, value or number of a
  forgotten memory as not, or no longer, the case is read too: "I live at 27
  Pine Lane now, not 14 Oak Street." beside a forgotten old address. For
  each target of the forget not marked `bundled` (a target that also states
  an unrelated fact, which goes whole), the rule takes one that names a name,
  number, code or e-mail address the target states as its own in a contrast
  beside another (", not ...", ", no longer ...", "rather than ...",
  "instead of ..."). It also takes one in which a clause whose subject is the
  user ("I", "we", "the user") names such a value after a word that says it
  ended ("no longer", "stopped", "quit", "used to", "previously",
  "formerly", "previous", "former", or "from" after "moved", "switched",
  "changed" or "transferred") or before "anymore": "I no longer work at
  ...", "we moved from ...". In a clause whose subject is someone or
  something else, those words take a forgotten number, code or e-mail
  address that follows them ("the previous one was 07700 900 552") or comes
  before "anymore". Chinese sentences are read with their own contrast and
  ending words. The model checks other contrasts
  against the forgotten memory or sentence. One that the rule or the model
  takes is reworded without the old value or forgotten with it, and the
  prepare lists which. A quote or a sentence that names the old value is
  withheld, the part that names it or the whole. The rule reads a number or
  code in other written forms too: with or without the spaces and hyphens
  between its digits, with one to three capital letters before it apart or
  together ("BA 2490" or "BA2490"), and, with eight digits or more, with or
  without a country code or a leading 0. A number in a date's shape ("12-07",
  "40-47") is read only as written, unless it has six digits or more and
  could not be a date ("40-47-84" is "404784").
- A memory that states the current value beside a forgotten one is reworded
  without the old value, or forgotten with it; the prepare lists which. A
  rewording leaves out the words that name the old value: a tail (", not 14
  Oak Street", "rather than Arkwright Media"), a clause before the current
  value ("no longer attends Qiming Kindergarten and has transferred to
  Lanwan Kindergarten") or the place it left ("moved from Shanghai to
  Hangzhou"). The old value is withheld from the memory's quotes, with the
  words there that name it as replaced ("not the fernmail one"), and the
  store checks the new wording before it writes it. A contrast that gives a
  code or number by its last characters ("not the 1234 one" for
  "13800001234") names that code. Such a memory marked sensitive
  (credentials, health) is forgotten whole, with every quote.
- When the old value is told in the same sentence as the new one ("I used to
  live at 14 Elm Street, now I live at 27 Birch Lane."), the old value's
  part of the sentence is withheld, or the sentence whole; the prepare lists
  the memories that rest on it.
- A request whose words the model matches to no memory, and that none of the
  rules above covers ("forget Dr. Lee" with no memory of a Dr. Lee), answers
  `nothing_matched`, and nothing is forgotten.
- The confirm then checks what it left. It answers `incomplete`, with a last
  outcome `left_readable` that names no memory, when one of these checks
  finds something:
  - a sentence its prepare listed is still wholly readable;
  - a memory it removed or hid, or a past memory it left, leaves its own
    sentence wholly readable with a value it shares with a memory it took,
    or with an older memory it removed or hid that shares one: a name,
    place, number or code one of them states as its own value, or, for a
    past memory, the place both give as where someone lives;
  - a memory its prepare listed as forgotten or unreadable is still read
    with such a name, value, number or code;
  - a current memory, a readable part of its quotes, or a readable saved
    sentence other than a forget request still names a name, value or number
    of a target it forgot that is not marked `bundled`, or of an older memory
    it removed or hid that shares a value with what it took, as not, or no
    longer, the case, as the rule above reads it;
  - the value a rewording left out is still shown by a current memory, or by
    a readable part of a message the plan withholds from or the reworded
    memory quotes.

  After `left_readable`, the plugins prepare the forget once more and report
  the answer to the user.

**Names.** In the `agent` mode, after a forget that took a name, each fact a
later Add proposes is read for that name before review:

- The name is read as the name of the one who bore it (the bearer),
  unless the Add gives it to someone else with a word such as "coworker" or
  "friend" ("My coworker Bob knocked my coffee over."), in a sentence that
  names nothing of the bearer's sort. When in doubt, it is the bearer's. A
  Chinese name, which may also be an ordinary word, is the bearer's in a
  sentence that names the bearer's sort or that a fact about the bearer
  quotes.
- A fact that shows the bearer's name is cut around it: its quotes leave the
  name out, and its statement names the bearer by its tie ("the user's cat")
  or says "someone" when nothing tells whom the name means ("Someone phoned
  the user from London last night."). A fact that cannot drop the name, or
  keeps no quote, is refused, and so is one that would say "someone" for a
  place or an organisation.
- The name's words that a cut or a refusal leaves out are withheld in the
  new message before review, whatever review then decides.
- Review then judges what is left as it judges any fact: it may keep it,
  refuse it or leave it pending. After "Forget my sister's name.", "Anna
  called me yesterday." can be cut to "Someone called the user yesterday."
  before review, which judges that.
- The user's own later, dated words that give the name ("My cat is called
  Bob.", "Her name is Anna.") are not cut, and review judges them. A naming
  of the name that the user's words do not give is refused.
- Words that give a new name and set the forgotten one aside ("Sorry, I got
  that wrong earlier: my cat is called Max, not Bob.") keep the new name, as
  "The user's cat is named Max."; the forgotten name is cut from the memory
  and its quote.

Search and the memory list drop a hit that names a name a forget recorded,
unless two or more remaining memories still state that name, the forget that
took it read and kept the hit, or the user's own later words give it again
(above). The plugins recall through Search.

A model decides which memories a request matches. The prepare returns
`confirmation_required` or `nothing_matched`, the matched memories in
`candidates`, and the other memories the forget changes in `affected`, so a
client can show the list before it confirms.
