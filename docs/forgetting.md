# Forgetting

A forget takes the fact out of Search, History, Provenance and the memory
list, and withholds its quoted words. Its tombstone then denies every later
restatement, with one exception in the `agent` Add mode: the fact is learned
again when the user states it again in their own words after the forget.
That statement must meet every one of these conditions:

- It is in a user message that the service received after the forget, by
  the service's clock. The message's own date must be after the forget too,
  so a late batch of older messages never counts.
- It states something: it is not a question or a forget request, and it does
  not name forgetting, erasing or the memory ("forget", "remember",
  "delete", "忘", "记住" and the like).
- It states the forgotten fact: it holds at least two of the fact's content
  words (all of them for a shorter fact) and at least half of them, or the
  message's embedding has a cosine of 0.6 or more to the forgotten statement.

Nothing else brings a forgotten fact back:

- the assistant's messages;
- the forget request and the words it quotes;
- a question;
- a message without a date, or one dated or received before the forget, such
  as a late batch from another tool;
- a later user sentence about something else.

The old messages stay withheld. The `lean-local` and `full` Add modes deny
every restatement.

In the `agent` Add mode, a forget also works inside sentences and on names:

- Forgetting one detail of a sentence keeps the other facts the sentence
  states, with the words that state them, in Chinese and mixed-script
  sentences too. After "请忘掉我的猫的名字" on "我的猫叫Bob，它三岁了", the
  memories that the user has a cat and that it is three years old stay.
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

A model decides which memories a request matches. The prepare returns
`confirmation_required` or `nothing_matched`, the matched memories in
`candidates`, and the other memories the forget changes in `affected`, so a
client can show the list before it confirms.
