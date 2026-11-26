# QuoteMem demo

`demo.mjs` tells one short story to a running QuoteMem service and prints what
the service does with it. It uses a new user id each time, so your own
memories are not touched.

0. It prints the store's Add mode and review, from `GET /agent/status`.
1. Chat 1: "I live in Beijing. My cat is called Bob. I work as a nurse."
2. Chat 2: "I moved from Beijing to Shanghai last month."
3. It lists the memories with their quotes and shows the history of the city.
   The old version is kept as `past`. It prints the link from the new memory
   to the one it replaces, and the byte range of the new memory's quote,
   checked against the message. It then asks `/memory/provenance` for the review behind the
   new memory: the relation it found and the reason.
4. It searches "Where does the user live now?" and prints the first hit with
   its quote.
5. Chat 3: "Forget my cat's name." It prepares the forget, prints the
   memories that state the fact, any memory the forget will reword, and the
   prepare's notice that a tombstone is kept, and confirms it. From the
   receipt it prints how many tombstones were written and sentences
   withheld, that nothing readable is left, and that the tombstone is kept
   only to block relearning and cannot be read.
6. It lists the memories again and searches for the cat's name.
   "The user has a cat." is there; the name is gone. It then reads the history
   of every memory from before the forget: none still holds the name. If the
   forget deleted a memory, that memory's history answers 404
   `memory_not_found`. A memory the forget reworded keeps its id, and its
   history shows the earlier versions redacted, without the name.
7. Chat 4 names the cat again in three messages: "My cat is called Bob."
   without a date, as in an old chat imported later; the question "Is my cat
   called Bob?"; and the assistant's answer. None of them is the user telling
   it again after the forget, so the tombstone keeps the name out. It prints
   the counts, the reason codes and whether "Bob" is in the memories or the
   search.

The Add answer counts updates, corrections and retractions together as
`corrected`; the reason code (`updated` in step 3) says which.

With `--relearn` it adds step 8: in chat 5 the user says, dated now, "For the
record, my cat is called Bob.", with the current time as its `timestamp`. In
the default `agent` Add mode that is the one way a forgotten fact comes back:
the name is stored again, quoting chat 5 only.

It calls the HTTP routes `GET /health`, `GET /agent/status`, `/memory/add`,
`/memory/list`, `/memory/history`, `/memory/provenance`, `/memory/search`,
`/memory/forget/prepare` and `/memory/forget/confirm`. At the end it runs
`quotemem memories` for the same user.

## Run it

The demo runs from a clone of the repository. You need Node 20.3 or later and
the `quotemem` command, set up once with `npm install -g quotemem` and
`quotemem setup`:

```bash
git clone https://github.com/edsml-jz2321/QuoteMem.git
cd QuoteMem
quotemem serve &
node examples/demo/demo.mjs
```

The script waits up to a minute for the service to answer `GET /health`, so
it can start right after `quotemem serve &`.

The script reads the service's token from its data folder
(`~/.local/share/quotemem/token`, or `$QUOTEMEM_DATA_DIR/token`). For a
service at another address, set `QUOTEMEM_URL` and `QUOTEMEM_TOKEN_FILE`:

```bash
QUOTEMEM_URL=http://127.0.0.1:9000 QUOTEMEM_TOKEN_FILE=/path/to/token node examples/demo/demo.mjs
```

`node examples/demo/demo.mjs <user id>` uses the user id you give, and
`node examples/demo/demo.mjs --relearn` adds step 8. Each run
calls the chat and embedding models of your service a few times; saving
takes a few seconds to a minute per chat.

To remove the demo user afterwards: `quotemem delete-user --user-id <user id> --yes`.

## What differs between runs

The model writes the statements, picks the quotes and gives the reasons, so
their wording can differ from run to run. How it splits chat 1 decides what
the forget does. In 12 runs on one service, with the default preset:

- 9 runs stored 4 facts from chat 1, among them "The user has a cat." and
  "The user's cat is named Bob.". The forget deleted the memory that gives
  the name, and its history answered 404 `memory_not_found`. The receipt
  showed 1 tombstone and 2 sentences withheld.
- 3 runs stored 3 facts, with "The user's cat is named Bob." as the only
  memory about the cat. The forget reworded it to "The user has a cat."
  (`prepare, reworded to: ...`), and it kept its id. The receipt showed 2
  tombstones and 1 sentence withheld.

Every run then showed no memory, search hit or history that names the cat,
and chat 4 was refused with `denied_by_forget`.
