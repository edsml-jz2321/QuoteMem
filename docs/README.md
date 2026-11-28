# QuoteMem documentation

QuoteMem is an auditable long-term memory service for conversational agents.
Each memory is a short statement tied to the exact words it came from. When a
fact changes, the old version is kept and marked as replaced. A user can ask the
agent to forget something: the memories the request matches are taken out of
reads, the words the forget removes are withheld, and later Adds are checked
against a tombstone of the fact.

## What it does

- **Quoted evidence.** Every memory points to the original message by exact
  UTF-8 byte ranges, so any answer can be traced back to the words that
  support it.
- **Versioned supersession.** An update or correction writes a new version and
  closes the old one. History stays readable through `/memory/history`.
- **Forgetting with tombstones.** Forgetting is a two-step prepare/confirm
  call. A forgotten fact leaves a tombstone that every later Add is checked
  against: a restatement with the same words is refused, and a close one goes
  to review beside the forgotten statement. In the `agent` Add mode the
  user's own later, dated statement of it, when it shows no name a forget
  took, is judged as a new fact (see [Forgetting](forgetting.md)).
- **Recorded review decisions.** Before a new memory is written, a review step
  compares it with the existing memories it may update. Each decision is
  stored with a record of the neighbours it was made against (their ids and
  scores), so it can be audited and replayed. `POST /memory/provenance` shows
  the decision behind any version of a memory.
- **Add modes and reviews.** By default `quotemem serve` creates a new store
  in the `agent` Add mode. In it you choose how strongly it reviews
  (`QUOTEMEM_REVIEW`, see [Review](configuration.md#review)):
  - `standard` (the default) sends a new memory to a judge model when it may
    repeat or change a stored one, or in the other cases
    [Review](architecture.md#review) lists, and reviews the rest with local
    rules;
  - `fast` reviews with local rules only;
  - `full` sends every new memory to a judge model.

  The benchmark Add modes of `quotemem bench serve` are `lean-local` (local
  rules, its default) and `full` (a judge model for every candidate). The Add
  mode and the review are fixed when a store is created.

[How QuoteMem works](architecture.md) explains each step of an Add, a read and
a forget, and what each file and table of a store holds.

## How it differs

Many memory layers store free-text summaries or embeddings
only. QuoteMem stores each memory together with its source quote, its version
history and its review decision, so every stored fact can be checked against
the conversation.

## Guides

| Page | What it covers |
|---|---|
| [How QuoteMem works](architecture.md) | A store's files and tables, an Add, reads, forgets |
| [Run the memory service](service.md) | Install, `quotemem serve`, the HTTP routes |
| [Configuration](configuration.md) | The review, and every setting of `quotemem serve` and `quotemem bench serve` |
| [Forgetting](forgetting.md) | Prepare and confirm, tombstones, what stays |
| [Commands and the memory list](cli.md) | The `quotemem` command, reaching the service, listing memories |
| [Use with your agents](agents.md) | Setup, the install matrix, what each agent does |
| [Benchmarks](benchmarks.md) | The AMLB result, the benchmark service and tools |
| [Development](development.md) | Tests, CI and the changelog |

The [demo](../examples/demo/README.md) runs one short story against your own service.
