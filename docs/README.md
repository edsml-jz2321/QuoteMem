# QuoteMem documentation

QuoteMem is an auditable long-term memory service for conversational agents.
Each memory is a short statement tied to the exact words it came from. When a
fact changes, the old version is kept and marked as replaced. A user can ask the
agent to forget something: the memories the request matches are taken out of
reads and are not learned again later, except from the user's own later, dated
statement of it in the `agent` Add mode.

## What it does

- **Quoted evidence.** Every memory points to the original message by exact
  UTF-8 byte ranges, so any answer can be traced back to the words that
  support it.
- **Versioned supersession.** An update or correction writes a new version and
  closes the old one. History stays readable through `/memory/history`.
- **Forgetting with tombstones.** Forgetting is a two-step prepare/confirm
  call. A forgotten fact leaves a tombstone that the review step checks, so
  a later restatement of it is not stored again. In the `agent` Add mode one
  thing brings it back: the user's own later, dated statement of it
  (see [Forgetting](forgetting.md)).
- **Recorded review decisions.** Before a new memory is written, a review step
  compares it with the existing memories it may update. Each decision is
  stored with the snapshot of neighbours it was made against, so it can be
  audited and replayed.
- **Review modes.** `agent` (the default of `quotemem serve`) sends a new
  memory to a judge model only when it may repeat, update or correct a stored
  one, and reviews the rest with local rules. `lean-local` (the default of
  `quotemem bench serve`) reviews with deterministic local rules. `full` sends
  every candidate to a judge model. The mode is fixed when a store is created.
  In the `agent` mode you choose how strongly it reviews: `fast`, `standard`
  (the default) or `full` (`QUOTEMEM_REVIEW`, see [Review](configuration.md#review)).

## How it differs

Many memory layers store free-text summaries or embeddings
only. QuoteMem stores each memory together with its source quote, its version
history and its review decision, so every stored fact can be checked against
the conversation.

## Guides

| Page | What it covers |
|---|---|
| [Run the memory service](service.md) | Install, `quotemem serve`, the HTTP routes |
| [Configuration](configuration.md) | The review, and every setting of `quotemem serve` and `quotemem bench serve` |
| [Forgetting](forgetting.md) | Prepare and confirm, tombstones, what stays |
| [Commands and the memory list](cli.md) | The `quotemem` command, reaching the service, listing memories |
| [Use with your agents](agents.md) | Setup, the install matrix, what each agent does |
| [Benchmarks](benchmarks.md) | The AMLB result, the benchmark service and tools |
| [Development](development.md) | Tests, CI and the changelog |

The [demo](../examples/demo/README.md) runs one short story against your own service.
