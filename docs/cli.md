# Commands and the memory list

## See what is remembered

Ask any agent, for example "what do you remember about me?": it lists the
memories with the `quotemem_memories` tool. In a terminal,
`quotemem memories` (or `npx quotemem memories`) prints each memory's status,
statement, dates and quotes; `--json` prints the same list as JSON, and
`--status current|past|possibly_outdated|all` and `--user` choose what to list.
`npx quotemem memories` lists the plugins' user (`QUOTEMEM_USER_ID`, else your
OS user name) unless `--user` is given; the service's own command needs
`--user` for that user.
`quotemem memories --html memories.html` writes one local, read-only page
with a search box: it holds no token, loads nothing from the network and only
you can read it. These only read the memory, from the running service or,
when it is stopped, from its data folder. Forgotten memories never appear.

## List the memories

`POST /memory/list` returns one page of a user's memories. It takes the same
bearer token as `/memory/search` (`TOKEN_FILE` is set as in the
[quickstart](service.md#quickstart-quotemem-serve)).

```bash
curl -s http://127.0.0.1:8765/memory/list \
  -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "status": "current", "limit": 50}'
```

- The body takes `user_id` (required), `status` (`current`, `past`,
  `possibly_outdated` or `all`, the default), `limit` (1 to 200, default 50)
  and `cursor` (the `next_cursor` of the previous page). Any other body
  answers 422 `invalid_request`.
- The answer holds `schema_version` (`1`), `memories`, `next_cursor`
  (`null` on the last page) and `counts`: how many memories have each status,
  whatever the filter.
- Each memory has `memory_id`, `statement`, `status`, `kind`,
  `created_at_ms`, `updated_at_ms`, `quotes` (each with `role`, `speaker`,
  `said_at_ms` and `text`), `supersedes` and `superseded_by`.
- The status is the one Search's `Status:` line gives. The newest change
  comes first (`updated_at_ms`, then `memory_id`), so a page boundary stays
  stable while memories are added.
- Nothing forgotten is listed, and withheld words are never shown. A user the
  store has never seen gets an empty list.

`quotemem memories` shows the same list in a terminal:

```bash
quotemem memories --user u1                       # text
quotemem memories --user u1 --json                # the /memory/list answer, pages merged, with its source
quotemem memories --user u1 --html memories.html  # one read-only HTML page
```

- The text shows each memory's status, statement, dates and up to three
  quotes, then the counts.
- `--user` (default: `QUOTEMEM_USER_ID`, else your OS user name, else `default`;
  pass `--user default` to see what `quotemem serve` stored for requests that
  name no user), `--status` (`all`, `current`, `past` or
  `possibly_outdated`) and `--limit` choose what to list. `--limit` defaults
  to 50; with `--html` every memory is listed.
- It asks the running service, or reads the stopped store read-only
  ([Reaching the service](#reaching-the-service)). It refuses while another
  process holds the store. Read from the store, a quote names only the
  speaker's role.
- `--html FILE` writes one self-contained page with a filter box (mode 0600).
  It holds no token and loads nothing from the network. A file already at
  `FILE` is replaced.

## Reaching the service

`quotemem memories`, `quotemem repair` and `quotemem delete-user` find the
service as the plugins do.

- The URL is `--url`, else `QUOTEMEM_URL`, else `http://127.0.0.1:8765`.
- The token is `QUOTEMEM_TOKEN` (`QUOTEMEM_HTTP_AUTH_SECRET` also works),
  then the file that `QUOTEMEM_TOKEN_FILE` names. For a service on this
  computer only, it is then `<data-dir>/token` (`--data-dir`, else
  `QUOTEMEM_DATA_DIR`).
- A service is on this computer when its host is `localhost`, `127.0.0.1` or
  `[::1]`. `127.0.0.2`, `0.0.0.0`, `[::]` and `localhost.` are not.
- A URL is refused with `schema_invalid` exactly when the plugins refuse it.
  After `http://` or `https://` it needs a host of letters, digits and
  `._~-`, or a bracketed IPv6 address. An optional `:port` of digits may
  follow, then `/`, `?`, `#` or the end. So a backslash, a space, `%` or `@`
  in the address is refused: a browser's URL parser and Python's would read
  another host.
- A host that a browser's parser would rewrite is refused too. An IPv4
  address must be four decimal numbers without a final dot (`127.0.0.1.`,
  `127.1` and `2130706433` are refused). An IPv6 address must be in its
  shortest form (`[0:0:0:0:0:0:0:1]` is refused).
- `npx quotemem <command>` hands these commands the plugins' settings
  ([Install matrix](agents.md#install-matrix)).

`quotemem memories` also decides where to read:

- When no service on this computer answers, it reads the stopped store in the
  data directory. A service on another computer that does not answer is
  `service_unreachable`.
- Without a token, it refuses with `authentication_failed` when a service
  answers at the URL, rather than read a store that may not be that
  service's.
- A service on this computer may serve (`/agent/ready`) and refuse the token
  while no process holds the data directory's store. That service serves
  another store, so `memories` reads the stopped one. While what answers
  there is not serving (a service still starting, or not a QuoteMem service),
  the refusal stands.
- The listing always says where it came from: a last `source:` line, and
  `source` in `--json` and in the `--html` report
  (`{"kind": "service", "url": ...}` or `{"kind": "store", "data_dir": ...}`).
