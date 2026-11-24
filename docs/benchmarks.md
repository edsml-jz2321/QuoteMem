# Benchmarks

**Result.** In the Agent Memory Leaderboard (AMLB) Cycle 2 official results,
QuoteMem scored **59.83** overall and ranked **23rd of 125** entries in
*Textual Memory · Closed-source Methods*, where it is listed as "QuoteMem"
with a link to this repository:
<https://agentmemoryleaderboard.ai/results/cycle-2/#industry-textual>.

## Benchmark service: `quotemem bench serve`

`quotemem bench serve` is the composition used for the Agent Memory
Leaderboard. It serves the AMLB `/add` and `/search` routes with the
benchmark Search contract.

```bash
export DASHSCOPE_API_KEY=...          # embeddings (Alibaba Cloud Model Studio)
export OPENAI_API_KEY=...             # chat model, through an OpenAI-compatible gateway
export OPENAI_BASE_URL=https://<your-gateway>/v1
export QUOTEMEM_HTTP_AUTH_SECRET=...  # clients send "Authorization: Bearer <secret>"
export QUOTEMEM_JOB_ID=local-1        # any non-empty identifier for this deployment

quotemem bench serve --workspace ./qm-workspace --host 127.0.0.1 --port 8000
```

Set these exports in the benchmark service shell. Never start an agent from
that shell. For the curl examples, set only `QUOTEMEM_HTTP_AUTH_SECRET` in a
separate client shell to the same bearer token.

`quotemem bench serve` creates and initializes the workspace on first start
(`quotemem bench init --workspace DIR --profile bench` only initializes it). Stop it with
Ctrl-C; it drains in-flight requests before it exits.

```bash
curl -s http://127.0.0.1:8000/add \
  -H "Authorization: Bearer $QUOTEMEM_HTTP_AUTH_SECRET" -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "session_id": "s1", "request_id": "r1",
       "messages": [{"role": "user", "speaker": "Rowan",
                     "content": "I moved to Lisbon last spring."}]}'

curl -s http://127.0.0.1:8000/search \
  -H "Authorization: Bearer $QUOTEMEM_HTTP_AUTH_SECRET" -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "query": "Where does Rowan live?", "top_k": 10}'
```

`/add` returns after the new memories are stored and searchable. The
benchmark service keeps its own route names and bodies: `/memory/history`,
`/memory/why`, `/memory/add-items`, the by-memory forget pair
`/memory/forget/prepare` and `/memory/forget`, and the by-statement pair
`/memory/forget/statement/prepare` and `/memory/forget/statement`. They take
the same request bodies as `/memory/history`, `/memory/provenance`,
`/memory/add-results` and the forget pair of `quotemem serve`.
`quotemem bench route --shard http://127.0.0.1:8001 --shard ...` spreads
users over several `quotemem bench serve` processes by `user_id`. Run
`quotemem bench serve --help` for the remaining flags.

## Benchmark tools

`python -m quotemem.bench.cli` downloads public benchmark files, replays them over
HTTP and computes local diagnostic scores. Benchmark data is not included in
this repository. [`benchmarks/`](../benchmarks) lists each source by URL, pinned
revision, SHA-256 and licence, and the tool downloads it from its publisher:

```bash
uv run --extra bench python -m quotemem.bench.cli \
  --sources benchmarks/sources.json --dataset ./data/longmemeval_oracle.json --variant oracle download
```

From a pip install, add the extra first: `pip install "quotemem[bench]>=0.3"`.

Each dataset keeps its own licence (see [NOTICE](../NOTICE)). LoCoMo-Refined is
CC BY-NC 4.0 (non-commercial) and BEAM data is CC BY-SA 4.0.
