# QuoteMem demo

`demo.mjs` tells one short story to a running QuoteMem service and prints what
the service does with it. It uses a new user id each time, so your own
memories are not touched.

1. Chat 1: "I live in Beijing. My cat is called Bob. I work as a nurse."
2. Chat 2: "I moved from Beijing to Shanghai last month."
3. It lists the memories with their quotes and shows the history of the city.
   The old version is kept as `past`.
4. It searches "Where does the user live now?" and prints the first hit with
   its quote.
5. Chat 3: "Forget my cat's name." It prepares the forget, prints what goes,
   and confirms it.
6. It lists the memories again and searches for the cat's name.
   "The user has a cat." stays; the name is gone.

It calls the HTTP routes `/memory/add`, `/memory/list`, `/memory/history`,
`/memory/search`, `/memory/forget/prepare` and `/memory/forget/confirm`. At
the end it runs `quotemem memories` for the same user.

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

`node examples/demo/demo.mjs <user id>` uses the user id you give. Each run
calls the chat and embedding models of your service a few times; saving
takes a few seconds to a minute per chat.

To remove the demo user afterwards: `quotemem delete-user --user-id <user id> --yes`.
