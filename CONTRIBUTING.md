# Contributing

Thank you for your help. Open an issue to discuss a larger change before you start.

## Set up

You need Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) and Node.js 20.3 or newer.

```bash
git clone https://github.com/edsml-jz2321/QuoteMem.git
cd QuoteMem
uv venv .venv
uv pip install --constraints .github/ci/constraints.txt --editable ".[dev,bench]"
```

## Run the checks

CI runs these checks on every pull request. Run them before you open one.

```bash
# Python: lint, strict type check, release script tests, server smoke test
uv run --no-sync ruff check src .github/ci .github/release
uv run --no-sync pyright
uv run --no-sync python -m unittest discover -s .github/release
uv run --no-sync python .github/ci/server_smoke.py

# Agent plugins: tests, then pack and install the npm package
npm test
npm run check
```

The smoke test uses a stub model provider. It needs no API key and no network.

## Pull requests

- Keep a pull request to one change.
- Add or update tests for the change.
- Describe a change that users notice in `CHANGELOG.md`.

## Releases

Maintainers release a new version `X.Y.Z` in one commit on `main`.

1. Set `X.Y.Z` in every file that names the version. CI and the release
   workflows check each one.
   - `pyproject.toml`: `version`
   - `src/quotemem/__init__.py`: `__version__` (what `quotemem --version` prints)
   - `package.json`: `version`
   - `server.json`: the top-level `version` and `packages[0].version`
   - `plugin.json`
   - `integrations/claude-code/.claude-plugin/plugin.json`
   - `integrations/codex/.codex-plugin/plugin.json`
   - `integrations/gemini-cli/gemini-extension.json`
   - `integrations/opencode/package.json`
   - `integrations/openclaw/package.json`
   - `integrations/goose/quotemem/plugin.json`
   - `integrations/continue/quotemem.yaml`: `version:`
   - `integrations/dsh/server.mjs`: `SERVER_VERSION`
   - `integrations/core/autostart.mjs`: `SERVICE_RELEASE` (the service release
     that the plugins start with `uvx --from quotemem==X.Y.Z`)
2. Copy the MCP server and the plugin core into every plugin folder that ships
   them, then check that every version and copy agrees:

   ```bash
   node integrations/check_package.mjs --write
   node integrations/check_package.mjs
   ```

3. Add a dated section `## X.Y.Z - YYYY-MM-DD` to `CHANGELOG.md`.

When that commit on `main` passes CI, the Tag release workflow creates the tag
`vX.Y.Z` and starts the Release workflow (`release.yml`) for it. The Release
workflow publishes to PyPI first, then to npm and the MCP Registry, and attaches
the files to a GitHub Release. Pushing the tag `vX.Y.Z` yourself starts the
same Release workflow.

If a Tag release run created the tag but did not start the Release workflow,
start it for the tag in one of two ways:

- Open that Tag release run and choose **Re-run failed jobs**. This reruns only
  the job that starts the Release workflow. (A full rerun finds the tag already
  in place and starts nothing.)
- Or run the Release workflow by hand with the tag as the ref:

  ```bash
  gh workflow run release.yml --ref vX.Y.Z
  ```

Every Release job can be rerun safely. A version that a registry already holds
with the same files is skipped.
