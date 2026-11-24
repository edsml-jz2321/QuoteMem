# Development

- **Tests.** The Node tests of the agent plugins are included under
  `integrations/` (`npm test`).
- **CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs ruff, a
  strict pyright check, a byte-compile and an import of every Python module,
  the benchmark tool's `--help`, a smoke test of `quotemem check` and
  `quotemem serve` against a stub provider, and a build whose sdist and wheel
  are checked file by file. It also runs the Node tests (`npm test`) and the
  npm package check (`npm run check`) on Node 20 and 24.
- Changes are listed in [CHANGELOG.md](../CHANGELOG.md).
- **Licences.** The included HeavenBase is MIT-licensed; its licence is in
  `src/quotemem/_vendor/heavenbase-0.1.2.3.dist-info/licenses/LICENSE`.
