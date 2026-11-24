'use strict';

// quotemem DeepSeek Harness bundle entry. DSH applies cordis.patch.yml, which
// requires this module to find the bundle's files wherever the package was
// installed from (the npm tarball or the Git repository root), then starts
// server.mjs with `node` and adds the bundle's skills to the profile. The same
// patch loads automem.mjs, a native plugin that recalls and saves memory
// automatically. server.mjs is a zero-dependency stdio MCP server that
// forwards explicit tool calls to a QuoteMem HTTP service that is already
// running (`quotemem serve`).
const path = require('node:path');

module.exports = {
  name: 'quotemem',
  kind: 'DSH bundle: automatic memory plugin, stdio MCP bridge to a QuoteMem service, and a skill',
  patch: path.join(__dirname, 'cordis.patch.yml'),
  autoMemory: path.join(__dirname, 'automem.mjs'),
  mcpServer: path.join(__dirname, 'server.mjs'),
  skillsDir: path.join(__dirname, 'skills'),
  tools: [
    'quotemem_search',
    'quotemem_memories',
    'quotemem_add',
    'quotemem_history',
    'quotemem_provenance',
    'quotemem_forget_prepare',
    'quotemem_forget_confirm',
    'quotemem_health',
  ],
};
