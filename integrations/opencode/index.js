// QuoteMem for OpenCode. One default export serves both plugin APIs:
// OpenCode 1.x (`opencode-ai` >= 1.18.29) calls `server`, OpenCode 2.x
// (`@opencode/cli`) calls `setup`. Keep this file to the default export only:
// OpenCode 1.x reads every export of a plugin module.
import { PLUGIN_ID, server, setup } from './adapter.js';

export default { id: PLUGIN_ID, setup, server };
