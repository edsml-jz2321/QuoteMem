// Map the Claude Code plugin options onto the QUOTEMEM_* variables.
//
// The plugin asks for three optional values when it is enabled (see
// .claude-plugin/plugin.json): the service URL, the token and the memory
// owner. Claude Code hands them to hooks as CLAUDE_PLUGIN_OPTION_<KEY> and to
// the MCP server through `${user_config.<key>}` in .mcp.json, which this
// plugin passes as QUOTEMEM_PLUGIN_OPTION_<KEY>. Both the hooks and the MCP
// server apply the same rule, so they always act on the same memory:
//
//   a plugin option that is set wins; otherwise QUOTEMEM_* from the
//   environment; otherwise the built-in default (in core/config.mjs).
//
// An option that is empty, or still the unexpanded `${user_config.…}` text,
// counts as unset.

/** Plugin option key → QuoteMem variable. */
export const OPTIONS = Object.freeze({
  URL: 'QUOTEMEM_URL',
  TOKEN: 'QUOTEMEM_TOKEN',
  USER_ID: 'QUOTEMEM_USER_ID',
});

export const HOOK_PREFIX = 'CLAUDE_PLUGIN_OPTION_';
export const MCP_PREFIX = 'QUOTEMEM_PLUGIN_OPTION_';

/** True when a plugin option carries a value. */
export function optionSet(value) {
  if (typeof value !== 'string') return false;
  const text = value.trim();
  return text.length > 0 && !text.includes('${user_config.') && !/^\$\{[A-Za-z_][A-Za-z0-9_]*\}$/.test(text);
}

/**
 * The environment with the plugin options applied, and every variable with
 * `prefix` removed (so the token is not copied into child processes twice).
 * @param {object} env - the process environment (not changed).
 * @param {string} prefix - HOOK_PREFIX or MCP_PREFIX.
 * @returns {object} a new environment object.
 */
export function applyOptions(env, prefix) {
  const out = {};
  for (const [key, value] of Object.entries(env ?? {})) {
    if (!key.startsWith(prefix) && value !== undefined) out[key] = value;
  }
  for (const [option, name] of Object.entries(OPTIONS)) {
    const value = env?.[`${prefix}${option}`];
    if (optionSet(value)) out[name] = value.trim();
  }
  return out;
}
