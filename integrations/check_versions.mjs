// Version numbers written in the documentation: every install or start command
// that names a QuoteMem release (`quotemem==X.Y.Z`, `quotemem@X.Y.Z`,
// `quotemem-X.Y.Z.tgz`, `quotemem[bench]==X.Y.Z`) names this package's version.
// check_package.mjs runs it, so a version change that leaves an example behind
// fails `npm test` and CI.
import fs from 'node:fs';
import path from 'node:path';

const MENTION = /\bquotemem(?:\[[a-z,]+\])?(?:==|@|-)(\d+\.\d+\.\d+)(?![\d])/g;
// Release history names earlier versions on purpose.
const HISTORY = new Set(['CHANGELOG.md']);

/** The versions a text names in release commands: [{line, version}]. */
export function versionMentions(text) {
  const out = [];
  for (const [index, line] of text.split('\n').entries()) {
    for (const match of line.matchAll(MENTION)) out.push({ line: index + 1, version: match[1] });
  }
  return out;
}

/** The Markdown files to check: README.md, docs/ and all of integrations/ and examples/. */
export function docFiles(root) {
  const files = [];
  const walk = (dir) => {
    let entries = [];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(file);
      else if (entry.isFile() && /\.mdc?$/.test(entry.name) && !HISTORY.has(entry.name)) {
        files.push(file);
      }
    }
  };
  for (const dir of ['integrations', 'examples']) walk(path.join(root, dir));
  for (const name of fs.readdirSync(path.join(root, 'docs')).filter((n) => n.endsWith('.md'))) {
    if (!HISTORY.has(name)) files.push(path.join(root, 'docs', name));
  }
  files.push(path.join(root, 'README.md'));
  return files.filter((file) => fs.existsSync(file)).sort();
}

/** Problems: one line per release command naming another version than `version`. */
export function docVersionProblems(root, version) {
  const problems = [];
  for (const file of docFiles(root)) {
    for (const mention of versionMentions(fs.readFileSync(file, 'utf8'))) {
      if (mention.version !== version) {
        // The path as the repository names it, with `/` on every system.
        const where = `${path.relative(root, file).split(path.sep).join('/')}:${mention.line}`;
        problems.push(`${where} names quotemem ${mention.version}, not ${version}`);
      }
    }
  }
  return problems;
}
