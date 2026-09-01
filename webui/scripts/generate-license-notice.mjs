import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const lock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));
const destination = path.resolve(root, '..', 'knowledge_forge', 'static', 'graph-explorer.LICENSE.txt');
const entries = [];

for (const [packagePath, metadata] of Object.entries(lock.packages || {})) {
  if (!packagePath.startsWith('node_modules/') || metadata.dev) continue;
  const directory = path.join(root, packagePath);
  const manifestPath = path.join(directory, 'package.json');
  if (!fs.existsSync(manifestPath)) continue;
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const licenseFile = fs.existsSync(directory)
    ? fs.readdirSync(directory).find(name => /^(licen[sc]e|copying|notice)(\..*)?$/i.test(name))
    : null;
  entries.push({
    name: manifest.name || packagePath.replace(/^node_modules\//, ''),
    version: manifest.version || metadata.version || 'unknown',
    license: manifest.license || metadata.license || 'SEE PACKAGE',
    repository: typeof manifest.repository === 'string'
      ? manifest.repository
      : (manifest.repository && manifest.repository.url) || '',
    text: licenseFile ? fs.readFileSync(path.join(directory, licenseFile), 'utf8').trim() : '',
  });
}

const grouped = new Map();
for (const entry of entries) {
  const key = entry.text || `License identifier: ${entry.license}`;
  const value = grouped.get(key) || {packages: [], text: key};
  value.packages.push(`${entry.name}@${entry.version} (${entry.license})${entry.repository ? ` — ${entry.repository}` : ''}`);
  grouped.set(key, value);
}

const output = [
  'Knowledge Forge graph explorer — third-party license notices',
  'Generated from webui/package-lock.json; production runtime dependencies only.',
  '',
  ...[...grouped.values()].flatMap(group => [
    '='.repeat(78),
    ...group.packages.sort(),
    '-'.repeat(78),
    group.text,
    '',
  ]),
].join('\n');

fs.mkdirSync(path.dirname(destination), {recursive: true});
fs.writeFileSync(destination, output, 'utf8');
