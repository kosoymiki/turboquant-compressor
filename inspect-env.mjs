#!/usr/bin/env node
import { execSync } from 'child_process';

console.log('Node.js version:', process.version);
console.log('Platform:', process.platform);
console.log('Architecture:', process.arch);
console.log('V8 version:', process.versions.v8);

try {
  const npmVersion = execSync('npm --version', { encoding: 'utf8' }).trim();
  console.log('npm version:', npmVersion);
} catch (e) {
  console.log('npm not available');
}