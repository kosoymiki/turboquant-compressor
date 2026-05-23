#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const statePath = join(forensicsDir, 'distillation-for-quantization-state.json');

const { buildDistillationForQuantizationState } = await import('../dist/core/distillation_for_quantization.js');

mkdirSync(forensicsDir, { recursive: true });

const state = buildDistillationForQuantizationState();
writeFileSync(statePath, JSON.stringify(state, null, 2));
console.log(JSON.stringify(state, null, 2));
