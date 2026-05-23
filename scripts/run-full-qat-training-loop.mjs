#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const statePath = join(forensicsDir, 'full-qat-training-loop-state.json');

const { buildFullQatTrainingLoopState } = await import('../dist/core/full_qat_training_loop.js');

mkdirSync(forensicsDir, { recursive: true });

const state = await buildFullQatTrainingLoopState(rootDir);
writeFileSync(statePath, JSON.stringify(state, null, 2));
console.log(JSON.stringify(state, null, 2));
