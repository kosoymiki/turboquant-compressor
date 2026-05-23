#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const statePath = join(forensicsDir, 'hybrid-cpu-gpu-training-runtime-state.json');

const { buildHybridCpuGpuTrainingRuntimeState } = await import('../dist/runtime/hybrid_training_runtime.js');

mkdirSync(forensicsDir, { recursive: true });

const state = await buildHybridCpuGpuTrainingRuntimeState(rootDir);
writeFileSync(statePath, JSON.stringify(state, null, 2));
console.log(JSON.stringify(state, null, 2));
