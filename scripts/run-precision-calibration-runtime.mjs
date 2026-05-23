#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const statePath = join(forensicsDir, 'precision-calibration-runtime-state.json');

const { buildPrecisionCalibrationRuntimeState } = await import('../dist/core/precision_calibration.js');

mkdirSync(forensicsDir, { recursive: true });

const state = buildPrecisionCalibrationRuntimeState();
writeFileSync(statePath, JSON.stringify(state, null, 2));
console.log(JSON.stringify(state, null, 2));
