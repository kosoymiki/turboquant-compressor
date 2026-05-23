import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { loadRuntimeContractBundle, resolveRuntimeContractPaths } from './runtime_contracts.js';

describe('runtime_contracts', () => {
  test('resolves canonical runtime contract paths from an explicit root', () => {
    const rootDir = mkdtempSync(join(tmpdir(), 'tq-runtime-contracts-'));

    try {
      const paths = resolveRuntimeContractPaths(rootDir);

      expect(paths.rootDir).toBe(rootDir);
      expect(paths.inferenceRuntimeContractPath).toBe(join(rootDir, 'forensics', 'adreno', 'opencl-inference-runtime-contract.json'));
      expect(paths.pagedKvRuntimeContractPath).toBe(join(rootDir, 'forensics', 'adreno', 'opencl-paged-kv-prefix-cache-contract.json'));
      expect(paths.precisionCalibrationStatePath).toBe(join(rootDir, 'forensics', 'precision-calibration-runtime-state.json'));
    } finally {
      rmSync(rootDir, { recursive: true, force: true });
    }
  });

  test('loads runtime contracts from explicit files without relying on implicit repo state', () => {
    const rootDir = mkdtempSync(join(tmpdir(), 'tq-runtime-contracts-'));
    const adrenoDir = join(rootDir, 'forensics', 'adreno');
    mkdirSync(adrenoDir, { recursive: true });
    writeFileSync(join(adrenoDir, 'opencl-inference-runtime-contract.json'), JSON.stringify({
      classification: 'module_ready',
      route: 'mesa_rusticl_kgsl',
    }));
    writeFileSync(join(adrenoDir, 'opencl-paged-kv-prefix-cache-contract.json'), JSON.stringify({
      classification: 'module_ready',
    }));
    writeFileSync(join(rootDir, 'forensics', 'precision-calibration-runtime-state.json'), JSON.stringify({
      activationStepSizeMean: 0.05,
      weightStepSizeMean: 0.11,
      kvGroupStepSizeMean: 0.09,
    }));

    try {
      const bundle = loadRuntimeContractBundle(rootDir);

      expect(bundle.inferenceRuntimeContract?.route).toBe('mesa_rusticl_kgsl');
      expect(bundle.pagedKvRuntimeContract?.classification).toBe('module_ready');
      expect(bundle.precisionCalibrationState?.weightStepSizeMean).toBe(0.11);
    } finally {
      rmSync(rootDir, { recursive: true, force: true });
    }
  });
});
