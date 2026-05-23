import { probeBackends } from '../../src/runtime/backend_probe.js';

describe('backend_probe', () => {
  it('quick mode returns structured result without hanging', () => {
    const start = Date.now();
    const result = probeBackends({ deep: false, timeoutMs: 250 });
    const elapsed = Date.now() - start;

    expect(result.node).toMatch(/^v\d+/);
    expect(result.probeMode).toBe('quick');
    expect(['available', 'missing']).toContain(result.python);
    expect(result.torch).toBe('not_checked');
    expect(result.triton).toBe('not_checked');
    expect(result.cuda).toBe('not_checked');
    expect(result.vllm).toBe('not_checked');
    expect(['diagnostic_only', 'typescript_cpu', 'mesa_rusticl_kgsl']).toContain(result.recommendedBackend);
    expect(['diagnostic_only', 'typescript_cpu', 'opencl_adreno', 'opencl_generic', 'mesa_rusticl_kgsl']).toContain(result.nativeBackend);
    expect(['native_cli_contract', 'diagnostic_runtime']).toContain(result.executionOwner);
    expect(['native_probe', 'production_policy_only']).toContain(result.admissionSource);
    expect(result.production.productionPolicy).toBe('custom_driver_stack_only');
    expect(typeof result.production.productionReady).toBe('boolean');
    expect(['diagnostic_only', 'mesa_rusticl_kgsl', 'turnip_vulkan_kgsl']).toContain(result.production.productionRoute);
    if (result.production.productionReady) {
      expect(['mesa_rusticl_kgsl', 'turnip_vulkan_kgsl']).toContain(result.production.productionRoute);
      expect(result.productionBackendAllowed).toBe(true);
    } else {
      expect(result.production.productionRoute).toBe('diagnostic_only');
      expect(result.productionBackendAllowed).toBe(false);
    }
    expect(result.production.forbiddenProductionBackends).toContain('typescript_cpu');
    expect(Array.isArray(result.warnings)).toBe(true);
    expect(elapsed).toBeLessThan(3000);
  });

  it('detects Termux environment', () => {
    const result = probeBackends({ deep: false });
    if (process.env['PREFIX']?.includes('com.termux')) {
      expect(result.termux).toBe(true);
    }
  });

  it('never throws even with minimal timeout', () => {
    expect(() => probeBackends({ deep: false, timeoutMs: 100 })).not.toThrow();
    expect(() => probeBackends({ deep: true, timeoutMs: 100 })).not.toThrow();
  });

  it('reports elapsedMs', () => {
    const result = probeBackends({ deep: false, timeoutMs: 250 });
    expect(result.elapsedMs).toBeGreaterThanOrEqual(0);
    expect(result.elapsedMs).toBeLessThan(3000);
  });
});
