import {
  classifyLoaderNamespaceContract,
  classifyRouteSelectionContract,
} from './route_selection_contract.js';

describe('route selection contracts', () => {
  test('classifies custom-ready route with vendor abi mismatch', () => {
    const loader = classifyLoaderNamespaceContract({
      vendorProbeStatus: 1,
      vendorProbeStderr:
        'CANNOT LINK EXECUTABLE "/data/local/tmp/tq_opencl_cli": cannot locate symbol "_ZTT..."',
      customRunAsProbeStatus: 0,
      customRunAsPlatformCount: 1,
      customRunAsDeviceCount: 1,
      runAsStageStatus: 0,
      renderNodeAccessible: false,
      kgslAccessible: true,
      customLinkerUsesStagedOpenCl: true,
    });
    const route = classifyRouteSelectionContract({
      vendorLoader: loader.classification,
      customLoader: loader.classification,
      customRunAsPlatformCount: 1,
      customRunAsDeviceCount: 1,
      vendorProbeUsable: false,
    });

    expect(loader.classification).toBe('vendor_abi_mismatch');
    expect(loader.ready).toBe(true);
    expect(route.classification).toBe('custom_preferred_vendor_abi_blocked');
  });

  test('classifies namespace or icd blocked custom route', () => {
    const loader = classifyLoaderNamespaceContract({
      customRunAsProbeStatus: 0,
      customRunAsPlatformCount: 0,
      customRunAsDeviceCount: 0,
      runAsStageStatus: 0,
      customLinkerUsesStagedOpenCl: true,
    });

    expect(loader.classification).toBe('namespace_or_icd_blocked');
    expect(loader.ready).toBe(false);
  });
});
