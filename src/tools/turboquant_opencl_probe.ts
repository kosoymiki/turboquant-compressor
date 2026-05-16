/**
 * MCP tool: turboquant_opencl_probe — OpenCL/Adreno availability detection.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { probeOpenCl } from '../native/opencl_probe.js';

export const turboquantOpenclProbeSchema = {
  description: 'Probe OpenCL availability on this device. Detects Adreno GPU, library paths, and capabilities. Use deep=true for full clinfo device enumeration.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      deep: { type: 'boolean', description: 'Run clinfo for full device details (slower)' },
      timeoutMs: { type: 'number', description: 'Timeout for deep probe in ms (100-5000)', minimum: 100, maximum: 5000 },
    },
    required: [] as string[],
  },
};

export function turboquantOpenclProbe(args: { deep?: boolean; timeoutMs?: number }): object {
  return probeOpenCl({ deep: args.deep, timeoutMs: args.timeoutMs });
}
