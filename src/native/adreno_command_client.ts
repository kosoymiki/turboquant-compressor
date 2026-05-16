/**
 * TurboQuant Adreno Command Client — MCP-facing command buffer submission.
 * Vortek-inspired: submit compute commands, backend routed by quirks.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { buildQuirksProfile, routeKernel, type DeviceProfile, type BackendRoute, type QuirksProfile } from './adreno_quirks.js';

export type CmdOp = 'mse_score' | 'qjl_score' | 'value_dequant' | 'fused_attention' | 'benchmark' | 'probe';

export interface CmdShape {
  tokens: number;
  heads: number;
  headDim: number;
  bits: number;
  groupSize?: number;
  projections?: number;
}

export interface CmdRequirements {
  allowFallback?: boolean;
  requireGpu?: boolean;
  requireParity?: boolean;
  maxLatencyMs?: number;
}

export interface Command {
  op: CmdOp;
  backend?: BackendRoute;
  shape: CmdShape;
  requirements?: CmdRequirements;
}

export interface CommandBuffer {
  version: 1;
  commands: Command[];
}

export interface CmdResult {
  op: CmdOp;
  backend: BackendRoute;
  success: boolean;
  parity: boolean;
  latencyMs: number;
  error?: string;
}

export interface CommandBufferResult {
  version: 1;
  allSuccess: boolean;
  results: CmdResult[];
}

export function createCommandBuffer(commands: Command[]): CommandBuffer {
  return { version: 1, commands };
}

export function routeCommandBuffer(buf: CommandBuffer, profile: QuirksProfile): CommandBuffer {
  return {
    version: 1,
    commands: buf.commands.map(cmd => ({
      ...cmd,
      backend: cmd.backend ?? routeKernel(profile, cmd.op, cmd.shape.tokens),
    })),
  };
}

export function buildProfileFromProbe(probe: {
  devices?: Array<{ name: string; vendor: string; version: string; hasFp16: boolean; hasSubgroups: boolean; globalMemBytes: number; computeUnits: number }>;
  recommendedBackend?: string;
}): QuirksProfile | null {
  const dev = probe.devices?.[0];
  if (!dev) return null;

  const device: DeviceProfile = {
    gpu: dev.name,
    opencl: dev.version,
    fp16: dev.hasFp16,
    subgroups: dev.hasSubgroups,
    maxWorkGroupSize: 1024,
    computeUnits: dev.computeUnits,
    globalMemBytes: dev.globalMemBytes,
  };

  return buildQuirksProfile(device);
}
