/**
 * TurboQuant Adreno Command Client — MCP-facing command buffer submission.
 * Vortek-inspired: submit compute commands, backend routed by quirks.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { buildQuirksProfile, routeKernel } from './adreno_quirks.js';
export function createCommandBuffer(commands) {
    return { version: 1, commands };
}
export function routeCommandBuffer(buf, profile) {
    return {
        version: 1,
        commands: buf.commands.map(cmd => ({
            ...cmd,
            backend: cmd.backend ?? routeKernel(profile, cmd.op, cmd.shape.tokens),
        })),
    };
}
export function buildProfileFromProbe(probe) {
    const dev = probe.devices?.[0];
    if (!dev)
        return null;
    const device = {
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
