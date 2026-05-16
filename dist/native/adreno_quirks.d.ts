/**
 * TurboQuant Adreno Quirks Registry — TypeScript dispatch policy layer.
 * Builds kernel routing from device probe + benchmark evidence.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export type BackendRoute = 'opencl_adreno' | 'opencl_tiled' | 'neon' | 'cpu_scalar' | 'vulkan_compute' | 'mesa_rusticl' | 'typescript_cpu' | 'auto';
export interface DeviceProfile {
    gpu: string;
    opencl: string;
    fp16: boolean;
    subgroups: boolean;
    maxWorkGroupSize: number;
    computeUnits: number;
    globalMemBytes: number;
}
export interface KernelPolicy {
    mse_score: BackendRoute;
    qjl_score: BackendRoute;
    value_dequant: BackendRoute;
    fused_attention: BackendRoute;
    fused_attention_neon_threshold_tokens: number;
    enable_subgroup_path: boolean;
    enable_fp16_values: boolean;
    three_bit_cross_byte_checked: boolean;
}
export interface QuirksProfile {
    device: DeviceProfile;
    policies: KernelPolicy;
}
export declare function buildQuirksProfile(device: DeviceProfile): QuirksProfile;
export declare function routeKernel(profile: QuirksProfile, kernel: string, tokens: number): BackendRoute;
