/**
 * TurboQuant Adreno Quirks Registry — TypeScript dispatch policy layer.
 * Builds kernel routing from device probe + benchmark evidence.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export function buildQuirksProfile(device) {
    const hasOpenCl = device.opencl.includes('OpenCL');
    const isAdreno = device.gpu.includes('Adreno') || device.gpu.includes('QUALCOMM');
    const policies = {
        mse_score: hasOpenCl ? 'opencl_adreno' : 'neon',
        qjl_score: hasOpenCl ? 'opencl_adreno' : 'neon',
        value_dequant: hasOpenCl ? 'opencl_adreno' : 'neon',
        fused_attention: hasOpenCl ? 'opencl_tiled' : 'neon',
        fused_attention_neon_threshold_tokens: 64,
        enable_subgroup_path: device.subgroups,
        enable_fp16_values: false,
        three_bit_cross_byte_checked: true,
    };
    // Adreno 7xx tuning
    if (isAdreno && device.computeUnits >= 4 && device.maxWorkGroupSize >= 1024) {
        policies.fused_attention_neon_threshold_tokens = 32;
    }
    else if (isAdreno && device.computeUnits >= 2) {
        policies.fused_attention_neon_threshold_tokens = 128;
    }
    return { device, policies };
}
export function routeKernel(profile, kernel, tokens) {
    if (kernel === 'mse_score')
        return profile.policies.mse_score;
    if (kernel === 'qjl_score')
        return profile.policies.qjl_score;
    if (kernel === 'value_dequant')
        return profile.policies.value_dequant;
    if (kernel === 'fused_attention') {
        if (tokens <= profile.policies.fused_attention_neon_threshold_tokens) {
            return 'neon';
        }
        return profile.policies.fused_attention;
    }
    return 'auto';
}
