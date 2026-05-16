/**
 * OpenCL probe for Termux/Android — detects OpenCL availability and Adreno GPU.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export interface OpenClProbeResult {
    available: boolean;
    libraryExists: boolean;
    loadable: boolean | null;
    loaderState: 'NO_LIBRARY' | 'LIBRARY_EXISTS_NOT_PROVEN_LOADABLE' | 'BLOCKED_BY_ANDROID_LINKER_NAMESPACE' | 'LOADABLE_NO_PLATFORMS' | 'READY';
    platformCount: number;
    deviceCount: number;
    deviceNames: string[];
    vendorNames: string[];
    versionStrings: string[];
    hasFp16: boolean;
    hasSubgroups: boolean;
    adrenoDetected: boolean;
    libraryCandidates: Array<{
        path: string;
        exists: boolean;
    }>;
    recommendedBackend: 'typescript_cpu' | 'opencl_adreno' | 'opencl_generic' | 'cpu_neon';
    warnings: string[];
    probeTimeMs: number;
}
export interface OpenClProbeOptions {
    /** Run clinfo for deep device info (may take >1s) */
    deep?: boolean;
    /** Timeout for clinfo subprocess in ms */
    timeoutMs?: number;
}
export declare function probeOpenCl(options?: OpenClProbeOptions): OpenClProbeResult;
