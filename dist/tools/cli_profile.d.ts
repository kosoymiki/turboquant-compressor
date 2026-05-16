/**
 * CLI profile tool implementation.
 * Returns MCP client profile information for any supported host.
 */
export interface CliProfileInput {
    host: string;
    includeSmokeTest?: boolean;
}
export interface CliProfileResult {
    host: string;
    supportStatus: 'verified' | 'documented' | 'unverified';
    transport: 'stdio' | 'http' | 'sse';
    configSnippet: string;
    smokeCommand: string;
    expectedTools: string[];
    warnings: string[];
}
/**
 * Get MCP client profile for specified host.
 */
export declare function getCliProfile(input: CliProfileInput): CliProfileResult;
/**
 * Get list of all supported hosts.
 */
export declare function listSupportedHosts(): string[];
