/**
 * MCP client profile types.
 */
export type Transport = 'stdio' | 'http' | 'sse';
export type SupportStatus = 'verified' | 'documented' | 'unverified';
export interface ClientProfile {
    id: string;
    name: string;
    transport: Transport;
    configSnippet: string;
    smokeCommand: string;
    expectedTools: string[];
    warnings: string[];
    supportStatus: SupportStatus;
}
export interface ClientRegistry {
    [key: string]: ClientProfile;
}
