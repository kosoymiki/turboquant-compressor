/**
 * CLI profile tool implementation.
 * Returns MCP client profile information for any supported host.
 */
import { getClientProfile, getClientIds } from '../profiles/clients/registry.js';
/**
 * Get MCP client profile for specified host.
 */
export function getCliProfile(input) {
    const profile = getClientProfile(input.host);
    if (!profile) {
        return {
            host: input.host,
            supportStatus: 'unverified',
            transport: 'stdio',
            configSnippet: '# Host not recognized. Configure manually.',
            smokeCommand: '# No smoke command available for unknown host',
            expectedTools: [],
            warnings: [
                `Unknown host: ${input.host}`,
                'Supported hosts: ' + getClientIds().join(', '),
            ],
        };
    }
    return {
        host: input.host,
        supportStatus: profile.supportStatus,
        transport: profile.transport,
        configSnippet: profile.configSnippet,
        smokeCommand: profile.smokeCommand,
        expectedTools: profile.expectedTools,
        warnings: profile.warnings,
    };
}
/**
 * Get list of all supported hosts.
 */
export function listSupportedHosts() {
    return getClientIds();
}
