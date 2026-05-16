/**
 * CLI profile tool implementation.
 * Returns MCP client profile information for any supported host.
 */

import { getClientProfile, getClientIds } from '../profiles/clients/registry.js';

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
export function getCliProfile(input: CliProfileInput): CliProfileResult {
  const profile = getClientProfile(input.host);

  if (!profile) {
    return {
      host: input.host,
      supportStatus: 'unverified',
      transport: 'stdio' as const,
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
export function listSupportedHosts(): string[] {
  return getClientIds();
}
