/**
 * MCP client profile registry.
 */
import { ClientProfile, ClientRegistry } from './types.js';
export declare const genericStdioProfile: ClientProfile;
export declare const claudeCodeProfile: ClientProfile;
export declare const codexCliProfile: ClientProfile;
export declare const cursorProfile: ClientProfile;
export declare const geminiCliProfile: ClientProfile;
export declare const opencodeProfile: ClientProfile;
export declare const crushProfile: ClientProfile;
export declare const clientRegistry: ClientRegistry;
export declare function getClientProfile(id: string): ClientProfile | undefined;
export declare function getClientIds(): string[];
