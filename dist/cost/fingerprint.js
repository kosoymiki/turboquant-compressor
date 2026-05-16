import { createHash } from 'node:crypto';
// Fast FNV-32 for local token hashing (not for manifest integrity)
export function fingerprintText(text) {
    let hash = 2166136261 >>> 0;
    for (let i = 0; i < text.length; i++) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
}
// SHA-256 for manifest integrity and forensic identity
export function sha256Text(text) {
    return createHash('sha256').update(text, 'utf8').digest('hex');
}
export function approximateTokens(text) {
    return Math.ceil(Buffer.byteLength(text, 'utf8') / 4);
}
