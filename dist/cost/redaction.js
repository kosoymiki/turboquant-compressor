// Key-value secret patterns that should be fully redacted
const SECRET_KEY_VALUE_PATTERNS = [
    // API keys, tokens, passwords in key=value format
    /\b(api[_-]?key|auth[_-]?token|token|password|secret|passwd|pwd)\s*[=:]\s*["']?([^"'\s]+)["']?/gi,
    // Bearer tokens
    /\b(Authorization|Bearer)\s*:\s*Bearer\s+([A-Za-z0-9_\-\.]+)/gi,
    // OpenAI/Anthropic API keys
    /\b(sk-ant-[A-Za-z0-9_-]{20,})\b/g,
    /\b(sk-[A-Za-z0-9_-]{20,})\b/g,
    // Private key blocks
    /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
    // Generic UUID-like secrets
    /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
];
// Simple word patterns for additional coverage
const SECRET_WORD_PATTERNS = [
    /\bapi[_-]?key\b/gi,
    /\bauth[_-]?token\b/gi,
    /\bsecret\b/gi,
    /\bpassword\b/gi,
];
export function redactSecrets(text) {
    let result = text;
    // Redact key-value pairs first
    for (const pattern of SECRET_KEY_VALUE_PATTERNS) {
        // Reset lastIndex for global patterns
        pattern.lastIndex = 0;
        result = result.replace(pattern, (match, key, value) => {
            // If we captured a key and value, redact the value
            if (value) {
                return `${key}=[REDACTED]`;
            }
            // For patterns without captured groups, redact the entire match
            return '[REDACTED]';
        });
    }
    // Additional word-level redaction for edge cases
    for (const pattern of SECRET_WORD_PATTERNS) {
        pattern.lastIndex = 0;
        result = result.replace(pattern, '[REDACTED]');
    }
    return result;
}
export function containsSecrets(text) {
    // Check if text contains any secret patterns
    for (const pattern of SECRET_KEY_VALUE_PATTERNS) {
        pattern.lastIndex = 0;
        if (pattern.test(text)) {
            return true;
        }
    }
    return false;
}
export function redactPath(path) {
    // Redact sensitive path components
    return path
        .replace(/\/api[_-]?keys?\//gi, '/[REDACTED]/')
        .replace(/\/secrets?\//gi, '/[REDACTED]/')
        .replace(/\/tokens?\//gi, '/[REDACTED]/');
}
