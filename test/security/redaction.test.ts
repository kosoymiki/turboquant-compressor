import { describe, it, expect } from '@jest/globals';

const PATTERNS = [
  /\b(ANTHROPIC_API_KEY|OPENAI_API_KEY|NVIDIA_API_KEY|GITHUB_TOKEN|HF_TOKEN|GOOGLE_API_KEY)\s*=\s*\S+/gi,
  /\b\w+_(TOKEN|SECRET|KEY|PASSWORD)\s*=\s*\S+/gi,
  /Authorization:\s*Bearer\s+\S+/gi,
  /\bsk-[A-Za-z0-9_-]{20,}/g,
  /\bant-[A-Za-z0-9_-]{20,}/g,
  /\bghp_[A-Za-z0-9]{36,}/g,
  /\bhf_[A-Za-z0-9]{20,}/g,
];

function redact(text: string): string {
  let out = text;
  for (const p of PATTERNS) {
    out = out.replace(p, (match) => {
      const eqIdx = match.indexOf('=');
      if (eqIdx !== -1) return match.slice(0, eqIdx + 1) + '[REDACTED]';
      return '[REDACTED]';
    });
  }
  return out;
}

describe('redact-forensics patterns', () => {
  it('redacts ANTHROPIC_API_KEY assignment', () => {
    const r = redact('ANTHROPIC_API_KEY=sk-ant-abc123xyz');
    expect(r).toBe('ANTHROPIC_API_KEY=[REDACTED]');
    expect(r).not.toMatch(/sk-ant/);
  });

  it('redacts OPENAI_API_KEY assignment', () => {
    const r = redact('OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz');
    expect(r).toContain('[REDACTED]');
    expect(r).not.toMatch(/sk-abc/);
  });

  it('redacts Bearer token in Authorization header', () => {
    const r = redact('Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9');
    expect(r).toContain('[REDACTED]');
    expect(r).not.toMatch(/eyJ/);
  });

  it('redacts sk- style API key inline', () => {
    const r = redact('key is sk-abcdefghijklmnopqrstuvwxyz123456');
    expect(r).not.toMatch(/sk-abc/);
    expect(r).toContain('[REDACTED]');
  });

  it('redacts ant- style key', () => {
    const r = redact('ant-api01-abcdefghijklmnopqrstuvwxyz12345678');
    expect(r).toContain('[REDACTED]');
  });

  it('redacts ghp_ GitHub token', () => {
    const r = redact('ghp_abcdefghijklmnopqrstuvwxyz1234567890ab');
    expect(r).toContain('[REDACTED]');
    expect(r).not.toMatch(/ghp_abc/);
  });

  it('does not redact normal log lines', () => {
    const line = 'Tests: 134 passed, 134 total';
    expect(redact(line)).toBe(line);
  });

  it('does not redact empty string', () => {
    expect(redact('')).toBe('');
  });

  it('redacts generic MY_SECRET= pattern', () => {
    const r = redact('MY_SECRET=supersecretvalue123');
    expect(r).toContain('[REDACTED]');
    expect(r).not.toMatch(/supersecret/);
  });
});
