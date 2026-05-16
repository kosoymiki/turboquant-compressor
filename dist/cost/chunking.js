export function chunkText(text, maxBytes) {
    const chunks = [];
    let current = '';
    for (const line of text.split(/\r?\n/)) {
        const next = current + line + '\n';
        if (Buffer.byteLength(next, 'utf8') > maxBytes && current.length > 0) {
            chunks.push(current);
            current = '';
        }
        current += line + '\n';
    }
    if (current.length > 0)
        chunks.push(current);
    return chunks;
}
