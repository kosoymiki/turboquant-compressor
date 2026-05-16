/**
 * CRC32 checksum with standard polynomial 0xEDB88320.
 */
const CRC32_TABLE = new Uint32Array(256);
function buildTable() {
    for (let i = 0; i < 256; i++) {
        let c = i;
        for (let j = 0; j < 8; j++) {
            c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        }
        CRC32_TABLE[i] = c;
    }
}
buildTable();
export function crc32(data) {
    let crc = 0xFFFFFFFF;
    for (const byte of data) {
        const index = (crc ^ byte) & 0xFF;
        crc = CRC32_TABLE[index] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
}
export function verifyCrc32(data, expectedCrc) {
    return crc32(data) === expectedCrc;
}
