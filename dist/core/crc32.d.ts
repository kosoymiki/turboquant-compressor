/**
 * CRC32 checksum with standard polynomial 0xEDB88320.
 */
export declare function crc32(data: Uint8Array): number;
export declare function verifyCrc32(data: Uint8Array, expectedCrc: number): boolean;
