#ifndef TQ_BASE64_H
#define TQ_BASE64_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace tq {
namespace core {
namespace base64 {

std::string encode(const uint8_t* data, size_t len);
bool decode(const std::string& encoded, uint8_t* out, size_t* out_len);
std::vector<uint8_t> decode(const std::string& encoded);

} // namespace base64
} // namespace core

// CRC32 standard polynomial 0xEDB88320
inline uint32_t crc32(const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (int j = 0; j < 256; j++) {
            uint32_t c = (uint32_t)j;
            for (int k = 0; k < 8; k++) c = (c >> 1) ^ (c & 1 ? 0xEDB88320U : 0);
            table[j] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFF;
}

// ── Bit packing for quantized vectors (inline, matches TS pack.ts) ─────────
inline void pack_bit_values(const uint32_t* indices, uint32_t count, int bits, uint8_t* out) {
    uint32_t mask = (1u << bits) - 1u;
    size_t bit_pos = 0;
    std::memset(out, 0, ((size_t)count * bits + 7) / 8);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t v = indices[i] & mask;
        size_t byte_pos = bit_pos >> 3;
        size_t bit_off = bit_pos & 7;
        out[byte_pos] |= (uint8_t)(v << bit_off);
        if (bit_off + bits > 8) out[byte_pos + 1] |= (uint8_t)(v >> (8 - bit_off));
        bit_pos += bits;
    }
}

} // namespace tq

#endif // TQ_BASE64_H
