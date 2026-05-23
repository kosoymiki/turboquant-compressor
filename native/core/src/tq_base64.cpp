/**
 * TurboQuant Core — Base64 encode/decode (no external dependencies)
 *
 * Pure C++17 implementation. Matches Node.js Buffer.from(x, 'base64') encoding.
 */

#include "tq_base64.h"
#include <cstdint>
#include <cstring>

namespace tq {
namespace core {
namespace base64 {

static const char ENCODED[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static inline int decode_char(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    if (c == '=') return 0;
    return -1;
}

std::string encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve((len + 2) / 3 * 4);

    size_t i = 0;
    while (i + 3 <= len) {
        uint32_t octet_a = data[i++];
        uint32_t octet_b = data[i++];
        uint32_t octet_c = data[i++];

        out.push_back(ENCODED[(octet_a >> 2) & 0x3F]);
        out.push_back(ENCODED[((octet_a << 4) | (octet_b >> 4)) & 0x3F]);
        out.push_back(ENCODED[((octet_b << 2) | (octet_c >> 6)) & 0x3F]);
        out.push_back(ENCODED[octet_c & 0x3F]);
    }

    size_t rem = len - i;
    if (rem == 1) {
        uint32_t octet_a = data[i];
        out.push_back(ENCODED[(octet_a >> 2) & 0x3F]);
        out.push_back(ENCODED[(octet_a << 4) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        uint32_t octet_a = data[i];
        uint32_t octet_b = data[i + 1];
        out.push_back(ENCODED[(octet_a >> 2) & 0x3F]);
        out.push_back(ENCODED[((octet_a << 4) | (octet_b >> 4)) & 0x3F]);
        out.push_back(ENCODED[(octet_b << 2) & 0x3F]);
        out.push_back('=');
    }

    return out;
}

bool decode(const std::string& encoded, uint8_t* out, size_t* out_len) {
    // Remove whitespace
    size_t len = encoded.size();
    size_t o = 0;

    for (size_t i = 0; i < len; i += 4) {
        int vals[4] = {-1, -1, -1, -1};
        int count = 0;

        for (int j = 0; j < 4 && (i + j) < len; j++) {
            char c = encoded[i + j];
            if (c == '=') { count++; continue; }
            vals[j] = decode_char(c);
            if (vals[j] < 0) {
                // skip padding/whitespace
                continue;
            }
            count++;
        }

        if (count >= 2 && vals[0] >= 0 && vals[1] >= 0) {
            out[o++] = (vals[0] << 2) | (vals[1] >> 4);
        }
        if (count >= 3 && vals[2] >= 0) {
            out[o++] = (vals[1] << 4) | (vals[2] >> 2);
        }
        if (count >= 4 && vals[3] >= 0) {
            out[o++] = (vals[2] << 6) | vals[3];
        }
    }

    if (out_len) *out_len = o;
    return true;
}

std::vector<uint8_t> decode(const std::string& encoded) {
    std::vector<uint8_t> out((encoded.size() * 3 + 3) / 4);
    size_t len = 0;
    decode(encoded, out.data(), &len);
    out.resize(len);
    return out;
}

} // namespace base64
} // namespace core
} // namespace tq
