#include "tq_core.h"
#include <vector>
#include "tq_base64.h"
#include <cstring>
#include <cstdint>
#include <cstdlib>

uint32_t tq_crc32(const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) { uint32_t c = i; for (int j = 0; j < 8; j++) c = (c>>1)^(c&1?0xEDB88320U:0); table[i]=c; }
        init = true;
    }
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) crc = table[(crc^data[i])&0xFF]^(crc>>8);
    return crc ^ 0xFFFFFFFF;
}

int tq_decode_header(const char* db_base64, tq_db_header* header) {
    if (!db_base64 || !header) return TQ_ERR_NULL;
    std::vector<uint8_t> buf = tq::core::base64::decode(std::string(db_base64));
    if (buf.size() < 80) return TQ_ERR_FORMAT;
    uint32_t magic; std::memcpy(&magic, buf.data(), 4);
    if (magic != 0x434D5154) return TQ_ERR_FORMAT;
    std::memcpy(&header->magic,            buf.data(),      4);
    std::memcpy(&header->version,          buf.data() + 4,  4);
    std::memcpy(&header->dimensions,        buf.data() + 8,  4);
    std::memcpy(&header->padded_dimensions, buf.data() + 12, 4);
    std::memcpy(&header->vector_count,     buf.data() + 16, 4);
    header->bits_per_value = buf[20];
    std::memcpy(&header->rotation_seed,    buf.data() + 24, 4);
    header->flags = buf[28];
    uint32_t ver = 0; std::memcpy(&ver, buf.data() + 4, 4);
    header->codebook_type = (ver >= 3) ? buf[29] : TQ_CODEBOOK_UNIFORM;
    std::memcpy(&header->qjl_dims,  buf.data() + 56, 4);
    std::memcpy(&header->qjl_bytes, buf.data() + 60, 4);
    header->has_qjl = (header->qjl_bytes > 0) ? 1 : 0;
    return TQ_OK;
}
