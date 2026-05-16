#include "../include/tq_opencl_types.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

namespace tq { void value_dequant_cpu_reference(const ValueDequantInput& input, float* output); }

static void pack_4bit(uint8_t* packed, int coord, uint8_t val) {
    int byte_idx = coord >> 1;
    int shift = (coord & 1) * 4;
    packed[byte_idx] |= (val & 0xF) << shift;
}

int main() {
    // 2 tokens, dim=4, 4-bit, group_size=2 (2 groups)
    const int tokens = 2, dim = 4, bits = 4, group_size = 2;
    int n_groups = 2;
    int packed_stride = (dim * bits + 7) / 8; // 2 bytes

    std::vector<uint8_t> packed(tokens * packed_stride, 0);
    // Token 0: values [3, 7, 1, 15]
    pack_4bit(packed.data(), 0, 3);
    pack_4bit(packed.data(), 1, 7);
    pack_4bit(packed.data(), 2, 1);
    pack_4bit(packed.data(), 3, 15);
    // Token 1: values [0, 0, 8, 8]
    pack_4bit(packed.data() + packed_stride, 0, 0);
    pack_4bit(packed.data() + packed_stride, 1, 0);
    pack_4bit(packed.data() + packed_stride, 2, 8);
    pack_4bit(packed.data() + packed_stride, 3, 8);

    // scales: [token0_g0, token0_g1, token1_g0, token1_g1]
    float scales[4] = {0.1f, 0.2f, 0.5f, 0.3f};
    float zeros[4] = {-1.0f, -2.0f, 0.0f, -1.0f};

    // Expected:
    // T0: [3*0.1-1.0, 7*0.1-1.0, 1*0.2-2.0, 15*0.2-2.0] = [-0.7, -0.3, -1.8, 1.0]
    // T1: [0*0.5+0.0, 0*0.5+0.0, 8*0.3-1.0, 8*0.3-1.0] = [0.0, 0.0, 1.4, 1.4]
    float expected[8] = {-0.7f, -0.3f, -1.8f, 1.0f, 0.0f, 0.0f, 1.4f, 1.4f};

    tq::ValueDequantInput input;
    input.packed_values = packed.data();
    input.scales = scales;
    input.zeros = zeros;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.group_size = group_size;

    float output[8] = {};
    tq::value_dequant_cpu_reference(input, output);

    int pass = 0, fail = 0;
    for (int i = 0; i < tokens * dim; i++) {
        float err = fabsf(output[i] - expected[i]);
        if (err < 1e-5f) { pass++; }
        else {
            printf("  FAIL [%d]: got %.6f expected %.6f\n", i, output[i], expected[i]);
            fail++;
        }
    }
    printf("Value dequant parity: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
