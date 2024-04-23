#include <array>
#include <immintrin.h>


bool Within_cpp(const NormalBox& normalBox1, const NormalBox& normalBox2) {
    return normalBox1.corner.x <= normalBox2.corner.x && normalBox1.corner.y <= normalBox2.corner.y;
}

unsigned int WithinPacked_cpp(const PackedHNormalBox& packedHNormalBox1,
                              const PackedHNormalBox& packedHNormalBox2) {
    unsigned int mask = 0;
    for (size_t i = 0; i < 4; i++) {
        if (Within_cpp(packedHNormalBox1.boxes[i], packedHNormalBox2.boxes[i])) {
            mask |= 1U << i;
        }
    }

    return mask;
}

extern "C" unsigned int WithinPacked_avx(const PackedHNormalBox& packedHNormalBox1,
                                         const PackedHNormalBox& packedHNormalBox2) {
    __m512d a = _mm512_load_pd(packedHNormalBox1.boxes.data());
    __m512d b = _mm512_load_pd(packedHNormalBox2.boxes.data());
    __mmask8 mask8 = _mm512_cmple_pd_mask(a, b);

    unsigned int mask = (mask8);

    // e.g. we have 11010011
    // now we should perform "horizontal" AND to get output mask (1001)

    unsigned int out_mask = 0;
    for (unsigned int i = 0; i < 4; i++) {
        out_mask |= ((mask >> (2 * i)) & (mask >> (2 * i + 1)) & 1U) << i;
    }
    return out_mask;
}
