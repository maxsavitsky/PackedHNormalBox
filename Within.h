#include <array>
#include <immintrin.h>

bool Within_cpp(const NormalBox& normal_box1, const NormalBox& normal_box2) {
    return normal_box1.corner.x <= normal_box2.corner.x &&
           normal_box1.corner.y <= normal_box2.corner.y;
}

unsigned int WithinPacked_cpp(const PackedHNormalBox& packed_h_normal_box1,
                              const PackedHNormalBox& packed_h_normal_box2) {
    unsigned int mask = 0;
    for (size_t i = 0; i < 4; i++) {
        if (Within_cpp(packed_h_normal_box1.boxes.at(i), packed_h_normal_box2.boxes.at(i))) {
            mask |= 1U << i;
        }
    }

    return mask;
}

extern "C" unsigned int WithinPacked_avx(const PackedHNormalBox& packed_h_normal_box1,
                                         const PackedHNormalBox& packed_h_normal_box2) {
    const __m512d a = _mm512_load_pd(packed_h_normal_box1.boxes.data());
    const __m512d b = _mm512_load_pd(packed_h_normal_box2.boxes.data());
    const __mmask8 mask8 = _mm512_cmple_pd_mask(a, b);

    unsigned int mask = (mask8);

    // e.g. we have 11010011
    // now we should perform "horizontal" AND to get output mask (1001)

    unsigned int out_mask = 0;
    for (unsigned int i = 0; i < 4; i++) {
        out_mask |= ((mask >> (2 * i)) & (mask >> (2 * i + 1)) & 1U) << i;
    }
    return out_mask;
}
