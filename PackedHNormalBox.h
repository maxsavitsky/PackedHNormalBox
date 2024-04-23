#include <array>
#include <immintrin.h>

struct alignas(64) Point {
    double x;
    double y;
};

struct alignas(64) NormalBox {
    Point corner;
};

struct PackedHNormalBox {
    std::array<NormalBox, 4> boxes;
};

double SquaredDistance_cpp(const Point& point, const NormalBox& normalBox) {
    double xdis = abs(point.x) - normalBox.corner.x;
    double ydis = abs(point.y) - normalBox.corner.y;
    if (xdis < 0) {
        xdis = 0;
    }
    if (ydis < 0) {
        ydis = 0;
    }
    return xdis * xdis + ydis * ydis;
}

std::array<double, 4> SquaredDistancePacked_cpp(const Point& point,
                                                const PackedHNormalBox& packedHNormalBox) {
    std::array<double, 4> ans;
    for (size_t i = 0; i < 4; i++) {
        ans[i] = SquaredDistance_cpp(point, packedHNormalBox.boxes[i]);
    }
    return ans;
}

extern "C" std::array<double, 4> SquaredDistancePacked_avx(const Point& point,
                                                           const PackedHNormalBox& packedHNormalBox) {
    __m512d point_x = _mm512_set1_pd(point.x);
    __m512d point_y = _mm512_set1_pd(point.y);
    __m512d point_coords = _mm512_mask_blend_pd((0b10101010), point_x, point_y);
    __m512d packed_normal_box = _mm512_load_pd(packedHNormalBox.boxes.data());
    __m512d difference = _mm512_abs_pd(_mm512_sub_pd(point_coords, packed_normal_box));
    __m512d sum = _mm512_abs_pd(_mm512_add_pd(point_coords, packed_normal_box));

    difference = _mm512_add_pd(difference, sum);
    difference = _mm512_sub_pd(difference, packed_normal_box);
    difference = _mm512_sub_pd(difference, packed_normal_box);

    difference = _mm512_mul_pd(difference, difference);
    __m512d sum_dif_shuffled = _mm512_permute_pd(difference, 0b01010101);
    difference = _mm512_add_pd(difference, sum_dif_shuffled);

    __m512d fours = _mm512_set1_pd(4.0);
    difference = _mm512_div_pd(difference, fours);

    std::array<double, 8> notanswer;
    _mm512_store_pd(notanswer.data(), difference);
    std::array<double, 4> answer = {notanswer[1], notanswer[3], notanswer[5], notanswer[7]};

    return answer;
}

double SquaredDeepDistance_cpp(const Point& point, const NormalBox& normalBox) {
    double xdis = abs(point.x) - normalBox.corner.x;
    double ydis = abs(point.y) - normalBox.corner.y;
    if (ydis < 0) {
        if (xdis < 0) {
            const double t = std::max(xdis, ydis);
            return t * t;
        }
        return xdis * xdis;
    }
    if (xdis < 0) {
        return ydis * ydis;
    }
    return xdis * xdis + ydis * ydis;
}

std::array<double, 4> SquaredDeepDistancePacked_cpp(const Point& point,
                                                    const PackedHNormalBox& packedHNormalBox) {
    std::array<double, 4> ans;
    for (size_t i = 0; i < 4; i++) {
        ans[i] = SquaredDeepDistance_cpp(point, packedHNormalBox.boxes[i]);
    }
    return ans;
}

extern "C" std::array<double, 4> SquaredDeepDistancePacked_avx(const Point& point,
                                                               const PackedHNormalBox& packedHNormalBox) {
    __m512d zeros = _mm512_setzero_pd();

    __m512d point_x = _mm512_abs_pd(_mm512_set1_pd(point.x)); //xxxxxxxx
    __m512d point_y = _mm512_abs_pd(_mm512_set1_pd(point.y)); //yyyyyyyy
    __m512d point_coords = _mm512_mask_blend_pd((0b10101010), point_x, point_y); //xyxyxyxy

    __m512d packed_normal_box = _mm512_load_pd(packedHNormalBox.boxes.data()); //lwlwlwlw
    __m512d difference = _mm512_sub_pd(point_coords, packed_normal_box); //dxdydxdydxdydxdy

    //00110110 00: both x and y outside; 01: x outside, y inside; 10: x inside, y outside; 11: both x and y inside
    __mmask8 mask_xy_inside = _mm512_cmple_pd_mask(difference,
                                                   zeros);              //xiyixiyixiyixiyi
    __mmask8 mask_xy_inside_x = _kshiftri_mask8(mask_xy_inside,
                                                1);                 // 0xiyixiyixiyixi
    __mmask8 mask_xy_outside = ~(mask_xy_inside);                                   //xoyoxoyoxoyoxoyo
    __mmask8 mask_xy_outside_x = _kshiftri_mask8(mask_xy_outside,
                                                 1);                // 0xoyoxoyoxoyoxo

    __mmask8 mask_full_inside = (mask_xy_inside &
                                 mask_xy_inside_x);                // 0xy??xy??xy??xy
    mask_full_inside = (mask_full_inside &
                        0b01010101);                                   //_0xy_0xy_0xy_0xy
    mask_full_inside = (mask_full_inside |
                        _kshiftli_mask8(mask_full_inside, 1));   //xyxyxyxyxyxyxyxy

    __mmask8 mask_x_inside = (mask_xy_outside & mask_xy_inside_x);
    mask_x_inside = (mask_x_inside & 0b01010101);
    mask_x_inside = (mask_x_inside | _kshiftli_mask8(mask_x_inside, 1));

    __mmask8 mask_y_inside = (mask_xy_inside & mask_xy_outside_x);
    mask_y_inside = (mask_y_inside & 0b01010101);
    mask_y_inside = (mask_y_inside | _kshiftli_mask8(mask_y_inside, 1));

    __mmask8 mask_no_inside = (mask_xy_outside & mask_xy_outside_x);
    mask_no_inside = (mask_no_inside & 0b01010101);
    mask_no_inside = (mask_no_inside | _kshiftli_mask8(mask_no_inside, 1));

    difference = _mm512_mul_pd(difference, difference);
    __m512d difference_shuffled = _mm512_permute_pd(difference, 0b01010101);
    __m512d difference_x = _mm512_permute_pd(difference, 0b00000000);
    __m512d difference_y = _mm512_permute_pd(difference, 0b11111111);

    __m512d ans = _mm512_min_pd(difference, difference_shuffled);
    ans = _mm512_mask_mov_pd(ans, mask_x_inside, difference_x);
    ans = _mm512_mask_mov_pd(ans, mask_y_inside, difference_y);
    difference = _mm512_maskz_add_pd(mask_no_inside, difference,
                                     _mm512_permute_pd(difference, 0b01010101));
    ans = _mm512_mask_mov_pd(ans, mask_no_inside, difference);

    std::array<double, 8> notanswer;
    _mm512_store_pd(notanswer.data(), ans);
    std::array<double, 4> answer = {notanswer[1], notanswer[3], notanswer[5], notanswer[7]};

    return answer;
}

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
