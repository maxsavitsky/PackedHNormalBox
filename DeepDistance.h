#include <array>
#include <immintrin.h>

double SquaredDeepDistance_cpp(const Point& point, const NormalBox& normalBox) {
    double xdis = abs(point.x) - normalBox.corner.x;
    double ydis = abs(point.y) - normalBox.corner.y;
    if (ydis < 0) {
        if (xdis < 0) {
            const double t = std::max(xdis, ydis);
            return -(t * t);
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
    __m512d neg = _mm512_set1_pd(-1.0);
    ans = _mm512_mul_pd(ans, neg);
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