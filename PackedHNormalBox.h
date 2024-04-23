#include <immintrin.h>
#include <stdio.h>
#include <iostream>

#include <cmath>
#include <array>

void Debug_print8(__m512d& mem) {
    std::array<double, 8> arr;
    _mm512_store_pd(arr.data(), mem);
    for (int i = 0; i < 8; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

struct Point {
    double x;
    double y;
};

struct NormalBox {
    double l;
    double w;
};

struct PackedHNormalBox {
    std::array<NormalBox, 4> boxes;
};

double SquaredDistancePointToNormalBox_cpp(const Point& point, const NormalBox& normalBox) {
    double xdis = abs(point.x) - normalBox.l;
    double ydis = abs(point.y) - normalBox.w;
    if (ydis < 0) {
        if (xdis < 0) {
            double t = std::min(std::abs(xdis), std::abs(ydis));
            return t * t;
        }
        return xdis * xdis;
    }
    if (xdis < 0) {
        return ydis * ydis;
    }
    return xdis * xdis + ydis * ydis;
}

std::array<double, 4> SquaredDistancePointToPackedHNormalBox_cpp(const Point& point,
                                                                 const PackedHNormalBox& packedHNormalBox) {
    std::array<double, 4> ans;
    for (size_t i = 0; i < 4; i++) {
        ans[i] = SquaredDistancePointToNormalBox_cpp(point, packedHNormalBox.boxes[i]);
    }
    return ans;
}

std::array<double, 4> SquaredDistancePointToPackedHNormalBox_avx(const Point& point,
                                                                 const PackedHNormalBox& packedHNormalBox) {
    __m512d zeros = _mm512_setzero_pd();

    __m512d point_x = _mm512_abs_pd(_mm512_set1_pd(point.x)); //xxxxxxxx
    __m512d point_y = _mm512_abs_pd(_mm512_set1_pd(point.y)); //yyyyyyyy
    __m512d point_coords = _mm512_mask_blend_pd((170), point_x, point_y); //xyxyxyxy

    __m512d packed_normal_box = _mm512_loadu_pd(packedHNormalBox.boxes.data()); //lwlwlwlw
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
                        (85));                                   //_0xy_0xy_0xy_0xy
    mask_full_inside = (mask_full_inside |
                        _kshiftli_mask8(mask_full_inside, 1));   //xyxyxyxyxyxyxyxy

    __mmask8 mask_x_inside = (mask_xy_outside & mask_xy_inside_x);
    mask_x_inside = (mask_x_inside & (85));
    mask_x_inside = (mask_x_inside | _kshiftli_mask8(mask_x_inside, 1));

    __mmask8 mask_y_inside = (mask_xy_inside & mask_xy_outside_x);
    mask_y_inside = (mask_y_inside & (85));
    mask_y_inside = (mask_y_inside | _kshiftli_mask8(mask_y_inside, 1));

    __mmask8 mask_no_inside = (mask_xy_outside & mask_xy_outside_x);
    mask_no_inside = (mask_no_inside & (85));
    mask_no_inside = (mask_no_inside | _kshiftli_mask8(mask_no_inside, 1));

    difference = _mm512_mul_pd(difference, difference);
    __m512d difference_shuffled = _mm512_permute_pd(difference, 85);
    __m512d difference_x = _mm512_permute_pd(difference, 0);
    __m512d difference_y = _mm512_permute_pd(difference, 255);

    __m512d ans = _mm512_min_pd(difference, difference_shuffled);
    ans = _mm512_mask_mov_pd(ans, mask_x_inside, difference_x);
    ans = _mm512_mask_mov_pd(ans, mask_y_inside, difference_y);
    difference = _mm512_maskz_add_pd(mask_no_inside, difference, _mm512_permute_pd(difference, 85));
    ans = _mm512_mask_mov_pd(ans, mask_no_inside, difference);

    std::array<double, 8> notanswer;
    _mm512_store_pd(notanswer.data(), ans);
    std::array<double, 4> answer = {notanswer[1], notanswer[3], notanswer[5], notanswer[7]};

    return answer;
}

bool NormalBoxWithinNormalBox_cpp(const NormalBox& normalBox1, const NormalBox& normalBox2) {
    return normalBox1.l <= normalBox2.l && normalBox1.w <= normalBox2.w;
}

unsigned int PackedHNormalBoxWithinPackedHNormalBox_cpp(const PackedHNormalBox& packedHNormalBox1,
                                                        const PackedHNormalBox& packedHNormalBox2) {
    unsigned int mask = 0;
    for (size_t i = 0; i < 4; i++) {
        if (NormalBoxWithinNormalBox_cpp(packedHNormalBox1.boxes[i], packedHNormalBox2.boxes[i])) {
            mask |= 1U << i;
        }
    }

    return mask;
}

unsigned int PackedHNormalBoxWithinPackedHNormalBox_avx(const PackedHNormalBox& packedHNormalBox1,
                                                        const PackedHNormalBox& packedHNormalBox2) {
    __m512d a = _mm512_loadu_pd(packedHNormalBox1.boxes.data());
    __m512d b = _mm512_loadu_pd(packedHNormalBox2.boxes.data());
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


//////////////////////////////////////////////////////////////////////


double SquaredDistancePointToNormalBox2_cpp(const Point& point, const NormalBox& normalBox) {
    double xdis = abs(point.x) - normalBox.l;
    double ydis = abs(point.y) - normalBox.w;
    if (xdis < 0) {
        xdis = 0;
    }
    if (ydis < 0) {
        ydis = 0;
    }
    return xdis * xdis + ydis * ydis;
}

std::array<double, 4> SquaredDistancePointToPackedHNormalBox2_cpp(const Point& point,
                                                                 const PackedHNormalBox& packedHNormalBox) {
    std::array<double, 4> ans;
    for (size_t i = 0; i < 4; i++) {
        ans[i] = SquaredDistancePointToNormalBox2_cpp(point, packedHNormalBox.boxes[i]);
    }
    return ans;
}

std::array<double, 4> SquaredDistancePointToPackedHNormalBox2_avx(const Point& point,
                                                                 const PackedHNormalBox& packedHNormalBox) {
    __m512d point_x = _mm512_abs_pd(_mm512_set1_pd(point.x)); //xxxxxxxx
    __m512d point_y = _mm512_abs_pd(_mm512_set1_pd(point.y)); //yyyyyyyy
    __m512d point_coords = _mm512_mask_blend_pd((170), point_x, point_y); //xyxyxyxy
    __m512d packed_normal_box = _mm512_loadu_pd(packedHNormalBox.boxes.data()); //lwlwlwlw
    __m512d difference1 = _mm512_sub_pd(point_coords, packed_normal_box); //dxdydxdydxdydxdy
    __m512d difference2 = _mm512_add_pd(point_coords, packed_normal_box);

    Debug_print8(point_coords);
    Debug_print8(packed_normal_box);

    Debug_print8(difference1);
    Debug_print8(difference2);

    __m512d sum_diff = _mm512_add_pd(difference1, difference2);
    Debug_print8(sum_diff);
    sum_diff = _mm512_sub_pd(sum_diff, packed_normal_box);
    sum_diff = _mm512_sub_pd(sum_diff, packed_normal_box);

    Debug_print8(sum_diff);

    sum_diff = _mm512_mul_pd(sum_diff, sum_diff);
    __m512d sum_dif_shuffled = _mm512_permute_pd(sum_diff, 85);
    sum_diff = _mm512_add_pd(sum_diff, sum_dif_shuffled);

    Debug_print8(sum_diff);

    __m512d fours = _mm512_set1_pd(4.0);
    sum_diff = _mm512_div_pd(sum_diff, fours);

    std::array<double, 8> notanswer;
    _mm512_store_pd(notanswer.data(), sum_diff);
    std::array<double, 4> answer = {notanswer[1], notanswer[3], notanswer[5], notanswer[7]};

    return answer;
}