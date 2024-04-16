#include <immintrin.h>
#include <stdio.h>
#include <iostream>

#include <cmath>
#include <array>

void debug_out(__mmask8 mask) {
    int n = static_cast<int>(mask);
    for (size_t i = 0; i < 8; i++)
    {
        std::cout << n % 2;
        n /= 2;
    }
    std::cout << std::endl;
}

double DistancePtoNB_cpp(double x, double y, double length, double width) {
    x = abs(x);
    y = abs(y);
    double xdis = x - length;
    double ydis = y - width;
    if (ydis < 0) {
        if (xdis < 0) {
            return std::min(std::abs(xdis), std::abs(ydis));
        }
        return xdis;
    }
    if (xdis < 0) {
        return ydis;
    }
    return sqrt(xdis*xdis + ydis*ydis);
}

std::array<double, 4> DistancePtoPackedHNB_cpp(double x, double y, std::array<double, 8>& NBCoordinates) {
    std::array<double, 4> ans;
    for (size_t i = 0; i < 4; i++) {
        ans[i] = DistancePtoNB_cpp(x, y, NBCoordinates[i * 2], NBCoordinates[i * 2 + 1]);
    }
    return ans;
}

std::array<double, 4> DistancePtoPackedHNB_avx(double x, double y, std::array<double, 8>& NBCoordinates) {
    __m512d zeros = _mm512_setzero_pd();

    x = abs(x);
    y = abs(y);
    // since NormalBox is symmetrical, we can consider Point to be in a 1st quarter

    __m512d point_x = _mm512_set1_pd(x); //xxxxxxxx
    __m512d point_y = _mm512_set1_pd(y); //yyyyyyyy

    __m512d point = _mm512_mask_blend_pd((170), point_x, point_y); //xyxyxyxy

    __m512d packed_normal_box = _mm512_loadu_pd(NBCoordinates.data()); //lwlwlwlw

    __m512d difference = _mm512_sub_pd(point, packed_normal_box); //dxdydxdydxdydxdy

    //00110110 00: both x and y outside; 01: x outside, y inside; 10: x inside, y outside; 11: both x and y inside
    __mmask8 mask_xy_inside = _mm512_cmple_pd_mask(difference, zeros);              //xiyixiyixiyixiyi
    __mmask8 mask_xy_inside_x = _kshiftri_mask8(mask_xy_inside, 1);                 // 0xiyixiyixiyixi
    __mmask8 mask_xy_outside = ~(mask_xy_inside);                                   //xoyoxoyoxoyoxoyo
    __mmask8 mask_xy_outside_x = _kshiftri_mask8(mask_xy_outside, 1);                // 0xoyoxoyoxoyoxo

    __mmask8 mask_full_inside = (mask_xy_inside & mask_xy_inside_x);                // 0xy??xy??xy??xy
    mask_full_inside = (mask_full_inside & (85));                                   //_0xy_0xy_0xy_0xy
    mask_full_inside = (mask_full_inside | _kshiftli_mask8(mask_full_inside, 1));   //xyxyxyxyxyxyxyxy

    __mmask8 mask_x_inside = (mask_xy_outside & mask_xy_inside_x);
    mask_x_inside = (mask_x_inside & (85));
    mask_x_inside = (mask_x_inside | _kshiftli_mask8(mask_x_inside, 1));

    __mmask8 mask_y_inside = (mask_xy_inside & mask_xy_outside_x);
    mask_y_inside = (mask_y_inside & (85));
    mask_y_inside = (mask_y_inside | _kshiftli_mask8(mask_y_inside, 1));

    __mmask8 mask_no_inside = (mask_xy_outside & mask_xy_outside_x);
    mask_no_inside = (mask_no_inside & (85));
    mask_no_inside = (mask_no_inside | _kshiftli_mask8(mask_no_inside, 1));

    difference = _mm512_abs_pd(difference);
    __m512d difference_shuffled = _mm512_permute_pd(difference, 85);
    __m512d difference_x = _mm512_permute_pd(difference, 0);
    __m512d difference_y = _mm512_permute_pd(difference, 255);

    __m512d ans = _mm512_min_pd(difference, difference_shuffled);
    ans = _mm512_mask_mov_pd(ans, mask_x_inside, difference_x);
    ans = _mm512_mask_mov_pd(ans, mask_y_inside, difference_y);

    difference = _mm512_maskz_mul_pd(mask_no_inside, difference, difference);
    __m512d ds = _mm512_permute_pd(difference, 85);
    difference = _mm512_maskz_add_pd(mask_no_inside, difference, ds);
    difference = _mm512_maskz_sqrt_pd(mask_no_inside, difference);

    ans = _mm512_mask_mov_pd(ans, mask_no_inside, difference);

    std::array<double, 8> notanswer;
    _mm512_store_pd(notanswer.data(), ans);
    std::array<double, 4> answer = {notanswer[1], notanswer[3], notanswer[5], notanswer[7]};

    return answer;
}

bool NormalBoxWithinNormalBox_cpp(double l1, double w1, double l2, double w2) {
    return l1 <= l2 && w1 <= w2;
}

unsigned int PackedHNBWithinPackedHNB_cpp(const std::array<double, 8>& NBCoordinates1, const std::array<double, 8>& NBCoordinates2) {
    unsigned int mask = 0;
    for (size_t i = 0; i < 4; i++) {
        if (NBCoordinates1[i * 2] <= NBCoordinates2[i * 2] &&
            NBCoordinates1[i * 2 + 1] <= NBCoordinates2[i * 2 + 1]) {
            mask |= 1U << i;
        }
    }

    return mask;
}

unsigned int PackedHNBWithinPackedHNB_avx(const std::array<double, 8>& NBCoordinates1, const std::array<double, 8>& NBCoordinates2) {
    __m512d a = _mm512_loadu_pd(NBCoordinates1.data());
    __m512d b = _mm512_loadu_pd(NBCoordinates2.data());
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







