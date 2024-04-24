#include <array>
#include <immintrin.h>
#include <iostream>

void Debug_print8(const __m512d& mem) {
    std::array<double, 8> arr {};
    _mm512_store_pd(arr.data(), mem);
    for (int i = 0; i < 8; ++i) {
        std::cout << arr.at(i) << " ";
    }
    std::cout << '\n';
}

double SquaredDeepDistance_cpp(const Point& point, const NormalBox& normal_box) {
    const double x_dis = std::abs(point.x) - normal_box.corner.x;
    const double y_dis = std::abs(point.y) - normal_box.corner.y;
    if (y_dis < 0) {
        if (x_dis < 0) {
            double t = std::max(x_dis, y_dis);
            return -1.0 * t * t;
        }
        return x_dis * x_dis;
    }
    if (x_dis < 0) {
        return y_dis * y_dis;
    }
    return x_dis * x_dis + y_dis * y_dis;
}

std::array<double, 4>
SquaredDeepDistancePacked_cpp(const Point& point, const PackedHNormalBox& packed_h_normal_box) {
    std::array<double, 4> ans {};
    for (size_t i = 0; i < 4; i++) {
        ans.at(i) = SquaredDeepDistance_cpp(point, packed_h_normal_box.boxes.at(i));
    }
    return ans;
}

extern "C" std::array<double, 4>
SquaredDeepDistancePacked_avx(const Point& point, const PackedHNormalBox& packed_h_normal_box) {
    const __m512d point_x = _mm512_abs_pd(_mm512_set1_pd(point.x));
    const __m512d point_y = _mm512_abs_pd(_mm512_set1_pd(point.y));
    const __m512d point_coords = _mm512_mask_blend_pd((0b10101010), point_x, point_y);

    const __m512d packed_normal_box = _mm512_load_pd(packed_h_normal_box.boxes.data());

    __m512d difference = _mm512_sub_pd(point_coords, packed_normal_box);

    const __mmask8 mask_xy_inside = _mm512_cmple_pd_mask(difference, _mm512_setzero_pd());
    const __mmask8 mask_xy_inside_x = _kshiftri_mask8(mask_xy_inside, 1);
    const __mmask8 mask_xy_outside = ~(mask_xy_inside);
    const __mmask8 mask_xy_outside_x = _kshiftri_mask8(mask_xy_outside, 1);

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
    const __m512d difference_shuffled = _mm512_permute_pd(difference, 0b01010101);
    const __m512d difference_x = _mm512_permute_pd(difference, 0b00000000);
    const __m512d difference_y = _mm512_permute_pd(difference, 0b11111111);

    __m512d ans = _mm512_min_pd(difference, difference_shuffled);
    ans = _mm512_mul_pd(ans, _mm512_set1_pd(-1.0));
    ans = _mm512_mask_mov_pd(ans, mask_x_inside, difference_x);
    ans = _mm512_mask_mov_pd(ans, mask_y_inside, difference_y);
    difference = _mm512_maskz_add_pd(mask_no_inside, difference, _mm512_permute_pd(difference, 0b01010101));
    ans = _mm512_mask_mov_pd(ans, mask_no_inside, difference);

    alignas(64) std::array<double, 8> not_answer{};
    _mm512_mask_store_pd(not_answer.data(), 0b01010101, ans);
    std::array<double, 4> answer = {not_answer[0], not_answer[2], not_answer[4], not_answer[6]};
    return answer;
}