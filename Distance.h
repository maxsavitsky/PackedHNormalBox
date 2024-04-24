#include <array>
#include <immintrin.h>

double SquaredDistance_cpp(const Point& point, const NormalBox& normal_box) {
    double x_dis = abs(point.x) - normal_box.corner.x;
    double y_dis = abs(point.y) - normal_box.corner.y;
    x_dis = std::max<double>(x_dis, 0);
    y_dis = std::max<double>(y_dis, 0);
    return x_dis * x_dis + y_dis * y_dis;
}

std::array<double, 4>
SquaredDistancePacked_cpp(const Point& point, const PackedHNormalBox& packed_h_normal_box) {
    std::array<double, 4> ans{};
    for (size_t i = 0; i < 4; i++) {
        ans.at(i) = SquaredDistance_cpp(point, packed_h_normal_box.boxes.at(i));
    }
    return ans;
}

extern "C" std::array<double, 4>
SquaredDistancePacked_avx(const Point& point, const PackedHNormalBox& packed_h_normal_box) {
    const __m512d point_x = _mm512_set1_pd(point.x);
    const __m512d point_y = _mm512_set1_pd(point.y);
    const __m512d point_coords = _mm512_mask_blend_pd((0b10101010), point_x, point_y);
    const __m512d packed_normal_box = _mm512_load_pd(packed_h_normal_box.boxes.data());
    __m512d difference = _mm512_abs_pd(_mm512_sub_pd(point_coords, packed_normal_box));
    const __m512d sum = _mm512_abs_pd(_mm512_add_pd(point_coords, packed_normal_box));

    difference = _mm512_add_pd(difference, sum);
    difference = _mm512_sub_pd(difference, packed_normal_box);
    difference = _mm512_sub_pd(difference, packed_normal_box);

    difference = _mm512_mul_pd(difference, difference);
    const __m512d sum_dif_shuffled = _mm512_permute_pd(difference, 0b01010101);
    difference = _mm512_add_pd(difference, sum_dif_shuffled);

    difference = _mm512_div_pd(difference, _mm512_set1_pd(4.0));

    std::array<double, 8> not_answer{};
    _mm512_mask_store_pd(not_answer.data(), 0b10101010, difference);
    std::array<double, 4> answer = {not_answer[0], not_answer[2], not_answer[4], not_answer[6]};


    return answer;
}