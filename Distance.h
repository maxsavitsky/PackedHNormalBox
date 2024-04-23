#include <array>
#include <immintrin.h>

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