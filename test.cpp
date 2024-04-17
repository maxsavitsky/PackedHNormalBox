#include "PackedHNormalBox.h"
#include <immintrin.h>
#include <iostream>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

TEST_CASE("Distance from Point to NormalBox") {

    REQUIRE(SquaredDistancePointToNormalBox_cpp(2, 0, 1, 1) == 1);
    REQUIRE(SquaredDistancePointToNormalBox_cpp(0, 2, 1, 1) == 1);

    REQUIRE(SquaredDistancePointToNormalBox_cpp(2, 2, 1, 1) == sqrt(2));
    REQUIRE(SquaredDistancePointToNormalBox_cpp(0, 0, 1, 1) == 1);

    REQUIRE(SquaredDistancePointToNormalBox_cpp(-2, 0, 1, 1) == 1);
    REQUIRE(SquaredDistancePointToNormalBox_cpp(0, 2, 1, 1) == 1);

    REQUIRE(SquaredDistancePointToNormalBox_cpp(2, -2, 1, 1) == sqrt(2));
}

void CompareArrays(std::array<double, 4> a, std::array<double, 4> b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a[i] - b[i]) <= 1e-5);
    }
}

TEST_CASE("Distance, CPP") {
    std::array<double, 8> v = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0};
    CompareArrays(SquaredDistancePointToPackedHNormalBox_cpp(2.0, 2.0, v), {sqrt(2), 0.0, 1.0, 1.0});
}

TEST_CASE("Distance, AVX") {
    std::array<double, 8> v = {1.0, 20.0, 1.0, 1.0, 10.0, 20.0, 10.0, 2.0};
    CompareArrays(SquaredDistancePointToPackedHNormalBox_cpp(3.0, 3.0, v),
                  SquaredDistancePointToPackedHNormalBox_avx(3.0, 3.0, v));
}

TEST_CASE("Random Distance") {
    {
        std::array<double, 8> arr = {1, 93, 72, 38, 26, 88, 16, 68};
        CompareArrays(DistancePtoPackedHNB_avx(-16, 92, arr), DistancePtoPackedHNB_cpp(-16, 92, arr));
    }
    {
        std::array<double, 8> arr = {83, 4, 32, 30, 56, 41, 75, 71};
        CompareArrays(DistancePtoPackedHNB_avx(72, -10, arr), DistancePtoPackedHNB_cpp(72, -10, arr));
    }
    {
        std::array<double, 8> arr = {37, 15, 96, 47, 53, 29, 62, 47};
        CompareArrays(DistancePtoPackedHNB_avx(-49, -45, arr), DistancePtoPackedHNB_cpp(-49, -45, arr));
    }
    {
        std::array<double, 8> arr = {11, 77, 10, 49, 58, 34, 74, 57};
        CompareArrays(DistancePtoPackedHNB_avx(-64, -69, arr), DistancePtoPackedHNB_cpp(-64, -69, arr));
    }
    {
        std::array<double, 8> arr = {83, 3, 81, 28, 82, 68, 92, 35};
        CompareArrays(DistancePtoPackedHNB_avx(-69, 27, arr), DistancePtoPackedHNB_cpp(-69, 27, arr));
    }
    {
        std::array<double, 8> arr = {30, 97, 57, 58, 86, 50, 15, 29};
        CompareArrays(DistancePtoPackedHNB_avx(-56, 39, arr), DistancePtoPackedHNB_cpp(-56, 39, arr));
    }
    {
        std::array<double, 8> arr = {29, 61, 39, 30, 3, 75, 45, 39};
        CompareArrays(DistancePtoPackedHNB_avx(-93, 96, arr), DistancePtoPackedHNB_cpp(-93, 96, arr));
    }
    {
        std::array<double, 8> arr = {91, 75, 40, 60, 9, 19, 41, 5};
        CompareArrays(DistancePtoPackedHNB_avx(-91, 99, arr), DistancePtoPackedHNB_cpp(-91, 99, arr));
    }
    {
        std::array<double, 8> arr = {35, 66, 20, 94, 67, 3, 27, 47};
        CompareArrays(DistancePtoPackedHNB_avx(-3, -88, arr), DistancePtoPackedHNB_cpp(-3, -88, arr));
    }
    {
        std::array<double, 8> arr = {26, 15, 31, 13, 62, 44, 86, 15};
        CompareArrays(DistancePtoPackedHNB_avx(-48, 97, arr), DistancePtoPackedHNB_cpp(-48, 97, arr));
    }
    {
        std::array<double, 8> arr = {35, 63, 95, 12, 30, 47, 95, 79};
        CompareArrays(DistancePtoPackedHNB_avx(65, 55, arr), DistancePtoPackedHNB_cpp(65, 55, arr));
    }
    {
        std::array<double, 8> arr = {81, 83, 12, 99, 11, 90, 80, 26};
        CompareArrays(DistancePtoPackedHNB_avx(90, 46, arr), DistancePtoPackedHNB_cpp(90, 46, arr));
    }
    {
        std::array<double, 8> arr = {12, 79, 96, 67, 55, 15, 37, 10};
        CompareArrays(DistancePtoPackedHNB_avx(79, 91, arr), DistancePtoPackedHNB_cpp(79, 91, arr));
    }
    {
        std::array<double, 8> arr = {56, 50, 50, 25, 18, 91, 4, 53};
        CompareArrays(DistancePtoPackedHNB_avx(-82, -3, arr), DistancePtoPackedHNB_cpp(-82, -3, arr));
    }
    {
        std::array<double, 8> arr = {47, 36, 50, 58, 93, 50, 81, 69};
        CompareArrays(DistancePtoPackedHNB_avx(-45, -40, arr), DistancePtoPackedHNB_cpp(-45, -40, arr));
    }
    {
        std::array<double, 8> arr = {77, 86, 57, 65, 37, 36, 75, 34};
        CompareArrays(DistancePtoPackedHNB_avx(98, 84, arr), DistancePtoPackedHNB_cpp(98, 84, arr));
    }
    {
        std::array<double, 8> arr = {21, 4, 38, 7, 58, 69, 5, 32};
        CompareArrays(DistancePtoPackedHNB_avx(-45, -19, arr), DistancePtoPackedHNB_cpp(-45, -19, arr));
    }
    {
        std::array<double, 8> arr = {23, 61, 34, 3, 78, 33, 4, 16};
        CompareArrays(DistancePtoPackedHNB_avx(-86, -74, arr), DistancePtoPackedHNB_cpp(-86, -74, arr));
    }
    {
        std::array<double, 8> arr = {50, 13, 70, 38, 45, 32, 78, 92};
        CompareArrays(DistancePtoPackedHNB_avx(-24, 86, arr), DistancePtoPackedHNB_cpp(-24, 86, arr));
    }
    {
        std::array<double, 8> arr = {29, 13, 64, 36, 21, 60, 26, 11};
        CompareArrays(DistancePtoPackedHNB_avx(23, -86, arr), DistancePtoPackedHNB_cpp(23, -86, arr));
    }
    {
        std::array<double, 8> arr = {78, 42, 11, 29, 58, 77, 34, 13};
        CompareArrays(DistancePtoPackedHNB_avx(13, -67, arr), DistancePtoPackedHNB_cpp(13, -67, arr));
    }
    {
        std::array<double, 8> arr = {75, 28, 56, 54, 40, 44, 11, 6};
        CompareArrays(DistancePtoPackedHNB_avx(-7, 82, arr), DistancePtoPackedHNB_cpp(-7, 82, arr));
    }
    {
        std::array<double, 8> arr = {71, 72, 26, 69, 21, 79, 51, 39};
        CompareArrays(DistancePtoPackedHNB_avx(32, 96, arr), DistancePtoPackedHNB_cpp(32, 96, arr));
    }
    {
        std::array<double, 8> arr = {27, 84, 25, 42, 63, 37, 43, 51};
        CompareArrays(DistancePtoPackedHNB_avx(-31, 76, arr), DistancePtoPackedHNB_cpp(-31, 76, arr));
    }
    {
        std::array<double, 8> arr = {12, 52, 59, 7, 1, 5, 15, 19};
        CompareArrays(DistancePtoPackedHNB_avx(-81, -18, arr), DistancePtoPackedHNB_cpp(-81, -18, arr));
    }
    {
        std::array<double, 8> arr = {42, 21, 95, 51, 42, 92, 80, 58};
        CompareArrays(DistancePtoPackedHNB_avx(67, -68, arr), DistancePtoPackedHNB_cpp(67, -68, arr));
    }
    {
        std::array<double, 8> arr = {62, 5, 15, 81, 55, 14, 43, 29};
        CompareArrays(DistancePtoPackedHNB_avx(-42, 56, arr), DistancePtoPackedHNB_cpp(-42, 56, arr));
    }
    {
        std::array<double, 8> arr = {7, 78, 52, 63, 67, 33, 5, 23};
        CompareArrays(DistancePtoPackedHNB_avx(-3, -78, arr), DistancePtoPackedHNB_cpp(-3, -78, arr));
    }
    {
        std::array<double, 8> arr = {86, 52, 5, 10, 74, 59, 29, 65};
        CompareArrays(DistancePtoPackedHNB_avx(37, -98, arr), DistancePtoPackedHNB_cpp(37, -98, arr));
    }
}

TEST_CASE("PackedNormalBox within PackedNormalBox") {
    {
        std::array<double, 8> v1 = {9, 24, 5, 11, 14, 19, 10, 24};
        std::array<double, 8> v2 = {17, 22, 15, 14, 19, 7, 23, 12};
        REQUIRE(0b0010 == PackedHNormalBoxWithinPackedHNormalBox_cpp(v1, v2));
    }

    {
        std::array<double, 8> v1 = {4, 19, 6, 2, 11, 3, 1, 4};
        std::array<double, 8> v2 = {10, 16, 18, 15, 12, 16, 4, 13};
        REQUIRE(0b1110 == PackedHNormalBoxWithinPackedHNormalBox_cpp(v1, v2));
    }
}

TEST_CASE("PackedNormalBox within PackedNormalBox (AVX)") {
    {
        std::array<double, 8> v1 = {9, 24, 5, 11, 14, 19, 10, 24};
        std::array<double, 8>  v2 = {17, 22, 15, 14, 19, 7, 23, 12};
        REQUIRE(PackedHNormalBoxWithinPackedHNormalBox_cpp(v1, v2) ==
                        PackedHNormalBoxWithinPackedHNormalBox_avx(v1, v2));
    }

    {
        std::array<double, 8>  v1 = {4, 19, 6, 2, 11, 3, 1, 4};
        std::array<double, 8>  v2 = {10, 16, 18, 15, 12, 16, 4, 13};
        REQUIRE(PackedHNormalBoxWithinPackedHNormalBox_cpp(v1, v2) ==
                        PackedHNormalBoxWithinPackedHNormalBox_avx(v1, v2));
    }
}
