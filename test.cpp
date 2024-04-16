#include "PackedHNormalBox.h"
#include <immintrin.h>
#include <iostream>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

TEST_CASE("Distance from Point to NormalBox") {
    REQUIRE(DistancePtoNB_cpp(2,0,1,1) == 1);
    REQUIRE(DistancePtoNB_cpp(0,2,1,1) == 1);

    REQUIRE(DistancePtoNB_cpp(2,2,1,1) == sqrt(2));
    REQUIRE(DistancePtoNB_cpp(0,0,1,1) == 1);

    REQUIRE(DistancePtoNB_cpp(-2,0,1,1) == 1);
    REQUIRE(DistancePtoNB_cpp(0,2,1,1) == 1);

    REQUIRE(DistancePtoNB_cpp(2,-2,1,1) == sqrt(2));
}

void CompareArrays(std::array<double, 4> a, std::array<double, 4> b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a[i] - b[i]) <= 1e-5);
    }
}

TEST_CASE("Distance, CPP") {
    std::array<double, 8> v = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0};
    CompareArrays(DistancePtoPackedHNB_cpp(2.0, 2.0, v), {sqrt(2), 0.0, 1.0, 1.0});
}

TEST_CASE("Distance, AVX") {
    std::array<double, 8> v = {1.0, 20.0, 1.0, 1.0, 10.0, 20.0, 10.0, 2.0};
    CompareArrays(DistancePtoPackedHNB_cpp(3.0, 3.0, v), DistancePtoPackedHNB_avx(3.0, 3.0, v));
}

TEST_CASE("Random Distance, AVX") {
    {
        std::array<double, 8> arr = {1, 93, 72, 38, 26, 88, 16, 68};
        std::array<double, 4> result = {91, 20, 66, 76};
        CompareArrays(DistancePtoPackedHNB_avx(-16, 92, arr), result);
    }
    {
        std::array<double, 8> arr = {83, 4, 32, 30, 56, 41, 75, 71};
        std::array<double, 4> result = {68, 42, 31, 1};
        CompareArrays(DistancePtoPackedHNB_avx(72, -10, arr), result);
    }
    {
        std::array<double, 8> arr = {37, 15, 96, 47, 53, 29, 62, 47};
        std::array<double, 4> result = {34.928498, 2, 20, 2};
        CompareArrays(DistancePtoPackedHNB_avx(-49, -45, arr), result);
    }
    {
        std::array<double, 8> arr = {11, 77, 10, 49, 58, 34, 74, 57};
        std::array<double, 4> result = {58, 60.876925, 31.953091, 7};
        CompareArrays(DistancePtoPackedHNB_avx(-64, -69, arr), result);
    }
    {
        std::array<double, 8> arr = {83, 3, 81, 28, 82, 68, 92, 35};
        std::array<double, 4> result = {66, 41, 1, 34};
        CompareArrays(DistancePtoPackedHNB_avx(-69, 27, arr), result);
    }
    {
        std::array<double, 8> arr = {30, 97, 57, 58, 86, 50, 15, 29};
        std::array<double, 4> result = {9, 2, 6, 36.124784};
        CompareArrays(DistancePtoPackedHNB_avx(-56, 39, arr), result);
    }
    {
        std::array<double, 8> arr = {29, 61, 39, 30, 3, 75, 45, 39};
        std::array<double, 4> result = {74.249579, 84.958814, 94.72592, 74.27651};
        CompareArrays(DistancePtoPackedHNB_avx(-93, 96, arr), result);
    }
    {
        std::array<double, 8> arr = {91, 75, 40, 60, 9, 19, 41, 5};
        std::array<double, 4> result = {17.888544, 66.648331, 115.256236, 103.73042};
        CompareArrays(DistancePtoPackedHNB_avx(-91, 99, arr), result);
    }
    {
        std::array<double, 8> arr = {35, 66, 20, 94, 67, 3, 27, 47};
        std::array<double, 4> result = {53, 68, 21, 61};
        CompareArrays(DistancePtoPackedHNB_avx(-3, -88, arr), result);
    }
    {
        std::array<double, 8> arr = {26, 15, 31, 13, 62, 44, 86, 15};
        std::array<double, 4> result = {78.294317, 74.706091, 35.22783, 34.785054};
        CompareArrays(DistancePtoPackedHNB_avx(-48, 97, arr), result);
    }
    {
        std::array<double, 8> arr = {35, 63, 95, 12, 30, 47, 95, 79};
        std::array<double, 4> result = {20.099751, 53, 30.805844, 14};
        CompareArrays(DistancePtoPackedHNB_avx(65, 55, arr), result);
    }
    {
        std::array<double, 8> arr = {81, 83, 12, 99, 11, 90, 80, 26};
        std::array<double, 4> result = {7, 34, 35, 64};
        CompareArrays(DistancePtoPackedHNB_avx(90, 46, arr), result);
    }
    {
        std::array<double, 8> arr = {12, 79, 96, 67, 55, 15, 37, 10};
        std::array<double, 4> result = {79, 12, 73.430239, 87.618491};
        CompareArrays(DistancePtoPackedHNB_avx(79, 91, arr), result);
    }
    {
        std::array<double, 8> arr = {56, 50, 50, 25, 18, 91, 4, 53};
        std::array<double, 4> result = {32, 57, 9, 29};
        CompareArrays(DistancePtoPackedHNB_avx(-82, -3, arr), result);
    }
    {
        std::array<double, 8> arr = {47, 36, 50, 58, 93, 50, 81, 69};
        std::array<double, 4> result = {9, 10, 5, 24};
        CompareArrays(DistancePtoPackedHNB_avx(-45, -40, arr), result);
    }
    {
        std::array<double, 8> arr = {77, 86, 57, 65, 37, 36, 75, 34};
        std::array<double, 4> result = {13.892444, 42.638011, 77.801028, 64.629715};
        CompareArrays(DistancePtoPackedHNB_avx(98, 84, arr), result);
    }
    {
        std::array<double, 8> arr = {21, 4, 38, 7, 58, 69, 5, 32};
        std::array<double, 4> result = {41, 38, 24, 19.104973};
        CompareArrays(DistancePtoPackedHNB_avx(-45, -19, arr), result);
    }
    {
        std::array<double, 8> arr = {23, 61, 34, 3, 78, 33, 4, 16};
        std::array<double, 4> result = {56.797887, 92.135769, 53, 98.994949};
        CompareArrays(DistancePtoPackedHNB_avx(-86, -74, arr), result);
    }
    {
        std::array<double, 8> arr = {50, 13, 70, 38, 45, 32, 78, 92};
        std::array<double, 4> result = {37.64306, 16, 41, 8};
        CompareArrays(DistancePtoPackedHNB_avx(-24, 86, arr), result);
    }
    {
        std::array<double, 8> arr = {29, 13, 64, 36, 21, 60, 26, 11};
        std::array<double, 4> result = {57.870545, 22, 65, 61.188234};
        CompareArrays(DistancePtoPackedHNB_avx(23, -86, arr), result);
    }
    {
        std::array<double, 8> arr = {78, 42, 11, 29, 58, 77, 34, 13};
        std::array<double, 4> result = {11, 56, 9, 33};
        CompareArrays(DistancePtoPackedHNB_avx(13, -67, arr), result);
    }
    {
        std::array<double, 8> arr = {75, 28, 56, 54, 40, 44, 11, 6};
        std::array<double, 4> result = {7, 26, 42, 71.007042};
        CompareArrays(DistancePtoPackedHNB_avx(-7, 82, arr), result);
    }
    {
        std::array<double, 8> arr = {71, 72, 26, 69, 21, 79, 51, 39};
        std::array<double, 4> result = {25, 70, 75, 45};
        CompareArrays(DistancePtoPackedHNB_avx(32, 96, arr), result);
    }
    {
        std::array<double, 8> arr = {27, 84, 25, 42, 63, 37, 43, 51};
        std::array<double, 4> result = {49, 51, 13, 33};
        CompareArrays(DistancePtoPackedHNB_avx(-31, 76, arr), result);
    }
    {
        std::array<double, 8> arr = {12, 52, 59, 7, 1, 5, 15, 19};
        std::array<double, 4> result = {29.614186, 74, 77.87811, 62.072538};
        CompareArrays(DistancePtoPackedHNB_avx(-81, -18, arr), result);
    }
    {
        std::array<double, 8> arr = {42, 21, 95, 51, 42, 92, 80, 58};
        std::array<double, 4> result = {52.839379, 16, 26, 9};
        CompareArrays(DistancePtoPackedHNB_avx(67, -68, arr), result);
    }
    {
        std::array<double, 8> arr = {62, 5, 15, 81, 55, 14, 43, 29};
        std::array<double, 4> result = {37, 41, 28.017851, 18.384776};
        CompareArrays(DistancePtoPackedHNB_avx(-42, 56, arr), result);
    }
    {
        std::array<double, 8> arr = {7, 78, 52, 63, 67, 33, 5, 23};
        std::array<double, 4> result = {71, 26, 11, 73};
        CompareArrays(DistancePtoPackedHNB_avx(-3, -78, arr), result);
    }
    {
        std::array<double, 8> arr = {86, 52, 5, 10, 74, 59, 29, 65};
        std::array<double, 4> result = {12, 96.840074, 24, 69};
        CompareArrays(DistancePtoPackedHNB_avx(37, -98, arr), result);
    }
}

TEST_CASE("PackedNormalBox within PackedNormalBox") {
    {
        std::array<double, 8> v1 = {9, 24, 5, 11, 14, 19, 10, 24};
        std::array<double, 8> v2 = {17, 22, 15, 14, 19, 7, 23, 12};
        REQUIRE(0b0010 == PackedHNBWithinPackedHNB_cpp(v1, v2));
    }

    {
        std::array<double, 8> v1 = {4, 19, 6, 2, 11, 3, 1, 4};
        std::array<double, 8> v2 = {10, 16, 18, 15, 12, 16, 4, 13};
        REQUIRE(0b1110 == PackedHNBWithinPackedHNB_cpp(v1, v2));
    }
}

TEST_CASE("PackedNormalBox within PackedNormalBox (AVX)") {
    {
        std::array<double, 8> v1 = {9, 24, 5, 11, 14, 19, 10, 24};
        std::array<double, 8>  v2 = {17, 22, 15, 14, 19, 7, 23, 12};
        REQUIRE(PackedHNBWithinPackedHNB_cpp(v1, v2) == PackedHNBWithinPackedHNB_avx(v1, v2));
    }

    {
        std::array<double, 8>  v1 = {4, 19, 6, 2, 11, 3, 1, 4};
        std::array<double, 8>  v2 = {10, 16, 18, 15, 12, 16, 4, 13};
        REQUIRE(PackedHNBWithinPackedHNB_cpp(v1, v2) == PackedHNBWithinPackedHNB_avx(v1, v2));
    }
}