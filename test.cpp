#include "PackedHNormalBox.h"
#include "DeepDistance.h"
#include "Distance.h"
#include "Within.h"

#include <catch2/catch_test_macros.hpp>

void CompareArrays(std::array<double, 4> a, std::array<double, 4> b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a[i] - b[i]) <= 1e-5);
    }
}

TEST_CASE("Random Distance") {
    {
        Point p = {-16, 92};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{1, 93}, NormalBox{72, 38}, NormalBox{26, 88}, NormalBox{16, 68}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {72, -10};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{83, 4}, NormalBox{32, 30}, NormalBox{56, 41}, NormalBox{75, 71}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-49, -45};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{37, 15}, NormalBox{96, 47}, NormalBox{53, 29}, NormalBox{62, 47}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-64, -69};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{11, 77}, NormalBox{10, 49}, NormalBox{58, 34}, NormalBox{74, 57}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-69, 27};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{83, 3}, NormalBox{81, 28}, NormalBox{82, 68}, NormalBox{92, 35}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-56, 39};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{30, 97}, NormalBox{57, 58}, NormalBox{86, 50}, NormalBox{15, 29}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-93, 96};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{29, 61}, NormalBox{39, 30}, NormalBox{3, 75}, NormalBox{45, 39}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-91, 99};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{91, 75}, NormalBox{40, 60}, NormalBox{9, 19}, NormalBox{41, 5}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-3, -88};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{35, 66}, NormalBox{20, 94}, NormalBox{67, 3}, NormalBox{27, 47}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-48, 97};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{26, 15}, NormalBox{31, 13}, NormalBox{62, 44}, NormalBox{86, 15}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {65, 55};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{35, 63}, NormalBox{95, 12}, NormalBox{30, 47}, NormalBox{95, 79}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {90, 46};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{81, 83}, NormalBox{12, 99}, NormalBox{11, 90}, NormalBox{80, 26}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {79, 91};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{12, 79}, NormalBox{96, 67}, NormalBox{55, 15}, NormalBox{37, 10}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-82, -3};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{56, 50}, NormalBox{50, 25}, NormalBox{18, 91}, NormalBox{4, 53}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-45, -40};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{47, 36}, NormalBox{50, 58}, NormalBox{93, 50}, NormalBox{81, 69}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {98, 84};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{77, 86}, NormalBox{57, 65}, NormalBox{37, 36}, NormalBox{75, 34}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-45, -19};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{21, 4}, NormalBox{38, 7}, NormalBox{58, 69}, NormalBox{5, 32}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-86, -74};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{23, 61}, NormalBox{34, 3}, NormalBox{78, 33}, NormalBox{4, 16}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-24, 86};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{50, 13}, NormalBox{70, 38}, NormalBox{45, 32}, NormalBox{78, 92}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {23, -86};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{29, 13}, NormalBox{64, 36}, NormalBox{21, 60}, NormalBox{26, 11}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {13, -67};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{78, 42}, NormalBox{11, 29}, NormalBox{58, 77}, NormalBox{34, 13}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-7, 82};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{75, 28}, NormalBox{56, 54}, NormalBox{40, 44}, NormalBox{11, 6}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {32, 96};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{71, 72}, NormalBox{26, 69}, NormalBox{21, 79}, NormalBox{51, 39}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-31, 76};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{27, 84}, NormalBox{25, 42}, NormalBox{63, 37}, NormalBox{43, 51}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-81, -18};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{12, 52}, NormalBox{59, 7}, NormalBox{1, 5}, NormalBox{15, 19}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {67, -68};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{42, 21}, NormalBox{95, 51}, NormalBox{42, 92}, NormalBox{80, 58}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-42, 56};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{62, 5}, NormalBox{15, 81}, NormalBox{55, 14}, NormalBox{43, 29}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {-3, -78};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{7, 78}, NormalBox{52, 63}, NormalBox{67, 33}, NormalBox{5, 23}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
    {
        Point p = {37, -98};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{86, 52}, NormalBox{5, 10}, NormalBox{74, 59}, NormalBox{29, 65}}};
        CompareArrays(SquaredDeepDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDeepDistancePacked_cpp(p, packed_h_normal_box));
    }
}

TEST_CASE("PackedNormalBox within PackedNormalBox") {
    {
        PackedHNormalBox packed_h_normal_box1 = {
                std::array<NormalBox, 4>{NormalBox{9, 24}, NormalBox{5, 11}, NormalBox{14, 19}, NormalBox{10, 24}}};
        PackedHNormalBox packed_h_normal_box2 = {
                std::array<NormalBox, 4>{NormalBox{17, 22}, NormalBox{15, 14}, NormalBox{19, 7}, NormalBox{23, 12}}};
        REQUIRE(0b0010 == WithinPacked_cpp(packed_h_normal_box1, packed_h_normal_box2));
    }

    {
        PackedHNormalBox packed_h_normal_box1 = {
                std::array<NormalBox, 4>{NormalBox{4, 19}, NormalBox{6, 2}, NormalBox{11, 3}, NormalBox{1, 4}}};
        PackedHNormalBox packed_h_normal_box2 = {
                std::array<NormalBox, 4>{NormalBox{10, 16}, NormalBox{18, 15}, NormalBox{12, 16}, NormalBox{4, 23}}};
        REQUIRE(0b1110 == WithinPacked_cpp(packed_h_normal_box1, packed_h_normal_box2));
    }
}

TEST_CASE("PackedNormalBox within PackedNormalBox (AVX)") {
    {
        PackedHNormalBox packed_h_normal_box1 = {
                std::array<NormalBox, 4>{NormalBox{9, 24}, NormalBox{5, 11}, NormalBox{14, 19}, NormalBox{10, 24}}};
        PackedHNormalBox packed_h_normal_box2 = {
                std::array<NormalBox, 4>{NormalBox{17, 22}, NormalBox{15, 14}, NormalBox{19, 7}, NormalBox{23, 12}}};
        REQUIRE(WithinPacked_avx(packed_h_normal_box1, packed_h_normal_box2) ==
                WithinPacked_cpp(packed_h_normal_box1, packed_h_normal_box2));
    }

    {
        PackedHNormalBox packed_h_normal_box1 = {
                std::array<NormalBox, 4>{NormalBox{4, 19}, NormalBox{6, 2}, NormalBox{11, 3}, NormalBox{1, 4}}};
        PackedHNormalBox packed_h_normal_box2 = {
                std::array<NormalBox, 4>{NormalBox{10, 16}, NormalBox{18, 15}, NormalBox{12, 16}, NormalBox{4, 23}}};
        REQUIRE(WithinPacked_avx(packed_h_normal_box1, packed_h_normal_box2) ==
                WithinPacked_cpp(packed_h_normal_box1, packed_h_normal_box2));
    }
}

TEST_CASE("Distance - zeros if inside") {
    {
        Point p = {5, 5};
        PackedHNormalBox packed_h_normal_box = {
                std::array<NormalBox, 4>{NormalBox{1, 1}, NormalBox{1, 10}, NormalBox{10, 1}, NormalBox{10, 10}}};
        CompareArrays(SquaredDistancePacked_avx(p, packed_h_normal_box),
                      SquaredDistancePacked_cpp(p, packed_h_normal_box));
    }
}

