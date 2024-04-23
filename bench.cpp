#include "PackedHNormalBox.h"
#include <immintrin.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>

struct TestCase {
    Point point;
    PackedHNormalBox box;
};

std::vector<TestCase> GenerateTestCases(int n) {
    std::random_device gen;
    std::mt19937 rnd(gen());
    std::uniform_int_distribution dist{-100, 100};

    std::vector<TestCase> cases(n);

    for (int i = 0; i < n; i++) {
        cases[i].point = {static_cast<double>(dist(rnd)), static_cast<double>(dist(rnd))};
        for (int j = 0; j < 4; j++) {
            cases[i].box.boxes[j] = NormalBox{static_cast<double>(dist(rnd)), static_cast<double>(dist(rnd))};
        }
    }

    return cases;
}


TEST_CASE("Bench CPP") {
    BENCHMARK_ADVANCED("Bench Distance, CPP")(Catch::Benchmark::Chronometer meter) {
        auto cases = GenerateTestCases(meter.runs());

        meter.measure([&](int i){
            TestCase& test_case = cases[i];
            return SquaredDistancePointToPackedHNormalBox_cpp(test_case.point, test_case.box);
        });
    };
}

TEST_CASE("Bench AVX"){
    BENCHMARK_ADVANCED("Bench Distance, AVX")(Catch::Benchmark::Chronometer meter) {
        auto cases = GenerateTestCases(meter.runs());

        meter.measure([&](int i){
            TestCase& test_case = cases[i];
            return SquaredDistancePointToPackedHNormalBox_avx(test_case.point, test_case.box);
        });
    };
}

TEST_CASE("Bench CPP_2") {
    BENCHMARK_ADVANCED("Bench Distance, CPP 2")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(meter.runs());

            meter.measure([&](int i){
                TestCase& test_case = cases[i];
                return SquaredDistancePointToPackedHNormalBox2_cpp(test_case.point, test_case.box);
            });
        };
}

TEST_CASE("Bench AVX_2"){
    BENCHMARK_ADVANCED("Bench Distance, AVX 2")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(meter.runs());

            meter.measure([&](int i){
                TestCase& test_case = cases[i];
                return SquaredDistancePointToPackedHNormalBox2_avx(test_case.point, test_case.box);
            });
        };
}