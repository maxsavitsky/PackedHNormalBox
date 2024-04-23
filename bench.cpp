#include "PackedHNormalBox.h"
#include "DeepDistance.h"
#include "Distance.h"
#include "Within.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
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
            return SquaredDeepDistancePacked_cpp(test_case.point, test_case.box);
        });
    };
}

TEST_CASE("Bench AVX"){
    BENCHMARK_ADVANCED("Bench Distance, AVX")(Catch::Benchmark::Chronometer meter) {
        auto cases = GenerateTestCases(meter.runs());

        meter.measure([&](int i){
            TestCase& test_case = cases[i];
            return SquaredDeepDistancePacked_avx(test_case.point, test_case.box);
        });
    };
}

TEST_CASE("Bench_2 CPP") {
    BENCHMARK_ADVANCED("Bench Distance2, CPP ")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(meter.runs());

            meter.measure([&](int i){
                TestCase& test_case = cases[i];
                return SquaredDistancePacked_cpp(test_case.point, test_case.box);
            });
        };
}

TEST_CASE("Bench_2 AVX"){
    BENCHMARK_ADVANCED("Bench Distance2, AVX")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(meter.runs());

            meter.measure([&](int i){
                TestCase& test_case = cases[i];
                return SquaredDistancePacked_avx(test_case.point, test_case.box);
            });
        };
}