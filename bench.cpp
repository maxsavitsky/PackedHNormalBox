#include "PackedHNormalBox.h"
#include "DeepDistance.h"
#include "Distance.h"
#include "Within.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

const uint32_t kTestCasesCount = 128; // should be power of 2 for performance
constexpr uint32_t kMod = kTestCasesCount - 1;

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
            cases[i].box.boxes.at(j) = NormalBox{static_cast<double>(std::abs(dist(rnd))),
                                                 static_cast<double>(std::abs(dist(rnd)))};
        }
    }

    return cases;
}


TEST_CASE("Bench Deep Distance") {
    BENCHMARK_ADVANCED("Bench Deep Distance, CPP")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(kTestCasesCount);

            meter.measure([&](int i) {
                TestCase &test_case = cases[i & kMod];
                return SquaredDeepDistancePacked_cpp(test_case.point, test_case.box);
            });
        };

    BENCHMARK_ADVANCED("Bench Deep Distance, AVX512")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(kTestCasesCount);

            meter.measure([&](int i) {
                TestCase &test_case = cases[i & kMod];
                return SquaredDeepDistancePacked_avx(test_case.point, test_case.box);
            });
        };
}

TEST_CASE("Bench Distance") {
    BENCHMARK_ADVANCED("Bench Distance, CPP")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(kTestCasesCount);

            meter.measure([&](int i) {
                TestCase &test_case = cases[i & kMod];
                return SquaredDistancePacked_cpp(test_case.point, test_case.box);
            });
        };
    BENCHMARK_ADVANCED("Bench Distance Old, AVX512")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(kTestCasesCount);

            meter.measure([&](int i) {
                TestCase &test_case = cases[i & kMod];
                return SquaredDistancePackedOld_avx(test_case.point, test_case.box);
            });
        };
    BENCHMARK_ADVANCED("Bench Distance, AVX512")(Catch::Benchmark::Chronometer meter) {
            auto cases = GenerateTestCases(kTestCasesCount);

            meter.measure([&](int i) {
                TestCase &test_case = cases[i & kMod];
                return SquaredDistancePacked_avx(test_case.point, test_case.box);
            });
        };
}

TEST_CASE("Bench Within") {
    const auto cases1 = GenerateTestCases(kTestCasesCount);
    const auto cases2 = GenerateTestCases(kTestCasesCount);
    BENCHMARK("Bench Within, CPP", i) {
        const TestCase &case1 = cases1[i & kMod];
        const TestCase &case2 = cases2[i & kMod];
        return WithinPacked_cpp(case1.box, case2.box);
    };

    BENCHMARK("Bench Within, AVX512", i) {
        const TestCase &case1 = cases1[i & kMod];
        const TestCase &case2 = cases2[i & kMod];
        return WithinPacked_avx(case1.box, case2.box);
    };
}