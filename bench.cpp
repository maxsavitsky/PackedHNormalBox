#include "PackedHNormalBox.h"
#include <immintrin.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>


TEST_CASE("Bench") {
    BENCHMARK_ADVANCED("Bench Distance, CPP")(Catch::Benchmark::Chronometer meter) {
        std::random_device gen;
        std::mt19937 rnd(gen());
        std::uniform_int_distribution dist{-100, 100};

        struct TestCase {
            double point_x;
            double point_y;
            std::array<double, 8> boxes;
        };

        std::vector<TestCase> cases(meter.runs());

        for (int i = 0; i < meter.runs(); i++) {
            cases[i].point_x = dist(rnd);
            cases[i].point_y = dist(rnd);
            for (int j = 0; j < 8; j++) {
                cases[i].boxes[j] = static_cast<double>(dist(rnd));
            }
        }

        meter.measure([&](int i){
            TestCase& test_case = cases[i];
            return DistancePtoPackedHNB_cpp(test_case.point_x, test_case.point_y, test_case.boxes);
        });
    };
}