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

        double point_x = dist(rnd);
        double point_y = dist(rnd);
        std::array<double, 8> boxes = {};
        for (int i = 0; i < 8; i++) {
            boxes[i] = static_cast<double>(dist(rnd));
        }
        meter.measure([&](){
            return DistancePtoPackedHNB_cpp(point_x, point_y, boxes);
        });
    };
}