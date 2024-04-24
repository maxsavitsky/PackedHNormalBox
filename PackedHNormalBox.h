#include <array>
#include <cmath>
#include <stdio.h>

struct alignas(64) Point {
    double x;
    double y;
};

struct alignas(64) NormalBox {
    Point corner;
};

struct alignas(64) PackedHNormalBox {
    std::array<NormalBox, 4> boxes;
};