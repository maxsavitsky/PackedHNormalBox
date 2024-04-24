#include <array>
#include <stdio.h>

struct Point {
    double x;
    double y;
};

struct NormalBox {
    Point corner;
};

struct alignas(64) PackedHNormalBox {
    std::array<NormalBox, 4> boxes;
};