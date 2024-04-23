#include <array>

struct alignas(64) Point {
    double x;
    double y;
};

struct alignas(64) NormalBox {
    Point corner;
};

struct PackedHNormalBox {
    std::array<NormalBox, 4> boxes;
};