load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "PackedHNormalBox",
    hdrs = ["PackedHNormalBox.h"],
)

cc_test(
    name = "benchmark_main",
    # size = "small",
    srcs = ["bench.cpp"],
    deps = [
        ":PackedHNormalBox",
        "//tools/bazel:catch2",
    ],
    linkstatic=True,
    copts = [
        "-m64",
        #"-mavx512f",
        #"-mavx512dq",
        #"-mprefer-vector-width=512",
        #"-masm=intel",
        #"-fverbose-asm",
        #"-O3",
        #"-march=skylake-avx512"
    ],
)

cc_test(
    name = "test_PackedHNormalBox",
    # size = "small",
    srcs = ["test.cpp"],
    deps = [
        ":PackedHNormalBox",
        "//tools/bazel:catch2",
    ],
    linkstatic=True,
    copts = [
        "-m64",
        #"-mavx512f",
        #"-mavx512dq",
        "-mprefer-vector-width=512",
        #"-masm=intel",
        #"-fverbose-asm",
        "-03",
    ],
)