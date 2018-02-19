#include <util/simd.hpp>
#include <util/simd/avx.hpp>

#include "common.hpp"

using namespace arb;

template <typename S>
struct simdfp: public ::testing::Test {};

TYPED_TEST_CASE_P(simdfp);

// Initialization and element access.

TYPED_TEST_P(simdfp, elements) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    // broadcast:
    simdfp a(2);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(2., a[i]);
    }

    // array initialization:
    fp bv[N];
    for (unsigned i = 0; i<N; ++i) {
        bv[i] = 1.5+i;
    }
    simdfp b(bv);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(1.5+i, b[i]);
    }

    // array rvalue initialization:
    fp cv_rvalue[N];
    for (unsigned i = 0; i<N; ++i) {
        cv_rvalue[i] = 3.5-i;
    }
    simdfp c(std::move(cv_rvalue));
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3.5-i, c[i]);
    }

    // std::array initialization:
    std::array<fp, N> dv;
    for (unsigned i = 0; i<N; ++i) {
        dv[i] = 6.5+i;
    }
    simdfp d(dv);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(6.5+i, d[i]);
    }

    // pointer initialization:
    fp ev[N];
    for (unsigned i = 0; i<N; ++i) {
        ev[i] = 7.5-i;
    }
    simdfp e(&ev[0]);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(7.5-i, e[i]);
    }
}

REGISTER_TYPED_TEST_CASE_P(simdfp, elements);

typedef ::testing::Types<
    simd<float, 2, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
#ifdef __AVX__
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<double, 4, simd_abi::avx2>,
#endif
    simd<double, 4, simd_abi::default_abi>
> simdfp_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(simdfp_tests, simdfp, simdfp_test_types);

