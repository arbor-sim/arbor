#include <array>
#include <gtest/gtest.h>
#include "backends/rand_impl.hpp"

bool verify_uniform(arb::cbprng::array_type counter, arb::cbprng::array_type key,
    std::array<arb::cbprng::value_type, 4> expected) {
    const auto r = arb::cbprng::generator{}(counter,key);
    return 
        (r[0] == expected[0]) &&
        (r[1] == expected[1]) &&
        (r[2] == expected[2]) &&
        (r[3] == expected[3]);
}

bool verify_normal(std::array<arb::cbprng::value_type, 4> uniform, std::array<long long, 4> expected) {
    const auto [a0, a1] = r123::boxmuller(uniform[0], uniform[1]);
    const auto [a2, a3] = r123::boxmuller(uniform[2], uniform[3]);
    return 
        ((long long)(a0*1000000) == expected[0]) &&
        ((long long)(a1*1000000) == expected[1]) &&
        ((long long)(a2*1000000) == expected[2]) &&
        ((long long)(a3*1000000) == expected[3]);
}

TEST(cbprng, uniform) {
    EXPECT_TRUE(verify_uniform(
        {0ull, 0ull, 0ull, 0ull},
        {0ull, 0ull, 0ull, 0ull}, 
        {29492327419918145ull, 4614276061115832004ull, 16925429801461668750ull, 5660986226915721659ull}));

    EXPECT_TRUE(verify_uniform(
        {1ull, 2ull, 3ull, 4ull},
        {5ull, 6ull, 7ull, 8ull}, 
        {1161205111990270403ull, 7490910075796229492ull, 3916163298891572586ull, 624723054006169054ull}));

    EXPECT_TRUE(verify_uniform(
        {6666ull, 77ull, 500ull, 8363839ull},
        {999ull, 137ull, 0xdeadf00dull, 0xdeadbeefull},
        {3717439728375325370ull, 14259638735392226729ull, 6108569366204981687ull, 8675995625794245694ull}));
}

TEST(cbprng, normal) {
    EXPECT_TRUE(verify_normal(
        {29492327419918145ull, 4614276061115832004ull, 16925429801461668750ull, 5660986226915721659ull},
        {16723ll, 1664687ll, -761307ll, 1335286ll}));

    EXPECT_TRUE(verify_normal(
        {1161205111990270403ull, 7490910075796229492ull, 3916163298891572586ull, 624723054006169054ull},
        {517262ll, 1238884ll, 2529374ll, 610685ll}));

    EXPECT_TRUE(verify_normal(
        {3717439728375325370ull, 14259638735392226729ull, 6108569366204981687ull, 8675995625794245694ull},
        {684541ll, 215202ll, 1072054ll, -599461ll}));
}
