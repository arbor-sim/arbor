#include <array>
#include <gtest/gtest.h>
#include "backends/rand_impl.hpp"

bool verify(arb::cbprng::array_type counter, arb::cbprng::array_type key,
    std::array<arb::cbprng::value_type, 4> expected) {
    const auto r = arb::cbprng::generator{}(counter,key);
    return 
        (r[0] == expected[0]) &&
        (r[1] == expected[1]) &&
        (r[2] == expected[2]) &&
        (r[3] == expected[3]);
}

TEST(cbprng, reproducibility) {
    EXPECT_TRUE(verify(
        {0ull, 0ull, 0ull, 0ull},
        {0ull, 0ull, 0ull, 0ull}, 
        {29492327419918145ull, 4614276061115832004ull, 16925429801461668750ull, 5660986226915721659ull}));

    EXPECT_TRUE(verify(
        {1ull, 2ull, 3ull, 4ull},
        {5ull, 6ull, 7ull, 8ull}, 
        {1161205111990270403ull, 7490910075796229492ull, 3916163298891572586ull, 624723054006169054ull}));

    EXPECT_TRUE(verify(
        {6666ull, 77ull, 500ull, 8363839ull},
        {999ull, 137ull, 0xdeadf00dull, 0xdeadbeefull},
        {3717439728375325370ull, 14259638735392226729ull, 6108569366204981687ull, 8675995625794245694ull}));
}
