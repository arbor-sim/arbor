#include "../gtest.h"

#include <vector>
#include <list>
#include <utility>

#include "util/unique.hpp"
#include "./common.hpp"

namespace {
auto same_parity = [](auto a, auto b) { return a%2 == b%2; };

template <typename C, typename Eq = std::equal_to<>>
void run_unique_in_place_test(C data, const C& expected, Eq eq = Eq{}) {
    arb::util::unique_in_place(data, eq);
    EXPECT_TRUE(testing::seq_eq(data, expected));
}

template <typename C>
void run_tests() {
    run_unique_in_place_test(C{}, C{});
    run_unique_in_place_test(C{1, 3, 2}, C{1, 3, 2});
    run_unique_in_place_test(C{1, 1, 1, 3, 2}, C{1, 3, 2});
    run_unique_in_place_test(C{1, 3, 3, 3, 2}, C{1, 3, 2});
    run_unique_in_place_test(C{1, 3, 2, 2, 2}, C{1, 3, 2});
    run_unique_in_place_test(C{1, 1, 3, 2, 2}, C{1, 3, 2});
    run_unique_in_place_test(C{3, 1, 3, 1, 1, 3, 1, 1}, C{3, 1, 3, 1, 3, 1});
    run_unique_in_place_test(C{1, 2, 4, 1, 3, 1, 2, 1}, C{1, 2, 1, 2, 1}, same_parity);
}
}

TEST(unique_in_place, vector) {
    run_tests<std::vector<int>>();
}

TEST(unique_in_place, list) {
    run_tests<std::list<int>>();
}

