#include "gtest.h"
#include "util/nop.hpp"

using namespace nest::mc::util;

TEST(nop, void_fn) {
    std::function<void ()> f(nop_function);

    EXPECT_TRUE(f);
    f(); // should do nothing

    bool flag = false;
    f = [&]() { flag = true; };
    f();
    EXPECT_TRUE(flag);

    flag = false;
    f = nop_function;
    f();
    EXPECT_FALSE(flag);

    // with some arguments
    std::function<void (int, int)> g(nop_function);
    EXPECT_TRUE(g);
    g(2, 3); // should do nothing

    int sum = 0;
    g = [&](int a, int b) { sum = a+b; };
    g(2, 3);
    EXPECT_EQ(5, sum);

    sum = 0;
    g = nop_function;
    g(2, 3);
    EXPECT_EQ(0, sum);
}

struct check_default {
    int value = 100;

    check_default() = default;
    explicit check_default(int n): value(n) {}
};

TEST(nop, default_return_fn) {
    std::function<check_default ()> f(nop_function);

    EXPECT_TRUE(f);
    auto result = f();
    EXPECT_EQ(result.value, 100);

    f = []() { return check_default(17); };
    result = f();
    EXPECT_EQ(result.value, 17);

    f = nop_function;
    result = f();
    EXPECT_EQ(result.value, 100);

    std::function<check_default (double, double)> g(nop_function);

    EXPECT_TRUE(g);
    result = g(1.4, 1.5);
    EXPECT_EQ(result.value, 100);

    g = [](double x, double y) { return check_default{(int)(x*y)}; };
    result = g(1.4, 1.5);
    EXPECT_EQ(result.value, 2);

    g = nop_function;
    result = g(1.4, 1.5);
    EXPECT_EQ(result.value, 100);

}

