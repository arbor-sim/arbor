#include <functional>

#include "../gtest.h"

#include <arbor/util/scope_exit.hpp>

using arb::util::on_scope_exit;

TEST(scope_exit, basic) {
    bool a = false;
    {
        auto guard = on_scope_exit([&a] { a = true; });
        EXPECT_FALSE(a);
    }
    EXPECT_TRUE(a);
}

TEST(scope_exit, noexceptcall) {
    auto guard1 = on_scope_exit([] {});
    using G1 = decltype(guard1);
    EXPECT_FALSE(noexcept(guard1.~G1()));

    auto guard2 = on_scope_exit([]() noexcept {});
    using G2 = decltype(guard2);
    EXPECT_TRUE(noexcept(guard2.~G2()));
}

TEST(scope_exit, function) {
    // on_scope_exit has a special overload for std::function
    // to work around its non-noexcept move ctor.
    bool a = false;
    std::function<void ()> setter = [&a] { a = true; };

    {
        auto guard = on_scope_exit(setter);
        EXPECT_FALSE(a);
    }
    EXPECT_TRUE(a);

    a = false;
    std::function<int ()> setter2 = [&a] { a = true; return 3; };

    {
        auto guard = on_scope_exit(setter2);
        EXPECT_FALSE(a);
    }
    EXPECT_TRUE(a);
}

