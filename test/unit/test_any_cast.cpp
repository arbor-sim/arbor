#include <any>
#include <type_traits>

#include <arbor/util/any_cast.hpp>

#include <gtest/gtest.h>

using namespace arb;

// Check std::any_cast follows std::any_cast semantics
// when applied to std::any objects.

#define ASSERT_ANY_CAST_SAME_TYPE(type, value)\
ASSERT_TRUE((std::is_same_v<decltype(std::any_cast<type>(value)), decltype(util::any_cast<type>(value))>))

TEST(any_cast, pointer) {
    std::any ai(42);

    ASSERT_ANY_CAST_SAME_TYPE(int, &ai);
    ASSERT_TRUE(util::any_cast<int>(&ai));
    EXPECT_EQ(42, *util::any_cast<int>(&ai));

    ASSERT_ANY_CAST_SAME_TYPE(const int, &ai);
    ASSERT_TRUE(util::any_cast<const int>(&ai));
    EXPECT_EQ(42, *util::any_cast<const int>(&ai));

    const auto& cai = ai;
    ASSERT_ANY_CAST_SAME_TYPE(int, &cai);
    ASSERT_TRUE(util::any_cast<int>(&cai));
    EXPECT_EQ(42, *util::any_cast<int>(&cai));

    // Assert type mismatch or empty any gives nullptr.

    std::any empty;
    EXPECT_EQ(util::any_cast<int>(&empty), nullptr);
    EXPECT_EQ(util::any_cast<float>(&ai), nullptr);
    EXPECT_EQ(util::any_cast<int>((std::any*)nullptr), nullptr);
}

TEST(any_cast, ref_and_value) {
    std::any ai(42);

    ASSERT_ANY_CAST_SAME_TYPE(int, ai);
    ASSERT_ANY_CAST_SAME_TYPE(int&, ai);
    ASSERT_ANY_CAST_SAME_TYPE(const int&, ai);
    ASSERT_ANY_CAST_SAME_TYPE(int&&, std::move(ai));

    EXPECT_EQ(42, util::any_cast<int>(ai));
    EXPECT_EQ(42, util::any_cast<int&>(ai));
    EXPECT_EQ(42, util::any_cast<const int&>(ai));
    EXPECT_EQ(42, util::any_cast<int&&>(std::move(ai)));
}
