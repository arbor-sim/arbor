#include <gtest/gtest.h>

#include <string>

#include <arbor/util/hash_def.hpp>

TEST(hash, string_eq) {
    ASSERT_EQ(arb::hash_value("foobar"), arb::hash_value(std::string{"foobar"}));
    ASSERT_EQ(arb::hash_value("foobar"), arb::hash_value("foobar"));
    ASSERT_NE(arb::hash_value("foobar"), arb::hash_value("barfoo"));
}

TEST(hash, doesnt_compile) {
    double foo = 42;
    // Sadly we cannot check static assertions... this shoudln't compile
    // EXPECT_ANY_THROW(arb::hash_value(&foo));
    // this should
    arb::hash_value((void*) &foo);
}

// check that we do not fall into the trap of the STL...
TEST(hash, integral_is_not_identity) {
    ASSERT_NE(arb::hash_value(42), 42);
}
