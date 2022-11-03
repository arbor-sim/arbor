#include <any>
#include <type_traits>
#include <typeinfo>

#include <arbor/util/any_cast.hpp>
#include <arbor/util/any_ptr.hpp>

#include <gtest/gtest.h>

using namespace arb;
using util::any_cast;
using util::any_ptr;

TEST(any_ptr, ctor_and_assign) {
    using util::any_ptr;

    any_ptr p;

    EXPECT_FALSE(p);
    EXPECT_FALSE(p.has_value());

    int x;
    any_ptr q(&x);

    EXPECT_TRUE(q);
    EXPECT_TRUE(q.has_value());

    p = q;

    EXPECT_TRUE(p);
    EXPECT_TRUE(p.has_value());

    p = nullptr;

    EXPECT_FALSE(p);
    EXPECT_FALSE(p.has_value());

    p = &x;

    EXPECT_TRUE(p);
    EXPECT_TRUE(p.has_value());

    p.reset();

    EXPECT_FALSE(p);
    EXPECT_FALSE(p.has_value());

    p.reset(&x);

    EXPECT_TRUE(p);
    EXPECT_TRUE(p.has_value());

    p.reset(nullptr);

    EXPECT_FALSE(p);
    EXPECT_FALSE(p.has_value());

    p = nullptr;

    EXPECT_FALSE(p);
    EXPECT_FALSE(p.has_value());
}

TEST(any_ptr, ordering) {
    int x[2];
    double y;

    any_ptr a(&x[0]);
    any_ptr b(&x[1]);

    EXPECT_LT(a, b);
    EXPECT_LE(a, b);
    EXPECT_NE(a, b);
    EXPECT_GE(b, a);
    EXPECT_GT(b, a);
    EXPECT_FALSE(a==b);

    any_ptr c(&y);

    EXPECT_NE(c, a);
    EXPECT_TRUE(a<c || a>c);
    EXPECT_FALSE(a==c);
}

TEST(any_ptr, as_and_type) {
    int x = 0;
    const int y = 0;
    any_ptr p;

    EXPECT_FALSE(p.as<int*>());

    p = &y;
    EXPECT_FALSE(p.as<int*>());
    EXPECT_TRUE(p.as<const int*>());
    EXPECT_EQ(typeid(const int*), p.type());

    p = &x;
    EXPECT_TRUE(p.as<int*>());
    EXPECT_FALSE(p.as<const int*>());
    EXPECT_EQ(typeid(int*), p.type());

    *p.as<int*>() = 3;
    EXPECT_EQ(3, x);
}

TEST(any_ptr, any_cast) {
    int x = 0;
    any_ptr p;

    auto c1 = any_cast<int*>(p);
    EXPECT_FALSE(c1);
    EXPECT_TRUE((std::is_same_v<int*, decltype(c1)>));

    p = &x;
    auto c2 = any_cast<int*>(p);
    EXPECT_TRUE(c2);

    auto c3 = any_cast<double*>(p);
    EXPECT_FALSE(c3);

    // Might want to reconsider these semantics, but here we are:
    auto c4 = any_cast<const int*>(p);
    EXPECT_FALSE(c4);

    p = (const int*)&x;
    auto c5 = any_cast<int*>(p);
    EXPECT_FALSE(c5);
}

